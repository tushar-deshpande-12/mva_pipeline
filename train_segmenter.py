import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage import exposure
from PIL import Image, ImageDraw, ImageFont


# ── FR-UNet Architecture ───────────────────────────────────────────────────────

class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = dilation
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch,  out_ch, dilation=dilation),
            _ConvBnRelu(out_ch, out_ch, dilation=dilation),
        )

    def forward(self, x):
        return self.conv(x)


class _FRUNet(nn.Module):
    """
    Full-Resolution U-Net for retinal vessel segmentation.

    No spatial downsampling — all feature maps remain at the input resolution.
    Multi-scale context is captured through four dilated-convolution stages
    (dilation rates 1, 2, 4, 8).  All stage outputs are concatenated and fused
    before the final 1×1 segmentation head.

    Reference
    ---------
    Wang et al. "FR-UNet: Towards High Performance Full-Resolution Network for
    Retinal Vessel Segmentation." MICCAI Workshop 2022.
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters

        # Encoder: four parallel dilated stages (full resolution throughout)
        self.stage1 = _DoubleConv(in_channels, f,      dilation=1)
        self.stage2 = _DoubleConv(f,           f * 2,  dilation=2)
        self.stage3 = _DoubleConv(f * 2,       f * 4,  dilation=4)
        self.stage4 = _DoubleConv(f * 4,       f * 8,  dilation=8)

        # Decoder: skip-concatenation and progressive channel reduction
        self.decode3 = _DoubleConv(f * 8 + f * 4, f * 4, dilation=4)
        self.decode2 = _DoubleConv(f * 4 + f * 2, f * 2, dilation=2)
        self.decode1 = _DoubleConv(f * 2 + f,     f,     dilation=1)

        # Aggregation fusion — concatenate all decoder outputs
        self.fuse = nn.Sequential(
            nn.Conv2d(f * 4 + f * 2 + f, f, kernel_size=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.stage1(x)           # (B, f,    H, W)
        s2 = self.stage2(s1)          # (B, 2f,   H, W)
        s3 = self.stage3(s2)          # (B, 4f,   H, W)
        s4 = self.stage4(s3)          # (B, 8f,   H, W)

        d3 = self.decode3(torch.cat([s4, s3], dim=1))   # (B, 4f, H, W)
        d2 = self.decode2(torch.cat([d3, s2], dim=1))   # (B, 2f, H, W)
        d1 = self.decode1(torch.cat([d2, s1], dim=1))   # (B,  f, H, W)

        fused = self.fuse(torch.cat([d3, d2, d1], dim=1))  # (B, f, H, W)
        return self.head(fused)                              # (B, 1, H, W)  (logits)


# ── Lightweight U-Net (fast validation model) ─────────────────────────────────

class _LightUNet(nn.Module):
    """
    Standard 3-level U-Net with MaxPool downsampling.

    Much faster than FR-UNet because feature maps shrink at each encoder level:
        384×384 → 192×192 → 96×96 → 48×48 (bottleneck)
    Use this for quick pipeline validation; swap in _FRUNet for final runs.
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=16):
        super().__init__()
        f = base_filters
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = _DoubleConv(in_channels, f)
        self.enc2 = _DoubleConv(f,     f * 2)
        self.enc3 = _DoubleConv(f * 2, f * 4)

        # Bottleneck
        self.bottleneck = _DoubleConv(f * 4, f * 8)

        # Decoder
        self.up3  = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = _DoubleConv(f * 8, f * 4)
        self.up2  = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = _DoubleConv(f * 4, f * 2)
        self.up1  = nn.ConvTranspose2d(f * 2, f,     kernel_size=2, stride=2)
        self.dec1 = _DoubleConv(f * 2, f)

        # Head
        self.head = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


_MODEL_REGISTRY = {
    "frunet": _FRUNet,
    "light":  _LightUNet,
}


# ── Combined Dice + BCE Loss ───────────────────────────────────────────────────

class _DiceBCELoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.
    Handles severe class imbalance (vessels ~10-15 % of pixels).
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth     = smooth
        self.bce        = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss  = self.bce(logits, targets)
        probs     = torch.sigmoid(logits)
        num       = (probs * targets).sum(dim=(2, 3)) * 2 + self.smooth
        den       = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        dice_loss = 1.0 - (num / den).mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ── Illumination correction ────────────────────────────────────────────────────

def _illumination_correct(images):
    """
    Extract green channel, remove low-frequency illumination gradient via
    median-blur background subtraction, and return a (N, H, W, 1) float32 array.

    Green channel carries the highest vessel-to-background contrast in fundus
    images.  A large median blur (kernel 51) estimates the slowly-varying
    illumination field; subtracting it flattens uneven brightness across the disc.
    """
    N, H, W = images.shape[:3]
    out = np.empty((N, H, W, 1), dtype=np.float32)
    for i in range(N):
        green = (images[i, :, :, 1] * 255).astype(np.uint8)
        bg    = cv2.medianBlur(green, 51)
        corr  = cv2.subtract(green, bg).astype(np.float32) / 255.0
        out[i, :, :, 0] = corr
    return out


# ── PyTorch Dataset wrapper ────────────────────────────────────────────────────

class _VesselDataset(Dataset):
    def __init__(self, images, masks):
        # images: (N, H, W, 3) float32  →  convert to (N, 3, H, W)
        # masks:  (N, H, W, 1) float32  →  convert to (N, 1, H, W)
        self.images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        self.masks  = torch.from_numpy(masks.transpose(0, 3, 1, 2))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


# ── ml_trainer class ──────────────────────────────────────────────────────────

class ml_trainer:
    """
    Trains a FR-UNet segmentation model on the combined retinal vessel dataset.

    Parameters
    ----------
    base_filters : int
        Number of filters in the first FR-UNet stage (doubles at each stage).
    lr : float
        Initial learning rate for AdamW optimiser.
    device : str or None
        "cuda" / "cpu" / None (auto-detect).

    Example
    -------
    from dataset_reader import process_data_folders
    from train_segmenter import ml_trainer

    proc    = process_data_folders(target_size=(512, 512))
    imgs_d, masks_d = proc.read_drive("dataset_artery_segmentation/archive (4)/DRIVE")
    imgs_h, masks_h = proc.read_hrf("dataset_artery_segmentation/archive (6)")
    # ... load other datasets ...

    X_train, X_test, y_train, y_test = proc.split_datasets(
        [(imgs_d, masks_d), (imgs_h, masks_h)], test_size=0.1
    )

    trainer = ml_trainer()
    trainer.load_data(X_train, X_test, y_train, y_test, batch_size=4)
    trainer.train(epochs=50, save_dir="checkpoints")
    """

    def __init__(self, base_filters=None, lr=1e-4, device=None,
                 model_type="light"):
        """
        Parameters
        ----------
        model_type : "light" | "frunet"
            "light"  — _LightUNet  (default, fast, good for validation)
            "frunet" — _FRUNet     (full-resolution, slower, better for final runs)
        base_filters : int or None
            Filter count for first stage.  Defaults to 16 for "light", 32 for "frunet".
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.model_type = model_type

        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(_MODEL_REGISTRY)}"
            )
        if base_filters is None:
            base_filters = 16 if model_type == "light" else 32
        self.base_filters = base_filters

        model_cls      = _MODEL_REGISTRY[model_type]
        self.model     = model_cls(in_channels=3, out_channels=1,
                                   base_filters=base_filters).to(self.device)
        self.criterion = _DiceBCELoss(bce_weight=0.5).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                                     weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        self.train_loader = None
        self.test_loader  = None
        self.best_dice    = 0.0

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  {model_cls.__name__} (base_filters={base_filters}) on {self.device}  |  "
              f"parameters: {n_params:,}")

    # ── Function 1: combine all dataset pairs into one collimated set ─────────

    @staticmethod
    def combine_datasets(dataset_entries):
        """
        Concatenate images and masks from all datasets into a single
        collimated array pair ready for splitting and training.

        Parameters
        ----------
        dataset_entries : list of (name, images, masks) tuples
            name   : str           — dataset label for reporting
            images : np.ndarray    — shape (N, H, W, 3)  float32 in [0, 1]
            masks  : np.ndarray    — shape (N, H, W, 1)  float32 in {0, 1}

        Returns
        -------
        all_images : np.ndarray  shape (N_total, H, W, 3)
        all_masks  : np.ndarray  shape (N_total, H, W, 1)
        """
        all_images, all_masks = [], []
        total = 0

        print(f"\n{'='*52}")
        print(f"  Combining datasets into unified training pool")
        print(f"{'='*52}")
        print(f"  {'Dataset':<12}  {'Images':>8}  {'Cumulative':>12}")
        print(f"  {'-'*38}")

        for name, images, masks in dataset_entries:
            if images.shape[0] == 0:
                print(f"  {name:<12}  {'(empty)':>8}  {'—':>12}  skipped")
                continue
            if images.shape[0] != masks.shape[0]:
                raise ValueError(
                    f"{name}: {images.shape[0]} images but "
                    f"{masks.shape[0]} masks — mismatch."
                )
            all_images.append(images)
            all_masks.append(masks)
            total += images.shape[0]
            print(f"  {name:<12}  {images.shape[0]:>8}  {total:>12}")

        if not all_images:
            raise ValueError("No datasets were provided or all were empty.")

        combined_images = np.concatenate(all_images, axis=0)
        combined_masks  = np.concatenate(all_masks,  axis=0)

        print(f"  {'-'*38}")
        print(f"  {'TOTAL':<12}  {combined_images.shape[0]:>8}")
        print(f"  Image shape : {combined_images.shape[1]}×{combined_images.shape[2]}×{combined_images.shape[3]}")
        print(f"  Mask  shape : {combined_masks.shape[1]}×{combined_masks.shape[2]}×{combined_masks.shape[3]}")
        print(f"{'='*52}\n")

        return combined_images, combined_masks

    # ── Function 2: parse and wrap datasets into DataLoaders ──────────────────

    def load_data(self, X_train, X_test, y_train, y_test, batch_size=4,
                  debug_preprocessing=False):
        """
        Wrap numpy arrays from process_data_folders.split_datasets() into
        PyTorch DataLoaders ready for training.

        Parameters
        ----------
        X_train, X_test : np.ndarray  (N, H, W, 3)  float32 in [0, 1]
        y_train, y_test : np.ndarray  (N, H, W, 1)  float32 in {0, 1}
        batch_size      : int  (keep low — FR-UNet holds full-resolution maps)
        """
        # ── Preprocessing ──────────────────────────────────────────────────────

        self.X_test_raw = X_test.copy()   # kept for progression sample display

        def _debug_save(img, step_label, debug_dir):
            """Save first training image after a preprocessing step."""
            arr = img.copy()
            # Rescale to [0, 255] for display regardless of current range
            lo, hi = arr.min(), arr.max()
            arr = ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            fname = os.path.join(debug_dir, f"{step_label}.png")
            Image.fromarray(arr).save(fname)
            print(f"  [debug] saved {fname}")

        if debug_preprocessing:
            debug_dir = os.path.join("artery_segmentation_output", "preprocessing_debug")
            os.makedirs(debug_dir, exist_ok=True)
            _debug_save(X_train[0], "step0_raw", debug_dir)

        # (1) Spatial size uniformisation.
        assert X_train.shape[1:3] == y_train.shape[1:3] == \
               X_test.shape[1:3]  == y_test.shape[1:3], (
            f"Spatial size mismatch — images {X_train.shape[1:3]} vs "
            f"masks {y_train.shape[1:3]}. Pass a consistent target_size "
            f"to process_data_folders()."
        )

        # (2) Per-image Z-score scaling.
        for X in (X_train, X_test):
            for i in range(len(X)):
                mu   = X[i].mean()
                std  = X[i].std() + 1e-6
                X[i] = (X[i] - mu) / std

        if debug_preprocessing:
            _debug_save(X_train[0], "step2_zscore", debug_dir)

        # (3) CLAHE on L channel only (LAB space) — avoids the per-channel colour
        #     shift that causes a bluish cast when equalising R/G/B independently.
        def _clahe(images, clip_limit=0.01, kernel_size=10):
            # Rescale per-image to [0,1] via min-max so Z-score outliers don't
            # collapse dark pixels to 0 the way hard clipping would.
            lo = images.min(axis=(1, 2, 3), keepdims=True)
            hi = images.max(axis=(1, 2, 3), keepdims=True)
            images = (images - lo) / (hi - lo + 1e-6)
            out = np.empty_like(images)
            for i in range(len(images)):
                img_u8 = (images[i] * 255).astype(np.uint8)
                lab    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
                l_f    = lab[:, :, 0].astype(np.float32) / 255.0
                l_eq   = exposure.equalize_adapthist(l_f, kernel_size=kernel_size,
                                                     clip_limit=clip_limit)
                lab[:, :, 0] = (l_eq * 255).astype(np.uint8)
                out[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
            return out

        print("  Applying Z-score + CLAHE to training images ...", end="\r")
        X_train = _clahe(X_train)
        print("  Applying Z-score + CLAHE to test images     ...", end="\r")
        X_test  = _clahe(X_test)
        print(" " * 50, end="\r")

        if debug_preprocessing:
            _debug_save(X_train[0], "step3_clahe", debug_dir)

        # ── DataLoaders ────────────────────────────────────────────────────────

        train_ds = _VesselDataset(X_train, y_train)
        test_ds  = _VesselDataset(X_test,  y_test)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=0, pin_memory=(self.device.type == "cuda")
        )
        self.test_loader = DataLoader(
            test_ds, batch_size=batch_size,
            shuffle=False, num_workers=0, pin_memory=(self.device.type == "cuda")
        )

        print(f"\n  Data loaded  (preprocessing: CLAHE clip_limit=0.02)")
        print(f"    Train : {len(train_ds)} images  |  {len(self.train_loader)} batches")
        print(f"    Test  : {len(test_ds)}  images  |  {len(self.test_loader)} batches")
        print(f"    Batch size : {batch_size}")
        print(f"    Image size : {X_train.shape[1]}×{X_train.shape[2]}\n")

    # ── Function 2: train with verbose + best-model checkpoint ────────────────

    def train(self, epochs=50, save_dir="checkpoints"):
        """
        Train the FR-UNet and save the best checkpoint.

        At each batch step verbose prints: epoch, step, loss, batch Dice.
        At each epoch end: mean train loss, mean val loss, val Dice, val IoU.
        The model is saved whenever the validation Dice score improves.

        Parameters
        ----------
        epochs   : int   Number of training epochs.
        save_dir : str   Directory where best_model.pth is saved.
        """
        if self.train_loader is None or self.test_loader is None:
            raise RuntimeError("Call load_data() before train().")

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "best_model.pth")
        prog_dir  = os.path.join("artery_segmentation_output", "model_progression")
        os.makedirs(prog_dir, exist_ok=True)

        print(f"{'='*65}")
        print(f"  Training FR-UNet   epochs={epochs}   device={self.device}")
        print(f"  Checkpoints → {save_path}")
        print(f"{'='*65}\n")

        for epoch in range(1, epochs + 1):
            # ── Training phase ───────────────────────────────────────────────
            self.model.train()
            epoch_start  = time.time()
            train_losses = []
            n_batches    = len(self.train_loader)

            for step, (imgs, masks) in enumerate(self.train_loader, start=1):
                imgs  = imgs.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)
                loss.backward()
                self.optimizer.step()

                # batch-level Dice for verbose
                with torch.no_grad():
                    probs      = torch.sigmoid(logits)
                    preds      = (probs > 0.5).float()
                    inter      = (preds * masks).sum().item()
                    batch_dice = (2 * inter + 1) / \
                                 (preds.sum().item() + masks.sum().item() + 1)

                train_losses.append(loss.item())

                print(
                    f"  Epoch [{epoch:>3}/{epochs}]  "
                    f"Step [{step:>3}/{n_batches}]  "
                    f"Loss: {loss.item():.4f}  "
                    f"Batch Dice: {batch_dice:.4f}",
                    end="\r"
                )

            # ── Validation phase ─────────────────────────────────────────────
            val_loss, val_dice, val_iou = self._validate()
            self.scheduler.step()

            elapsed = time.time() - epoch_start
            mean_train_loss = np.mean(train_losses)

            print(
                f"  Epoch [{epoch:>3}/{epochs}]  "
                f"Train Loss: {mean_train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Dice: {val_dice:.4f}  "
                f"Val IoU: {val_iou:.4f}  "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}  "
                f"[{elapsed:.1f}s]"
            )

            # ── Save best model + progression sample ────────────────────────
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                torch.save({
                    "epoch":                epoch,
                    "model_type":           self.model_type,
                    "base_filters":         self.base_filters,
                    "model_state_dict":     self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_dice":             val_dice,
                    "val_iou":              val_iou,
                    "val_loss":             val_loss,
                }, save_path)
                print(f"  *** Best model saved  (Val Dice: {val_dice:.4f}) → {save_path}")
                self._save_progression_sample(epoch, val_dice, prog_dir)

        print(f"\n{'='*65}")
        print(f"  Training complete.  Best Val Dice: {self.best_dice:.4f}")
        print(f"  Best model: {save_path}")
        print(f"{'='*65}\n")

    # ── internal: progression sample ──────────────────────────────────────────

    def _save_progression_sample(self, epoch, val_dice, out_dir):
        """Save a 3-panel PNG (raw | overlay | mask) for the first test image."""
        self.model.eval()
        img_t, _ = next(iter(self.test_loader))   # first batch
        img_t = img_t[:1].to(self.device)         # just 1 image

        with torch.no_grad():
            prob = torch.sigmoid(self.model(img_t))[0, 0].cpu().numpy()

        mask = (prob > 0.5)
        orig = (self.X_test_raw[0] * 255).astype(np.uint8)   # true raw image

        overlay = orig.copy()
        overlay[mask] = (220, 30, 30)

        mask_rgb = np.stack([mask.astype(np.uint8) * 255] * 3, axis=-1)

        h, w   = orig.shape[:2]
        sep    = 4
        canvas = Image.new("RGB", (w * 3 + sep * 2, h + 24), color=(40, 40, 40))

        for i, (arr, label) in enumerate(
            [(orig, "Raw"), (overlay, "Arteries"), (mask_rgb, "Mask")]
        ):
            panel = Image.fromarray(arr)
            x     = i * (w + sep)
            canvas.paste(panel, (x, 24))
            draw  = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except OSError:
                font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw   = bbox[2] - bbox[0]
            draw.text((x + (w - tw) // 2, 4), label, fill=(220, 220, 220), font=font)

        fname = f"epoch_{epoch:03d}_dice_{val_dice:.4f}.png"
        canvas.save(os.path.join(out_dir, fname))
        print(f"  >>> Progression sample saved → {os.path.join(out_dir, fname)}")

    # ── internal: validation pass ──────────────────────────────────────────────

    def _validate(self):
        self.model.eval()
        losses, dices, ious = [], [], []
        n_batches = len(self.test_loader)

        print(f"\n  Validating ... (0/{n_batches})", end="\r")

        with torch.no_grad():
            for step, (imgs, masks) in enumerate(self.test_loader, start=1):
                imgs  = imgs.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)
                losses.append(loss.item())

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                inter     = (preds * masks).sum(dim=(1, 2, 3))
                union     = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                iou_union = union - inter

                dices.extend(((2 * inter + 1) / (union + 1)).cpu().tolist())
                ious.extend(((inter + 1) / (iou_union + 1)).cpu().tolist())

                print(f"  Validating ... ({step}/{n_batches})  "
                      f"loss: {loss.item():.4f}", end="\r")

        # Clear the validation line before the epoch summary prints
        print(" " * 60, end="\r")
        return np.mean(losses), np.mean(dices), np.mean(ious)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dataset_reader import process_data_folders

    DEBUG = False   # set False to skip saving preprocessing debug images

    # ── Dataset paths ──────────────────────────────────────────────────────────
    BASE  = os.path.dirname(os.path.abspath(__file__))
    SEG   = os.path.join(BASE, "dataset_artery_segmentation")

    DRIVE_PATH = os.path.join(SEG, "DRIVE", "DRIVE")   # nested subfolder
    CHASE_PATH = os.path.join(SEG, "CHASE")
    HRF_PATH   = os.path.join(SEG, "HRF")

    # ── Step 1: Load all three datasets ───────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1 — Loading datasets  (DRIVE + CHASE + HRF)")
    print("="*60)

    proc = process_data_folders(target_size=(384, 384))

    def _try_load(fn, path):
        """Load a dataset; return None on failure."""
        try:
            ds = fn(path)
            print(f"  {ds.summary()}")
            return ds
        except FileNotFoundError as exc:
            print(f"  NOT FOUND — {exc}")
            return None

    print()
    drive = _try_load(proc.read_drive, DRIVE_PATH)
    chase = _try_load(proc.read_chase, CHASE_PATH)
    hrf   = _try_load(proc.read_hrf,   HRF_PATH)

    all_ds = [ds for ds in [drive, chase, hrf] if ds is not None]

    if DEBUG:
        debug_dir = os.path.join(BASE, "artery_segmentation_output", "preprocessing_debug")
        os.makedirs(debug_dir, exist_ok=True)
        for ds in all_ds:
            if not ds.has_vessel:
                continue
            imgs, masks = ds.vessel_data
            img_u8  = (imgs[0] * 255).astype(np.uint8)
            mask_u8 = (masks[0, :, :, 0] * 255).astype(np.uint8)
            Image.fromarray(img_u8).save(
                os.path.join(debug_dir, f"{ds.name}_image.png"))
            Image.fromarray(mask_u8).save(
                os.path.join(debug_dir, f"{ds.name}_mask.png"))
            print(f"  [debug] {ds.name} → image + mask saved")

    # ── Step 2: Vessel segmentation train/test split ──────────────────────────
    print("\n" + "="*60)
    print("  STEP 2 — Vessel dataset  (artery/vessel segmentation)")
    print("="*60)

    X_tr_ves, X_te_ves, y_tr_ves, y_te_ves = proc.split_vessel(
        all_ds, test_size=0.2, seed=42
    )

    # ── Step 3: Train vessel segmentation model ───────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3 — Training vessel segmentation model")
    print("="*60)

    # MODEL SELECTION — change model_type here:
    #   "light"  → _LightUNet  (fast, use for validation)
    #   "frunet" → _FRUNet     (full-resolution, use for final training)
    ves_trainer = ml_trainer(model_type="light", lr=1e-4, device=None)
    ves_trainer.load_data(X_tr_ves, X_te_ves, y_tr_ves, y_te_ves, batch_size=4,
                          debug_preprocessing=DEBUG)

    if DEBUG:
        print("\n  DEBUG mode — preprocessing images saved. Training skipped.")
        raise SystemExit(0)

    ves_trainer.train(
        epochs=500,  # EPOCHS — change here (5-10 for quick validation, 50+ for final)
        save_dir=os.path.join(BASE, "checkpoints", "vessel"),
    )
