"""
run_artery_segmentation.py
──────────────────────────
Runs the trained vessel segmentation model on every test-split image and
saves side-by-side comparison PNGs to:

    mva_pipeline/artery_segmentation_output/output_test_images/

The test split is reproduced with the same seed (42) and ratio (20 %) used
during training so the model is evaluated on images it never saw.

Output layout per image
───────────────────────
┌──────────────────────┬──────────────────────┐
│   Original image     │  Artery segmentation │
│  (colour fundus)     │  (vessels highlighted│
│                      │   in red on original)│
└──────────────────────┴──────────────────────┘

Vessel pixels in the right panel are coloured red so they stand out against
the dark fundus background while preserving spatial context from the original.

Usage
─────
    python run_artery_segmentation.py
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from skimage import exposure

from train_segmenter import _FRUNet, _LightUNet, _MODEL_REGISTRY, _VesselDataset
from dataset_reader   import process_data_folders

# ── Configuration ─────────────────────────────────────────────────────────────

BASE       = os.path.dirname(os.path.abspath(__file__))
SEG        = os.path.join(BASE, "dataset_artery_segmentation")

DRIVE_PATH = os.path.join(SEG, "DRIVE", "DRIVE")
CHASE_PATH = os.path.join(SEG, "CHASE")
HRF_PATH   = os.path.join(SEG, "HRF")

CHECKPOINT   = os.path.join(BASE, "checkpoints", "vessel", "best_model.pth")
OUT_DIR      = os.path.join(BASE, "artery_segmentation_output", "output_test_images")

TARGET_SIZE  = (384, 384)   # must match what was used during training
BASE_FILTERS = 32
BATCH_SIZE   = 4
THRESHOLD    = 0.5
TEST_SPLIT   = 0.1
SEED         = 42

LABEL_HEIGHT = 28
FONT_SIZE    = 16
SEPARATOR_W  = 4

# Highlight colour for detected vessel pixels (R, G, B)
VESSEL_COLOUR = (220, 30, 30)   # red

# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_clahe(images, clip_limit=0.02):
    """CLAHE on L channel only (LAB) — matches ml_trainer.load_data() preprocessing."""
    import cv2
    out = np.empty_like(images)
    for i in range(len(images)):
        img_u8 = (images[i] * 255).astype(np.uint8)
        lab    = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
        l_f    = lab[:, :, 0].astype(np.float32) / 255.0
        l_eq   = exposure.equalize_adapthist(l_f, clip_limit=clip_limit)
        lab[:, :, 0] = (l_eq * 255).astype(np.uint8)
        out[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    return out


def _load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        sys.exit(
            f"\n  ERROR: Checkpoint not found:\n  {checkpoint_path}\n"
            "  Run train_vessel_only.py first.\n"
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type   = ckpt.get("model_type",  "frunet")
    base_filters = ckpt.get("base_filters", 32)
    model_cls    = _MODEL_REGISTRY.get(model_type, _FRUNet)

    model = model_cls(in_channels=3, out_channels=1,
                      base_filters=base_filters).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  Architecture : {model_cls.__name__}  (base_filters={base_filters})")
    print(f"  Epoch        : {ckpt.get('epoch', '?')}")
    print(f"  Val Dice     : {ckpt.get('val_dice', float('nan')):.4f}")
    print(f"  Val IoU      : {ckpt.get('val_iou',  float('nan')):.4f}")
    return model


def _overlay_vessels(orig_np, vessel_mask, colour):
    """
    Return a copy of orig_np (H, W, 3 uint8) with vessel pixels
    replaced by `colour` (R, G, B).
    """
    result = orig_np.copy()
    result[vessel_mask] = colour
    return result


def _labelled_panel(img_np, label, label_height, font_size):
    h, w = img_np.shape[:2]
    canvas = Image.new("RGB", (w, h + label_height), color=(40, 40, 40))
    canvas.paste(Image.fromarray(img_np), (0, label_height))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox  = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (label_height - th) // 2),
              label, fill=(220, 220, 220), font=font)
    return canvas


def _make_side_by_side(orig_np, overlay_np, mask_np, label_height, separator_w, font_size):
    h, w = orig_np.shape[:2]
    left   = _labelled_panel(orig_np,    "Original",            label_height, font_size)
    middle = _labelled_panel(overlay_np, "Artery Segmentation", label_height, font_size)
    right  = _labelled_panel(mask_np,    "Predicted Mask",       label_height, font_size)

    canvas = Image.new("RGB", (w * 3 + separator_w * 2, h + label_height),
                       color=(255, 255, 255))
    canvas.paste(left,   (0,                       0))
    canvas.paste(middle, (w + separator_w,          0))
    canvas.paste(right,  (w * 2 + separator_w * 2, 0))
    return canvas


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("  Artery Segmentation — Test Image Output")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Resolution : {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  Output dir : {OUT_DIR}\n")

    # ── 1. Load datasets ───────────────────────────────────────────────────────
    print("─" * 60)
    print("  Step 1 — Loading datasets  (DRIVE + CHASE + HRF)")
    print("─" * 60)

    proc = process_data_folders(target_size=TARGET_SIZE)

    def _try_load(fn, path, name):
        try:
            ds = fn(path)
            print(f"  {ds.summary()}")
            return ds
        except FileNotFoundError:
            print(f"  [{name}] not found at {path} — skipping.")
            return None

    drive = _try_load(proc.read_drive, DRIVE_PATH, "DRIVE")
    chase = _try_load(proc.read_chase, CHASE_PATH, "CHASE")
    hrf   = _try_load(proc.read_hrf,   HRF_PATH,   "HRF")

    all_ds = [ds for ds in [drive, chase, hrf] if ds is not None]
    if not all_ds:
        sys.exit("\n  ERROR: No vessel datasets loaded.  Check dataset paths.\n")

    # ── 2. Reproduce the exact test split ──────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Step 2 — Reproducing vessel test split  "
          f"(seed={SEED}, test_size={TEST_SPLIT})")
    print("─" * 60)

    _, X_test, _, _ = proc.split_vessel(all_ds, test_size=TEST_SPLIT, seed=SEED)
    n_test = len(X_test)
    print(f"  Test images : {n_test}")

    # ── 3. Load model ──────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 3 — Loading model")
    print("─" * 60)

    model = _load_model(CHECKPOINT, device)

    # ── 4. Preprocessing (must match ml_trainer.load_data) ────────────────────
    print("\n" + "─" * 60)
    print("  Step 4 — Preprocessing")
    print("─" * 60)

    # (1) [DISABLED] Colour normalisation
    # X_test = (X_test - ch_mean) / ch_std

    # (2) [DISABLED] Spatial size uniformisation
    # assert X_test.shape[1:3] == (TARGET_SIZE[0], TARGET_SIZE[1])

    # (2) Per-image Z-score scaling
    X_test_proc = X_test.copy()
    for i in range(len(X_test_proc)):
        mu             = X_test_proc[i].mean()
        std            = X_test_proc[i].std() + 1e-6
        X_test_proc[i] = (X_test_proc[i] - mu) / std

    # (3) CLAHE — clip to [0,1] first since Z-scoring can push values outside
    print("  Applying Z-score + CLAHE to test images ...", end="\r")
    lo = X_test_proc.min(axis=(1, 2, 3), keepdims=True)
    hi = X_test_proc.max(axis=(1, 2, 3), keepdims=True)
    X_test_proc = _apply_clahe((X_test_proc - lo) / (hi - lo + 1e-6), clip_limit=0.02)
    print("  Done.                                     ")

    # ── 5. Run inference ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Step 5 — Running inference")
    print("─" * 60)

    dummy_masks = np.zeros(
        (n_test, TARGET_SIZE[0], TARGET_SIZE[1], 1), dtype=np.float32
    )
    dataset = _VesselDataset(X_test_proc, dummy_masks)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for step, (imgs, _) in enumerate(loader, start=1):
            logits = model(imgs.to(device))
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            print(f"  Inference batch {step}/{len(loader)}", end="\r")

    print(" " * 50, end="\r")

    all_probs    = np.concatenate(all_probs, axis=0).squeeze(1)   # (N, H, W)
    vessel_masks = (all_probs > THRESHOLD)                         # bool (N, H, W)

    # ── 6. Save side-by-side images ────────────────────────────────────────────
    print(f"\n" + "─" * 60)
    print(f"  Step 6 — Saving {n_test} side-by-side images")
    print("─" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    vessel_pixel_pcts = []

    for i in range(n_test):
        # Use the original (pre-CLAHE) image for display so colours look natural
        orig_np   = (X_test[i] * 255).astype(np.uint8)

        # Overlay detected vessel pixels in VESSEL_COLOUR on original
        overlay_np = _overlay_vessels(orig_np, vessel_masks[i], VESSEL_COLOUR)

        # Binary predicted mask: white vessels on black background
        mask_np = (vessel_masks[i].astype(np.uint8) * 255)
        mask_np = np.stack([mask_np, mask_np, mask_np], axis=-1)

        composite = _make_side_by_side(
            orig_np, overlay_np, mask_np,
            label_height=LABEL_HEIGHT,
            separator_w=SEPARATOR_W,
            font_size=FONT_SIZE,
        )

        out_path = os.path.join(OUT_DIR, f"artery_{i+1:04d}.png")
        composite.save(out_path)

        pct = vessel_masks[i].mean() * 100
        vessel_pixel_pcts.append(pct)
        print(f"  [{i+1:>4}/{n_test}]  vessels={pct:.1f}%  "
              f"→ {os.path.basename(out_path)}", end="\r")

    print(" " * 70, end="\r")
    print(f"\n  Done — {n_test} images saved to:")
    print(f"  {OUT_DIR}")

    # ── 7. Summary stats ───────────────────────────────────────────────────────
    pcts = np.array(vessel_pixel_pcts)
    print(f"\n  Detected vessel coverage (% pixels classified as vessel):")
    print(f"    Mean : {pcts.mean():.2f}%")
    print(f"    Min  : {pcts.min():.2f}%")
    print(f"    Max  : {pcts.max():.2f}%")
    print(f"    Std  : {pcts.std():.2f}%")
    print()


if __name__ == "__main__":
    main()
