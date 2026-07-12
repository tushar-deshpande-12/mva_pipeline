import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image

from train_segmenter import _FRUNet, _VesselDataset
from torch.utils.data import DataLoader


class model_evaluator:
    """
    Loads the best saved FR-UNet checkpoint and evaluates its artery/vessel
    segmentation accuracy on a held-out test set.

    Metrics reported
    ----------------
    - Pixel Accuracy      : (TP + TN) / total pixels
    - Sensitivity         : TP / (TP + FN)  — vessel recall
    - Specificity         : TN / (TN + FP)  — background recall
    - Dice Coefficient    : 2TP / (2TP + FP + FN)
    - IoU (Jaccard)       : TP / (TP + FP + FN)
    - AUC-ROC             : area under the ROC curve (continuous probability)
    - F1 Score            : identical to Dice for binary masks

    Two models are trained — one per task:
      checkpoints/fov/best_model.pth    — eye-extraction (FOV) model
      checkpoints/vessel/best_model.pth — vessel segmentation model

    Example
    -------
    from model_eval import model_evaluator

    # Evaluate the vessel segmentation model
    evaluator = model_evaluator(checkpoint_path="checkpoints/vessel/best_model.pth")
    evaluator.load_model()

    results = evaluator.evaluate(X_test, y_test, batch_size=4)
    evaluator.print_report(results)
    evaluator.save_predictions(X_test, y_test, out_dir="eval_outputs")
    """

    def __init__(self, checkpoint_path, base_filters=32, device=None):
        """
        Parameters
        ----------
        checkpoint_path : str
            Path to the .pth file saved by ml_trainer.train().
        base_filters : int
            Must match the value used during training (default 32).
        device : str or None
            "cuda" / "cpu" / None (auto-detect).
        """
        self.checkpoint_path = checkpoint_path
        self.base_filters    = base_filters

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model  = None

    # ── load the best checkpoint ───────────────────────────────────────────────

    def load_model(self):
        """
        Instantiate FR-UNet and load weights from the best checkpoint.
        Prints the epoch and Val Dice the checkpoint was saved at.
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                "Run ml_trainer.train() first to generate a checkpoint."
            )

        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device,
                                weights_only=False)

        self.model = _FRUNet(in_channels=3, out_channels=1,
                             base_filters=self.base_filters).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print(f"\n  Model loaded from: {self.checkpoint_path}")
        print(f"    Saved at epoch : {checkpoint.get('epoch', '?')}")
        print(f"    Val Dice (ckpt): {checkpoint.get('val_dice', '?'):.4f}")
        print(f"    Val IoU  (ckpt): {checkpoint.get('val_iou',  '?'):.4f}")
        print(f"    Device         : {self.device}\n")

    # ── main evaluation function ───────────────────────────────────────────────

    def evaluate(self, X_test, y_test, batch_size=4, threshold=0.5):
        """
        Run the best FR-UNet on the test set and compute all segmentation metrics.

        Parameters
        ----------
        X_test    : np.ndarray  (N, H, W, 3)  float32 in [0, 1]
        y_test    : np.ndarray  (N, H, W, 1)  float32 in {0, 1}
        batch_size : int
        threshold  : float  Binarisation threshold for predicted probabilities.

        Returns
        -------
        results : dict with keys:
            pixel_accuracy, sensitivity, specificity, dice, iou, f1, auc_roc
            + per_image_dice (list of per-image Dice scores)
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before evaluate().")

        dataset = _VesselDataset(X_test, y_test)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        all_probs  = []
        all_preds  = []
        all_targets = []

        print(f"  Evaluating on {len(dataset)} test images ...")

        with torch.no_grad():
            for imgs, masks in loader:
                imgs  = imgs.to(self.device)
                logits = self.model(imgs)
                probs  = torch.sigmoid(logits).cpu().numpy()   # (B, 1, H, W)
                preds  = (probs > threshold).astype(np.float32)
                tgts   = masks.numpy()                          # (B, 1, H, W)

                all_probs.append(probs)
                all_preds.append(preds)
                all_targets.append(tgts)

        all_probs   = np.concatenate(all_probs,   axis=0).squeeze(1)  # (N, H, W)
        all_preds   = np.concatenate(all_preds,   axis=0).squeeze(1)  # (N, H, W)
        all_targets = np.concatenate(all_targets, axis=0).squeeze(1)  # (N, H, W)

        results = self._compute_metrics(all_probs, all_preds, all_targets)
        return results

    # ── metric computation ─────────────────────────────────────────────────────

    def _compute_metrics(self, probs, preds, targets):
        """
        Compute global and per-image segmentation metrics.

        Parameters — all float32 numpy arrays of shape (N, H, W)
        """
        eps = 1e-7

        # Flatten to (N, H*W) for per-image stats, then (N*H*W,) for global
        flat_preds   = preds.reshape(len(preds), -1)
        flat_targets = targets.reshape(len(targets), -1)
        flat_probs   = probs.reshape(len(probs), -1)

        # ── per-image Dice ────────────────────────────────────────────────────
        inter    = (flat_preds * flat_targets).sum(axis=1)
        union    = flat_preds.sum(axis=1) + flat_targets.sum(axis=1)
        per_dice = (2 * inter + eps) / (union + eps)

        # ── global pixel counts ───────────────────────────────────────────────
        p = flat_preds.ravel()
        t = flat_targets.ravel()

        TP = (p * t).sum()
        FP = (p * (1 - t)).sum()
        FN = ((1 - p) * t).sum()
        TN = ((1 - p) * (1 - t)).sum()
        total = TP + FP + FN + TN

        pixel_accuracy = (TP + TN) / (total + eps)
        sensitivity    = TP / (TP + FN + eps)   # recall / true positive rate
        specificity    = TN / (TN + FP + eps)   # true negative rate
        dice           = (2 * TP) / (2 * TP + FP + FN + eps)
        iou            = TP / (TP + FP + FN + eps)
        f1             = dice                    # equivalent for binary masks

        # ── AUC-ROC (via trapezoidal rule over probability thresholds) ────────
        auc_roc = self._compute_auc(flat_probs.ravel(), t)

        return {
            "pixel_accuracy": float(pixel_accuracy),
            "sensitivity":    float(sensitivity),
            "specificity":    float(specificity),
            "dice":           float(dice),
            "iou":            float(iou),
            "f1":             float(f1),
            "auc_roc":        float(auc_roc),
            "per_image_dice": per_dice.tolist(),
        }

    def _compute_auc(self, probs, targets, n_thresholds=100):
        """Trapezoidal AUC-ROC over n_thresholds equally spaced thresholds."""
        thresholds = np.linspace(0, 1, n_thresholds)
        tprs, fprs = [], []
        eps = 1e-7
        for thresh in thresholds:
            p  = (probs > thresh).astype(np.float32)
            TP = (p * targets).sum()
            FP = (p * (1 - targets)).sum()
            FN = ((1 - p) * targets).sum()
            TN = ((1 - p) * (1 - targets)).sum()
            tprs.append(TP / (TP + FN + eps))
            fprs.append(FP / (FP + TN + eps))
        # Sort by FPR ascending for trapz
        fprs, tprs = zip(*sorted(zip(fprs, tprs)))
        return float(np.trapz(tprs, fprs))

    # ── report printer ─────────────────────────────────────────────────────────

    def print_report(self, results):
        """
        Print a formatted table of all segmentation metrics.

        Parameters
        ----------
        results : dict  returned by evaluate()
        """
        per_dice = results["per_image_dice"]
        print(f"\n{'='*52}")
        print(f"  Artery / Vessel Segmentation — Evaluation Report")
        print(f"{'='*52}")
        print(f"  {'Metric':<25}  {'Value':>10}")
        print(f"  {'-'*38}")
        print(f"  {'Pixel Accuracy':<25}  {results['pixel_accuracy']:>10.4f}")
        print(f"  {'Sensitivity (Recall)':<25}  {results['sensitivity']:>10.4f}")
        print(f"  {'Specificity':<25}  {results['specificity']:>10.4f}")
        print(f"  {'Dice Coefficient':<25}  {results['dice']:>10.4f}")
        print(f"  {'IoU (Jaccard)':<25}  {results['iou']:>10.4f}")
        print(f"  {'F1 Score':<25}  {results['f1']:>10.4f}")
        print(f"  {'AUC-ROC':<25}  {results['auc_roc']:>10.4f}")
        print(f"  {'-'*38}")
        print(f"  {'Per-image Dice (mean)':<25}  {np.mean(per_dice):>10.4f}")
        print(f"  {'Per-image Dice (std)':<25}  {np.std(per_dice):>10.4f}")
        print(f"  {'Per-image Dice (min)':<25}  {np.min(per_dice):>10.4f}")
        print(f"  {'Per-image Dice (max)':<25}  {np.max(per_dice):>10.4f}")
        print(f"{'='*52}\n")

    # ── save side-by-side prediction images ────────────────────────────────────

    def save_predictions(self, X_test, y_test, out_dir="eval_outputs",
                         batch_size=4, threshold=0.5, n_save=None):
        """
        Save side-by-side comparison images:  input | ground truth | prediction.

        Parameters
        ----------
        X_test, y_test : numpy arrays from process_data_folders.split_datasets()
        out_dir        : directory to write PNG files to
        n_save         : number of images to save (None = all)
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before save_predictions().")

        os.makedirs(out_dir, exist_ok=True)
        dataset = _VesselDataset(X_test, y_test)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        saved = 0
        limit = n_save if n_save is not None else len(dataset)

        with torch.no_grad():
            for imgs_t, masks_t in loader:
                logits = self.model(imgs_t.to(self.device))
                probs  = torch.sigmoid(logits).cpu().numpy()

                for i in range(len(imgs_t)):
                    if saved >= limit:
                        break

                    # Input image (H, W, 3) → uint8
                    img_np = (imgs_t[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    # Ground truth mask (H, W) → uint8 grayscale
                    gt_np  = (masks_t[i, 0].numpy() * 255).astype(np.uint8)
                    # Predicted mask (H, W) → uint8 grayscale
                    pred_binary = (probs[i, 0] > threshold).astype(np.uint8) * 255

                    # Compose side-by-side: input | GT | pred
                    h, w = img_np.shape[:2]
                    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
                    canvas[:, :w]          = img_np
                    canvas[:, w:2*w]       = np.stack([gt_np]*3, axis=-1)
                    canvas[:, 2*w:3*w]     = np.stack([pred_binary]*3, axis=-1)

                    out_path = os.path.join(out_dir, f"pred_{saved+1:04d}.png")
                    Image.fromarray(canvas).save(out_path)
                    saved += 1

                if saved >= limit:
                    break

        print(f"  {saved} prediction images saved to: {out_dir}")
        print(f"  Layout: [Input Image | Ground Truth | Prediction]")
