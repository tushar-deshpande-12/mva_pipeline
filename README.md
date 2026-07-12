# MVA Pipeline — Microvascular Analysis for Retinal Vessel Segmentation

This pipeline trains a deep learning model to segment blood vessels in colour fundus
photographs and is part of the broader CKD (Chronic Kidney Disease) paper. Retinal
microvascular changes — measured as CRAE, CRVE, AVR, fractal dimension, branching
angles, and tortuosity — are extracted from the predicted vessel masks and used as
features in the CKD classification model.

---

## Repository Structure

```
mva_pipeline/
├── train_segmenter.py          — Model training (FR-UNet or LightUNet)
├── run_artery_segmentation.py  — Inference on test images, 3-panel output
├── dataset_reader.py           — Dataset loaders (DRIVE, CHASE, HRF)
├── image_png_converter.py      — Export all dataset images/masks as PNG for inspection
├── model_eval.py               — Evaluate a saved checkpoint (Dice, IoU, AUC-ROC, ...)
├── dataset_readme.md           — Full citation and structure info for all datasets
│
├── dataset_artery_segmentation/
│   ├── DRIVE/                  — 20 training images with vessel GT masks
│   ├── CHASE/                  — 28 images with vessel GT masks
│   ├── HRF/                    — 45 images with vessel GT masks
│   ├── AV-20191104T162310Z-001/— ~100 images with artery/vein masks (future AV stage)
│   └── Fundus-AVSeg/           — 100 images with AV annotations (future AV stage)
│
├── checkpoints/
│   ├── fov/best_model.pth      — Saved FOV (eye-extraction) model
│   └── vessel/best_model.pth   — Saved vessel segmentation model
│
└── artery_segmentation_output/
    ├── output_test_images/     — 3-panel PNGs from run_artery_segmentation.py
    ├── model_progression/      — Per-epoch best-model sample saves during training
    └── preprocessing_debug/    — Debug images from each preprocessing step
```

---

## Models

Two architectures are available, selected via `model_type` in `train_segmenter.py`:

| `model_type` | Class | Description | Use when |
|---|---|---|---|
| `"light"` | `_LightUNet` | 3-level U-Net with MaxPool (16 base filters) | Quick validation runs |
| `"frunet"` | `_FRUNet` | Full-Resolution U-Net, dilated convolutions (rates 1/2/4/8, 32 base filters) | Final training |

The active model type and `base_filters` are saved inside each checkpoint so inference
always loads the correct architecture automatically.

---

## Training Datasets

Binary vessel GT masks come from three datasets (93 images total, 90/10 split):

| Dataset | Images | Resolution | Format |
|---------|--------|------------|--------|
| DRIVE   | 20     | 768 × 584  | TIFF + GIF masks |
| CHASE_DB1 | 28   | 1280 × 960 | JPEG + PNG masks |
| HRF     | 45     | 3504 × 2336 | JPEG + TIFF masks |

AV-STARE and Fundus-AVSeg contain artery/vein class labels for the future
artery/vein classification stage and are not used in the current vessel model.

See `dataset_readme.md` for full citations and folder layouts.

---

## Preprocessing Pipeline

Applied identically during training and inference:

1. **Size uniformisation** — all images resized to `384 × 384`
2. **Per-image Z-score** — `(x − μ) / (σ + 1e-6)` computed per image independently
3. **Min-max rescale to [0, 1]** — preserves full dynamic range before CLAHE
4. **CLAHE on L channel (LAB)** — contrast enhancement without colour cast
   (`clip_limit=0.01`, applied only to the luminance channel)

---

## Quickstart

### 1. Train the vessel segmentation model

```bash
python train_segmenter.py
```

**Key settings to change at the top of the `if __name__ == "__main__":` block:**

| Variable | Location | Description |
|---|---|---|
| `DEBUG` | line ~628 | `True` → save preprocessing debug images and stop before training |
| `model_type` | `ml_trainer(model_type=...)` | `"light"` for validation, `"frunet"` for final |
| `epochs` | `ves_trainer.train(epochs=...)` | 5–10 for quick check, 500 for final |

Checkpoints are saved to `checkpoints/vessel/best_model.pth` whenever validation
Dice improves. A 3-panel progression sample is saved to
`artery_segmentation_output/model_progression/` at each best-model epoch.

### 2. Run inference on test images

```bash
python run_artery_segmentation.py
```

Reproduces the exact 90/10 test split (seed 42) used during training and saves
side-by-side PNGs (`Original | Red vessel overlay | Predicted mask`) to
`artery_segmentation_output/output_test_images/`.

### 3. Evaluate a checkpoint

```python
from model_eval import model_evaluator
from dataset_reader import process_data_folders

evaluator = model_evaluator(checkpoint_path="checkpoints/vessel/best_model.pth")
evaluator.load_model()
results = evaluator.evaluate(X_test, y_test, batch_size=4)
evaluator.print_report(results)
```

Metrics: Pixel Accuracy, Sensitivity, Specificity, Dice, IoU, AUC-ROC, F1.

### 4. Inspect dataset images as PNG

```bash
python image_png_converter.py
```

Exports all dataset image + mask pairs to `dataset_png/train/` and `dataset_png/test/`
using the same split as training. Useful for spotting faulty masks visually.

### 5. Debug preprocessing

Set `DEBUG = True` in `train_segmenter.py`. The pipeline will:
- Save a raw image and mask for each dataset to `artery_segmentation_output/preprocessing_debug/`
- Save intermediate images after each preprocessing step
- Stop before training begins

---

## Loss Function

Combined Dice + BCE loss (`bce_weight=0.5`) defined in `_DiceBCELoss`. Handles the
severe class imbalance inherent in vessel segmentation (~10–15 % vessel pixels).

---

## Requirements

```
torch
torchvision
numpy
Pillow
scikit-image
scikit-learn
opencv-python (cv2)
```
