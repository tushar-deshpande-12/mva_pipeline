"""
image_png_converter.py
──────────────────────
Converts all dataset images and arterial masks to PNG for visual inspection.
Reproduces the exact train/test split used during training (seed=42, 20%).

Output layout
─────────────
dataset_png/
    train/
        image/   DRIVE_000.png  CHASE_DB1_000.png  ...
        mask/    DRIVE_000.png  CHASE_DB1_000.png  ...
    test/
        image/   ...
        mask/    ...

Usage
─────
    python image_png_converter.py
"""

import os
import numpy as np
from PIL import Image
from dataset_reader import process_data_folders

BASE       = os.path.dirname(os.path.abspath(__file__))
SEG        = os.path.join(BASE, "dataset_artery_segmentation")
OUT_DIR    = os.path.join(BASE, "dataset_png")
TARGET_SIZE = (384, 384)
TEST_SIZE   = 0.1
SEED        = 42

DRIVE_PATH = os.path.join(SEG, "DRIVE", "DRIVE")
CHASE_PATH = os.path.join(SEG, "CHASE")
HRF_PATH   = os.path.join(SEG, "HRF")


def _save(arr, path):
    lo, hi = arr.min(), arr.max()
    img_u8 = ((arr - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
    if img_u8.ndim == 3 and img_u8.shape[-1] == 1:
        img_u8 = img_u8[:, :, 0]   # grayscale mask
    Image.fromarray(img_u8).save(path)


def main():
    # ── Make output folders ────────────────────────────────────────────────────
    for split in ("train", "test"):
        for kind in ("image", "mask"):
            os.makedirs(os.path.join(OUT_DIR, split, kind), exist_ok=True)

    # ── Load datasets ──────────────────────────────────────────────────────────
    proc = process_data_folders(target_size=TARGET_SIZE)

    datasets = []
    for name, fn, path in [
        ("DRIVE",    proc.read_drive, DRIVE_PATH),
        ("CHASE",    proc.read_chase, CHASE_PATH),
        ("HRF",      proc.read_hrf,   HRF_PATH),
    ]:
        try:
            ds = fn(path)
            datasets.append(ds)
            print(f"  Loaded  {ds.summary()}")
        except FileNotFoundError:
            print(f"  SKIP    {name} — not found at {path}")

    if not datasets:
        raise RuntimeError("No datasets found.")

    # ── Split (must match train_segmenter.py exactly) ─────────────────────────
    X_tr, X_te, y_tr, y_te = proc.split_vessel(
        datasets, test_size=TEST_SIZE, seed=SEED
    )

    # ── We need per-image dataset labels for filenames ─────────────────────────
    # Rebuild the same ordering split_vessel uses so we can name files correctly.
    all_imgs, all_masks, all_names = [], [], []
    for ds in datasets:
        if not ds.has_vessel:
            continue
        imgs, masks = ds.vessel_data
        for i in range(len(imgs)):
            all_imgs.append(imgs[i])
            all_masks.append(masks[i])
            all_names.append(f"{ds.name}_{i:03d}")

    from sklearn.model_selection import train_test_split
    idx = list(range(len(all_imgs)))
    idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                      random_state=SEED, shuffle=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    def save_split(indices, split_name):
        for pos, i in enumerate(indices):
            name = all_names[i]
            _save(all_imgs[i],  os.path.join(OUT_DIR, split_name, "image", f"{name}.png"))
            _save(all_masks[i], os.path.join(OUT_DIR, split_name, "mask",  f"{name}.png"))
            print(f"  [{split_name}]  {pos+1:>3}/{len(indices)}  {name}", end="\r")
        print(f"  [{split_name}]  {len(indices)} files saved.              ")

    save_split(idx_tr, "train")
    save_split(idx_te, "test")

    print(f"\n  Done → {OUT_DIR}")
    print(f"    train : {len(idx_tr)} image+mask pairs")
    print(f"    test  : {len(idx_te)} image+mask pairs")


if __name__ == "__main__":
    main()
