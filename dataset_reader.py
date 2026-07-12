import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
from sklearn.model_selection import train_test_split


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class DatasetResult:
    """
    Holds all loaded data for one retinal vessel segmentation dataset.

    Two distinct mask types are tracked separately:

    fov_data : (images, masks) — eye vs background (field-of-view) masks.
               The circular FOV mask separates the illuminated retinal disc
               from the black background.  Used to train an eye-extraction
               model that crops or masks the fundus image before further
               processing.

    vessel_data : (images, masks) — binary artery/vessel segmentation masks.
                  Ground-truth labels hand-annotated by ophthalmologists.
                  Used to train the vessel segmentation model.

    Both tuples pair arrays of equal length: fov_data[0][i] ↔ fov_data[1][i].
    The two tuples may have DIFFERENT lengths (e.g. DRIVE has FOV masks for
    all 40 images but vessel GT for only 20 training images).

    Either field is None when that mask type is not available for the dataset.

    Mask availability per dataset
    -----------------------------
    DRIVE   : fov_data=40 images,   vessel_data=20 images (training split only)
    CHASE   : fov_data=None,        vessel_data=28 images
    HRF     : fov_data=None,        vessel_data=45 images
    """

    name: str
    fov_data: Optional[Tuple[np.ndarray, np.ndarray]]
    vessel_data: Optional[Tuple[np.ndarray, np.ndarray]]

    @property
    def has_fov(self) -> bool:
        return self.fov_data is not None and len(self.fov_data[0]) > 0

    @property
    def has_vessel(self) -> bool:
        return self.vessel_data is not None and len(self.vessel_data[0]) > 0

    def summary(self) -> str:
        fov_n = len(self.fov_data[0]) if self.has_fov else 0
        ves_n = len(self.vessel_data[0]) if self.has_vessel else 0
        return (f"{self.name:<12}  FOV masks: {fov_n:>3}    "
                f"Vessel GT: {ves_n:>3}")


# ── Main reader class ─────────────────────────────────────────────────────────

class process_data_folders:
    """
    Reads four retinal vessel segmentation datasets and returns DatasetResult
    objects that carry both mask types (FOV and vessel GT) where available.

    Each read_*() method returns a DatasetResult.  Use split_fov() or
    split_vessel() to build the corresponding train/test numpy arrays.

    Quick start
    -----------
    proc  = process_data_folders(target_size=(512, 512))
    drive = proc.read_drive("dataset_artery_segmentation/DRIVE/DRIVE")
    chase = proc.read_chase("dataset_artery_segmentation/CHASE")
    hrf   = proc.read_hrf  ("dataset_artery_segmentation/HRF")

    all_ds = [drive, chase, hrf]

    # FOV / eye-extraction dataset  (DRIVE only has these)
    X_tr_fov, X_te_fov, y_tr_fov, y_te_fov = proc.split_fov(all_ds)

    # Vessel segmentation dataset  (DRIVE training + CHASE + HRF)
    X_tr_ves, X_te_ves, y_tr_ves, y_te_ves = proc.split_vessel(all_ds)
    """


    def __init__(self, target_size=(512, 512)):
        """
        Parameters
        ----------
        target_size : (H, W)
            All images and masks are resized to this resolution before
            returning.  Keeping a common size is required because numpy
            stacking needs uniform array shapes.
        """
        self.target_size = target_size

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_image(self, path):
        """RGB float32 in [0, 1], resized to target_size."""
        img = Image.open(path).convert("RGB")
        img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _load_mask(self, path):
        """Binary float32 in {0, 1}, shape (H, W, 1), resized to target_size."""
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        arr  = np.array(mask, dtype=np.float32)
        return (arr > 127).astype(np.float32)[..., np.newaxis]

    def _sorted_files(self, folder, extensions):
        """Sorted list of files in folder matching the given extensions."""
        folder = Path(folder)
        if not folder.exists():
            return []
        exts  = {e.lower() for e in extensions}
        files = [f for f in folder.iterdir()
                 if f.is_file() and f.suffix.lower() in exts]
        return sorted(files, key=lambda f: f.name)

    # ── Dataset 1: DRIVE ─────────────────────────────────────────────────────

    def read_drive(self, folder_path) -> DatasetResult:
        """
        Read the DRIVE dataset (DRIVE/DRIVE/ nested subfolder).

        Image format : TIFF (.tif)
        Mask formats : GIF (.gif) for both FOV and vessel GT

        Folder structure
        ----------------
        folder_path/
        ├── training/
        │   ├── images/       20 .tif colour images  (numbered 21–40)
        │   ├── 1st_manual/   20 .gif vessel GT masks
        │   └── mask/         20 .gif FOV masks
        └── test/
            ├── images/       20 .tif colour images  (numbered 01–20)
            └── mask/         20 .gif FOV masks
            (no vessel GT provided for the test split)

        Returns
        -------
        DatasetResult
            fov_data    : (40 images, 40 FOV masks)      — training + test
            vessel_data : (20 images, 20 vessel GT masks) — training only
        """
        root = Path(folder_path)
        _img_exts  = {".tif", ".png", ".jpg", ".ppm"}
        _mask_exts = {".gif", ".png", ".tif"}

        fov_imgs,    fov_msks    = [], []
        vessel_imgs, vessel_msks = [], []

        # ── test split: FOV mask only ─────────────────────────────────────────
        test_imgs = self._sorted_files(root / "test" / "images", _img_exts)
        test_fovs = self._sorted_files(root / "test" / "mask",   _mask_exts)

        if len(test_imgs) != len(test_fovs):
            raise ValueError(
                f"DRIVE test: {len(test_imgs)} images but "
                f"{len(test_fovs)} FOV masks — folder incomplete."
            )
        for img_p, fov_p in zip(test_imgs, test_fovs):
            fov_imgs.append(self._load_image(img_p))
            fov_msks.append(self._load_mask(fov_p))

        # ── training split: FOV mask AND vessel GT ────────────────────────────
        tr_imgs = self._sorted_files(root / "training" / "images",     _img_exts)
        tr_fovs = self._sorted_files(root / "training" / "mask",       _mask_exts)
        tr_vess = self._sorted_files(root / "training" / "1st_manual", _mask_exts)

        if len(tr_imgs) != len(tr_fovs) or len(tr_imgs) != len(tr_vess):
            raise ValueError(
                f"DRIVE training: images={len(tr_imgs)}, FOV masks={len(tr_fovs)}, "
                f"vessel GT={len(tr_vess)} — folder incomplete."
            )
        for img_p, fov_p, ves_p in zip(tr_imgs, tr_fovs, tr_vess):
            img = self._load_image(img_p)
            fov_imgs.append(img)
            fov_msks.append(self._load_mask(fov_p))
            vessel_imgs.append(img)
            vessel_msks.append(self._load_mask(ves_p))

        if not fov_imgs:
            raise FileNotFoundError(
                f"No DRIVE images found in {folder_path}."
            )

        return DatasetResult(
            name="DRIVE",
            fov_data=(np.stack(fov_imgs), np.stack(fov_msks)),
            vessel_data=(
                (np.stack(vessel_imgs), np.stack(vessel_msks))
                if vessel_imgs else None
            ),
        )

    # ── Dataset 2: CHASE_DB1 ─────────────────────────────────────────────────

    def read_chase(self, folder_path) -> DatasetResult:
        """
        Read the CHASE_DB1 dataset (CHASE/ folder).

        Image format : JPEG (.jpg)
        Mask format  : PNG (.png) — vessel GT only; no FOV masks provided

        Folder structure
        ----------------
        folder_path/
            Image_01L.jpg          retinal image, subject 01, left eye
            Image_01R.jpg          retinal image, subject 01, right eye
            Image_01L_1stHO.png    1st annotator vessel GT mask
            Image_01L_2ndHO.png    2nd annotator vessel GT mask
            Image_01R_1stHO.png
            ...  (28 images total, 14 subjects × 2 eyes)

        The 1st annotator masks (_1stHO) are used as vessel ground truth.

        Returns
        -------
        DatasetResult
            fov_data    : None  (not provided)
            vessel_data : (28 images, 28 vessel GT masks)
        """
        root = Path(folder_path)

        # Accept .jpg (standard) and .tif (alternative releases)
        img_files = sorted(
            [f for f in list(root.glob("Image_*.jpg")) +
                         list(root.glob("Image_*.tif"))
             if "HO" not in f.name],
            key=lambda f: f.name,
        )

        vessel_imgs, vessel_msks = [], []
        for img_path in img_files:
            stem = img_path.stem
            mask_candidates = [
                root / f"{stem}_1stHO.png",
                root / f"{stem}_1stHO.PNG",
            ]
            mask_path = next((p for p in mask_candidates if p.exists()), None)
            if mask_path is None:
                print(f"  [CHASE] Warning: no 1stHO mask for "
                      f"{img_path.name} — skipping.")
                continue
            vessel_imgs.append(self._load_image(img_path))
            vessel_msks.append(self._load_mask(mask_path))

        if not vessel_imgs:
            raise FileNotFoundError(
                f"No CHASE_DB1 images found in {folder_path}.\n"
                "Expected: Image_01L.jpg, Image_01L_1stHO.png, ...\n"
                "Download from https://blogs.kingston.ac.uk/retinal/chasedb1/"
            )

        return DatasetResult(
            name="CHASE_DB1",
            fov_data=None,
            vessel_data=(np.stack(vessel_imgs), np.stack(vessel_msks)),
        )

    # ── Dataset 3: HRF ───────────────────────────────────────────────────────

    def read_hrf(self, folder_path) -> DatasetResult:
        """
        Read the HRF (High Resolution Fundus) dataset (HRF/ folder).

        Image format : JPEG (.jpg / .JPG)
        Mask format  : TIFF (.tif) — vessel GT only; no FOV masks provided

        Folder structure
        ----------------
        folder_path/
        ├── images/
        │   ├── 01_h.jpg    healthy
        │   ├── 01_dr.JPG   diabetic retinopathy
        │   ├── 01_g.jpg    glaucomatous
        │   └── ... (45 images: 15 per category)
        └── masks/
            ├── 01_h.tif
            ├── 01_dr.tif
            ├── 01_g.tif
            └── ... (45 vessel GT masks, matching image stems)

        Returns
        -------
        DatasetResult
            fov_data    : None  (not provided)
            vessel_data : (45 images, 45 vessel GT masks)
        """
        root     = Path(folder_path)
        img_dir  = root / "images"
        mask_dir = root / "masks"

        img_files = self._sorted_files(
            img_dir, {".jpg", ".jpeg", ".png", ".tif"}
        )

        vessel_imgs, vessel_msks = [], []
        for img_path in img_files:
            stem = img_path.stem
            mask_candidates = [
                mask_dir / f"{stem}.tif",
                mask_dir / f"{stem}.png",
            ]
            mask_path = next((p for p in mask_candidates if p.exists()), None)
            if mask_path is None:
                print(f"  [HRF] Warning: no mask for "
                      f"{img_path.name} — skipping.")
                continue
            vessel_imgs.append(self._load_image(img_path))
            vessel_msks.append(self._load_mask(mask_path))

        if not vessel_imgs:
            raise FileNotFoundError(
                f"No HRF images found in {folder_path}.\n"
                "Expected subfolders: images/ and masks/"
            )

        return DatasetResult(
            name="HRF",
            fov_data=None,
            vessel_data=(np.stack(vessel_imgs), np.stack(vessel_msks)),
        )

    # ── Train/test split — FOV task ───────────────────────────────────────────

    def split_fov(self, datasets, test_size=0.1, seed=42):
        """
        Build a train/test split for the eye-extraction (FOV) task.

        Only datasets that have fov_data are included.  Currently only
        DRIVE provides FOV masks.

        Parameters
        ----------
        datasets  : list[DatasetResult]
        test_size : float  fraction held out for testing
        seed      : int

        Returns
        -------
        X_train, X_test : (N, H, W, 3) float32
        y_train, y_test : (N, H, W, 1) float32  binary FOV masks
        """
        all_imgs, all_msks = [], []

        print(f"\n{'='*55}")
        print(f"  FOV (eye-extraction) dataset")
        print(f"{'='*55}")

        for ds in datasets:
            if not ds.has_fov:
                print(f"  {ds.name:<12} : no FOV masks — skipped")
                continue
            imgs, msks = ds.fov_data
            all_imgs.append(imgs)
            all_msks.append(msks)
            print(f"  {ds.name:<12} : {len(imgs)} images")

        if not all_imgs:
            raise ValueError(
                "No datasets with FOV masks were found.  "
                "Only DRIVE provides FOV masks."
            )

        X = np.concatenate(all_imgs, axis=0)
        y = np.concatenate(all_msks, axis=0)

        print(f"  {'─'*40}")
        print(f"  Total          : {len(X)} images  "
              f"{X.shape[1]}×{X.shape[2]}")

        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, shuffle=True
        )
        print(f"  Train          : {len(tr_idx)} images")
        print(f"  Test           : {len(te_idx)} images")
        print(f"{'='*55}\n")

        return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    # ── Train/test split — vessel segmentation task ───────────────────────────

    def split_vessel(self, datasets, test_size=0.1, seed=42):
        """
        Build a train/test split for the vessel segmentation task.

        Only datasets that have vessel_data are included.
        (DRIVE training, CHASE, HRF)

        Parameters
        ----------
        datasets  : list[DatasetResult]
        test_size : float  fraction held out for testing
        seed      : int

        Returns
        -------
        X_train, X_test : (N, H, W, 3) float32
        y_train, y_test : (N, H, W, 1) float32  binary vessel masks
        """
        all_imgs, all_msks = [], []

        print(f"\n{'='*55}")
        print(f"  Vessel segmentation dataset")
        print(f"{'='*55}")

        for ds in datasets:
            if not ds.has_vessel:
                print(f"  {ds.name:<12} : no vessel GT — skipped")
                continue
            imgs, msks = ds.vessel_data
            all_imgs.append(imgs)
            all_msks.append(msks)
            print(f"  {ds.name:<12} : {len(imgs)} images")

        if not all_imgs:
            raise ValueError(
                "No datasets with vessel GT masks were found."
            )

        X = np.concatenate(all_imgs, axis=0)
        y = np.concatenate(all_msks, axis=0)

        print(f"  {'─'*40}")
        print(f"  Total          : {len(X)} images  "
              f"{X.shape[1]}×{X.shape[2]}")

        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, shuffle=True
        )
        print(f"  Train          : {len(tr_idx)} images")
        print(f"  Test           : {len(te_idx)} images")
        print(f"{'='*55}\n")

        return X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
