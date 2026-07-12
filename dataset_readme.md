# MVA Pipeline — Retinal Vessel Segmentation Datasets

This file records all datasets used to train and evaluate the vessel segmentation model
inside the Microvascular Analysis (MVA) pipeline. The MVA pipeline extracts quantitative
retinal vascular parameters (CRAE, CRVE, AVR, FD, ARTBA, VEINBA, ARTTOR, AVGVEINTOR)
from colour fundus photographs. The segmentation model underpinning this extraction was
trained on the publicly available benchmark datasets described below.

---

## Local Folder Structure

```
mva_pipeline/
└── dataset_artery_segmentation/
    ├── DRIVE/                    →  DRIVE        (images in DRIVE/DRIVE/training/ and DRIVE/DRIVE/test/)
    ├── CHASE/                    →  CHASE_DB1    (28 .jpg images + .png vessel GT masks, flat folder)
    ├── HRF/                      →  HRF          (images/ + masks/ subfolders, _dr/_g/_h convention)
    ├── AV-20191104T162310Z-001/  →  AV-STARE     (AV/ subfolder, one IM######/ folder per image)
    └── Fundus-AVSeg/             →  Fundus-AVSeg (images/ + annotation/ + training.txt/testing.txt)
```

---

## Dataset 1 — DRIVE (Digital Retinal Images for Vessel Extraction)

| Property | Detail |
|----------|--------|
| Local folder | `dataset_artery_segmentation/DRIVE/DRIVE/` (full: training + test splits) |
| Total images | 40 colour fundus photographs |
| Split | 20 training / 20 test |
| Resolution | 768 × 584 pixels |
| Field of view | 45° |
| Image format | TIFF (`.tif`) |
| Ground truth | Two manual vessel segmentations for test set (1st and 2nd annotator); 1st manual for training |
| FOV masks | Provided per image (circular field-of-view mask) |
| Patient population | Diabetic retinopathy screening programme, The Netherlands |
| Access | https://drive.grand-challenge.org |

### Contents

```
archive (4)/DRIVE/
├── training/
│   ├── images/         (20 .tif colour images)
│   ├── 1st_manual/     (20 .gif binary vessel maps)
│   └── mask/           (20 .gif FOV masks)
└── test/
    ├── images/         (20 .tif colour images)
    ├── 1st_manual/     (20 .gif binary vessel maps — primary ground truth)
    ├── 2nd_manual/     (20 .gif binary vessel maps — inter-annotator reference)
    └── mask/           (20 .gif FOV masks)
```

### Citation

> Staal J, Abràmoff MD, Niemeijer M, Viergever MA, van Ginneken B.
> Ridge-based vessel segmentation in color images of the retina.
> *IEEE Transactions on Medical Imaging*. 2004;23(4):501–509.
> https://doi.org/10.1109/TMI.2004.825627

---

## Dataset 2 — CHASE_DB1 (Child Heart and Health Study in England)

| Property | Detail |
|----------|--------|
| Local folder | `dataset_artery_segmentation/CHASE/` (28 .jpg images + .png vessel GT masks) |
| Total images | 28 colour retinal images |
| Subjects | 14 school children (left eye + right eye per child) |
| Resolution | 1280 × 960 pixels |
| Field of view | 30° |
| Image format | JPEG (`.jpg`) |
| Ground truth | Two manual vessel segmentations per image (1st and 2nd annotator) |
| FOV masks | Not provided separately (full rectangular image used) |
| Patient population | Primary school children, UK |
| Access | https://blogs.kingston.ac.uk/retinal/chasedb1/ |

### Filename Convention

```
Image_01L.jpg   — subject 01, left eye
Image_01R.jpg   — subject 01, right eye
Image_01L_1stHO.png  — 1st annotator vessel map
Image_01L_2ndHO.png  — 2nd annotator vessel map
```

### Citation

> Fraz MM, Remagnino P, Hoover A, Uyyanonvara B, Rudnicka AR, Owen CG, Barman SA.
> An ensemble classification-based approach applied to retinal blood vessel segmentation.
> *IEEE Transactions on Biomedical Engineering*. 2012;59(9):2538–2548.
> https://doi.org/10.1109/TBME.2012.2205687

---

## Dataset 3 — HRF (High Resolution Fundus)

| Property | Detail |
|----------|--------|
| Local folder | `dataset_artery_segmentation/HRF/` |
| Total images | 45 colour fundus photographs |
| Groups | 15 healthy (`_h`), 15 diabetic retinopathy (`_dr`), 15 glaucomatous (`_g`) |
| Resolution | 3504 × 2336 pixels (high resolution) |
| Field of view | 45° |
| Image format | JPEG (`.jpg` / `.JPG`) |
| Ground truth | Binary vessel segmentation masks (`.tif`) per image |
| FOV masks | Provided per image |
| Patient population | Mixed: healthy controls, DR patients, glaucoma patients |
| Access | https://www5.cs.fau.de/research/data/fundus-images/ |

### Contents

```
archive (6)/
├── images/    (45 .jpg files: 01_h.jpg, 01_dr.JPG, 01_g.jpg, ..., 15_h.jpg, ...)
└── masks/     (45 .tif binary vessel segmentation ground truth files)
```

### Filename Convention

| Suffix | Category | Count |
|--------|----------|-------|
| `_h`   | Healthy | 15 |
| `_dr`  | Diabetic Retinopathy | 15 |
| `_g`   | Glaucomatous | 15 |

### Citation

> Budai A, Bock R, Maier A, Hornegger J, Michelson G.
> Robust vessel segmentation in fundus images.
> *International Journal of Biomedical Imaging*. 2013;2013:154860.
> https://doi.org/10.1155/2013/154860

---

## Dataset 4 — AV-STARE (Retinal Artery/Vein Segmentation on STARE images)

| Property | Detail |
|----------|--------|
| Local folder | `dataset_artery_segmentation/AV-20191104T162310Z-001/AV/` |
| Total images | ~100 colour fundus photographs |
| Resolution | Variable (STARE-derived, typically 700 × 605) |
| Field of view | 35° |
| Image format | JPEG (`.JPG`) for images; JPEG for masks |
| Ground truth | Per-image artery mask (`--artry.jpg`), vein mask (`--veins.jpg`), combined vessel mask (`--vessels.jpg`) |
| FOV masks | Not provided |
| Patient population | STARE image subset — mixed pathology (DR, AMD, hypertensive retinopathy) |
| Access | https://data.mendeley.com/datasets/3csr652p9y/2 |

### Contents

```
AV-20191104T162310Z-001/
└── AV/
    ├── IM000001/
    │   ├── IM000001.JPG           ← original fundus image
    │   ├── IM000001--artry.jpg    ← artery segmentation mask
    │   ├── IM000001--veins.jpg    ← vein segmentation mask
    │   ├── IM000001--vessels.jpg  ← all vessels combined mask
    │   ├── IM000001--both.jpg     ← artery + vein overlay
    │   └── IM000001.ai            ← Adobe Illustrator source annotation
    ├── IM000004/
    │   └── ...
    └── ...  (~100 IM###### folders total)
```

### Citation

> Dataset: Retinal Artery/Vein Segmentation.
> Mendeley Data, V2. 2019.
> https://doi.org/10.17632/3csr652p9y.2

---

## Dataset 5 — Fundus-AVSeg

| Property | Detail |
|----------|--------|
| Local folder | `dataset_artery_segmentation/Fundus-AVSeg/` |
| Total images | 100 colour fundus photographs |
| Resolution | Variable |
| Image format | PNG (`.png`) |
| Ground truth | Per-image AV segmentation annotation (PNG, matching filename) |
| Pathology labels | G = Glaucoma, N = Normal, D = Diabetic Retinopathy, A = AMD |
| Train/test split | Provided in `training.txt` / `testing.txt` |
| Metadata | `metadata.xlsx` — per-image clinical labels |
| FOV masks | Not provided |
| Access | https://www.nature.com/articles/s41597-025-05381-2 |

### Contents

```
Fundus-AVSeg/
├── images/           (100 PNG fundus images:  001_G.png, 002_N.png, ...)
├── annotation/       (100 PNG AV masks:       001_G.png, 002_N.png, ...)
├── training.txt      (image IDs in training split)
├── testing.txt       (image IDs in test split)
└── metadata.xlsx     (per-image clinical metadata)
```

### Filename Convention

```
NNN_X.png   — NNN = zero-padded index, X = pathology label
              G → Glaucoma
              N → Normal
              D → Diabetic Retinopathy
              A → AMD (Age-related Macular Degeneration)
```

### Citation

> Fundus-AVSeg: A fundus image dataset for artery/vein segmentation.
> *Scientific Data*. 2025.
> https://www.nature.com/articles/s41597-025-05381-2

---

## Combined Dataset Summary

| Dataset | Images | Vessel GT masks | AV masks | Resolution | Format |
|---------|--------|-----------------|----------|------------|--------|
| DRIVE       | 40  | 20 (training)  | —   | 768 × 584    | TIFF + GIF  |
| CHASE_DB1   | 28  | 28             | —   | 1280 × 960   | JPEG + PNG  |
| HRF         | 45  | 45             | —   | 3504 × 2336  | JPEG + TIFF |
| AV-STARE    | ~100 | —             | ~100 artery + vein | ~700 × 605 | JPEG |
| Fundus-AVSeg | 100 | —             | 100 | Variable     | PNG         |
| **Total**   | **~313** | **~93**   | **~200** | — | — |

---

## Usage in the MVA Pipeline

The vessel segmentation model is trained on datasets with binary vessel GT masks:
- **Vessel model**: DRIVE training (20) + CHASE (28) + HRF (45) = **93 images** (90/10 train/test split)

AV-STARE and Fundus-AVSeg provide artery/vein class labels (not binary vessel masks) and
are intended for the future artery/vein classification stage of the pipeline.

The trained model is used in the MVA pipeline to:
1. Segment the retinal vessel tree from an input colour fundus image
2. Separate arterioles and venules (artery/vein classification)
3. Measure calibres (CRAE, CRVE), ratios (AVR), branching angles (ARTBA, VEINBA),
   tortuosity (ARTTOR, AVGVEINTOR), and fractal dimension (FD)

---

## Licensing Notes

| Dataset | Licence |
|---------|---------|
| DRIVE | Research use only — contact University Medical Center Utrecht |
| CHASE_DB1 | Research use only — Kingston University |
| HRF | Research use only — Friedrich-Alexander-Universität Erlangen-Nürnberg |
| AV-STARE | CC BY 4.0 — Mendeley Data |
| Fundus-AVSeg | See article terms — *Scientific Data* 2025 |

All datasets are de-identified. For clinical deployment or commercial use, review
the licensing terms of each contributing institution.

---

## References

1. Staal J et al. Ridge-based vessel segmentation in color images of the retina.
   *IEEE Trans Med Imaging*. 2004;23(4):501–509.
   https://doi.org/10.1109/TMI.2004.825627

2. Fraz MM et al. An ensemble classification-based approach applied to retinal blood
   vessel segmentation. *IEEE Trans Biomed Eng*. 2012;59(9):2538–2548.
   https://doi.org/10.1109/TBME.2012.2205687

3. Budai A et al. Robust vessel segmentation in fundus images.
   *Int J Biomed Imaging*. 2013;2013:154860.
   https://doi.org/10.1155/2013/154860

4. Retinal Artery/Vein Segmentation dataset.
   Mendeley Data, V2. 2019.
   https://doi.org/10.17632/3csr652p9y.2

5. Fundus-AVSeg: A fundus image dataset for artery/vein segmentation.
   *Scientific Data*. 2025.
   https://www.nature.com/articles/s41597-025-05381-2
