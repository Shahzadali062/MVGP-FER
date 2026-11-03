# Training

These scripts train the **reference ablation models**, not the authors' original unpublished implementation. Accuracy values in the manuscript are **not** reproduced or guaranteed here.

## Setup

Run from the repository root:

```powershell
python -m pip install -r training/requirements.txt
```

Prepare RAF-DB or FER2013 with `dataset_preprocess`. Provide a `manifest.csv` with `visual_npy`, `paper_label`, and `split` columns. Image arrays must be preprocessed float32 arrays of shape `(3,224,224)`.

## Train M1 or M2

```powershell
python -m training.train --variant M1 --dataset rafdb --manifest "D:/MVGP-output/rafdb/manifest.csv" --out "D:/MVGP-private/checkpoints/M1.pt"
python -m training.train --variant M2 --dataset fer2013 --manifest "D:/MVGP-output/fer2013/manifest.csv" --out "D:/MVGP-private/checkpoints/M2.pt"
```

The defaults are batch size 8, Adam at 3e-4, up to 100 epochs, seed 42, and early stopping after 20 epochs. CUDA uses mixed precision to reduce GPU memory.

For RAF-DB, validation is a **new stratified 10% split from its training partition**; the official test split is untouched. For FER2013, `PublicTest` is used as validation and `PrivateTest` stays held out. These are reference choices; the original manuscript checkpoint-selection procedure is not specified.

## M3 inputs

M3 also needs two things the public preprocessing package does **not** provide:

- The verified class-prototype matrix with shape `(7,16)`, saved outside this public repository as JSON or NPY.
- The 16 real geometric cues per sample, with the matching manifest column `geometric_cues_16_npy`. The preprocessing package's `landmarks_88_npy` plus these 16 cues forms the raw 104-feature vector. As an alternative, provide `geometry_104_npy` containing the full raw 104 features.

Only valid training samples are used to fit standardization statistics. The model falls back to visual-only predictions for missing geometry. Never fill missing real cues with invented values.

```powershell
python -m training.train --variant M3 --dataset rafdb --manifest "D:/MVGP-output/rafdb/manifest.csv" --prototypes "D:/MVGP-private/prototypes.npy" --out "D:/MVGP-private/checkpoints/M3.pt"
```

## Evaluate

```powershell
python -m training.evaluate --checkpoint "D:/MVGP-private/checkpoints/M1.pt" --manifest "D:/MVGP-output/rafdb/manifest.csv" --split test
```

Checkpoints, data, prototypes, experiment reports, and results belong outside the public Git repository. Before claiming published metrics, validate this code against the **original** implementation, sample exclusions, and checkpoints.
