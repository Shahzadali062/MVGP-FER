# Dataset preprocessing

The scripts here prepare RAF-DB and FER2013 images for MVGP-FER. Download the datasets yourself using the links in `dataset/`. Nothing is downloaded automatically.

## Setup

Run these commands from the repository root:

```powershell
python -m pip install -r dataset_preprocess/requirements.txt
```

## Run

For a quick check with one image:

```powershell
python -m dataset_preprocess.run image --input "D:/data/face.jpg" --out "D:/MVGP-output/face.npy"
```

For the original FER2013 CSV:

```powershell
python -m dataset_preprocess.run fer2013 --csv "D:/data/FER2013/fer2013.csv" --out "D:/MVGP-output/fer2013"
```

For RAF-DB with its original `EmoLabel` and `Image/aligned` folders:

```powershell
python -m dataset_preprocess.run rafdb --root "D:/data/RAF-DB" --out "D:/MVGP-output/rafdb"
```

Each run saves float32 image arrays (`3 x 224 x 224`) and a `manifest.csv` listing the source split and class. Keep output outside the Git repository.

## Landmarks (optional)

Landmark extraction needs MediaPipe, a Face Landmarker `.task` model, and the **exact 44 landmark indices** from the original experiment. Install MediaPipe with `python -m pip install mediapipe`, then add both flags to a dataset command:

```powershell
--landmarker-model "D:/models/face_landmarker.task" --landmark-indices "D:/private/landmark_indices.json"
```

If landmarks cannot be detected, the manifest records `missing`. The image remains in the output.

The manuscript does not specify the complete 44-index selection or the individual formulas for the 16 geometric cues. This code does **not** invent them or produce a complete 104-feature vector. `standardize.py` contains general training-only statistics functions for use once the full feature definitions are available.

To run the checks: `python -m unittest discover -s dataset_preprocess/tests -v`.
