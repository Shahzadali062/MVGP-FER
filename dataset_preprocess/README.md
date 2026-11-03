# Dataset preprocessing

These scripts convert RAF-DB and FER2013 images into NumPy arrays and a CSV manifest. Get the datasets from the links in `dataset/`; they are not included here.

From the repository root, install the dependencies:

```powershell
python -m pip install -r dataset_preprocess/requirements.txt
```

## One image

```powershell
python -m dataset_preprocess.run image --input "D:/data/face.jpg" --out "D:/output/face.npy"
```

## FER2013

```powershell
python -m dataset_preprocess.run fer2013 --csv "D:/data/FER2013/fer2013.csv" --out "D:/output/fer2013"
```

## RAF-DB

The input directory needs the original `EmoLabel` and `Image/aligned` folders.

```powershell
python -m dataset_preprocess.run rafdb --root "D:/data/RAF-DB" --out "D:/output/rafdb"
```

Output contains `manifest.csv` and normalized image arrays of shape `(3, 224, 224)`. Keep it outside this Git repository.

**Landmarks:** To extract them, install MediaPipe and provide both a Face Landmarker `.task` file and your verified list of 44 landmark indices:

```powershell
python -m pip install mediapipe
python -m dataset_preprocess.run rafdb --root "D:/data/RAF-DB" --out "D:/output/rafdb" --landmarker-model "D:/models/face_landmarker.task" --landmark-indices "D:/private/indices.json"
```

The manuscript does not list all 44 indices or the formulas for its 16 geometric cues. Those details must come from the original experiment; the current scripts do not create the complete 104-feature vector.

To run the preprocessing tests:

```powershell
python -m unittest discover -s dataset_preprocess/tests -v
```
