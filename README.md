# MVGP-FER

**Multi-Context Visual Representation and Geometric Prototype-Guided Learning for Facial Expression Recognition**

MVGP-FER presents lightweight facial expression recognition using visual features and facial landmark geometry. Its design combines a compact image encoder, multi-context attention, a small landmark network, class-specific geometric prototypes, and adaptive logit fusion.


## Repository

| Folder | Contents |
| --- | --- |
| [`dataset/`](dataset/) | Links to obtain RAF-DB and FER2013 |
| [`dataset_preprocess/`](dataset_preprocess/) | Dataset readers, image preprocessing, optional landmark extraction, tests |
| [`Model/`](Model/) | Stem, residual blocks, attention, spatial mixer, geometry, fusion |
| [`training/`](training/) | Training  |


## Getting started

Install the dependencies with Python 3.10 or later:

```powershell
python -m pip install -r dataset_preprocess/requirements.txt
python -m pip install -r training/requirements.txt
```

Prepare the dataset you have downloaded from its original source:

```powershell
python -m dataset_preprocess.run rafdb --root "D:/data/RAF-DB" --out "D:/prepared/rafdb"
```

Or use the original FER2013 CSV:

```powershell
python -m dataset_preprocess.run fer2013 --csv "D:/data/FER2013/fer2013.csv" --out "D:/prepared/fer2013"
```

Train model:

```powershell
python -m training.train"
```
