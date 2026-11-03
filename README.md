# MVGP-FER

**Multi-Context Visual Representation and Geometric Prototype-Guided Learning for Facial Expression Recognition**

MVGP-FER studies lightweight facial expression recognition using facial appearance and landmark-based geometry. The proposed method combines a compact visual pathway, expression-related geometric cues, class-specific prototypes, and adaptive prediction fusion.

This repository contains dataset preparation tools, reusable model components, and a training script for the **visual-only M1 baseline**. It is a partial code release, not the complete implementation used for the manuscript experiments.

## Repository structure

| Folder | Contents |
| --- | --- |
| [`dataset/`](dataset/) | Official access information for RAF-DB and FER2013 |
| [`dataset_preprocess/`](dataset_preprocess/) | Image preprocessing, dataset readers, optional landmark extraction, and tests |
| [`Model/`](Model/) | Convolutional blocks, spatial mixer, visual branch, and fusion interface |
| [`training/`](training/) | M1 baseline training, evaluation, and tests |

## Getting started

Use Python 3.10 or later. Run these commands from the repository root.

Install the preprocessing and training dependencies:

```powershell
python -m pip install -r dataset_preprocess/requirements.txt
python -m pip install -r training/requirements.txt
```

Prepare FER2013 from the original CSV:

```powershell
python -m dataset_preprocess.run fer2013 --csv "D:/data/FER2013/fer2013.csv" --out "D:/prepared/fer2013"
```

Or prepare aligned RAF-DB images:

```powershell
python -m dataset_preprocess.run rafdb --root "D:/data/RAF-DB" --out "D:/prepared/rafdb"
```

Train the available visual baseline:

```powershell
python -m training.train --dataset fer2013 --manifest "D:/prepared/fer2013/manifest.csv" --out "D:/experiments/M1.pt"
```

See [`dataset_preprocess/README.md`](dataset_preprocess/README.md) and [`training/README.md`](training/README.md) for the input format, validation split, and evaluation commands. Replace the example paths with your local directories.

## Data and release status

RAF-DB and FER2013 must be obtained from their original providers. No dataset images or annotations are included here.

The manuscript describes 44 selected facial landmarks and 16 geometric cues. The exact landmark selection, complete cue definitions, attention implementation, and class-prototype configuration are **not part of this release**. The full geometry-guided model and M2/M3 experiments therefore cannot be reproduced with this repository alone. The public training workflow supports M1 only.

Keep downloaded datasets, generated arrays, and checkpoints outside the Git repository. The code has been checked with synthetic inputs, but the manuscript results have not been independently reproduced with this public release.

## Citation and use

For now, refer to the project as **MVGP-FER**. A formal citation will be added when the paper is published.

No repository-wide open-source license has been provided. The original dataset providers retain their own access and usage terms.
