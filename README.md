# MVGP-FER

**Multi-Context Visual Representation and Geometric Prototype-Guided Learning for Facial Expression Recognition**

MVGP-FER explores lightweight facial expression recognition using visual features and facial landmark geometry. Its design combines a compact image encoder, multi-context attention, a small landmark network, class-specific geometric prototypes, and adaptive logit fusion.

This repository provides a **reference implementation** and utilities for preparing RAF-DB and FER2013. The code follows the architecture and settings described in the manuscript; it has not been verified as the original experiment source or used here to reproduce the manuscript's reported performance.

## Repository

| Folder | Contents |
| --- | --- |
| [`dataset/`](dataset/) | Links to obtain RAF-DB and FER2013 |
| [`dataset_preprocess/`](dataset_preprocess/) | Dataset readers, image preprocessing, optional landmark extraction, tests |
| [`Model/`](Model/) | Stem, residual blocks, attention, spatial mixer, geometry, fusion |
| [`training/`](training/) | Training and evaluation scripts for M1, M2 and M3 |

## Model configurations

- **M1:** visual baseline with fused MBConv, inverted residual blocks and gated spatial mixing.
- **M2:** M1 with multi-context coordinate attention.
- **M3:** M2 with geometric prototype guidance and adaptive logit fusion.

An `Ablations/` folder is not needed: the `--variant` argument selects the configuration.

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

Train the visual baseline:

```powershell
python -m training.train --variant M1 --dataset rafdb --manifest "D:/prepared/rafdb/manifest.csv" --out "D:/experiments/M1.pt"
```

M2 can use the same prepared images. M3 additionally requires the original **16 geometric cue values per image** and a verified **7 x 16 prototype matrix**. The currently available preprocessing scripts do not calculate those cues. See [`training/README.md`](training/README.md) before attempting M3 training.

## Research status

No dataset images or pretrained weights are included. The code contains a reference interpretation of the manuscript's attention and geometry modules, but the exact attention implementation, 44 selected landmark indices, complete geometric cue formulas, prototype values, and original split/validation records have not been established from the available materials. Consequently this repository alone cannot reproduce the published accuracies.

Keep raw data, generated arrays, private configuration values, and experiment checkpoints outside Git. The source has been exercised with synthetic inputs; that is a functionality check, not an experimental validation.

## Citation and terms

Use the project title, **MVGP-FER**, until a publication citation is available. No repository-wide license has been supplied. RAF-DB and FER2013 have their own access and usage terms.
