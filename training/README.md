# Training

This folder trains the visual-only M1 baseline. The attention and geometry modules used by the full model are not part of the current repository, so M2 and M3 training is unavailable.

Run the commands below from the repository root.

## Install

```powershell
python -m pip install -r training/requirements.txt
```

## Prepare the data

Use `dataset_preprocess` to create a `manifest.csv` and NumPy image arrays. The manifest must contain `split`, `paper_label` and `visual_npy`. Each image array must have shape `(3, 224, 224)` and dtype `float32`.

Keep datasets and checkpoints outside the Git repository.

## Train

```powershell
python -m training.train --dataset rafdb --manifest "D:/data/rafdb/manifest.csv" --out "D:/experiments/M1.pt"
```

For FER2013, change `--dataset` to `fer2013` and use its manifest.

Defaults: Adam, learning rate `3e-4`, batch size `8`, up to `100` epochs, patience `20`, seed `42`.

RAF-DB training uses a stratified 10% holdout for validation. FER2013 uses `PublicTest` for validation. These are choices made for this training script; they are not a verified copy of the manuscript's validation protocol.

## Evaluate

```powershell
python -m training.evaluate --checkpoint "D:/experiments/M1.pt" --manifest "D:/data/rafdb/manifest.csv" --split test
```

These scripts do not reproduce the paper's published results without the original experiment setup and model implementation.
