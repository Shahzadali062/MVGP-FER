# Training

The `--variant` option selects one of three model configurations:

- **M1:** visual backbone and gated spatial mixer
- **M2:** M1 with multi-context coordinate attention
- **M3:** M2 with landmark MLP, fixed class prototypes, and adaptive logit fusion

This is a reference implementation of the architecture described in the manuscript. It has not been verified against the original training source or published scores.

Run all commands from the repository root.

## Install

```powershell
python -m pip install -r dataset_preprocess/requirements.txt
python -m pip install -r training/requirements.txt
```

## Prepare data

Use `dataset_preprocess/` to create a `manifest.csv` and normalized image arrays. Set `--manifest` to that CSV. The required columns are `split`, `paper_label`, and `visual_npy`. Input arrays are `float32` with shape `(3, 224, 224)`.

For M3, each row with valid landmarks must additionally contain either `geometry_104_npy`, or both `landmarks_88_npy` and `geometric_cues_16_npy`. These refer to raw, unstandardized arrays. `landmark_status` should be `valid` when the features are present. The script calculates the 104-dimensional normalization statistics from training samples only.

**The current preprocessing tools do not calculate all 16 geometric cues.** Supply them using your verified original code. You must also supply the original **7 x 16** fixed prototype matrix as a JSON array. Do not substitute arbitrary values when reporting research results.

## Train M1 or M2

```powershell
python -m training.train --variant M1 --dataset rafdb --manifest "D:/prepared/rafdb/manifest.csv" --out "D:/experiments/M1.pt"
python -m training.train --variant M2 --dataset fer2013 --manifest "D:/prepared/fer2013/manifest.csv" --out "D:/experiments/M2.pt"
```

## Train M3

```powershell
python -m training.train --variant M3 --dataset rafdb --manifest "D:/prepared/rafdb/manifest.csv" --prototypes "D:/private/prototypes.json" --out "D:/experiments/M3.pt"
```

The script defaults to Adam, learning rate `3e-4`, batch size `8`, up to `100` epochs, patience `20`, and random seed `42`.

RAF-DB uses a stratified 10% holdout from training samples for validation. FER2013 uses `PublicTest` for validation. These are explicit choices in this release and are **not verified** as the original manuscript protocol.

## Evaluate

```powershell
python -m training.evaluate --checkpoint "D:/experiments/M1.pt" --manifest "D:/prepared/rafdb/manifest.csv" --split test
python -m training.evaluate --checkpoint "D:/experiments/M2.pt" --manifest "D:/prepared/fer2013/manifest.csv" --split PrivateTest
```

Keep raw datasets, generated arrays, prototypes, and checkpoints outside the public repository.

## Tests

```powershell
python -m unittest discover -s training/tests -v
python -m unittest discover -s dataset_preprocess/tests -v
```
