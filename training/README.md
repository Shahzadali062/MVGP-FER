## Install

```powershell
python -m pip install -r dataset_preprocess/requirements.txt
python -m pip install -r training/requirements.txt
```

## Prepare data

Use `dataset_preprocess/` to create a `manifest.csv` and normalized image arrays. Set `--manifest` to that CSV. The required columns are `split`, `paper_label`, and `visual_npy`. Input arrays are `float32` with shape `(3, 224, 224)`.

The script defaults to Adam, learning rate `3e-4`, batch size `8`, up to `100` epochs, patience `20`, and random seed `42`.

## Evaluate

```powershell
python -m training.evaluate --checkpoint "D:/experiments/M1.pt" --manifest "D:/prepared/rafdb/manifest.csv" --split test
python -m training.evaluate --checkpoint "D:/experiments/M2.pt" --manifest "D:/prepared/fer2013/manifest.csv" --split PrivateTest
```


## Tests

```powershell
python -m unittest discover -s training/tests -v
python -m unittest discover -s dataset_preprocess/tests -v
```
