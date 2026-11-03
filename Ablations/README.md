# Ablation models

- **M1**: visual baseline (fused MBConv, inverted residual blocks, gated spatial mixer).
- **M2**: M1 with multi-context coordinate attention.
- **M3**: M2 with landmark MLP, fixed geometric prototypes, and adaptive logit fusion.

These are reference implementations from the manuscript description, not the authors' original experiment source. Exact attention details, some hyperparameters, and prototype values are not provided by the paper. No reported accuracy or parameter counts are asserted for this code.

Run from the repository root with `python -m training.train --variant M1 ...` (see `training/README.md`).
