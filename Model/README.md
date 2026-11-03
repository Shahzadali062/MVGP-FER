# Model

The model code is split into reusable parts:

- `blocks.py` — convolutional stem and residual blocks
- `mixer.py` — spatial token mixing
- `visual.py` — visual feature extractor and seven-class classifier
- `fusion.py` — adaptive logit fusion
- `mvgp_fer.py` — assembly interface for externally supplied model components

`attention.py` and `geometry.py` have been removed. The visual-only baseline still works, but the full MVGP-FER model needs the original attention module, landmark model and class prototypes. These are not included here. The public training script supports the visual baseline only.
