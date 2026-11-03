# Model

This folder holds the network components used in the reference implementation.

| File | Purpose |
| --- | --- |
| `blocks.py` | Fused MBConv stem and inverted residual blocks |
| `attention.py` | Multi-context coordinate attention |
| `mixer.py` | Gated facial spatial mixing |
| `visual.py` | Visual branch and classifier |
| `geometry.py` | Landmark MLP and prototype matching |
| `fusion.py` | Adaptive logit fusion |
| `mvgp_fer.py` | M3 model assembly |



The architecture follows the manuscript, but some implementation details cannot be checked against the original experimental source. In particular, the exact attention parameterization, prototype matrix, and geometric-cue formulas are not available here. This code has not been validated as a reproduction of the manuscript results.
