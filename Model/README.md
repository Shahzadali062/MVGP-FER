# Model components

Shared components for the three ablation variants:

- `blocks.py`: stem and inverted residual blocks
- `attention.py`: directional and global attention
- `mixer.py`: gated spatial mixer
- `visual.py`: visual feature extraction and classifier
- `geometry.py`: landmark MLP and fixed prototype distance
- `fusion.py`: sample-dependent residual fusion
- `mvgp_fer.py`: combined model (M3)

The original trained architecture has not been verified against this implementation. M3 needs genuine 104-dimensional landmark features and a genuine class-prototype matrix. No placeholder prototypes are supplied.
