"""Full M3 reference model; requires genuine prototype and geometry inputs."""

import torch
from torch import nn

from .visual import VisualBranch
from .geometry import LandmarkMLP, PrototypeComparison
from .fusion import AdaptiveLogitFusion


class MVGPFER(nn.Module):
    def __init__(self, prototypes: torch.Tensor):
        super().__init__()
        self.visual = VisualBranch(use_attention=True)
        self.landmark_mlp = LandmarkMLP()
        self.prototype_matcher = PrototypeComparison(prototypes)
        self.fusion = AdaptiveLogitFusion()

    def forward(self, images: torch.Tensor, geometry: torch.Tensor | None = None,
                cues: torch.Tensor | None = None, valid: torch.Tensor | None = None):
        features, visual_logits = self.visual(images)
        b = images.shape[0]
        if geometry is None or cues is None or valid is None:
            raise ValueError("M3 requires standardized (B,104) geometry, raw (B,16) cues and valid mask")
        if geometry.shape != (b, 104) or cues.shape != (b, 16) or valid.shape != (b,):
            raise ValueError("Incorrect geometry/cues/valid shapes")
        if not torch.isfinite(geometry).all() or not torch.isfinite(cues).all():
            raise ValueError("Non-finite geometric inputs")
        # BatchNorm1d in the landmark MLP requires more than one valid row
        # in training mode. When only one is valid, use batch statistics from
        # the remaining zeroed placeholders, and mask its output afterward.
        lm_logits = self.landmark_mlp(geometry)
        proto_logits, alpha = self.prototype_matcher(cues)
        corrected = lm_logits + alpha * proto_logits
        logits, gate = self.fusion(features, visual_logits, corrected, proto_logits, valid)
        return {"logits": logits, "visual_logits": visual_logits,
                "visual_features": features, "landmark_logits": lm_logits,
                "prototype_logits": proto_logits, "gate": gate}
