

import torch
from torch import nn

from .attention import MultiContextCoordinateAttention
from .fusion import AdaptiveLogitFusion
from .geometry import LandmarkMLP, PrototypeComparison
from .visual import VisualBranch


class MVGPFER(nn.Module):
    """M3 reference implementation.

    Expects genuine standardized landmark features and raw geometric cues.
    Class prototype values must come from the original experiment records.
    """

    def __init__(self, prototypes: torch.Tensor):
        super().__init__()
        self.visual = VisualBranch(attention_factory=MultiContextCoordinateAttention)
        self.landmark_mlp = LandmarkMLP()
        self.prototype_matcher = PrototypeComparison(prototypes)
        self.fusion = AdaptiveLogitFusion()

    def forward(self, images: torch.Tensor, geometry: torch.Tensor,
                cues: torch.Tensor, valid: torch.Tensor):
        features, visual_logits = self.visual(images)
        batch = images.shape[0]
        if geometry.shape != (batch, 104) or cues.shape != (batch, 16):
            raise ValueError("Expected geometry (B,104) and raw cues (B,16)")
        if valid.shape != (batch,) or valid.dtype != torch.bool:
            raise ValueError("Expected a bool landmark-valid mask with shape (B,)")
        if not bool(torch.isfinite(geometry).all()) or not bool(torch.isfinite(cues).all()):
            raise ValueError("Landmark features and cues must be finite")

        if not bool(valid.any()):
            zero = visual_logits.new_zeros((batch, 7))
            gate = visual_logits.new_zeros((batch, 1))
            return {
                "logits": visual_logits,
                "visual_logits": visual_logits,
                "visual_features": features,
                "landmark_logits": zero,
                "prototype_logits": zero,
                "gate": gate,
            }

        landmark_logits = self.landmark_mlp(geometry)
        prototype_logits, alpha = self.prototype_matcher(cues)
        correction = landmark_logits + alpha * prototype_logits
        logits, gate = self.fusion(features, visual_logits, correction, prototype_logits, valid)
        return {
            "logits": logits,
            "visual_logits": visual_logits,
            "visual_features": features,
            "landmark_logits": landmark_logits,
            "prototype_logits": prototype_logits,
            "gate": gate,
        }
