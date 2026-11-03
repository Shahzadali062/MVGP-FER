"""Model assembly for projects with their own attention and geometry modules.

The public repository does not include the attention implementation, landmark
model or fixed prototype values. Supply those components explicitly.
"""

from collections.abc import Callable

import torch
from torch import nn

from .fusion import AdaptiveLogitFusion
from .visual import VisualBranch


class MVGPFER(nn.Module):
    def __init__(
        self,
        landmark_mlp: nn.Module,
        prototype_matcher: nn.Module,
        attention_factory: Callable[[int], nn.Module],
    ):
        super().__init__()
        if not isinstance(landmark_mlp, nn.Module):
            raise TypeError("landmark_mlp must be a torch.nn.Module")
        if not isinstance(prototype_matcher, nn.Module):
            raise TypeError("prototype_matcher must be a torch.nn.Module")
        if not callable(attention_factory):
            raise TypeError("attention_factory must be callable")

        self.visual = VisualBranch(attention_factory=attention_factory)
        self.landmark_mlp = landmark_mlp
        self.prototype_matcher = prototype_matcher
        self.fusion = AdaptiveLogitFusion()

    def forward(
        self,
        images: torch.Tensor,
        geometry: torch.Tensor,
        cues: torch.Tensor,
        valid: torch.Tensor,
    ):
        features, visual_logits = self.visual(images)
        batch = images.shape[0]
        if geometry.shape != (batch, 104) or cues.shape != (batch, 16):
            raise ValueError("Expected geometry (B,104) and cues (B,16)")
        if valid.shape != (batch,) or valid.dtype != torch.bool:
            raise ValueError("Expected a boolean validity mask of shape (B,)")
        if not torch.isfinite(geometry).all() or not torch.isfinite(cues).all():
            raise ValueError("Geometry and cue values must be finite")

        landmark_logits = self.landmark_mlp(geometry)
        prototype_logits, alpha = self.prototype_matcher(cues)
        if landmark_logits.shape != (batch, 7) or prototype_logits.shape != (batch, 7):
            raise ValueError("Landmark and prototype modules must return (B,7) logits")
        corrected_logits = landmark_logits + alpha * prototype_logits
        logits, gate = self.fusion(
            features, visual_logits, corrected_logits, prototype_logits, valid
        )
        return {
            "logits": logits,
            "visual_logits": visual_logits,
            "visual_features": features,
            "landmark_logits": landmark_logits,
            "prototype_logits": prototype_logits,
            "gate": gate,
        }
