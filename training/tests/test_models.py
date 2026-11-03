"""Checks for the currently available model components."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from Model.mvgp_fer import MVGPFER
from Model.visual import VisualOnly
from training.build_model import make_model
from training.dataset import compute_train_statistics


class ConstantPrototypes(nn.Module):
    def forward(self, cues):
        batch = cues.shape[0]
        return cues.new_zeros((batch, 7)), cues.new_tensor(0.35)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_visual_baseline(self):
        model = make_model("M1").eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(output["logits"].shape, (2, 7))
        self.assertEqual(output["visual_features"].shape, (2, 640))
        self.assertIsInstance(model, VisualOnly)

    def test_removed_variants_rejected(self):
        for name in ("M2", "M3"):
            with self.assertRaisesRegex(ValueError, "not included"):
                make_model(name)

    def test_model_assembly_accepts_supplied_modules(self):
        model = MVGPFER(
            landmark_mlp=nn.Linear(104, 7),
            prototype_matcher=ConstantPrototypes(),
            attention_factory=lambda channels: nn.Identity(),
        ).eval()
        images = torch.randn(2, 3, 64, 64)
        geometry = torch.randn(2, 104)
        cues = torch.randn(2, 16)
        valid = torch.tensor([True, False])
        with torch.no_grad():
            outputs = model(images, geometry, cues, valid)
        self.assertEqual(outputs["logits"].shape, (2, 7))
        torch.testing.assert_close(outputs["logits"][1], outputs["visual_logits"][1])
        self.assertEqual(outputs["gate"][1].item(), 0.0)

    def test_training_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.save(root / "a.npy", np.full(104, 10, dtype=np.float32))
            np.save(root / "b.npy", np.full(104, 20, dtype=np.float32))
            train = [
                {"geometry_104_npy": "a.npy", "landmark_status": "valid"},
                {"geometry_104_npy": "b.npy", "landmark_status": "valid"},
            ]
            mean, std = compute_train_statistics(root / "manifest.csv", train)
            np.testing.assert_allclose(mean, 15.0)
            np.testing.assert_allclose(std, 5.0)


if __name__ == "__main__":
    unittest.main()
