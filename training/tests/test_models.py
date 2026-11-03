import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Model.attention import MultiContextCoordinateAttention
from Model.geometry import LandmarkMLP, PrototypeComparison
from training.build_model import load_prototypes, make_model
from training.dataset import compute_train_statistics


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_m1_and_m2_forward(self):
        images = torch.randn(2, 3, 64, 64)
        for variant in ("M1", "M2"):
            with self.subTest(variant=variant):
                model = make_model(variant).eval()
                with torch.no_grad():
                    result = model(images)
                self.assertEqual(result["logits"].shape, (2, 7))
                self.assertEqual(result["visual_features"].shape, (2, 640))

    def test_attention_preserves_shape(self):
        layer = MultiContextCoordinateAttention(32).eval()
        with torch.no_grad():
            output = layer(torch.randn(2, 32, 12, 9))
        self.assertEqual(output.shape, (2, 32, 12, 9))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_m3_forward_and_visual_fallback(self):
        # Random values are used only to test tensor shapes and control flow.
        model = make_model("M3", torch.randn(7, 16)).eval()
        images = torch.randn(2, 3, 64, 64)
        geometry = torch.randn(2, 104)
        cues = torch.randn(2, 16)
        valid = torch.tensor([True, False])
        with torch.no_grad():
            output = model(images, geometry, cues, valid)
        self.assertEqual(output["logits"].shape, (2, 7))
        torch.testing.assert_close(output["logits"][1], output["visual_logits"][1])
        self.assertEqual(float(output["gate"][1]), 0)

    def test_m3_all_landmarks_missing(self):
        model = make_model("M3", torch.randn(7, 16)).train()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            output = model(images, torch.zeros(2, 104), torch.zeros(2, 16), torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(output["logits"], output["visual_logits"])
        self.assertEqual(int(torch.count_nonzero(output["gate"])), 0)

    def test_m3_needs_real_prototypes(self):
        with self.assertRaisesRegex(ValueError, "prototype"):
            make_model("M3")
        with self.assertRaisesRegex(ValueError, "prototypes"):
            PrototypeComparison(torch.ones(7, 15))

    def test_prototype_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prototypes.json"
            path.write_text(json.dumps([[0.1] * 16] * 7), encoding="utf-8")
            self.assertEqual(load_prototypes(path).shape, (7, 16))

    def test_m3_gradient_flow(self):
        model = make_model("M3", torch.randn(7, 16)).train()
        output = model(torch.randn(2, 3, 32, 32), torch.randn(2, 104),
                       torch.randn(2, 16), torch.ones(2, dtype=torch.bool))
        torch.nn.functional.cross_entropy(output["logits"], torch.tensor([0, 1])).backward()
        self.assertIsNotNone(model.landmark_mlp.layers[0].weight.grad)
        self.assertIsNotNone(model.visual.classifier.weight.grad)

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
