"""Shape and fallback tests; these do not establish research accuracy."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from Ablations.M1.model import M1
from Ablations.M2.model import M2
from Ablations.M3.model import M3
from training.dataset import PreparedFERDataset, compute_train_statistics


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_variants_shapes(self):
        images = torch.randn(2, 3, 64, 64)
        for klass in (M1, M2):
            model = klass().eval()
            with torch.no_grad():
                output = model(images)
            self.assertEqual(tuple(output["logits"].shape), (2, 7))
            self.assertEqual(tuple(output["visual_features"].shape), (2, 640))

    def test_m3_fallback(self):
        model = M3(torch.zeros(7, 16)).eval()  # Test-only shape fixture, not real prototypes.
        images = torch.randn(2, 3, 64, 64)
        geometry = torch.randn(2, 104)
        cues = torch.randn(2, 16)
        with torch.no_grad():
            outputs = model(images, geometry, cues, torch.tensor([True, False]))
        self.assertEqual(tuple(outputs["logits"].shape), (2, 7))
        torch.testing.assert_close(outputs["logits"][1], outputs["visual_logits"][1])
        self.assertEqual(float(outputs["gate"][1]), 0.0)

    def test_training_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            np.save(p / "c1.npy", np.full(104, 10, dtype=np.float32))
            np.save(p / "c2.npy", np.full(104, 20, dtype=np.float32))
            np.save(p / "c3.npy", np.full(104, 200, dtype=np.float32))
            train = [dict(geometry_104_npy="c1.npy", landmark_status="valid"),
                     dict(geometry_104_npy="c2.npy", landmark_status="valid")]
            mean, std = compute_train_statistics(p / "manifest.csv", train)
            np.testing.assert_allclose(mean, 15.0)
            np.testing.assert_allclose(std, 5.0)
            self.assertNotAlmostEqual(float(mean[0]), 200.0)


if __name__ == "__main__":
    unittest.main()
