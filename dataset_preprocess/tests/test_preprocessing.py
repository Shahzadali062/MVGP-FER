"""Small checks that do not require downloading a dataset."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_preprocess.export import prepare_dataset
from dataset_preprocess.images import preprocess_image
from dataset_preprocess.landmarks import normalize_landmarks
from dataset_preprocess.labels import FER2013, RAFDB
from dataset_preprocess.standardize import apply_training_stats, fit_training_stats


class PreprocessingTests(unittest.TestCase):
    def test_image(self):
        source = Image.new("RGB", (50, 40), (100, 125, 150))
        array, rgb = preprocess_image(source)
        self.assertEqual(array.shape, (3, 224, 224))
        self.assertEqual(array.dtype, np.float32)
        self.assertEqual(rgb.shape, (224, 224, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])
        self.assertTrue(np.isfinite(array).all())

    def test_labels(self):
        self.assertEqual(FER2013[6], 4)
        self.assertEqual(RAFDB[1], 6)

    def test_landmark_center_and_scale(self):
        points = np.zeros((468, 2), dtype=np.float64)
        points[33] = [0.2, 0.5]
        points[263] = [0.8, 0.5]
        points[10] = [0.5, 0.1]
        points[152] = [0.5, 0.9]
        coords = normalize_landmarks(points, tuple(range(44)))
        self.assertEqual(coords.shape, (88,))
        np.testing.assert_allclose(coords[66:68], [-0.5, 0.0], atol=1e-6)
        self.assertTrue(np.isfinite(coords).all())

    def test_training_stats(self):
        train = np.array([[1., 2.], [3., 6.]], dtype=np.float64)
        mean, std = fit_training_stats(train)
        np.testing.assert_allclose(mean, [2., 4.])
        np.testing.assert_allclose(apply_training_stats(train, mean, std).mean(axis=0), [0., 0.])

    def test_fer2013_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "fer2013.csv"
            pixels = " ".join(["150"] * (48 * 48))
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["emotion", "pixels", "Usage"])
                writer.writeheader()
                writer.writerow({"emotion": "6", "pixels": pixels, "Usage": "PublicTest"})
            output = root / "result"
            self.assertEqual(prepare_dataset("fer2013", csv_path, output), 1)
            with (output / "manifest.csv").open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["paper_class"], "neutral")
            self.assertEqual(row["split"], "PublicTest")
            arr = np.load(output / row["visual_npy"], allow_pickle=False)
            self.assertEqual(arr.shape, (3, 224, 224))

    def test_rafdb_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "EmoLabel").mkdir()
            (root / "Image" / "aligned").mkdir(parents=True)
            (root / "EmoLabel" / "list_patition_label.txt").write_text("train_0001.jpg 1\n", encoding="utf-8")
            Image.new("RGB", (36, 36), (100, 110, 120)).save(root / "Image" / "aligned" / "train_0001_aligned.jpg")
            output = root / "result"
            self.assertEqual(prepare_dataset("rafdb", root, output), 1)
            with (output / "manifest.csv").open(encoding="utf-8", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["paper_class"], "surprise")
            self.assertEqual(row["split"], "train")


if __name__ == "__main__":
    unittest.main()
