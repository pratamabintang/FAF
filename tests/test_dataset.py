"""
Unit tests for Dataset Loaders (FusionModelDataset and PST900Dataset).
Uses synthetic mock datasets to test loading, augmentation, and collation.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from FusionModelDataset import FusionModelDataset
    HAS_FUSION_DATASET = True
except ImportError:
    HAS_FUSION_DATASET = False

try:
    from PST900Dataset import PST900Dataset, get_pst900_palette
    HAS_PST900_DATASET = True
except ImportError:
    HAS_PST900_DATASET = False


class TestDatasetLoaders(unittest.TestCase):
    """Test suite for FusionModelDataset and PST900Dataset loaders."""

    def create_mock_fusion_dataset(self, temp_dir: str, num_samples: int = 4):
        """Creates a mock dataset folder structure for FusionModelDataset."""
        images_dir = os.path.join(temp_dir, "images")
        labels_dir = os.path.join(temp_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        sample_names = []
        for i in range(num_samples):
            name = f"sample_{i:03d}"
            sample_names.append(name)

            # Create 4-channel image (RGB + IR) of size 64x64
            img4 = np.random.randint(0, 256, (64, 64, 4), dtype=np.uint8)
            Image.fromarray(img4).save(os.path.join(images_dir, f"{name}.png"))

            # Create 1-channel label mask (values 0, 1, 2) of size 64x64
            mask = np.random.randint(0, 3, (64, 64), dtype=np.uint8)
            Image.fromarray(mask).save(os.path.join(labels_dir, f"{name}.png"))

        # Write split text files
        for split in ["train_", "test"]:
            with open(os.path.join(temp_dir, f"{split}.txt"), "w") as f:
                f.write("\n".join(sample_names))

        return sample_names

    def create_mock_pst900_dataset(self, temp_dir: str, num_samples: int = 3):
        """Creates a mock dataset folder structure for PST900Dataset."""
        for split in ["train", "test"]:
            rgb_dir = os.path.join(temp_dir, split, "rgb")
            thermal_dir = os.path.join(temp_dir, split, "thermal")
            labels_dir = os.path.join(temp_dir, split, "labels")
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(thermal_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            for i in range(num_samples):
                name = f"pst_{split}_{i:03d}"
                # RGB image 64x64
                rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(os.path.join(rgb_dir, f"{name}.png"))

                # Grayscale thermal image 64x64
                thermal = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
                Image.fromarray(thermal).save(os.path.join(thermal_dir, f"{name}.png"))

                # 1-channel label mask (values 0-4)
                lbl = np.random.randint(0, 5, (64, 64), dtype=np.uint8)
                Image.fromarray(lbl).save(os.path.join(labels_dir, f"{name}.png"))

    @unittest.skipUnless(HAS_FUSION_DATASET, "albumentations not installed")
    def test_fusion_model_dataset_train_split(self):
        """Verify FusionModelDataset training split with augmentations."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_names = self.create_mock_fusion_dataset(tmp_dir, num_samples=4)

            dataset = FusionModelDataset(
                data_dir=tmp_dir,
                split="train_",
                have_label=True,
                rgb_size=(64, 64),
                ir_size=(64, 64)
            )

            self.assertEqual(len(dataset), 4)

            rgb, ir, mask, name = dataset[0]

            # Verify shapes
            self.assertEqual(rgb.shape, torch.Size([3, 64, 64]))
            self.assertEqual(ir.shape, torch.Size([1, 64, 64]))
            self.assertEqual(mask.shape, torch.Size([64, 64]))
            self.assertIn(name, sample_names)

            # Verify tensor data types
            self.assertEqual(rgb.dtype, torch.float32)
            self.assertEqual(ir.dtype, torch.float32)
            self.assertEqual(mask.dtype, torch.long)

    @unittest.skipUnless(HAS_FUSION_DATASET, "albumentations not installed")
    def test_fusion_model_dataset_eval_split(self):
        """Verify FusionModelDataset test split (deterministic evaluation mode)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_mock_fusion_dataset(tmp_dir, num_samples=3)

            dataset = FusionModelDataset(
                data_dir=tmp_dir,
                split="test",
                have_label=True,
                rgb_size=(64, 64),
                ir_size=(64, 64)
            )

            self.assertEqual(len(dataset), 3)
            rgb, ir, mask, name = dataset[0]
            self.assertEqual(rgb.shape, torch.Size([3, 64, 64]))
            self.assertEqual(ir.shape, torch.Size([1, 64, 64]))

    @unittest.skipUnless(HAS_PST900_DATASET, "albumentations not installed")
    def test_pst900_dataset(self):
        """Verify PST900Dataset loading, thermal 1-channel shape and label palette."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_mock_pst900_dataset(tmp_dir, num_samples=3)

            dataset = PST900Dataset(
                data_dir=tmp_dir,
                split="train",
                rgb_size=(64, 64),
                thermal_size=(64, 64),
                use_augmentation=False
            )

            self.assertEqual(len(dataset), 3)

            rgb, thermal, mask, name = dataset[0]
            self.assertEqual(rgb.shape, torch.Size([3, 64, 64]))
            self.assertEqual(thermal.shape, torch.Size([1, 64, 64]))
            self.assertEqual(mask.shape, torch.Size([64, 64]))

            palette = get_pst900_palette()
            self.assertEqual(len(palette), 5)

    @unittest.skipUnless(HAS_FUSION_DATASET, "albumentations not installed")
    def test_dataloader_batch_collation(self):
        """Verify PyTorch DataLoader batch collation on synthetic dataset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_mock_fusion_dataset(tmp_dir, num_samples=4)

            dataset = FusionModelDataset(
                data_dir=tmp_dir,
                split="train_",
                have_label=True,
                rgb_size=(64, 64),
                ir_size=(64, 64)
            )

            loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

            batch_count = 0
            for rgb_b, ir_b, mask_b, names in loader:
                self.assertEqual(rgb_b.shape, torch.Size([2, 3, 64, 64]))
                self.assertEqual(ir_b.shape, torch.Size([2, 1, 64, 64]))
                self.assertEqual(mask_b.shape, torch.Size([2, 64, 64]))
                self.assertEqual(len(names), 2)
                batch_count += 1

            self.assertEqual(batch_count, 2)


if __name__ == '__main__':
    unittest.main()
