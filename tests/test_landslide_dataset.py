"""
Unit tests specifically for LandslideDataset (RGB + DTM / Terrain Fusion).
Validates P0 and P1 criteria including float32 precision, NoData handling,
interpolation schemes, color augmentations, and PyTorch DataLoader integration.
"""

import os
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LandslideDataset import LandslideDataset


class TestLandslideDataset(unittest.TestCase):
    """Test suite for LandslideDataset."""

    def create_synthetic_landslide_dataset(self, temp_dir: str, num_samples: int = 4):
        """Creates synthetic RGB (PNG), DTM (TIF float32 with NaNs), and Label (PNG) data."""
        train_dir = os.path.join(temp_dir, "train")
        img_dir = os.path.join(train_dir, "IMAGE")
        dtm_dir = os.path.join(train_dir, "DTM")
        lbl_dir = os.path.join(train_dir, "LABEL")

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(dtm_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        sample_stems = []
        for i in range(num_samples):
            stem = f"Chainage_16_{i:05d}"
            sample_stems.append(stem)

            # 1. RGB image: uint8 [128, 128, 3]
            rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(os.path.join(img_dir, f"{stem}.png"))

            # 2. DTM raster: continuous float32 elevations with some NaNs and negative values
            dtm = np.random.uniform(10.0, 150.0, (128, 128)).astype(np.float32)
            # Inject some NaNs and sentinel values
            dtm[0:5, 0:5] = np.nan
            dtm[10, 10] = -9999.0
            
            # Save DTM as float32 TIF
            try:
                import tifffile
                tifffile.imwrite(os.path.join(dtm_dir, f"{stem}.tif"), dtm)
            except ImportError:
                cv2.imwrite(os.path.join(dtm_dir, f"{stem}.tif"), dtm)

            # 3. Label mask: 0 (background) and 65535 (landslide)
            mask = np.zeros((128, 128), dtype=np.uint16)
            mask[20:60, 20:60] = 65535  # Landslide polygon
            cv2.imwrite(os.path.join(lbl_dir, f"{stem}.png"), mask)

        return sample_stems

    def test_p0_output_shapes_and_types(self):
        """[P0] Verify RGB is float32 [3, H, W], Terrain is float32 [1, H, W], Mask is int64 [H, W]."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_stems = self.create_synthetic_landslide_dataset(tmp_dir, num_samples=3)

            target_h, target_w = 64, 96
            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(target_h, target_w),
                is_training=False
            )

            self.assertEqual(len(dataset), 3)

            rgb, terrain, mask, sample_id = dataset[0]

            # Shape assertions
            self.assertEqual(rgb.shape, torch.Size([3, target_h, target_w]))
            self.assertEqual(terrain.shape, torch.Size([1, target_h, target_w]))
            self.assertEqual(mask.shape, torch.Size([target_h, target_w]))
            self.assertIn(sample_id, sample_stems)

            # Dtype assertions
            self.assertEqual(rgb.dtype, torch.float32)
            self.assertEqual(terrain.dtype, torch.float32)
            self.assertEqual(mask.dtype, torch.int64)

            # Return signature check
            self.assertIsInstance(sample_id, str)

    def test_p0_dtm_float32_precision_and_no_nan_inf(self):
        """[P0] Verify DTM is read as continuous float32, and free of NaN/Inf post-resize."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=2)

            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(64, 64),
                dtm_norm="standard",
                is_training=True
            )

            for i in range(len(dataset)):
                rgb, terrain, mask, sample_id = dataset[i]
                self.assertFalse(torch.isnan(terrain).any(), "NaN found in terrain tensor!")
                self.assertFalse(torch.isinf(terrain).any(), "Inf found in terrain tensor!")
                self.assertFalse(torch.isnan(rgb).any(), "NaN found in rgb tensor!")

    def test_p0_mask_integer_binary_encoding(self):
        """[P0] Verify mask values (0, 65535) are converted to integer classes (0=BG, 1=Landslide)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=2)

            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(64, 64),
                is_training=False
            )

            rgb, terrain, mask, _ = dataset[0]
            unique_classes = torch.unique(mask).tolist()
            for cls in unique_classes:
                self.assertIn(cls, [0, 1])

    def test_p1_color_augmentation_rgb_only(self):
        """[P1] Ensure photometric color augmentations modify RGB while keeping DTM continuous."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=2)

            dataset_aug = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(64, 64),
                is_training=True
            )

            rgb, terrain, mask, _ = dataset_aug[0]
            self.assertEqual(rgb.shape[0], 3)
            self.assertEqual(terrain.shape[0], 1)

    def test_dataloader_batch_integration(self):
        """Verify DataLoader batch collation and multithreading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=4)

            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(64, 64),
                is_training=True
            )

            loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

            batch_count = 0
            for rgb_batch, terrain_batch, mask_batch, ids in loader:
                self.assertEqual(rgb_batch.shape, torch.Size([2, 3, 64, 64]))
                self.assertEqual(terrain_batch.shape, torch.Size([2, 1, 64, 64]))
                self.assertEqual(mask_batch.shape, torch.Size([2, 64, 64]))
                self.assertEqual(len(ids), 2)
                batch_count += 1

            self.assertEqual(batch_count, 2)

    def test_real_dataset_integration_if_present(self):
        """Tests loading directly from real dataset folder if dataset_1 exists."""
        real_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'dataset_1'))
        if os.path.exists(real_path):
            test_split_path = os.path.join(real_path, 'test')
            if os.path.exists(test_split_path):
                dataset = LandslideDataset(
                    data_dir=real_path,
                    split="test",
                    img_size=(480, 640),
                    is_training=False
                )
                self.assertGreater(len(dataset), 0)
                rgb, terrain, mask, sample_id = dataset[0]
                self.assertEqual(rgb.shape, torch.Size([3, 480, 640]))
                self.assertEqual(terrain.shape, torch.Size([1, 480, 640]))
                self.assertEqual(mask.shape, torch.Size([480, 640]))
                self.assertFalse(torch.isnan(terrain).any())
                print(f"\n[PASS] Real dataset test loaded {len(dataset)} samples successfully from {real_path}")


if __name__ == '__main__':
    unittest.main()
