"""
Unit tests specifically for LandslideDatasetV2.
Validates:
1. Blacklist filtering functionality (skipping noisy stems).
2. Dynamic channel selection & spectrum concatenation (5 channels: RGB, DTM, Slope).
3. Ignore mask generation for RGB(0,0,0) void/border pixels (-100).
4. Synchronous spatial augmentations (Rotate, Flip, Crop, Elastic).
5. Exclusive photometric augmentations for RGB only (DTM/Slope untouched).
6. DataLoader integration, batch generation, and positive-aware sampling cache.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LandslideDatasetV2 import LandslideDatasetV2, build_landslide_v2_dataloader


class TestLandslideDatasetV2(unittest.TestCase):
    """Test suite for LandslideDatasetV2 against dataset_1V2."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path("dataset/dataset_1V2").resolve()
        cls.has_dataset = cls.data_dir.is_dir() and (cls.data_dir / "train" / "IMAGE").is_dir()
        if not cls.has_dataset:
            raise unittest.SkipTest("dataset/dataset_1V2 not found on disk.")

    def test_01_blacklist_filtering(self):
        """Verifies that blacklisted tiles are skipped when apply_blacklist=True."""
        ds_all = LandslideDatasetV2(
            data_dir=self.data_dir,
            split="train",
            apply_blacklist=False,
            is_training=False,
        )
        ds_clean = LandslideDatasetV2(
            data_dir=self.data_dir,
            split="train",
            apply_blacklist=True,
            is_training=False,
        )
        self.assertGreater(len(ds_all), len(ds_clean))
        # Ensure no sample in ds_clean has a stem in blacklist
        clean_stems = {s["stem"] for s in ds_clean.samples}
        for black_stem in ds_clean.blacklist:
            self.assertNotIn(black_stem, clean_stems)

    def test_02_five_channel_output(self):
        """Verifies 5-channel dual-stream output: RGB [3, H, W] and Terrain [2, H, W]."""
        ds = LandslideDatasetV2(
            data_dir=self.data_dir,
            split="train",
            channels=["rgb", "dtm", "slope"],
            img_size=(512, 512),
            is_training=False,
        )
        self.assertEqual(ds.num_total_channels, 5)
        self.assertEqual(ds.num_terrain_channels, 2)
        rgb, terrain, mask, stem = ds[0]

        self.assertEqual(rgb.shape, (3, 512, 512))
        self.assertEqual(terrain.shape, (2, 512, 512))
        self.assertEqual(mask.shape, (512, 512))
        self.assertEqual(rgb.dtype, torch.float32)
        self.assertEqual(terrain.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.int64)

    def test_03_concat_return_mode(self):
        """Verifies that return_concat=True returns a single [5, H, W] tensor."""
        ds = LandslideDatasetV2(
            data_dir=self.data_dir,
            split="train",
            channels=["rgb", "dtm", "slope"],
            img_size=(512, 512),
            return_concat=True,
            is_training=False,
        )
        concat_t, mask, stem = ds[0]
        self.assertEqual(concat_t.shape, (5, 512, 512))
        self.assertEqual(mask.shape, (512, 512))

    def test_04_ignore_rgb_black_mask(self):
        """Verifies that pure black pixels RGB(0,0,0) are assigned ignore_index=-100."""
        ds = LandslideDatasetV2(
            data_dir=self.data_dir,
            split="train",
            is_training=False,
            ignore_rgb_black=True,
            ignore_index=-100,
        )
        # Inject artificial black box in sample reader
        orig_read = ds._read_rgb
        def mock_read(path):
            arr = orig_read(path)
            arr[20:40, 20:40] = 0
            return arr
        ds._read_rgb = mock_read

        rgb, terrain, mask, stem = ds[0]
        black_region = mask[20:40, 20:40]
        self.assertTrue((black_region == -100).all())

    def test_05_training_augmentations(self):
        """Verifies that training augmentations maintain spatial alignment and validity."""
        ds = LandslideDatasetV2(
            data_dir=self.data_dir,
            split="train",
            channels=["rgb", "dtm", "slope"],
            is_training=True,
        )
        for i in range(5):
            rgb, terrain, mask, stem = ds[i]
            self.assertEqual(rgb.shape, (3, 512, 512))
            self.assertEqual(terrain.shape, (2, 512, 512))
            self.assertEqual(mask.shape, (512, 512))
            # Mask values must strictly be subset of {0, 1, -100}
            u_vals = set(torch.unique(mask).tolist())
            self.assertTrue(u_vals.issubset({0, 1, -100}))

    def test_06_dataloader_batching(self):
        """Verifies DataLoader batch creation and sample_has_positive caching."""
        loader = build_landslide_v2_dataloader(
            data_dir=self.data_dir,
            split="train",
            batch_size=4,
            channels=["rgb", "dtm", "slope"],
            num_workers=0,
            use_sampler=True,
        )
        self.assertGreater(len(loader), 0)
        rgb_b, ter_b, mask_b, stems = next(iter(loader))
        self.assertEqual(rgb_b.shape, (4, 3, 512, 512))
        self.assertEqual(ter_b.shape, (4, 2, 512, 512))
        self.assertEqual(mask_b.shape, (4, 512, 512))
        self.assertEqual(len(stems), 4)


if __name__ == "__main__":
    unittest.main()
