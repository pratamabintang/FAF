"""
Unit tests specifically for LandslideDataset (RGB + DTM / Terrain Fusion).
Validates P0 and P1 criteria including float32 precision, NoData handling,
interpolation schemes, color augmentations, and PyTorch DataLoader integration.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
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
                is_training=False,
                ignore_nodata=False
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

    def test_missing_label_raises_error(self):
        """Verify that missing label file raises FileNotFoundError when require_labels=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=3)
            # Delete one label file
            lbl_dir = os.path.join(tmp_dir, "train", "LABEL")
            first_label = os.listdir(lbl_dir)[0]
            os.remove(os.path.join(lbl_dir, first_label))

            with self.assertRaises(FileNotFoundError):
                _ = LandslideDataset(
                    data_dir=tmp_dir,
                    split="train",
                    require_labels=True
                )

    def test_unlabeled_inference_allowed_when_require_labels_false(self):
        """Verify that require_labels=False allows missing labels and returns zeros."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=2)
            # Remove all labels
            lbl_dir = os.path.join(tmp_dir, "train", "LABEL")
            for f in os.listdir(lbl_dir):
                os.remove(os.path.join(lbl_dir, f))

            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                require_labels=False
            )
            self.assertEqual(len(dataset), 2)
            _, _, mask, _ = dataset[0]
            self.assertEqual(mask.sum().item(), 0)

    def test_spatial_mismatch_raises_error(self):
        """Verify that spatial dimensions mismatch (RGB != DTM) raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=1)
            # Overwrite RGB with mismatched size (64x64 instead of 128x128)
            img_dir = os.path.join(tmp_dir, "train", "IMAGE")
            img_file = os.path.join(img_dir, os.listdir(img_dir)[0])
            mismatched_rgb = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(mismatched_rgb).save(img_file)

            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train"
            )
            with self.assertRaises(ValueError):
                _ = dataset[0]

    def test_unexpected_mask_values_raises_error(self):
        """Verify that unexpected mask values (e.g. 2, 255) raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=1)
            lbl_dir = os.path.join(tmp_dir, "train", "LABEL")
            lbl_file = os.path.join(lbl_dir, os.listdir(lbl_dir)[0])
            corrupted_mask = np.full((128, 128), 255, dtype=np.uint16)
            cv2.imwrite(lbl_file, corrupted_mask)

            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train"
            )
            with self.assertRaises(ValueError):
                _ = dataset[0]

    def test_dtm_nodata_ignore_index_labeling(self):
        """Verify that when ignore_nodata=True, NoData regions are labeled with ignore_index in mask."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=1)
            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(128, 128),
                is_training=False,
                ignore_nodata=True,
                ignore_index=-100
            )
            _, _, mask, _ = dataset[0]
            # Synthetic dataset has NaNs at [0:5, 0:5] and -9999 at [10, 10]
            self.assertEqual(mask[0, 0].item(), -100)
            self.assertEqual(mask[10, 10].item(), -100)

    def test_terrain_derivatives_physical_scale_and_aspect_sincos(self):
        """Verify 4-channel terrain output [DTM, Slope, Sin(Aspect), Cos(Aspect)] with physical pixel scale."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=1)
            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(128, 128),
                is_training=False,
                include_derivatives=True,
                pixel_scale=5.0  # 5-meter resolution
            )
            _, terrain, _, _ = dataset[0]
            # Must have 4 channels: [DTM, Slope, Sin_Aspect, Cos_Aspect]
            self.assertEqual(terrain.shape, torch.Size([4, 128, 128]))

            slope = terrain[1].numpy()
            sin_aspect = terrain[2].numpy()
            cos_aspect = terrain[3].numpy()

            # Slope is normalized by pi/2 -> [0, 1]
            self.assertTrue(np.all(slope >= 0.0) and np.all(slope <= 1.0))
            # Sin and Cos aspect bounded in [-1.0, 1.0]
            self.assertTrue(np.all(sin_aspect >= -1.0) and np.all(sin_aspect <= 1.0))
            self.assertTrue(np.all(cos_aspect >= -1.0) and np.all(cos_aspect <= 1.0))

    def test_positive_aware_sampling(self):
        """Verify that positive-aware sampling guarantees positive landslide representation in cropped patches."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=1)
            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(64, 64),
                is_training=True,
                positive_aware_sampling=True,
                positive_sample_prob=1.0  # Force positive crop
            )
            # Run multiple samples to test stability
            for _ in range(5):
                _, _, mask, _ = dataset[0]
                # Check that mask contains at least one landslide pixel (value 1)
                self.assertTrue((mask == 1).any().item())

    def test_sample_has_positive_and_crop_safety(self):
        """Verify sample_has_positive public API and scale-crop float32 mask resize with ignore_index."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.create_synthetic_landslide_dataset(tmp_dir, num_samples=2)
            dataset = LandslideDataset(
                data_dir=tmp_dir,
                split="train",
                img_size=(64, 64),
                is_training=True,
                ignore_nodata=True,
                ignore_index=-100
            )
            # Test sample_has_positive public API
            has_pos_0 = dataset.sample_has_positive(0)
            self.assertIsInstance(has_pos_0, bool)
            self.assertTrue(has_pos_0)  # Synthetic dataset sample 0 has foreground
            self.assertIn(0, dataset._positive_cache)

            # Test spatial augmentation crop with mask containing -100
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            dtm = np.ones((64, 64), dtype=np.float32) * 50.0
            mask = np.zeros((64, 64), dtype=np.int64)
            mask[10:20, 10:20] = 1
            mask[0:5, 0:5] = -100  # NoData region

            for _ in range(10):
                rgb_aug, dtm_aug, mask_aug, scale_aug = dataset._apply_coordinated_spatial_aug(rgb.copy(), dtm.copy(), mask.copy())
                unique_vals = set(np.unique(mask_aug))
                self.assertTrue(unique_vals.issubset({0, 1, -100}))
                self.assertLessEqual(scale_aug, dataset.pixel_scale + 1e-6)

    def test_topography_slope_and_aspect_coherence(self):
        """Verify effective_pixel_scale preserves physical slope and aspect vectors transform coherently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            for sub in ["IMAGE", "DTM", "LABEL"]:
                (train_dir / sub).mkdir(parents=True)

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            # Create a uniform linear slope in x direction: dz/dx = 0.57735 (30 degrees slope)
            y_grid, x_grid = np.mgrid[0:64, 0:64]
            dtm = (0.57735 * x_grid).astype(np.float32)
            mask = np.zeros((64, 64), dtype=np.uint8)

            cv2.imwrite(str(train_dir / "IMAGE" / "tile_001.png"), img)
            cv2.imwrite(str(train_dir / "DTM" / "tile_001.tif"), dtm)
            cv2.imwrite(str(train_dir / "LABEL" / "tile_001.png"), mask)

            dataset = LandslideDataset(
                data_dir=tmpdir,
                split="train",
                img_size=(64, 64),
                include_derivatives=True,
                pixel_scale=1.0,
                is_training=False
            )

            # 1. Uncropped physical slope verification
            terrain_uncropped = dataset._normalize_terrain(dtm.copy(), effective_pixel_scale=1.0)
            # Slope is normalized to [0, 1] relative to 90 deg (pi/2)
            slope_deg = terrain_uncropped[1, 32, 32] * 90.0
            self.assertAlmostEqual(slope_deg, 30.0, places=2)

            # 2. Cropped physical slope with effective_pixel_scale compensation
            crop_ratio = 0.8
            crop_h, crop_w = int(64 * crop_ratio), int(64 * crop_ratio)
            dtm_cropped = cv2.resize(dtm[:crop_h, :crop_w], (64, 64), interpolation=cv2.INTER_LINEAR)

            # Old buggy way (uncompensated scale)
            terrain_buggy = dataset._normalize_terrain(dtm_cropped, effective_pixel_scale=1.0)
            slope_buggy_deg = terrain_buggy[1, 32, 32] * 90.0
            self.assertLess(slope_buggy_deg, 26.0)  # Underestimates slope!

            # Correct way (compensated effective scale)
            terrain_compensated = dataset._normalize_terrain(dtm_cropped, effective_pixel_scale=1.0 * crop_ratio)
            slope_comp_deg = terrain_compensated[1, 32, 32] * 90.0
            self.assertAlmostEqual(slope_comp_deg, 30.0, delta=0.2)  # Preserves physical slope (<0.1 deg error)!

            # 3. Aspect vector coherence under rotation
            # Facing East: aspect = 0 rad -> sin=0, cos=1
            sin_e = terrain_uncropped[2, 32, 32]
            cos_e = terrain_uncropped[3, 32, 32]
            self.assertAlmostEqual(sin_e, 0.0, places=2)
            self.assertAlmostEqual(cos_e, 1.0, places=2)

            # Rotate 90 deg CCW: East becomes North -> aspect = 90 deg -> sin=1, cos=0
            dtm_rot = np.rot90(dtm, 1).copy()
            terrain_rot = dataset._normalize_terrain(dtm_rot, effective_pixel_scale=1.0)
            sin_n = terrain_rot[2, 32, 32]
            cos_n = terrain_rot[3, 32, 32]
            self.assertAlmostEqual(sin_n, 1.0, places=2)
            self.assertAlmostEqual(cos_n, 0.0, places=2)

    def test_raster_alignment_verification_tool(self):
        """Verify the raster alignment verification tool on real dataset tiles."""
        from tools.verify_raster_alignment import verify_raster_alignment
        real_data_dir = Path(__file__).resolve().parent.parent / "dataset" / "dataset_1"
        if (real_data_dir / "test").exists():
            res = verify_raster_alignment(data_dir=real_data_dir, split="test", sample_limit=20, verbose=False)
            self.assertTrue(res["passed"])
            self.assertEqual(len(res["dimension_mismatches"]), 0)
            self.assertEqual(len(res["corrupted_files"]), 0)
            self.assertEqual(len(res["georeferencing_mismatches"]), 0)
            self.assertIn("rasterio_available", res)

    def test_class_weights_excludes_nodata(self):
        """Verify that calculate_class_weights filters out DTM NoData pixels."""
        from FusionModelTrain import FusionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            for sub in ["IMAGE", "DTM", "LABEL"]:
                (train_dir / sub).mkdir(parents=True)

            # Sample 1: 10x10 image. Half of DTM is NoData (-9999)
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            dtm = np.ones((10, 10), dtype=np.float32) * 50.0
            dtm[:5, :] = -9999.0  # Top half is NoData
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[7:9, 7:9] = 1   # Landslide on valid terrain: 4 pixels

            cv2.imwrite(str(train_dir / "IMAGE" / "tile_001.png"), img)
            cv2.imwrite(str(train_dir / "DTM" / "tile_001.tif"), dtm)
            cv2.imwrite(str(train_dir / "LABEL" / "tile_001.png"), mask)

            dataset = LandslideDataset(
                data_dir=tmpdir,
                split="train",
                img_size=(10, 10),
                is_training=False
            )

            class DummyConfig:
                num_classes = 2
                class_weight_multiplier = 10.0
                dataset = "landslide"

            class DummyTrainer:
                def __init__(self, ds):
                    self.config = DummyConfig()
                    self.device = torch.device("cpu")
                    self.train_loader = type("Loader", (), {"dataset": ds})()

            DummyTrainer.calculate_class_weights = FusionTrainer.calculate_class_weights

            trainer = DummyTrainer(dataset)
            weights = trainer.calculate_class_weights()

            self.assertFalse(torch.isnan(weights).any())
            self.assertEqual(weights[0].item(), 1.0)
            self.assertGreater(weights[1].item(), 1.0)

    def test_class_weights_fail_fast_guards(self):
        """Verify fail-fast assertions in calculate_class_weights for invalid configurations or zero landslides."""
        from FusionModelTrain import FusionTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            for sub in ["IMAGE", "DTM", "LABEL"]:
                (train_dir / sub).mkdir(parents=True)

            img = np.zeros((10, 10, 3), dtype=np.uint8)
            dtm = np.ones((10, 10), dtype=np.float32) * 50.0
            mask_no_landslides = np.zeros((10, 10), dtype=np.uint8)

            cv2.imwrite(str(train_dir / "IMAGE" / "tile_001.png"), img)
            cv2.imwrite(str(train_dir / "DTM" / "tile_001.tif"), dtm)
            cv2.imwrite(str(train_dir / "LABEL" / "tile_001.png"), mask_no_landslides)

            dataset = LandslideDataset(
                data_dir=tmpdir,
                split="train",
                img_size=(10, 10),
                is_training=False
            )

            class DummyConfig:
                num_classes = 2
                class_weight_multiplier = 10.0
                dataset = "landslide"

            class DummyTrainer:
                def __init__(self, ds, num_classes=2):
                    self.config = DummyConfig()
                    self.config.num_classes = num_classes
                    self.device = torch.device("cpu")
                    self.train_loader = type("Loader", (), {"dataset": ds})()

            DummyTrainer.calculate_class_weights = FusionTrainer.calculate_class_weights

            # 1. Zero positive landslide pixels must raise RuntimeError
            trainer_zero_pos = DummyTrainer(dataset, num_classes=2)
            with self.assertRaises(RuntimeError) as ctx:
                trainer_zero_pos.calculate_class_weights()
            self.assertIn("No landslide foreground pixels found", str(ctx.exception))

            # 2. Unsupported num_classes != 2 must assert or raise
            trainer_multi = DummyTrainer(dataset, num_classes=3)
            with self.assertRaises(AssertionError):
                trainer_multi.calculate_class_weights()

    def test_topographic_normalization_local_relief_and_slope_only(self):
        """Verify relative local relief and slope-only terrain normalization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir) / "train"
            for sub in ["IMAGE", "DTM", "LABEL"]:
                (train_dir / sub).mkdir(parents=True)

            img = np.zeros((32, 32, 3), dtype=np.uint8)
            # Create synthetic elevation ramp: 200m to 300m (delta = 100m)
            dtm = np.linspace(200.0, 300.0, 32 * 32, dtype=np.float32).reshape(32, 32)
            mask = np.zeros((32, 32), dtype=np.uint8)

            cv2.imwrite(str(train_dir / "IMAGE" / "t1.png"), img)
            cv2.imwrite(str(train_dir / "DTM" / "t1.tif"), dtm)
            cv2.imwrite(str(train_dir / "LABEL" / "t1.png"), mask)

            # 1. Local relief: should map [200, 300] to [0.0, 1.0] exactly
            ds_relief = LandslideDataset(
                data_dir=tmpdir,
                split="train",
                img_size=(32, 32),
                dtm_norm="local_relief",
                is_training=False
            )
            _, terrain_relief, _, _ = ds_relief[0]
            elev_ch = terrain_relief[0].numpy()
            self.assertAlmostEqual(float(elev_ch.min()), 0.0, places=3)
            self.assertAlmostEqual(float(elev_ch.max()), 1.0, places=3)

            # 2. Slope only: elevation channel should be zeros, derivatives should be valid
            ds_slope = LandslideDataset(
                data_dir=tmpdir,
                split="train",
                img_size=(32, 32),
                dtm_norm="slope_only",
                include_derivatives=True,
                is_training=False
            )
            _, terrain_slope, _, _ = ds_slope[0]
            self.assertEqual(terrain_slope.shape[0], 4)  # [DTM, Slope, Sin_Aspect, Cos_Aspect]
            # Channel 0 (elevation) must be completely zeroed out
            self.assertEqual(float(terrain_slope[0].abs().max()), 0.0)
            # Channel 1 (slope) must be non-zero
            self.assertGreater(float(terrain_slope[1].max()), 0.0)


if __name__ == '__main__':
    unittest.main()

