"""
================================================================================
Tests for Inference, Utilities, Visualizer, and Config-Driven Model Loading
================================================================================
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2
import torch

from FusionModelUtils import compute_results, get_palette, colorize_mask, visualize
from FusionModel import FusionModel


class TestInferenceAndUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compute_results_metrics(self):
        """Verify precision, recall, IoU, and F1 calculations from confusion matrix."""
        # 2x2 confusion matrix:
        # Rows = Ground Truth, Columns = Predictions
        # [[TP_0, FN_0],
        #  [FP_0, TP_1]] -> GT 0: 80 true, 20 predicted as 1; GT 1: 10 predicted as 0, 90 true.
        conf = np.array([
            [80, 20],  # GT 0
            [10, 90]   # GT 1
        ], dtype=np.int64)

        precision, recall, iou, f1 = compute_results(conf)

        # For Class 0:
        # TP = 80, FP = 10, FN = 20
        # Precision = 80 / 90 = 0.888889
        # Recall = 80 / 100 = 0.800000
        # IoU = 80 / (80 + 10 + 20) = 80 / 110 = 0.727273
        self.assertAlmostEqual(precision[0], 80.0 / 90.0, places=5)
        self.assertAlmostEqual(recall[0], 80.0 / 100.0, places=5)
        self.assertAlmostEqual(iou[0], 80.0 / 110.0, places=5)

        # For Class 1:
        # TP = 90, FP = 20, FN = 10
        # Precision = 90 / 110 = 0.818182
        # Recall = 90 / 100 = 0.900000
        # IoU = 90 / (90 + 20 + 10) = 90 / 120 = 0.750000
        self.assertAlmostEqual(precision[1], 90.0 / 110.0, places=5)
        self.assertAlmostEqual(recall[1], 90.0 / 100.0, places=5)
        self.assertAlmostEqual(iou[1], 90.0 / 120.0, places=5)

    def test_compute_results_zero_division(self):
        """Verify compute_results handles empty classes gracefully without throwing."""
        conf_empty = np.zeros((2, 2), dtype=np.int64)
        precision, recall, iou, f1 = compute_results(conf_empty)
        self.assertEqual(precision[0], 0.0)
        self.assertEqual(recall[0], 0.0)
        self.assertEqual(iou[0], 0.0)
        self.assertEqual(f1[0], 0.0)

    def test_get_palette(self):
        """Verify distinct color palettes for datasets."""
        palette_ls = get_palette("landslide", num_classes=2)
        self.assertEqual(palette_ls.shape, (2, 3))
        # Landslide background is black, landslide is reddish
        np.testing.assert_array_equal(palette_ls[0], [0, 0, 0])
        self.assertEqual(palette_ls[1, 0], 255)

    def test_visualize_rgb_denormalization(self):
        """Verify that normalized RGB input is properly denormalized and saved in multi-panel diagnostic."""
        h, w = 64, 64
        # Create synthetic normalized image (ImageNet normalized)
        # Simulated raw RGB around 128 (gray) -> normalized: (0.5 - mean) / std ~ 0.2
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        raw_rgb = np.full((3, h, w), 0.5, dtype=np.float32)
        norm_rgb = (raw_rgb - mean) / std

        rgb_tensor = torch.from_numpy(norm_rgb).unsqueeze(0)  # [1, 3, H, W]
        ir_tensor = torch.zeros((1, 1, h, w), dtype=torch.float32)
        pred_tensor = torch.zeros((1, h, w), dtype=torch.long)
        pred_tensor[0, 10:20, 10:20] = 1  # simulated landslide region
        lbl_tensor = pred_tensor.clone()

        visualize(
            image_name=["sample_01"],
            predictions=pred_tensor,
            weight_name="test_run",
            rgb=rgb_tensor,
            ir=ir_tensor,
            labels=lbl_tensor,
            dataset="landslide",
            save_dir=self.temp_dir,
            save_side_by_side=True
        )

        diag_path = os.path.join(self.temp_dir, "sample_01_test_run_diagnostic.png")
        self.assertTrue(os.path.exists(diag_path), f"File {diag_path} does not exist.")

        # Read saved image
        diag_img = cv2.imread(diag_path)
        self.assertIsNotNone(diag_img)
        # 4 panels: [RGB | DTM | GT | Overlay] -> width should be 4 * 64 = 256
        self.assertEqual(diag_img.shape[0], h)
        self.assertEqual(diag_img.shape[1], 4 * w)

        # Check that RGB panel (first 64 columns) is NOT pure black (< 5 mean)
        rgb_panel = diag_img[:, :w, :]
        mean_intensity = np.mean(rgb_panel)
        self.assertGreater(mean_intensity, 50.0, "RGB panel was improperly denormalized (near black)!")

    def test_config_driven_model_instantiation_and_strict_loading(self):
        """Verify that checkpoint configuration dynamically drives model construction and strict loading succeeds."""
        resolution = (64, 64)
        config_dict = {
            "rgb_arch": "convnextv2_nano.fcmae_ft_in22k_in1k_384",
            "ir_arch": "convnextv2_nano.fcmae_ft_in22k_in1k_384",
            "num_classes": 2,
            "context_dim": "[80,160,320,640]",
            "img_height": 64,
            "img_width": 64,
            "decoder_type": "panet",
            "deep_supervision": False,
            "enhanced_fusion": False,
            "use_safd": False,
            "use_cafg": False,
            "use_tpsw": False,
            "dataset": "landslide"
        }

        # Instantiate reference model and get weights
        model_orig = FusionModel(
            rgb_arch=config_dict["rgb_arch"],
            ir_arch=config_dict["ir_arch"],
            num_classes=config_dict["num_classes"],
            context_dim=[80, 160, 320, 640],
            input_resolution=resolution,
            rgb_backbone_resolution=resolution,
            ir_backbone_resolution=resolution,
            output_resolution=resolution,
            decoder_type=config_dict["decoder_type"],
            deep_supervision=config_dict["deep_supervision"]
        )

        ckpt_path = os.path.join(self.temp_dir, "test_checkpoint.pth")
        torch.save({
            "model_state_dict": model_orig.state_dict(),
            "config": config_dict,
            "best_miou": 0.85,
            "best_landslide_iou": 0.78
        }, ckpt_path)

        # Now simulate FusionModelRunDemo loading
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        recovered_cfg = ckpt["config"]

        # Build model using recovered configuration
        cd = [int(x.strip()) for x in recovered_cfg["context_dim"].strip("[]").split(",")]
        res = (recovered_cfg["img_height"], recovered_cfg["img_width"])
        model_loaded = FusionModel(
            rgb_arch=recovered_cfg["rgb_arch"],
            ir_arch=recovered_cfg["ir_arch"],
            num_classes=recovered_cfg["num_classes"],
            context_dim=cd,
            input_resolution=res,
            rgb_backbone_resolution=res,
            ir_backbone_resolution=res,
            output_resolution=res,
            decoder_type=recovered_cfg["decoder_type"],
            deep_supervision=recovered_cfg["deep_supervision"]
        )

        # strict=True MUST succeed without error
        model_loaded.load_state_dict(ckpt["model_state_dict"], strict=True)

        # Verify that mismatched architecture fails with strict=True
        model_mismatch = FusionModel(
            rgb_arch=recovered_cfg["rgb_arch"],
            ir_arch=recovered_cfg["ir_arch"],
            num_classes=recovered_cfg["num_classes"],
            context_dim=cd,
            input_resolution=res,
            rgb_backbone_resolution=res,
            ir_backbone_resolution=res,
            output_resolution=res,
            decoder_type="fpn"  # Mismatched decoder
        )
        with self.assertRaises(RuntimeError):
            model_mismatch.load_state_dict(ckpt["model_state_dict"], strict=True)


if __name__ == '__main__':
    unittest.main()
