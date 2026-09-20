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
from eval import MODELS


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

    def test_colorize_mask_nodata_rendering(self):
        """Verify that NoData pixels (-100) are rendered in neutral gray ([128, 128, 128])."""
        palette = get_palette("landslide", num_classes=2)
        # Create mask with background (0), landslide (1), and nodata (-100)
        mask = np.array([
            [0, 1],
            [-100, -1]
        ], dtype=np.int64)
        colored = colorize_mask(mask, palette, ignore_index=-100)
        # Background -> [0, 0, 0]
        np.testing.assert_array_equal(colored[0, 0], [0, 0, 0])
        # Landslide -> [255, 50, 50]
        np.testing.assert_array_equal(colored[0, 1], [255, 50, 50])
        # NoData (-100) -> [128, 128, 128]
        np.testing.assert_array_equal(colored[1, 0], [128, 128, 128])
        # Negative -> [128, 128, 128]
        np.testing.assert_array_equal(colored[1, 1], [128, 128, 128])

    def test_visualize_rgb_denormalization(self):
        """Verify that normalized RGB input is properly denormalized and saved in multi-panel diagnostic."""
        h, w = 64, 64
        # Create synthetic normalized image (ImageNet normalized)
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

    def test_visualize_prediction_only_suppresses_gt_panel(self):
        """Verify that prediction-only mode (labels=None) suppresses ground truth panel (3 panels instead of 4)."""
        h, w = 64, 64
        rgb_tensor = torch.zeros((1, 3, h, w), dtype=torch.float32)
        ir_tensor = torch.zeros((1, 1, h, w), dtype=torch.float32)
        pred_tensor = torch.zeros((1, h, w), dtype=torch.long)
        pred_tensor[0, 5:15, 5:15] = 1

        visualize(
            image_name=["pred_only_sample"],
            predictions=pred_tensor,
            weight_name="pred_mode",
            rgb=rgb_tensor,
            ir=ir_tensor,
            labels=None,  # Prediction-only mode
            dataset="landslide",
            save_dir=self.temp_dir,
            save_side_by_side=True
        )

        diag_path = os.path.join(self.temp_dir, "pred_only_sample_pred_mode_diagnostic.png")
        self.assertTrue(os.path.exists(diag_path))
        diag_img = cv2.imread(diag_path)
        self.assertIsNotNone(diag_img)
        # Prediction-only: exactly 3 panels [RGB | DTM | Overlay], width = 3 * w
        self.assertEqual(diag_img.shape[0], h)
        self.assertEqual(diag_img.shape[1], 3 * w)

        mask_path = os.path.join(self.temp_dir, "pred_only_sample_pred_mode_mask.png")
        self.assertTrue(os.path.exists(mask_path))

    def test_visualize_nodata_diagnostic_rendering(self):
        """Verify that NoData pixels (-100) are cleanly rendered as neutral gray ([128, 128, 128]) in all panels."""
        h, w = 64, 64
        rgb_tensor = torch.zeros((1, 3, h, w), dtype=torch.float32)
        ir_tensor = torch.ones((1, 1, h, w), dtype=torch.float32) * 50.0
        pred_tensor = torch.ones((1, h, w), dtype=torch.long)  # Model predicted landslide everywhere
        lbl_tensor = torch.zeros((1, h, w), dtype=torch.long)
        lbl_tensor[0, :32, :] = -100  # Top half is NoData survey boundary

        visualize(
            image_name=["nodata_sample"],
            predictions=pred_tensor,
            weight_name="nodata_run",
            rgb=rgb_tensor,
            ir=ir_tensor,
            labels=lbl_tensor,
            dataset="landslide",
            save_dir=self.temp_dir,
            save_side_by_side=True,
            ignore_index=-100
        )

        diag_path = os.path.join(self.temp_dir, "nodata_sample_nodata_run_diagnostic.png")
        diag_img = cv2.imread(diag_path)
        self.assertIsNotNone(diag_img)
        # 4 panels: [RGB | DTM | GT | Overlay]
        self.assertEqual(diag_img.shape[1], 4 * w)

        # Panel 2 (DTM): columns [w : 2*w] -> top half should be neutral gray [128, 128, 128]
        dtm_panel = diag_img[:, w:2*w, :]
        np.testing.assert_array_equal(dtm_panel[10, 10], [128, 128, 128])

        # Panel 3 (GT): columns [2*w : 3*w] -> top half should be neutral gray [128, 128, 128]
        gt_panel = diag_img[:, 2*w:3*w, :]
        np.testing.assert_array_equal(gt_panel[10, 10], [128, 128, 128])

        # Panel 4 (Overlay): columns [3*w : 4*w] -> top half should be neutral gray [128, 128, 128]
        overlay_panel = diag_img[:, 3*w:4*w, :]
        np.testing.assert_array_equal(overlay_panel[10, 10], [128, 128, 128])

        # Check mask image
        mask_path = os.path.join(self.temp_dir, "nodata_sample_nodata_run_mask.png")
        mask_img = cv2.imread(mask_path)
        self.assertIsNotNone(mask_img)
        np.testing.assert_array_equal(mask_img[10, 10], [128, 128, 128])

    def test_demo_models_registry_dispatch(self):
        """Verify that MODELS registry contains FusionModel and rejects unknown architecture strings."""
        self.assertIn("FusionModel", MODELS)
        self.assertEqual(MODELS["FusionModel"], FusionModel)
        unknown = "UnknownArbitraryCodeModel"
        self.assertNotIn(unknown, MODELS)

    def test_demo_cli_boolean_optional_actions(self):
        """Verify that CLI flags correctly toggle boolean flags via BooleanOptionalAction."""
        from eval import build_parser
        parser = build_parser()
        args_eval = parser.parse_args(["--have-test-labels", "--visualize"])
        self.assertTrue(args_eval.have_test_labels)
        self.assertTrue(args_eval.visualize)

        args_pred = parser.parse_args(["--no-have-test-labels", "--no-visualize"])
        self.assertFalse(args_pred.have_test_labels)
        self.assertFalse(args_pred.visualize)

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
            pretrained=False,
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
            pretrained=False,
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
            pretrained=False,
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
