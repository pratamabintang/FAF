"""
Unit tests for Model Components, Backbones, Decoders, and Novel Fusion Modules.
"""

import os
import sys
import gc
import unittest
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FusionModel import (
    FusionModel,
    get_backbone_context_dim,
    BACKBONE_CONTEXT_DIMS,
    SceneAdaptiveSigmaPredictor,
    AdaptiveGaussianLowPass,
    ComplementarityAwareFusionGate,
    ThermalPriorModule,
    GaussianLowPass,
    DualPoolSpatialAttention,
    ScalarConfidenceGate,
    SafeResidualFusion,
    EnhancedSemanticFusion,
)


class TestModelArchitecture(unittest.TestCase):
    """Test suite for FusionModel architecture, backbones, and modules."""

    def setUp(self):
        torch.manual_seed(42)

    def tearDown(self):
        gc.collect()

    def test_backbone_context_dim_mapping(self):
        """Verify context_dim mappings for Nano, Tiny, and Base backbones."""
        self.assertEqual(get_backbone_context_dim('convnextv2_nano.fcmae_ft_in22k_in1k_384'), [80, 160, 320, 640])
        self.assertEqual(get_backbone_context_dim('convnextv2_tiny.fcmae_ft_in22k_in1k_384'), [96, 192, 384, 768])
        self.assertEqual(get_backbone_context_dim('convnextv2_base.fcmae_ft_in22k_in1k_384'), [128, 256, 512, 1024])
        # Default fallback
        self.assertEqual(get_backbone_context_dim('unknown_arch'), [96, 192, 384, 768])

    def test_safd_isolated_modules(self):
        """Verify SceneAdaptiveSigmaPredictor and AdaptiveGaussianLowPass."""
        B, C, H, W = 2, 64, 32, 32
        num_bands = 4
        rgb_feat = torch.randn(B, C, H, W)
        ir_feat = torch.randn(B, C, H, W)

        # Test Sigma Predictor
        predictor = SceneAdaptiveSigmaPredictor(channels=C, num_bands=num_bands)
        weights = predictor(rgb_feat, ir_feat)
        self.assertEqual(weights.shape, torch.Size([B, num_bands, 1, 1]))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(B, 1, 1), atol=1e-5))
        self.assertFalse(torch.isnan(weights).any())

        # Test Adaptive LowPass
        lowpass = AdaptiveGaussianLowPass(channels=C, k=7, num_bands=num_bands)
        smoothed = lowpass(ir_feat, weights)
        self.assertEqual(smoothed.shape, torch.Size([B, C, H, W]))
        self.assertFalse(torch.isnan(smoothed).any())
        self.assertFalse(torch.isinf(smoothed).any())

    def test_safd_batch_size_one(self):
        """Ensure SAFD works when batch_size=1 during training mode."""
        B, C, H, W = 1, 64, 32, 32
        predictor = SceneAdaptiveSigmaPredictor(channels=C, num_bands=4)
        predictor.train()
        rgb_feat = torch.randn(B, C, H, W)
        ir_feat = torch.randn(B, C, H, W)
        weights = predictor(rgb_feat, ir_feat)
        self.assertEqual(weights.shape, torch.Size([1, 4, 1, 1]))

    def test_cafg_module_isolated(self):
        """Verify ComplementarityAwareFusionGate."""
        B, C, H, W = 2, 96, 32, 32
        rgb_feat = torch.randn(B, C, H, W)
        ir_feat = torch.randn(B, C, H, W)

        cafg = ComplementarityAwareFusionGate(channels=C)
        out = cafg(rgb_feat, ir_feat)
        self.assertEqual(out.shape, torch.Size([B, C, H, W]))
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_tpsw_module_isolated(self):
        """Verify ThermalPriorModule."""
        B, H, W = 2, 128, 128
        raw_ir = torch.randn(B, 1, H, W)

        tpsw = ThermalPriorModule()
        priors = tpsw(raw_ir)
        self.assertEqual(len(priors), 4)
        for p in priors:
            self.assertEqual(p.shape[0], B)
            self.assertEqual(p.shape[1], 1)
            self.assertFalse(torch.isnan(p).any())

    def test_forward_fpn_decoder_landslide(self):
        """Verify forward pass with FPN decoder for 2-class landslide segmentation."""
        B, H, W = 1, 128, 128
        num_classes = 2
        model = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=num_classes,
            input_resolution=(H, W),
            output_resolution=(H, W),
            decoder_type='fpn',
            pretrained=False
        )
        model.eval()

        with torch.no_grad():
            rgb = torch.randn(B, 3, H, W)
            ir = torch.randn(B, 1, H, W)
            logits, aux = model(rgb, ir)

        self.assertEqual(logits.shape, torch.Size([B, num_classes, H, W]))
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isinf(logits).any())

    def test_forward_panet_deep_supervision(self):
        """Verify forward pass with PANet decoder and deep supervision."""
        B, H, W = 1, 128, 128
        num_classes = 2
        model = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=num_classes,
            input_resolution=(H, W),
            output_resolution=(H, W),
            decoder_type='panet',
            deep_supervision=True,
            use_safd=True,
            use_cafg=True,
            pretrained=False
        )
        model.train()

        rgb = torch.randn(B, 3, H, W)
        ir = torch.randn(B, 1, H, W)
        logits, aux_dict = model(rgb, ir)

        self.assertEqual(logits.shape, torch.Size([B, num_classes, H, W]))
        self.assertIsInstance(aux_dict, dict)
        self.assertIn('aux', aux_dict)
        self.assertIn('deep', aux_dict)
        self.assertFalse(torch.isnan(logits).any())
        self.assertFalse(torch.isnan(aux_dict['aux']).any())
        for d in aux_dict['deep']:
            self.assertFalse(torch.isnan(d).any())

    def test_backward_and_optimizer_step(self):
        """Verify backward pass and parameter update."""
        B, H, W = 1, 64, 64
        num_classes = 2
        model = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=num_classes,
            input_resolution=(H, W),
            output_resolution=(H, W),
            decoder_type='fpn',
            use_safd=True,
            pretrained=False
        )
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        rgb = torch.randn(B, 3, H, W, requires_grad=True)
        ir = torch.randn(B, 1, H, W, requires_grad=True)
        targets = torch.randint(0, num_classes, (B, H, W), dtype=torch.long)

        optimizer.zero_grad()
        logits, _ = model(rgb, ir)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_grad)

        # Snapshot parameter before step
        param = next(p for p in model.parameters() if p.requires_grad)
        before = param.data.clone()
        optimizer.step()
        after = param.data

        # Verify update happened
        self.assertGreater((after - before).abs().sum().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
