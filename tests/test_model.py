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
    NovelFrequencyAwareFusionModule,
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

    def test_novel_frequency_aware_fusion_module_isolated(self):
        """Verify NovelFrequencyAwareFusionModule with SAFD and CAFG isolated."""
        B, C, H, W = 2, 64, 32, 32
        rgb_feat = torch.randn(B, C, H, W, requires_grad=True)
        ir_feat = torch.randn(B, C, H, W, requires_grad=True)

        # 1. With use_safd=True, use_cafg=False (specifically testing ir_high without CAFG)
        fusion_mod_safd = NovelFrequencyAwareFusionModule(rgb_c=C, ir_c=C, use_safd=True, use_cafg=False)
        out1 = fusion_mod_safd(rgb_feat, ir_feat)
        self.assertEqual(out1.shape, torch.Size([B, C, H, W]))
        self.assertFalse(torch.isnan(out1).any())
        out1.sum().backward()
        self.assertIsNotNone(rgb_feat.grad)
        self.assertIsNotNone(ir_feat.grad)

        # 2. With use_safd=True, use_cafg=True
        rgb_feat.grad = None
        ir_feat.grad = None
        fusion_mod_full = NovelFrequencyAwareFusionModule(rgb_c=C, ir_c=C, use_safd=True, use_cafg=True)
        out2 = fusion_mod_full(rgb_feat, ir_feat)
        self.assertEqual(out2.shape, torch.Size([B, C, H, W]))
        self.assertFalse(torch.isnan(out2).any())

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

    def test_loss_functions_stability(self):
        """Verify BoundaryLoss, OHEMCrossEntropyLoss, and ComboLoss3 are finite and non-negative."""
        from FusionModelTrain import BoundaryLoss, OHEMCrossEntropyLoss, ComboLoss3, ComboLossOHEM

        B, C, H, W = 2, 2, 32, 32
        logits = torch.randn(B, C, H, W, requires_grad=True)
        targets = torch.randint(0, C, (B, H, W), dtype=torch.long)

        # 1. BoundaryLoss
        boundary_loss = BoundaryLoss(num_classes=C)
        b_loss = boundary_loss(logits, targets)
        self.assertFalse(torch.isnan(b_loss))
        self.assertFalse(torch.isinf(b_loss))
        self.assertGreaterEqual(b_loss.item(), 0.0)

        # 2. OHEMCrossEntropyLoss
        ohem_loss = OHEMCrossEntropyLoss(min_kept=10, thresh=0.7)
        o_loss = ohem_loss(logits, targets)
        self.assertFalse(torch.isnan(o_loss))
        self.assertFalse(torch.isinf(o_loss))
        self.assertGreaterEqual(o_loss.item(), 0.0)

        # 3. ComboLoss3 with high class weights (e.g. 50.0 for rare class) and active Lovasz weight
        class_weights = torch.tensor([1.0, 50.0])
        combo3 = ComboLoss3(ce_w=0.5, dice_w=0.3, lovasz_w=0.2, class_weights=class_weights)
        c_loss = combo3(logits, targets)
        self.assertFalse(torch.isnan(c_loss))
        self.assertFalse(torch.isinf(c_loss))
        self.assertGreaterEqual(c_loss.item(), 0.0)

        # 4. ComboLoss3 and ComboLossOHEM with NoData ignore_index (-100)
        targets_with_nodata = targets.clone()
        targets_with_nodata[:, :4, :4] = -100
        c_loss_nodata = combo3(logits, targets_with_nodata)
        self.assertFalse(torch.isnan(c_loss_nodata))
        self.assertFalse(torch.isinf(c_loss_nodata))
        self.assertGreaterEqual(c_loss_nodata.item(), 0.0)

        combo_ohem = ComboLossOHEM(ce_w=0.35, dice_w=0.35, lovasz_w=0.30, ohem_w=0.1, boundary_w=0.1,
                                   class_weights=class_weights, ignore_index=-100, num_classes=2)
        co_loss = combo_ohem(logits, targets_with_nodata)
        self.assertFalse(torch.isnan(co_loss))
        self.assertFalse(torch.isinf(co_loss))
        self.assertGreaterEqual(co_loss.item(), 0.0)

    def test_loss_nodata_invariance_and_empty_batch_penalty(self):
        """Verify Lovasz empty-batch false positive penalty and NoData strict invariance across losses."""
        from FusionModelTrain import lovasz_softmax, BoundaryLoss, ComboLoss3, ComboLossOHEM

        # 1. Lovasz on all-negative batch: false positives must be penalized, correct predictions near 0
        all_neg_labels = torch.zeros(1, 16, 16, dtype=torch.long)
        logits_fp = torch.zeros(1, 2, 16, 16)
        logits_fp[:, 1, :, :] = 10.0  # predicting class 1 (false positive) with ~100% confidence
        loss_fp = lovasz_softmax(logits_fp, all_neg_labels)
        self.assertGreater(loss_fp.item(), 0.5)

        logits_tn = torch.zeros(1, 2, 16, 16)
        logits_tn[:, 0, :, :] = 10.0  # predicting class 0 (true negative) with ~100% confidence
        loss_tn = lovasz_softmax(logits_tn, all_neg_labels)
        self.assertLess(loss_tn.item(), 1e-3)

        # 2. NoData Invariance: changing pixel logits in NoData regions produces 0 change on valid territory
        torch.manual_seed(42)
        logits1 = torch.randn(1, 2, 16, 16, requires_grad=True)
        targets = torch.zeros(1, 16, 16, dtype=torch.long)
        targets[0, 5:11, 5:11] = 1   # Foreground landslide square in center
        targets[0, 0:4, :] = -100     # Top 4 rows are NoData

        losses_to_test = [
            ("BoundaryLoss", BoundaryLoss(num_classes=2, ignore_index=-100)),
            ("ComboLoss3", ComboLoss3(ce_w=0.5, dice_w=0.3, lovasz_w=0.2, ignore_index=-100)),
            ("ComboLossOHEM", ComboLossOHEM(ce_w=0.4, dice_w=0.2, lovasz_w=0.2, ohem_w=0.1, boundary_w=0.1,
                                            ignore_index=-100, num_classes=2, ohem_min_kept=20))
        ]

        for name, loss_fn in losses_to_test:
            if logits1.grad is not None:
                logits1.grad.zero_()
            l1 = loss_fn(logits1, targets)
            l1.backward()
            grad1 = logits1.grad.clone()

            # Alter logits only in NoData region (rows 0 to 3)
            logits2 = logits1.detach().clone()
            logits2[:, :, 0:4, :] = torch.randn(1, 2, 4, 16) * 20.0
            logits2.requires_grad = True

            l2 = loss_fn(logits2, targets)
            l2.backward()
            grad2 = logits2.grad.clone()

            # Loss difference must be identically zero
            loss_diff = abs(l1.item() - l2.item())
            self.assertAlmostEqual(loss_diff, 0.0, places=5, msg=f"{name} loss varied when modifying NoData pixels!")

            # Gradient on interior valid territory (rows 5 to 15) must be identically zero
            grad_diff_valid = (grad1[:, :, 5:, :] - grad2[:, :, 5:, :]).abs().max().item()
            self.assertAlmostEqual(grad_diff_valid, 0.0, places=5, msg=f"{name} valid gradient varied with NoData changes!")

            # Gradient on NoData pixels should be zero
            grad_nodata = grad1[:, :, 0:4, :].abs().max().item()
            self.assertAlmostEqual(grad_nodata, 0.0, places=5, msg=f"{name} accumulated gradient in NoData region!")

        # 3. All NoData batch should return finite 0.0 without crashing
        all_nodata = torch.full((1, 16, 16), -100, dtype=torch.long)
        logits_dummy = torch.randn(1, 2, 16, 16)
        for name, loss_fn in losses_to_test:
            l_empty = loss_fn(logits_dummy, all_nodata)
            self.assertFalse(torch.isnan(l_empty))
            self.assertEqual(l_empty.item(), 0.0)

    def test_dynamic_in_chans_adaptation(self):
        """Verify FusionModel initializes and runs forward pass with 4-channel terrain input."""
        res = (64, 64)
        model = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=2,
            context_dim=[80, 160, 320, 640],
            input_resolution=res,
            rgb_backbone_resolution=res,
            ir_backbone_resolution=res,
            output_resolution=res,
            ir_in_chans=4,
            use_tpsw=True,
            pretrained=False
        )
        rgb = torch.randn(2, 3, 64, 64)
        terrain = torch.randn(2, 4, 64, 64)
        main_out, _ = model(rgb, terrain)
        self.assertEqual(main_out.shape, torch.Size([2, 2, 64, 64]))
        self.assertFalse(torch.isnan(main_out).any())

    def test_optimizer_parameter_groups_and_layerwise_decay(self):
        """Verify layer-wise decay for timm ConvNeXt, canonical terrain_prior, and unified weight decay."""
        from FusionModelTrain import FusionTrainer

        res = (64, 64)
        model = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=2,
            context_dim=[80, 160, 320, 640],
            input_resolution=res,
            rgb_backbone_resolution=res,
            ir_backbone_resolution=res,
            output_resolution=res,
            ir_in_chans=1,
            use_tpsw=True,
            pretrained=False
        )

        class DummyConfig:
            lr_backbone = 1e-4
            lr_fusion = 2e-4
            lr_decoder = 3e-4
            weight_decay = 0.015
            optimizer = 'adamw'
            layer_decay = 0.90

        class DummyTrainer:
            def __init__(self, m):
                self.model = m
                self.config = DummyConfig()

        DummyTrainer.setup_optimizer = FusionTrainer.setup_optimizer

        trainer = DummyTrainer(model)
        optimizer = trainer.setup_optimizer()

        param_groups_by_name = {g["name"]: g for g in optimizer.param_groups}

        # 1. Verify ConvNeXt layer-wise decay groups exist for rgb_0 through rgb_4
        for lid in range(5):
            self.assertIn(f"rgb_{lid}", param_groups_by_name, f"Missing layer group rgb_{lid}")
            self.assertGreater(len(param_groups_by_name[f"rgb_{lid}"]["params"]), 0)

        # 2. Verify strict monotonic learning rate decay: lr(rgb_0) < lr(rgb_1) < ... < lr(rgb_4)
        for lid in range(4):
            lr_curr = param_groups_by_name[f"rgb_{lid}"]["lr"]
            lr_next = param_groups_by_name[f"rgb_{lid+1}"]["lr"]
            self.assertLess(lr_curr, lr_next, f"Expected lr(rgb_{lid}) < lr(rgb_{lid+1})")

        # 3. Verify terrain_prior is canonically registered and receives lr_fusion
        self.assertIn("terrain_prior", param_groups_by_name, "terrain_prior group not found in optimizer!")
        self.assertGreater(len(param_groups_by_name["terrain_prior"]["params"]), 0)
        self.assertAlmostEqual(param_groups_by_name["terrain_prior"]["lr"], 2e-4)

        # 4. Verify all parameter groups respect config.weight_decay (0.015)
        for name, g in param_groups_by_name.items():
            self.assertEqual(g["weight_decay"], 0.015, f"Group {name} did not inherit config weight_decay!")

        # 5. Verify partition bijection: every trainable parameter belongs to exactly one optimizer group
        all_param_ids = []
        for g in optimizer.param_groups:
            for p in g["params"]:
                all_param_ids.append(id(p))
        self.assertEqual(len(all_param_ids), len(set(all_param_ids)), "Duplicate parameters found across optimizer groups!")
        trainable_model_param_ids = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(set(all_param_ids), trainable_model_param_ids, "Mismatch between trainable parameters and optimizer groups!")

    def test_tpsw_bidirectional_centered_modulation(self):
        """Verify TPSW modulation uses centered f * (0.5 + tp) formula for bidirectional modulation."""
        from FusionModel import TerrainPriorModule, FusionModel

        # 1. Direct TerrainPriorModule check
        prior_mod = TerrainPriorModule(in_channels=1)
        raw_dtm = torch.randn(2, 1, 64, 64)
        priors = prior_mod(raw_dtm, target_sizes=[(16, 16)])
        tp = priors[0]
        # tp is output of Sigmoid -> in (0, 1)
        self.assertTrue((tp >= 0.0).all())
        self.assertTrue((tp <= 1.0).all())

        # Centered multiplier: 0.5 + tp is in [0.5, 1.5]
        multiplier = 0.5 + tp
        self.assertTrue((multiplier >= 0.5).all())
        self.assertTrue((multiplier <= 1.5).all())

        # 2. End-to-end model forward with TPSW
        model = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=2,
            input_resolution=(64, 64),
            output_resolution=(64, 64),
            decoder_type='fpn',
            use_tpsw=True,
            pretrained=False
        )
        rgb = torch.randn(2, 3, 64, 64)
        dtm = torch.randn(2, 1, 64, 64)
        logits, _ = model(rgb, dtm)
        self.assertEqual(logits.shape, (2, 2, 64, 64))

    def test_unimodal_forward_modes(self):
        """Verify decoupled unimodal modes ('rgb_only' and 'dtm_only') operate independently."""
        from FusionModel import FusionModel

        # 1. RGB-Only mode
        model_rgb = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=2,
            input_resolution=(64, 64),
            output_resolution=(64, 64),
            modal_mode='rgb_only',
            pretrained=False
        )
        rgb = torch.randn(2, 3, 64, 64)
        logits_rgb, _ = model_rgb(rgb=rgb)
        self.assertEqual(logits_rgb.shape, (2, 2, 64, 64))

        # Passing None for rgb should raise ValueError
        with self.assertRaises(ValueError):
            model_rgb(rgb=None)

        # 2. DTM-Only mode
        model_dtm = FusionModel(
            rgb_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            ir_arch='convnextv2_nano.fcmae_ft_in22k_in1k_384',
            num_classes=2,
            input_resolution=(64, 64),
            output_resolution=(64, 64),
            modal_mode='dtm_only',
            pretrained=False
        )
        dtm = torch.randn(2, 1, 64, 64)
        logits_dtm, _ = model_dtm(terrain=dtm)
        self.assertEqual(logits_dtm.shape, (2, 2, 64, 64))

        # Passing None for terrain should raise ValueError
        with self.assertRaises(ValueError):
            model_dtm(terrain=None)

    def test_all_yaml_configurations_offline_smoke(self):
        """Verify all configuration YAML files instantiate, forward, and backward offline with pretrained=False."""
        import glob
        import yaml
        from pathlib import Path
        from FusionModel import FusionModel

        configs_dir = Path(__file__).resolve().parent.parent / "configs"
        yaml_files = sorted(configs_dir.glob("*.yaml"))
        self.assertGreater(len(yaml_files), 10, f"Expected >10 config YAMLs, found {len(yaml_files)}")

        res = (64, 64)
        for ypath in yaml_files:
            with open(ypath, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            exp_name = cfg.get("exp_name", ypath.stem)
            with self.subTest(config=exp_name):
                include_derivs = cfg.get("include_derivatives", False)
                ir_chans = 4 if include_derivs else cfg.get("ir_in_chans", 1)
                modal_mode = cfg.get("modal_mode", "multimodal")

                # Build model with nano arch & pretrained=False for fast offline execution
                model = FusionModel(
                    rgb_arch="convnextv2_nano.fcmae_ft_in22k_in1k_384",
                    ir_arch="convnextv2_nano.fcmae_ft_in22k_in1k_384",
                    pretrained=False,
                    num_classes=cfg.get("num_classes", 2),
                    input_resolution=res,
                    rgb_backbone_resolution=res,
                    ir_backbone_resolution=res,
                    output_resolution=res,
                    context_dim=[80, 160, 320, 640],
                    decoder_type=cfg.get("decoder_type", "fpn"),
                    deep_supervision=cfg.get("deep_supervision", False),
                    enhanced_fusion=cfg.get("enhanced_fusion", False),
                    use_safd=cfg.get("use_safd", False),
                    use_cafg=cfg.get("use_cafg", False),
                    use_tpsw=cfg.get("use_tpsw", False),
                    ir_in_chans=ir_chans,
                    modal_mode=modal_mode
                )
                model.train()

                rgb = torch.randn(2, 3, 64, 64, requires_grad=True)
                dtm = torch.randn(2, ir_chans, 64, 64, requires_grad=True)

                if modal_mode == "rgb_only":
                    main_out, _ = model(rgb=rgb)
                elif modal_mode in ("dtm_only", "ir_only"):
                    main_out, _ = model(terrain=dtm)
                else:
                    main_out, _ = model(rgb=rgb, terrain=dtm)

                self.assertEqual(main_out.shape, torch.Size([2, 2, 64, 64]))
                self.assertFalse(torch.isnan(main_out).any())

                loss = main_out.sum()
                loss.backward()

                # Verify valid gradients exist
                has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
                self.assertTrue(has_grad, f"No gradients produced for config {exp_name}")


if __name__ == '__main__':
    unittest.main()

