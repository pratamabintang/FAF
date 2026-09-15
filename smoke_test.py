"""
================================================================================
Frequency-Aware Fusion (FAF) - Comprehensive Smoke Test Suite
================================================================================
Test Specification:
  - Input RGB  : [B, 3, H, W]  (default B=2, H=480, W=640)
  - Input DTM  : [B, 1, H, W]  (default B=2, H=480, W=640)
  - Output Logits : [B, 2, H, W]  (Binary Landslide Segmentation)

Verification Criteria (P0):
  [P0] 1. Output shape strictly matches [B, 2, H, W].
  [P0] 2. Primary output and auxiliary outputs are 100% free of NaN and Inf values.
  [P0] 3. Backward pass executes cleanly, gradients exist and contain 0 NaN / 0 Inf.
  [P0] 4. Optimizer step successfully updates trainable model parameters.

Usage:
  python smoke_test.py
  python smoke_test.py --device cuda --batch_size 2 --img_height 480 --img_width 640
  python smoke_test.py --arch convnextv2_tiny.fcmae_ft_in22k_in1k_384 --use_safd --use_cafg
================================================================================
"""

import os
import sys
import gc
import argparse
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def check_tensor_health(tensor, name: str = "tensor"):
    """Validates that a tensor contains no NaN or Inf values."""
    if tensor is None:
        return
    assert not torch.isnan(tensor).any(), f"[P0 FAIL] NaN detected in {name}!"
    assert not torch.isinf(tensor).any(), f"[P0 FAIL] Inf (Infinity) detected in {name}!"

def run_single_smoke_test(
    model_name: str,
    rgb_arch: str,
    ir_arch: str,
    decoder_type: str,
    use_safd: bool,
    use_cafg: bool,
    use_tpsw: bool,
    deep_supervision: bool,
    batch_size: int,
    height: int,
    width: int,
    device: str,
    num_classes: int = 2,
) -> bool:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from FusionModel import FusionModel, get_backbone_context_dim

    print("\n" + "-" * 80)
    print(f"RUNNING TEST CONFIGURATION: {model_name}")
    print(f"  Backbone: RGB={rgb_arch} | IR={ir_arch}")
    print(f"  Decoder: {decoder_type.upper()} (deep_supervision={deep_supervision})")
    print(f"  Novel Modules: SAFD={use_safd}, CAFG={use_cafg}, TPSW={use_tpsw}")
    print(f"  Target Device: {device.upper()}")
    print("-" * 80)

    try:
        # Determine context_dim based on backbone
        ctx_dim = get_backbone_context_dim(rgb_arch)

        # 1. Initialize Model
        model = FusionModel(
            rgb_arch=rgb_arch,
            ir_arch=ir_arch,
            pretrained=False,
            num_classes=num_classes,
            input_resolution=(height, width),
            rgb_backbone_resolution=(height, width),
            ir_backbone_resolution=(height, width),
            output_resolution=(height, width),
            context_dim=ctx_dim,
            decoder_type=decoder_type,
            deep_supervision=deep_supervision,
            use_safd=use_safd,
            use_cafg=use_cafg,
            use_tpsw=use_tpsw,
        )
        
        target_device = torch.device(device)
        model = model.to(target_device)
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        # 2. Prepare Synthetic Data (RGB, DTM, Labels)
        torch.manual_seed(42)
        rgb = torch.randn(batch_size, 3, height, width, device=target_device, requires_grad=True)
        dtm = torch.randn(batch_size, 1, height, width, device=target_device, requires_grad=True)
        targets = torch.randint(0, num_classes, (batch_size, height, width), device=target_device, dtype=torch.long)

        print(f"  [1/6] Inputs generated: RGB={list(rgb.shape)}, DTM={list(dtm.shape)}")

        # 3. Forward Pass
        optimizer.zero_grad()
        main_logits, aux_logits = model(rgb, dtm)

        # 4. [P0] Assert Output Shape [B, 2, H, W]
        expected_shape = torch.Size([batch_size, num_classes, height, width])
        assert main_logits.shape == expected_shape, (
            f"[FAIL] Output shape mismatch! Expected {expected_shape}, got {main_logits.shape}"
        )
        print(f"  [2/6] [PASS] Output Shape verified: {main_logits.shape}")

        # 5. [P0] Assert NaN and Inf Checks on Primary Output
        check_tensor_health(main_logits, "main_logits")
        print("  [3/6] [PASS] [P0] Primary output is 100% free of NaN and Inf.")

        # Compute Loss (including auxiliary heads if enabled)
        loss = criterion(main_logits, targets)

        if aux_logits is not None:
            if isinstance(aux_logits, dict):
                if 'aux' in aux_logits and aux_logits['aux'] is not None:
                    check_tensor_health(aux_logits['aux'], "aux_logits['aux']")
                    loss = loss + 0.2 * criterion(aux_logits['aux'], targets)
                if 'deep' in aux_logits and aux_logits['deep'] is not None:
                    for d_idx, d_out in enumerate(aux_logits['deep']):
                        check_tensor_health(d_out, f"deep_supervision_level_{d_idx}")
                        loss = loss + 0.1 * criterion(d_out, targets)
            elif isinstance(aux_logits, (list, tuple)):
                for idx, aux in enumerate(aux_logits):
                    check_tensor_health(aux, f"aux_logits[{idx}]")
                    loss = loss + 0.2 * criterion(aux, targets)
            elif isinstance(aux_logits, torch.Tensor):
                check_tensor_health(aux_logits, "aux_logits")
                loss = loss + 0.2 * criterion(aux_logits, targets)

        check_tensor_health(loss, "calculated loss")
        print(f"  [4/6] [PASS] Loss computation successful: {loss.item():.6f}")

        # 6. [P0] Backward Pass
        loss.backward()
        print("  [5/6] [PASS] [P0] Backward pass executed successfully.")

        # Verify parameter gradients
        has_grads = 0
        nan_grads = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads += 1
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    nan_grads += 1

        assert has_grads > 0, "[P0 FAIL] No gradients found on trainable parameters!"
        assert nan_grads == 0, f"[P0 FAIL] Found {nan_grads} parameter tensors with NaN/Inf gradients!"
        print(f"  [PASS] [P0] Gradients verified for {has_grads} parameter tensors (0 NaN/Inf).")

        # 7. [P0] Optimizer Step
        # Store a snapshot of a parameter to verify weights change
        sample_param = next(p for p in model.parameters() if p.requires_grad)
        param_before = sample_param.data.clone()

        optimizer.step()

        param_after = sample_param.data
        weight_diff = (param_after - param_before).abs().sum().item()
        assert weight_diff > 0.0, "[P0 FAIL] Optimizer step did not update parameter weights!"
        print(f"  [6/6] [PASS] [P0] Optimizer step completed (Weight delta = {weight_diff:.6e}).")

        print(f"\n[TEST RESULT] '{model_name}' PASSED ALL P0 CHECKS!")
        
        del model, optimizer, criterion, rgb, dtm, targets, main_logits, aux_logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return True

    except Exception as e:
        print(f"\n[TEST RESULT] '{model_name}' FAILED!")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="FAF Landslide Fusion Model Smoke Test Suite")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--img_height", type=int, default=480, help="Input height (default: 480)")
    parser.add_argument("--img_width", type=int, default=640, help="Input width (default: 640)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (default: 2 for binary segmentation)")
    parser.add_argument("--arch", type=str, default="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
                        help="Backbone architecture (nano, tiny, or base)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Execution device: auto (detect GPU, else CPU), cpu, or cuda")
    parser.add_argument("--all_configs", action="store_true", default=False,
                        help="Run full matrix of configurations (Baseline + Novel + Deep Supervision)")
    args = parser.parse_args()

    # Determine device
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 80)
    print("       FREQUENCY-AWARE FUSION (FAF) - SMOKE TEST SUITE (P0 CRITICAL)          ")
    print("=" * 80)
    print(f"  PyTorch Version : {torch.__version__}")
    print(f"  CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device     : {torch.cuda.get_device_name(0)}")
    print(f"  Selected Device : {device.upper()}")
    print(f"  Input Dims      : RGB=[{args.batch_size}, 3, {args.img_height}, {args.img_width}], DTM=[{args.batch_size}, 1, {args.img_height}, {args.img_width}]")
    print(f"  Output Dims     : Logits=[{args.batch_size}, {args.num_classes}, {args.img_height}, {args.img_width}]")
    print("=" * 80)

    if args.all_configs:
        test_matrix = [
            {
                "model_name": "Config 1: Standard FPN Baseline",
                "rgb_arch": args.arch,
                "ir_arch": args.arch,
                "decoder_type": "fpn",
                "use_safd": False,
                "use_cafg": False,
                "use_tpsw": False,
                "deep_supervision": False,
            },
            {
                "model_name": "Config 2: Novel FAF (PANet + SAFD + CAFG + TPSW + Deep Supervision)",
                "rgb_arch": args.arch,
                "ir_arch": args.arch,
                "decoder_type": "panet",
                "use_safd": True,
                "use_cafg": True,
                "use_tpsw": True,
                "deep_supervision": True,
            },
            {
                "model_name": "Config 3: Novel FAF (PANet + SAFD Only)",
                "rgb_arch": args.arch,
                "ir_arch": args.arch,
                "decoder_type": "panet",
                "use_safd": True,
                "use_cafg": False,
                "use_tpsw": False,
                "deep_supervision": False,
            },
        ]
    else:
        test_matrix = [
            {
                "model_name": "Primary Smoke Test: Novel FAF (PANet + SAFD + CAFG + TPSW)",
                "rgb_arch": args.arch,
                "ir_arch": args.arch,
                "decoder_type": "panet",
                "use_safd": True,
                "use_cafg": True,
                "use_tpsw": True,
                "deep_supervision": True,
            }
        ]

    results = []
    for test_cfg in test_matrix:
        passed = run_single_smoke_test(
            model_name=test_cfg["model_name"],
            rgb_arch=test_cfg["rgb_arch"],
            ir_arch=test_cfg["ir_arch"],
            decoder_type=test_cfg["decoder_type"],
            use_safd=test_cfg["use_safd"],
            use_cafg=test_cfg["use_cafg"],
            use_tpsw=test_cfg["use_tpsw"],
            deep_supervision=test_cfg["deep_supervision"],
            batch_size=args.batch_size,
            height=args.img_height,
            width=args.img_width,
            device=device,
            num_classes=args.num_classes,
        )
        results.append((test_cfg["model_name"], passed))

    # Summary Report
    print("\n" + "=" * 80)
    print("                       SMOKE TEST SUMMARY REPORT                                ")
    print("=" * 80)
    all_success = True
    for name, success in results:
        status_str = "[PASS]" if success else "[FAIL]"
        if not success:
            all_success = False
        print(f"  {status_str} : {name}")
    print("=" * 80)

    if all_success:
        print("[FINAL STATUS] SUCCESS: All forward, backward, NaN/Inf and optimizer checks PASSED.")
        sys.exit(0)
    else:
        print("[FINAL STATUS] FAILURE: One or more smoke tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
