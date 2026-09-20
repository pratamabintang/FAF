import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np 
from torch.autograd import Variable
from torch.utils.data import DataLoader
try:
    from FusionModelDataset import FusionModelDataset
except ImportError:
    FusionModelDataset = None

try:
    from PST900Dataset import PST900Dataset, get_pst900_palette
except ImportError:
    PST900Dataset = None
    get_pst900_palette = None

from LandslideDataset import LandslideDataset
try:
    from LandslideDatasetV2 import LandslideDatasetV2
except ImportError:
    LandslideDatasetV2 = None
from FusionModelUtils import compute_results, get_palette, visualize
from sklearn.metrics import confusion_matrix
from scipy.io import savemat 
import torch.nn.functional as F 
from FusionModel import FusionModel, get_backbone_context_dim
from train import ModelEMA
from tent import Tent, collect_params, configure_model

# Explicit model class registry (P0: replace unsafe eval())
MODELS = {
    "FusionModel": FusionModel,
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Test with pytorch')
    parser.add_argument('--model_name', '-m', type=str, default='FusionModel')
    parser.add_argument('--weight_name', '-w', type=str, default='checkpoints')
    parser.add_argument('--file_name', '-f', type=str, default='best_model.ema.pth')
    parser.add_argument('--dataset_split', '-d', type=str, default='test') # test, val, test_day, test_night
    parser.add_argument('--have-test-labels', '--have_test_labels', '-htl',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Whether ground truth test labels are available for quantitative evaluation')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--img_height', '-ih', type=int, default=512) 
    parser.add_argument('--img_width', '-iw', type=int, default=512)  
    parser.add_argument('--num_workers', '-j', type=int, default=4)
    parser.add_argument('--n_class', '-nc', type=int, default=2)
    parser.add_argument('--context_dim', type=str, default='[96,192,384,768]')
    parser.add_argument('--dataset', type=str, default='landslide', choices=['landslide', 'landslide_v2', 'mfnet', 'pst900'],
                        help='Dataset to evaluate: landslide, landslide_v2 (2 classes), mfnet (9 classes) or pst900 (5 classes)')
    parser.add_argument('--channels', type=str, default=None,
                        help='Input channels for LandslideDatasetV2 (comma-separated, e.g. rgb,dtm,slope)')
    parser.add_argument('--apply_blacklist', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply blacklist filtering to skip noisy tiles in LandslideDatasetV2')
    parser.add_argument('--blacklist_path', type=str, default=None,
                        help='Path to blacklist file (defaults to dataset/dataset_1V2/black_list.txt)')
    parser.add_argument('--ignore_rgb_black', action=argparse.BooleanOptionalAction, default=True,
                        help='Ignore RGB(0,0,0) void/border pixels during evaluation')
    parser.add_argument('--data_dir', '-dr', type=str, default='./dataset/dataset_1')
    parser.add_argument('--model_dir', '-wd', type=str, default='Experiments/faf_landslide_experiment_v1')
    parser.add_argument('--visualize',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Save prediction masks and side-by-side diagnostic panels')
    # ---- DTM Preprocessing Overrides (defaults recovered dynamically from checkpoint) ----
    parser.add_argument('--dtm_norm', type=str, default=None, choices=['standard', 'minmax', 'local_relief', 'relative', 'slope_only', 'none'],
                        help='DTM elevation normalization method (defaults to checkpoint config)')
    parser.add_argument('--dtm_mean', type=float, default=None,
                        help='Empirical mean elevation for DTM standardization (defaults to checkpoint config)')
    parser.add_argument('--dtm_std', type=float, default=None,
                        help='Empirical std elevation for DTM standardization (defaults to checkpoint config)')
    parser.add_argument('--nodata_value', type=float, default=None,
                        help='Sentinel value for NoData in DTM rasters (defaults to checkpoint config)')
    parser.add_argument('--ignore_index', type=int, default=None,
                        help='Target mask label index to ignore during evaluation (defaults to checkpoint config)')
    parser.add_argument('--modal_mode', type=str, default=None, choices=['multimodal', 'rgb_only', 'dtm_only'],
                        help='Modality operating mode (defaults to checkpoint config: multimodal, rgb_only, dtm_only)')
    # ---- TTA argümanları ----
    parser.add_argument('--tta', action='store_true', default=False,
                        help='Enable test-time augmentation')
    parser.add_argument('--tta_scales', type=float, nargs='+', default=[1.0],
                        help='TTA scales, e.g. 0.75 1.0 1.25')
    parser.add_argument('--tta_flip', action='store_true', default=False,
                        help='Add horizontal flip to TTA')
    # ---- TENT argümanları ----
    parser.add_argument('--use_tent', action='store_true', default=False, help='Enable TENT test-time adaptation')
    # ---- Decoder/Fusion architecture ----
    parser.add_argument('--rgb_arch', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                        help='timm model name for RGB backbone')
    parser.add_argument('--ir_arch', type=str, default='convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                        help='timm model name for IR backbone')
    parser.add_argument('--decoder_type', type=str, default='panet', choices=['fpn', 'panet'],
                        help='Decoder type: fpn or panet')
    parser.add_argument('--deep_supervision', action='store_true', default=False,
                        help='Enable deep supervision for PANet decoder')
    parser.add_argument('--enhanced_fusion', action='store_true', default=False,
                        help='Use EnhancedSemanticFusion for stages 3-4')
    parser.add_argument('--use_safd', action='store_true', default=False,
                        help='Novel: Scene-Adaptive Frequency Decomposition')
    parser.add_argument('--use_cafg', action='store_true', default=False,
                        help='Novel: Complementarity-Aware Fusion Gate')
    parser.add_argument('--use_tpsw', action='store_true', default=False,
                        help='Novel: Thermal Prior-Guided Spatial Weighting')
    return parser


def tta_inference(model, rgb, ir, scales=[1.0], do_flip=False, output_size=(480, 640)):
    """
    Apply test-time augmentation (TTA) by scaling and flipping.
    Args:
        model: segmentation model
        rgb: RGB input tensor [1, C, H, W]
        ir: IR input tensor [1, C, H, W]
        scales: list of scale factors (e.g., [0.75, 1.0, 1.25])
        do_flip: whether to apply horizontal flip
        output_size: desired final output size (height, width)
    Returns:
        Averaged logits after TTA
    """
    prob_list = []

    for scale in scales:
        # --- Resize inputs ---
        if scale != 1.0:
            # We must ensure new dimensions are divisible by 32 for ConvNeXt
            new_h = int(rgb.shape[2] * scale / 32) * 32
            new_w = int(rgb.shape[3] * scale / 32) * 32
            rgb_scaled = F.interpolate(rgb, size=(new_h, new_w), mode='bilinear', align_corners=False)
            ir_scaled  = F.interpolate(ir,  size=(new_h, new_w), mode='bilinear', align_corners=False)
        else:
            rgb_scaled = rgb
            ir_scaled = ir

        # --- Normal forward ---
        logits, _ = model(rgb_scaled, ir_scaled)
        logits = F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)
        prob = F.softmax(logits, dim=1)
        prob_list.append(prob)

        # --- Horizontal flip ---
        if do_flip:
            rgb_flipped = torch.flip(rgb_scaled, dims=[3])
            ir_flipped  = torch.flip(ir_scaled, dims=[3])
            logits_flip, _ = model(rgb_flipped, ir_flipped)
            logits_flip = torch.flip(logits_flip, dims=[3])  # flip back
            logits_flip = F.interpolate(logits_flip, size=output_size, mode='bilinear', align_corners=False)
            prob_flip = F.softmax(logits_flip, dim=1)
            prob_list.append(prob_flip)

    # --- Average all probabilities ---
    final_prob = torch.stack(prob_list, dim=0).mean(dim=0)
    return final_prob


def main(args=None):
    if args is None:
        parser = build_parser()
        args = parser.parse_args()
  
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("\nthe pytorch version:", torch.__version__)
        print("the gpu count:", torch.cuda.device_count())
        print("the current used gpu:", torch.cuda.current_device(), '\n')
    else:
        print("\nthe pytorch version:", torch.__version__)
        print("running on CPU mode.\n")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Prepare isolated demo output directory (preserve existing runs)
    save_demo_dir = os.path.join("./runs", "demo_results")
    os.makedirs(save_demo_dir, exist_ok=True)
    try:
        os.chmod(save_demo_dir, stat.S_IRWXU)
    except Exception:
        pass

    model_dir = os.path.join(args.model_dir, args.weight_name)
    if not os.path.exists(model_dir):
        sys.exit(f"The model directory {model_dir} does not exist.")
    model_file = os.path.join(model_dir, args.file_name)
    if not os.path.exists(model_file):
        sys.exit(f"No checkpoint file found at: {model_file}")

    print(f"Testing {args.model_name}: {args.weight_name} on {device} with PyTorch")
    print(f"Loading checkpoint file: {model_file} ...")
    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    # Recover model configuration from checkpoint
    ckpt_config = checkpoint.get("config", {})
    if isinstance(ckpt_config, object) and hasattr(ckpt_config, "__dict__"):
        ckpt_config = vars(ckpt_config)
    elif not isinstance(ckpt_config, dict):
        ckpt_config = {}

    rgb_arch = ckpt_config.get("rgb_arch", args.rgb_arch)
    ir_arch = ckpt_config.get("ir_arch", args.ir_arch)
    num_classes = ckpt_config.get("num_classes", args.n_class)
    decoder_type = ckpt_config.get("decoder_type", args.decoder_type)
    deep_supervision = ckpt_config.get("deep_supervision", args.deep_supervision)
    enhanced_fusion = ckpt_config.get("enhanced_fusion", args.enhanced_fusion)
    use_safd = ckpt_config.get("use_safd", args.use_safd)
    use_cafg = ckpt_config.get("use_cafg", args.use_cafg)
    use_tpsw = ckpt_config.get("use_tpsw", args.use_tpsw)
    include_derivatives = ckpt_config.get("include_derivatives", getattr(args, "include_derivatives", False))
    ir_in_chans = 4 if include_derivatives else ckpt_config.get("ir_in_chans", getattr(args, "ir_in_chans", 1))

    if "context_dim" in ckpt_config:
        raw_cd = ckpt_config["context_dim"]
        if isinstance(raw_cd, str):
            ctx_dim = [int(x.strip()) for x in raw_cd.strip('[]').split(',')]
        else:
            ctx_dim = list(raw_cd)
    else:
        ctx_dim = [int(x.strip()) for x in args.context_dim.strip('[]').split(',')]

    img_h = ckpt_config.get("img_height", args.img_height)
    img_w = ckpt_config.get("img_width", args.img_width)
    resolution = (img_h, img_w)
    dataset_type = ckpt_config.get("dataset", args.dataset)

    # Dynamic DTM preprocessing and NoData extraction from checkpoint configuration
    dtm_norm = args.dtm_norm if args.dtm_norm is not None else ckpt_config.get("dtm_norm", "standard")
    dtm_mean = args.dtm_mean if args.dtm_mean is not None else ckpt_config.get("dtm_mean", 72.82)
    dtm_std = args.dtm_std if args.dtm_std is not None else ckpt_config.get("dtm_std", 58.01)
    nodata_value = args.nodata_value if args.nodata_value is not None else ckpt_config.get("nodata_value", -9999.0)
    ignore_index = args.ignore_index if args.ignore_index is not None else ckpt_config.get("ignore_index", -100)
    ignore_nodata = ckpt_config.get("ignore_nodata", True)
    pixel_scale = ckpt_config.get("pixel_scale", 1.0)
    modal_mode = getattr(args, "modal_mode", None) or ckpt_config.get("modal_mode", "multimodal")

    # Synchronize args with checkpoint configuration
    args.n_class = num_classes
    args.dataset = dataset_type
    args.img_height = img_h
    args.img_width = img_w

    print(f"[CONFIG] Instantiating {args.model_name} from checkpoint configuration:")
    print(f"  rgb_arch: {rgb_arch} | ir_arch: {ir_arch} | ir_in_chans: {ir_in_chans}")
    print(f"  decoder_type: {decoder_type} | deep_supervision: {deep_supervision} | modal_mode: {modal_mode}")
    print(f"  novel_fusion: SAFD={use_safd}, CAFG={use_cafg}, TPSW={use_tpsw}")
    print(f"  resolution: {resolution} | num_classes: {num_classes} | dataset: {dataset_type}")
    print(f"  dtm_norm: {dtm_norm} (mean={dtm_mean}, std={dtm_std}) | pixel_scale={pixel_scale}")
    print(f"  nodata: value={nodata_value}, ignore_index={ignore_index}, ignore_nodata={ignore_nodata}")

    if args.model_name not in MODELS:
        raise ValueError(
            f"Unknown model_name '{args.model_name}'. Available models in registry: {list(MODELS.keys())}"
        )
    model = MODELS[args.model_name](
        rgb_arch=rgb_arch,
        ir_arch=ir_arch,
        num_classes=num_classes,
        context_dim=ctx_dim,
        input_resolution=resolution,
        rgb_backbone_resolution=resolution,
        ir_backbone_resolution=resolution,
        output_resolution=resolution,
        decoder_type=decoder_type,
        deep_supervision=deep_supervision,
        enhanced_fusion=enhanced_fusion,
        use_safd=use_safd,
        use_cafg=use_cafg,
        use_tpsw=use_tpsw,
        ir_in_chans=ir_in_chans,
        modal_mode=modal_mode
    ).to(device)

    # Extract state dict and strip DataParallel 'module.' prefix if present
    raw_state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw_state_dict.items()
    }

    # Strict weight loading to guarantee architectural congruence
    try:
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] Model weights successfully loaded with strict=True.")
    except Exception as e:
        print(f"[ERROR] strict=True weight loading failed: {e}")
        raise

    # If EMA weights exist in checkpoint, apply shadow weights
    if checkpoint.get('ema_state_dict') is not None:
        ema = ModelEMA(model)
        ema.load_state_dict(checkpoint['ema_state_dict'], device)
        ema.apply_shadow(model) 
        print("[INFO] EMA weights loaded and applied.")
    else:
        print("[INFO] Using base checkpoint weights.")

    best_miou = checkpoint.get('best_miou', 0.0)
    best_ls_iou = checkpoint.get('best_landslide_iou', None)
    if best_ls_iou is not None:
        print(f"[INFO] Checkpoint records -> Best Landslide IoU: {best_ls_iou:.4f} | Best mIoU: {best_miou:.4f}")
    else:
        print(f"[INFO] Checkpoint records -> Best mIoU: {best_miou:.4f}")

    # Load dataset based on configuration
    if args.dataset in ('landslide', 'landslide_v2'):
        use_v2 = (
            args.dataset == 'landslide_v2' or
            ckpt_config.get("dataset") == 'landslide_v2' or
            '1v2' in str(args.data_dir).lower() or
            'channels' in ckpt_config
        ) and (LandslideDatasetV2 is not None)

        if use_v2:
            channels = args.channels or ckpt_config.get("channels", ['rgb', 'dtm', 'slope'])
            apply_bl = getattr(args, 'apply_blacklist', ckpt_config.get("apply_blacklist", True))
            bl_path = getattr(args, 'blacklist_path', ckpt_config.get("blacklist_path", None))
            ignore_rgb_black = getattr(args, 'ignore_rgb_black', ckpt_config.get("ignore_rgb_black", True))

            print(f"[INFO] Loading LandslideDatasetV2 ({args.n_class} classes) with channels={channels}")
            test_dataset = LandslideDatasetV2(
                data_dir=args.data_dir,
                split=args.dataset_split,
                img_size=resolution,
                channels=channels,
                apply_blacklist=apply_bl,
                blacklist_path=bl_path,
                is_training=False,
                ignore_rgb_black=ignore_rgb_black,
                ignore_index=ignore_index,
                require_labels=args.have_test_labels
            )
        else:
            print(f"[INFO] Loading Landslide dataset ({args.n_class} classes)")
            test_dataset = LandslideDataset(
                data_dir=args.data_dir,
                split=args.dataset_split,
                img_size=resolution,
                is_training=False,
                dtm_norm=dtm_norm,
                dtm_mean=dtm_mean,
                dtm_std=dtm_std,
                include_derivatives=include_derivatives,
                pixel_scale=pixel_scale,
                nodata_value=nodata_value,
                ignore_nodata=ignore_nodata,
                ignore_index=ignore_index,
                require_labels=args.have_test_labels
            )
    elif args.dataset == 'pst900':
        if PST900Dataset is None:
            raise ImportError("PST900Dataset requires 'albumentations' which is not installed.")
        print(f"[INFO] Loading PST900 dataset ({args.n_class} classes)")
        test_dataset = PST900Dataset(
            data_dir=args.data_dir,
            split=args.dataset_split,
            rgb_size=resolution,
            thermal_size=resolution,
            have_label=args.have_test_labels,
            use_augmentation=False
        )
    elif args.dataset == 'mfnet':
        if FusionModelDataset is None:
            raise ImportError("Failed to import FusionModelDataset / MFNetDataset.")
        print(f"[INFO] Loading MFNet dataset ({args.n_class} classes)")
        test_dataset = FusionModelDataset(
            data_dir=args.data_dir,
            split=args.dataset_split,
            have_label=args.have_test_labels
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
        
    batch_size = 1
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    if args.tta:
        mult = 2 if args.tta_flip else 1
        print(f"[INFO] TTA enabled | scales={args.tta_scales} | flip={args.tta_flip} "
              f"| total-forwards-per-image={len(args.tta_scales)*mult}")    

    conf_total = np.zeros((num_classes, num_classes))
    ave_time_cost = 0.0
    timed_frames = 0
    use_cuda_events = torch.cuda.is_available()
    if use_cuda_events:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    if args.use_tent:
        model = configure_model(model)
        tent_params, param_names = collect_params(model)
        for p, name in zip(tent_params, param_names):
            if not p.requires_grad:
                print(f"[WARN] Param {name} does not have requires_grad=True")

        tent_optimizer = torch.optim.SGD(tent_params, lr=1e-3) 
        tent = Tent(model, tent_optimizer)
        print("[INFO] TENT enabled | optimizing params:", param_names)

    if not args.use_tent:
        model.eval()

    for it, batch_data in enumerate(test_loader):
        rgb, ir, raw_labels, img_name = batch_data
        rgb = rgb.to(device)
        ir = ir.to(device)
        labels = raw_labels.to(device) if args.have_test_labels else None
        
        t0 = time.perf_counter()
        if use_cuda_events:
            starter.record()
            
        if args.tta:
            with torch.no_grad():
                logits = tta_inference(model, rgb, ir, scales=args.tta_scales, do_flip=args.tta_flip, output_size=resolution)
                aux = None
        elif args.use_tent:
            logits = tent(rgb, ir)
            aux = None
        else:
            with torch.no_grad():
                logits, aux = model(rgb, ir)
                
        if use_cuda_events:
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
        else:
            curr_time = (time.perf_counter() - t0) * 1000.0
            
        if len(test_loader) > 5:
            if it >= 5: # ignore warmup frames
                ave_time_cost += curr_time
                timed_frames += 1
        else:
            ave_time_cost += curr_time
            timed_frames += 1

        # Only accumulate confusion matrix if real ground-truth labels are present
        if args.have_test_labels and labels is not None:
            lbl_flat = labels.cpu().numpy().squeeze().flatten()
            pred_flat = logits.argmax(1).cpu().numpy().squeeze().flatten()
            valid_mask = (lbl_flat != ignore_index) & (lbl_flat >= 0) & (lbl_flat < num_classes)
            if np.any(valid_mask):
                conf = confusion_matrix(
                    y_true=lbl_flat[valid_mask],
                    y_pred=pred_flat[valid_mask],
                    labels=list(range(num_classes))
                )
                conf_total += conf
        
        if args.visualize:
            sample_name = img_name[0] if isinstance(img_name, (list, tuple)) else str(img_name)
            visualize(
                image_name=[str(sample_name)],
                predictions=logits.argmax(1),
                weight_name=args.weight_name,
                rgb=rgb,
                ir=ir,
                labels=labels,
                dataset=args.dataset,
                save_dir=save_demo_dir,
                save_side_by_side=True,
                ignore_index=ignore_index
            )
        mode_str = "eval" if args.have_test_labels else "predict"
        print("%s, %s, frame %d/%d, %s, mode: %s, time cost: %.2f ms, demo result processed."
              % (args.model_name, args.weight_name, it+1, len(test_loader), str(it), mode_str, curr_time))

    mean_time_ms = ave_time_cost / max(timed_frames, 1)
    fps = (1000.0 / mean_time_ms) if mean_time_ms > 0 else 0.0

    if args.have_test_labels:
        precision_per_class, recall_per_class, iou_per_class, f1score = compute_results(conf_total)
        conf_total_matfile = os.path.join(save_demo_dir, 'conf_' + args.weight_name + '.mat')
        savemat(conf_total_matfile, {'conf': conf_total})

        if dataset_type in ('landslide', 'landslide_v2') or num_classes == 2:
            class_names = ["Background", "Landslide"]
        elif dataset_type == 'pst900' or num_classes == 5:
            class_names = ["Background", "Fire Extinguisher", "Backpack", "Hand Drill", "Rescue Randy"]
        elif dataset_type == 'mfnet' or num_classes == 9:
            class_names = ["Unlabeled", "Car", "Person", "Bike", "Curve", "Car Stop", "Guardrail", "Color Cone", "Bump"]
        else:
            class_names = [f"Class_{i}" for i in range(num_classes)]

        device_name = torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else 'CPU'
        print('\n###########################################################################')
        print('\n%s: %s test results (with batch size %d) on %s using %s:' % (args.model_name, args.weight_name, batch_size, datetime.date.today(), device_name)) 
        print('\n* Tested dataset: %s (split: %s)' % (dataset_type, args.dataset_split))
        print('* Tested image count: %d' % len(test_loader))
        print('* Tested image size: %d x %d' % (img_h, img_w)) 
        print('* Weight dir: %s' % args.weight_name) 
        print('* Checkpoint file: %s' % args.file_name)

        print('\n* Per-class Recall:')
        for i in range(num_classes):
            cname = class_names[i] if i < len(class_names) else f"Class_{i}"
            print(f'    [{i}] {cname:<18}: {recall_per_class[i]:.6f}')

        print('\n* Per-class IoU:')
        for i in range(num_classes):
            cname = class_names[i] if i < len(class_names) else f"Class_{i}"
            print(f'    [{i}] {cname:<18}: {iou_per_class[i]:.6f}')

        print('\n* Per-class F1-score:')
        for i in range(num_classes):
            cname = class_names[i] if i < len(class_names) else f"Class_{i}"
            print(f'    [{i}] {cname:<18}: {f1score[i]:.6f}')

        print("\n* Mean metrics across all classes:")
        print("  Recall: %.6f | IoU (mIoU): %.6f | Precision: %.6f | F1-score: %.6f"
              % (recall_per_class.mean(), iou_per_class.mean(), precision_per_class.mean(), f1score.mean()))

        if dataset_type == 'landslide' or num_classes == 2:
            print(f"\n* [PRIMARY METRIC] Landslide (Class 1) IoU: {iou_per_class[1]:.4f} | F1-Score: {f1score[1]:.4f}")

        print(f'\n* Average time cost per frame: {mean_time_ms:.2f} ms (over {timed_frames} timed frames) -> {fps:.2f} FPS')
        print('###########################################################################')
    else:
        # Dedicated prediction-only mode summary
        device_name = torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else 'CPU'
        print('\n###########################################################################')
        print('\n%s: %s prediction-only inference on %s using %s:' % (args.model_name, args.weight_name, datetime.date.today(), device_name)) 
        print('\n* Mode: Dedicated Prediction-Only (Ground-truth labels suppressed)')
        print('* Tested dataset: %s (split: %s)' % (dataset_type, args.dataset_split))
        print('* Inferred image count: %d' % len(test_loader))
        print('* Inferred image size: %d x %d' % (img_h, img_w)) 
        print('* Weight dir: %s' % args.weight_name) 
        print('* Checkpoint file: %s' % args.file_name)
        print('* Output directory: %s' % save_demo_dir)
        print(f'* Average time cost per frame: {mean_time_ms:.2f} ms (over {timed_frames} timed frames) -> {fps:.2f} FPS')
        print('###########################################################################')


if __name__ == '__main__':
    main()