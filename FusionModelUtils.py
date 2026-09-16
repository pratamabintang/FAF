"""
================================================================================
Fusion Model Utilities: Metrics, Palettes, and Multimodal Visualizations
================================================================================
Provides:
  1. compute_results: Per-class Precision, Recall, IoU, and F1-Score from Confusion Matrix.
  2. get_palette: Distinct color palettes for Landslide, PST900, and MFNet datasets.
  3. visualize: Generates side-by-side multimodal diagnostic grids and prediction overlays.
================================================================================
"""

import os
from typing import Tuple, List, Optional, Union
import numpy as np
import cv2
from PIL import Image
import torch


def compute_results(conf_total: np.ndarray):
    """
    Compute per-class Precision, Recall, IoU, and F1-Score from confusion matrix.
    conf_total: shape (n_class, n_class) where rows are ground truth, columns are predictions.
    """
    n_class = conf_total.shape[0]
    precision_per_class = np.zeros(n_class, dtype=np.float64)
    recall_per_class = np.zeros(n_class, dtype=np.float64)
    iou_per_class = np.zeros(n_class, dtype=np.float64)
    f1score = np.zeros(n_class, dtype=np.float64)

    for i in range(n_class):
        tp = float(conf_total[i, i])
        fp = float(np.sum(conf_total[:, i]) - tp)
        fn = float(np.sum(conf_total[i, :]) - tp)

        precision_per_class[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_per_class[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou_per_class[i] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        p = precision_per_class[i]
        r = recall_per_class[i]
        f1score[i] = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

    return precision_per_class, recall_per_class, iou_per_class, f1score


def get_palette(dataset: str = "landslide", num_classes: int = 2) -> np.ndarray:
    """
    Returns RGB color palette tailored for specific datasets:
      - Landslide (2 classes): 0=Background (Black), 1=Landslide (Bright Red)
      - PST900 (5 classes): Background, Fire Extinguisher, Backpack, Hand Drill, Rescue Randy
      - MFNet (9 classes): Unlabeled, Car, Person, Bike, Curve, Car Stop, Guardrail, Cone, Bump
    """
    dataset_lower = str(dataset).lower()

    if dataset_lower == "landslide" or num_classes == 2:
        palette = [
            [0, 0, 0],         # 0: Background (Black)
            [255, 50, 50],     # 1: Landslide (Vivid Red)
        ]
    elif dataset_lower == "pst900" or num_classes == 5:
        palette = [
            [0, 0, 0],         # 0: Background (Black)
            [255, 0, 0],       # 1: Fire Extinguisher (Red)
            [0, 255, 0],       # 2: Backpack (Green)
            [0, 0, 255],       # 3: Hand Drill (Blue)
            [255, 255, 0],     # 4: Rescue Randy (Yellow)
        ]
    else:
        # Default 9-class MFNet palette
        palette = [
            [0, 0, 0],         # 0: Unlabeled / Background
            [64, 0, 128],      # 1: Car
            [64, 64, 0],       # 2: Person
            [0, 128, 192],     # 3: Bike
            [0, 0, 192],       # 4: Curve
            [128, 128, 0],     # 5: Car Stop
            [64, 64, 128],     # 6: Guardrail
            [192, 128, 128],   # 7: Color Cone
            [192, 64, 0],      # 8: Bump
        ]

    if num_classes > len(palette):
        np.random.seed(42)
        extra = np.random.randint(0, 255, size=(num_classes - len(palette), 3)).tolist()
        palette.extend(extra)

    return np.array(palette[:num_classes], dtype=np.uint8)


def colorize_mask(
    mask: np.ndarray,
    palette: np.ndarray,
    ignore_index: int = -100,
    nodata_color: Union[Tuple[int, int, int], List[int], np.ndarray] = (128, 128, 128)
) -> np.ndarray:
    """
    Applies RGB palette to integer segmentation mask.
    Renders NoData pixels (ignore_index or negative values) in neutral gray ([128, 128, 128]).
    """
    is_nodata = (mask == ignore_index) | (mask < 0)
    safe_mask = np.where(is_nodata, 0, mask)
    mask_clipped = np.clip(safe_mask, 0, len(palette) - 1).astype(np.uint8)
    colored = palette[mask_clipped].copy()
    if np.any(is_nodata):
        colored[is_nodata] = np.array(nodata_color, dtype=np.uint8)
    return colored


def visualize(
    image_name: Union[List[str], str],
    predictions: Union[torch.Tensor, np.ndarray],
    weight_name: str = "demo",
    rgb: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ir: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    dataset: str = "landslide",
    save_dir: str = "./runs/demo_results",
    save_side_by_side: bool = True,
    ignore_index: int = -100,
    nodata_color: Tuple[int, int, int] = (128, 128, 128)
):
    """
    Saves color-mapped predictions and comprehensive side-by-side diagnostic panels:
      - With ground truth: [RGB | DTM/IR Normalized | Ground Truth | Prediction Overlay] (4 panels)
      - Prediction-only (labels=None): [RGB | DTM/IR Normalized | Prediction Overlay] (3 panels)
    Renders NoData pixels (ignore_index = -100) in neutral gray ([128, 128, 128]) across all panels
    to eliminate false-positive visual artifacts outside valid survey terrain.
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(image_name, str):
        image_name = [image_name]

    if hasattr(predictions, "cpu"):
        preds = predictions.detach().cpu().numpy()
    else:
        preds = np.asarray(predictions)

    if preds.ndim == 4:
        preds = np.argmax(preds, axis=1)

    num_classes = int(np.max(preds)) + 1 if len(preds) > 0 else 2
    num_classes = max(num_classes, 2 if dataset == "landslide" else 9)
    palette = get_palette(dataset=dataset, num_classes=num_classes)

    # Convert tensors to numpy if provided
    rgb_np = rgb.detach().cpu().numpy() if hasattr(rgb, "cpu") else (np.asarray(rgb) if rgb is not None else None)
    ir_np  = ir.detach().cpu().numpy() if hasattr(ir, "cpu") else (np.asarray(ir) if ir is not None else None)
    lbl_np = labels.detach().cpu().numpy() if hasattr(labels, "cpu") else (np.asarray(labels) if labels is not None else None)

    for idx, name in enumerate(image_name):
        pred_mask = preds[idx].astype(np.int64)

        # Check if ground truth labels are provided for this sample
        cur_lbl = None
        nodata_mask = None
        if lbl_np is not None:
            cur_lbl = lbl_np[idx]
            if cur_lbl.ndim == 3:
                cur_lbl = cur_lbl[0]
            nodata_mask = (cur_lbl == ignore_index) | (cur_lbl < 0)

        # 1. Save raw colored prediction mask
        color_pred = colorize_mask(pred_mask, palette, ignore_index=ignore_index, nodata_color=nodata_color)
        if nodata_mask is not None and np.any(nodata_mask):
            color_pred[nodata_mask] = np.array(nodata_color, dtype=np.uint8)

        out_mask_path = os.path.join(save_dir, f"{name}_{weight_name}_mask.png")
        cv2.imwrite(out_mask_path, cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))

        # 2. If RGB available and side_by_side requested, save multi-panel comparison
        if save_side_by_side and rgb_np is not None:
            cur_rgb = rgb_np[idx]
            if cur_rgb.ndim == 3 and cur_rgb.shape[0] == 3:
                cur_rgb = np.transpose(cur_rgb, (1, 2, 0))
            # Denormalize standard ImageNet normalized RGB
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            if cur_rgb.dtype != np.uint8:
                if cur_rgb.min() < 0.0 or cur_rgb.max() <= 5.0:
                    # Standardized with (x - mean) / std
                    cur_rgb = np.clip((cur_rgb * std + mean) * 255.0, 0, 255).astype(np.uint8)
                elif cur_rgb.max() <= 1.0:
                    cur_rgb = np.clip(cur_rgb * 255.0, 0, 255).astype(np.uint8)
                else:
                    cur_rgb = np.clip(cur_rgb, 0, 255).astype(np.uint8)
            else:
                cur_rgb = np.clip(cur_rgb, 0, 255).astype(np.uint8)

            H, W = cur_rgb.shape[:2]

            # Prepare DTM visualization (Colormapped elevation / hillshade)
            if ir_np is not None:
                cur_ir = ir_np[idx]
                if cur_ir.ndim == 3:
                    cur_ir = cur_ir[0]
                # Min-max normalization for visualization on valid pixels
                if nodata_mask is not None and np.any(nodata_mask):
                    valid_ir = cur_ir[~nodata_mask]
                    if len(valid_ir) > 0 and np.max(valid_ir) > np.min(valid_ir):
                        ir_min, ir_max = float(np.min(valid_ir)), float(np.max(valid_ir))
                        ir_vis = np.clip((cur_ir - ir_min) / (ir_max - ir_min) * 255.0, 0, 255).astype(np.uint8)
                    else:
                        ir_vis = np.zeros((H, W), dtype=np.uint8)
                else:
                    ir_min, ir_max = float(np.min(cur_ir)), float(np.max(cur_ir))
                    if ir_max > ir_min:
                        ir_vis = ((cur_ir - ir_min) / (ir_max - ir_min) * 255.0).astype(np.uint8)
                    else:
                        ir_vis = np.zeros((H, W), dtype=np.uint8)

                ir_vis_color = cv2.applyColorMap(ir_vis, cv2.COLORMAP_TURBO)
                ir_vis_rgb = cv2.cvtColor(ir_vis_color, cv2.COLOR_BGR2RGB)
                if nodata_mask is not None and np.any(nodata_mask):
                    ir_vis_rgb[nodata_mask] = np.array(nodata_color, dtype=np.uint8)
            else:
                ir_vis_rgb = np.zeros((H, W, 3), dtype=np.uint8)

            # Ground Truth Color (if labels provided)
            color_gt = None
            if cur_lbl is not None:
                color_gt = colorize_mask(cur_lbl, palette, ignore_index=ignore_index, nodata_color=nodata_color)

            # Overlay Prediction on RGB (Alpha = 0.5)
            overlay = cur_rgb.copy()
            landslide_pixels = (pred_mask > 0)
            if nodata_mask is not None and np.any(nodata_mask):
                landslide_pixels = landslide_pixels & (~nodata_mask)
            if np.any(landslide_pixels):
                overlay[landslide_pixels] = (
                    0.5 * cur_rgb[landslide_pixels] + 0.5 * color_pred[landslide_pixels]
                ).astype(np.uint8)
            if nodata_mask is not None and np.any(nodata_mask):
                overlay[nodata_mask] = np.array(nodata_color, dtype=np.uint8)

            # Assemble diagnostic grid:
            # - If ground truth available: 4 panels [RGB | DTM | Ground Truth | Prediction Overlay]
            # - Prediction-only: 3 panels [RGB | DTM | Prediction Overlay] (suppressing fake Ground Truth)
            if color_gt is not None:
                panel = np.hstack([cur_rgb, ir_vis_rgb, color_gt, overlay])
            else:
                panel = np.hstack([cur_rgb, ir_vis_rgb, overlay])

            panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            out_panel_path = os.path.join(save_dir, f"{name}_{weight_name}_diagnostic.png")
            cv2.imwrite(out_panel_path, panel_bgr)
