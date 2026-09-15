"""
================================================================================
Landslide Dataset Loader (RGB + DTM / Terrain Fusion)
================================================================================
Dedicated PyTorch Dataset for Multimodal Landslide Semantic Segmentation.

Design & Specifications:
  [P0] Reads RGB as 3 channels (float32 [3, H, W], normalized).
  [P0] Reads DTM (Digital Terrain Model) as float32 [K, H, W], NOT uint8.
  [P0] Reads segmentation mask as integer (int64 [H, W]: 0=Background, 1=Landslide).
  [P0] Returns: rgb, terrain, mask, sample_id
  [P0] Uses bilinear interpolation for RGB (cv2.INTER_LINEAR).
  [P0] Uses bilinear interpolation for continuous terrain/DTM (cv2.INTER_LINEAR).
  [P0] Uses nearest-neighbor interpolation for mask and validity map (cv2.INTER_NEAREST).
  [P0] Handles NoData / NaN before and after resize with validity masking.
  [P1] Coordinated spatial transformations (synchronous crop/flip/rotate90 on RGB, Terrain, Mask).
  [P1] Color augmentations applied ONLY to RGB.
  [P1] No thermal brightness/contrast distortion on DTM.
  [P1] Avoids elastic, perspective, and grid distortion on baseline.

Dataset Directory Structure:
  <data_root>/
  ├── train/ (or val, test)
  │   ├── IMAGE/   (RGB images: .png, .jpg, .tif)
  │   ├── DTM/     (DTM elevation rasters: .tif, .png)
  │   └── LABEL/   (Mask labels: .png, .tif - 0=BG, >0=Landslide)
================================================================================
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional, Union, Dict

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Prevent OpenCV thread contention across DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Default Normalization Statistics (Empirical statistics from dataset_1 train split)
DEFAULT_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_DTM_MEAN = 72.82   # Dataset empirical elevation mean (meters)
DEFAULT_DTM_STD  = 58.01   # Dataset empirical elevation std (meters)


class LandslideDataset(Dataset):
    """
    Multimodal Landslide Segmentation Dataset (RGB + DTM/Elevation).
    
    Args:
        data_dir: Path to dataset root or split folder (e.g. dataset/dataset_1).
        split: Dataset split ('train', 'val', 'test').
        img_size: Target (Height, Width) for model input. Default (480, 640).
        is_training: If True, applies data augmentations (flips, rotations, color jitter).
        dtm_norm: Normalization method for DTM ('standard', 'minmax', 'none'). Default 'standard'.
        dtm_mean: Mean elevation for DTM standardization. Default 72.82.
        dtm_std: Std elevation for DTM standardization. Default 58.01.
        rgb_mean: Mean for RGB normalization. Default ImageNet mean.
        rgb_std: Std for RGB normalization. Default ImageNet std.
        include_derivatives: If True, computes slope and aspect channels (K=3: [DTM, Slope, Aspect]).
        nodata_value: Sentinel value for NoData in DTM rasters. Default -9999.0.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        img_size: Tuple[int, int] = (480, 640),
        is_training: Optional[bool] = None,
        dtm_norm: str = "standard",
        dtm_mean: float = DEFAULT_DTM_MEAN,
        dtm_std: float = DEFAULT_DTM_STD,
        rgb_mean: np.ndarray = DEFAULT_RGB_MEAN,
        rgb_std: np.ndarray = DEFAULT_RGB_STD,
        include_derivatives: bool = False,
        nodata_value: float = -9999.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.split = split
        self.target_h, self.target_w = img_size
        self.is_training = (split == "train") if is_training is None else is_training
        self.dtm_norm = dtm_norm
        self.dtm_mean = float(dtm_mean)
        self.dtm_std = max(float(dtm_std), 1e-6)
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.include_derivatives = include_derivatives
        self.nodata_value = nodata_value

        # Resolve split directory
        split_candidates = [
            self.data_dir / split,
            self.data_dir
        ]
        self.split_dir = None
        for cand in split_candidates:
            if (cand / "IMAGE").exists() and (cand / "DTM").exists():
                self.split_dir = cand
                break

        if self.split_dir is None:
            raise FileNotFoundError(
                f"Could not find valid IMAGE and DTM folders in {self.data_dir} for split '{split}'."
            )

        self.img_dir = self.split_dir / "IMAGE"
        self.dtm_dir = self.split_dir / "DTM"
        self.lbl_dir = self.split_dir / "LABEL"

        # Index and match triplets
        self.samples = self._index_dataset_triplets()
        print(f"[LandslideDataset] Split: {split.upper()} | Samples: {len(self.samples)} | Dir: {self.split_dir}")

    def _index_dataset_triplets(self) -> List[Dict[str, Path]]:
        """Finds matching RGB, DTM, and optional LABEL file paths."""
        valid_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        img_files = {
            f.stem: f for f in self.img_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        }
        dtm_files = {
            f.stem: f for f in self.dtm_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        }

        lbl_files = {}
        if self.lbl_dir.exists():
            lbl_files = {
                f.stem: f for f in self.lbl_dir.iterdir()
                if f.suffix.lower() in valid_extensions
            }

        common_stems = sorted(list(set(img_files.keys()) & set(dtm_files.keys())))
        if not common_stems:
            raise RuntimeError(f"No matching IMAGE and DTM triplets found in {self.split_dir}!")

        triplets = []
        for stem in common_stems:
            triplets.append({
                "stem": stem,
                "img_path": img_files[stem],
                "dtm_path": dtm_files[stem],
                "lbl_path": lbl_files.get(stem, None)
            })
        return triplets

    def _read_rgb(self, path: Path) -> np.ndarray:
        """Reads RGB image as 3-channel uint8 array in RGB color order."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback with PIL
            pil_img = Image.open(str(path)).convert("RGB")
            img = np.array(pil_img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # Shape: (H, W, 3), uint8

    def _read_dtm(self, path: Path) -> np.ndarray:
        """Reads DTM raster preserving continuous float32 values."""
        dtm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if dtm is None:
            try:
                import tifffile
                dtm = tifffile.imread(str(path))
            except ImportError:
                dtm = np.array(Image.open(str(path)))

        dtm = np.asarray(dtm, dtype=np.float32)
        if dtm.ndim == 3:
            dtm = dtm[:, :, 0]  # Single channel elevation
        return dtm  # Shape: (H, W), float32

    def _read_mask(self, path: Optional[Path], default_shape: Tuple[int, int]) -> np.ndarray:
        """Reads label mask as integer array: 0=Background, 1=Landslide."""
        if path is None or not path.exists():
            return np.zeros(default_shape, dtype=np.int64)

        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.array(Image.open(str(path)))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Map non-zero values (e.g. 65535, 255, 1) to integer class 1 (Landslide)
        binary_mask = (mask > 0).astype(np.int64)
        return binary_mask  # Shape: (H, W), int64

    def _handle_nodata_and_resize(
        self,
        rgb: np.ndarray,
        dtm: np.ndarray,
        mask: np.ndarray,
        target_h: int,
        target_w: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies bilinear interpolation for continuous data (RGB, DTM) and
        nearest-neighbor for discrete data (Mask, Validity Map).
        Handles NaNs and NoData prior to and post-interpolation.
        Fast-path avoids redundant resize and median calculation when data is already clean and correct size.
        """
        # Fast check: does DTM have invalid / NaN / NoData pixels?
        has_invalid = np.isnan(dtm).any() or np.isinf(dtm).any() or (dtm <= self.nodata_value).any()

        if has_invalid:
            valid_mask = (~np.isnan(dtm)) & (~np.isinf(dtm)) & (dtm > self.nodata_value)
            valid_pixels = dtm[valid_mask]
            fill_val = float(np.median(valid_pixels)) if len(valid_pixels) > 0 else 0.0
            dtm_clean = np.where(valid_mask, dtm, fill_val)
        else:
            dtm_clean = dtm
            fill_val = 0.0

        h, w = rgb.shape[:2]
        if (h, w) != (target_h, target_w):
            # Bilinear Interpolation for RGB
            rgb_resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Bilinear Interpolation for continuous DTM
            dtm_resized = cv2.resize(dtm_clean, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Nearest-Neighbor Interpolation for discrete Mask
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

            if has_invalid:
                valid_mask_resized = cv2.resize(
                    valid_mask.astype(np.uint8),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                dtm_resized = np.nan_to_num(dtm_resized, nan=fill_val, posinf=fill_val, neginf=fill_val)
                dtm_resized = np.where(valid_mask_resized, dtm_resized, fill_val)
            else:
                dtm_resized = np.nan_to_num(dtm_resized, nan=fill_val, posinf=fill_val, neginf=fill_val)
        else:
            rgb_resized = rgb
            dtm_resized = np.nan_to_num(dtm_clean, nan=fill_val, posinf=fill_val, neginf=fill_val)
            mask_resized = mask.astype(np.int64)

        return rgb_resized, dtm_resized, mask_resized

    def _apply_coordinated_spatial_aug(
        self,
        rgb: np.ndarray,
        dtm: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies strictly coordinated spatial transformations (flip, rotate90, crop)
        across RGB, DTM, and Mask simultaneously.
        """
        # Random Horizontal Flip
        if random.random() > 0.5:
            rgb = np.fliplr(rgb).copy()
            dtm = np.fliplr(dtm).copy()
            mask = np.fliplr(mask).copy()

        # Random Vertical Flip
        if random.random() > 0.5:
            rgb = np.flipud(rgb).copy()
            dtm = np.flipud(dtm).copy()
            mask = np.flipud(mask).copy()

        # Random 90-degree Rotation
        rot_k = random.choice([0, 1, 2, 3])
        if rot_k > 0:
            rgb = np.rot90(rgb, rot_k).copy()
            dtm = np.rot90(dtm, rot_k).copy()
            mask = np.rot90(mask, rot_k).copy()

        # Random Coordinated Scale Crop (Between 80% and 100% of spatial dimension)
        if random.random() > 0.5:
            h, w = dtm.shape
            crop_ratio = random.uniform(0.8, 1.0)
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)

            rgb_cropped = rgb[top:top+crop_h, left:left+crop_w]
            dtm_cropped = dtm[top:top+crop_h, left:left+crop_w]
            mask_cropped = mask[top:top+crop_h, left:left+crop_w]

            # Bilinear for RGB and DTM, Nearest for Mask
            rgb = cv2.resize(rgb_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            dtm = cv2.resize(dtm_cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask_cropped.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        return rgb, dtm, mask

    def _apply_rgb_color_aug(self, rgb: np.ndarray) -> np.ndarray:
        """Applies photometric color augmentations ONLY to RGB channels."""
        # Random Brightness and Contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.uniform(-20, 20)    # Brightness
            rgb = np.clip(alpha * rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Random HSV / Hue Saturation Shift
        if random.random() > 0.5:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            h_shift = random.uniform(-10, 10)
            s_scale = random.uniform(0.85, 1.15)
            hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Random Gaussian Blur
        if random.random() > 0.7:
            k = random.choice([3, 5])
            rgb = cv2.GaussianBlur(rgb, (k, k), 0)

        return rgb

    def _normalize_terrain(self, dtm: np.ndarray) -> np.ndarray:
        """Normalizes DTM elevation into standardized float32 range."""
        if self.dtm_norm == "standard":
            dtm_norm = (dtm - self.dtm_mean) / self.dtm_std
        elif self.dtm_norm == "minmax":
            d_min, d_max = np.min(dtm), np.max(dtm)
            dtm_norm = (dtm - d_min) / (d_max - d_min + 1e-6)
        else:
            dtm_norm = dtm

        # Ensure no NaNs or Infs persist in terrain
        dtm_norm = np.nan_to_num(dtm_norm, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        # Expand channel dimension -> (K, H, W) where K=1 (or K=3 if derivatives enabled)
        if self.include_derivatives:
            gy, gx = np.gradient(dtm)
            slope = np.arctan(np.sqrt(gx**2 + gy**2)).astype(np.float32)
            aspect = np.arctan2(-gy, gx).astype(np.float32)
            terrain = np.stack([dtm_norm, slope, aspect], axis=0)  # [3, H, W]
        else:
            terrain = dtm_norm[np.newaxis, :, :]  # [1, H, W]

        return terrain

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Loads and returns a synchronized multimodal sample.
        
        Returns:
            rgb: float32 Tensor [3, H, W]
            terrain: float32 Tensor [K, H, W] (K=1 or 3)
            mask: int64 Tensor [H, W] (0=Background, 1=Landslide)
            sample_id: str (Stem identifier)
        """
        sample_meta = self.samples[index]
        sample_id = sample_meta["stem"]

        # 1. Read Raw Modalities
        rgb_raw = self._read_rgb(sample_meta["img_path"])
        dtm_raw = self._read_dtm(sample_meta["dtm_path"])
        mask_raw = self._read_mask(sample_meta["lbl_path"], default_shape=dtm_raw.shape)

        # 2. Synchronous Spatial Resizing with NoData Handling
        rgb_res, dtm_res, mask_res = self._handle_nodata_and_resize(
            rgb_raw, dtm_raw, mask_raw, self.target_h, self.target_w
        )

        # 3. Training Augmentations
        if self.is_training:
            # [P1] Synchronized spatial crop / flip / rotate across all modalities
            rgb_res, dtm_res, mask_res = self._apply_coordinated_spatial_aug(
                rgb_res, dtm_res, mask_res
            )
            # [P1] Photometric color augmentation ONLY on RGB (NOT on DTM)
            rgb_res = self._apply_rgb_color_aug(rgb_res)

        # 4. Normalize RGB -> float32 [3, H, W]
        rgb_float = (rgb_res.astype(np.float32) / 255.0 - self.rgb_mean) / self.rgb_std
        rgb_tensor = torch.from_numpy(rgb_float.transpose(2, 0, 1)).contiguous().float()

        # 5. Normalize Terrain -> float32 [K, H, W]
        terrain_np = self._normalize_terrain(dtm_res)
        terrain_tensor = torch.from_numpy(terrain_np).contiguous().float()

        # 6. Mask -> int64 [H, W]
        mask_tensor = torch.from_numpy(mask_res).contiguous().long()

        return rgb_tensor, terrain_tensor, mask_tensor, sample_id
