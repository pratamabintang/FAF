"""
================================================================================
Landslide Dataset Loader V2 (Multimodal RGB + Topographic Fusion)
================================================================================
Modernized, modular PyTorch Dataset for Multimodal Landslide Detection.

Designed specifically for preprocessed datasets (e.g., dataset_1V2):
  - Pre-imputed NaNs, anti-aliased DTM, and pre-extracted Slope rasters.
  - Blacklist filtering for skipping noisy / corrupted tiles.
  - Flexible channel selection & spectrum concatenation (RGB, DTM, Slope, Aspect, etc.).
  - Coordinated synchronous spatial augmentations across all 5+ channels & masks:
      * 2.1 Random 90°, 180°, 270° Rotation (p=0.75) to resolve aspect blind spots
      * 2.2 Random Horizontal & Vertical Flip (p=0.5)
      * 2.3 Random Crop (448 to 480) & Resize to target (p=0.5)
      * 2.4 Elastic Distortion (p=0.2, alpha=1.0, sigma=50.0, alpha_affine=10.0)
  - Exclusive photometric augmentations for RGB only (DTM & Slope preserved):
      * 2.5 Color Jitter / Brightness & Contrast (p=0.5)
      * 2.6 CLAHE (p=0.3, clip=2.0, tile=(8, 8))
      * 2.7 Gaussian / Motion Blur (p=0.2, k=3 or 5)
  - Automatic ignore_mask for RGB(0,0,0) void/border pixels for loss & evaluation.
  - Compatible with standard dual-stream FusionModel and single-stream concat models.
================================================================================
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# Prevent OpenCV thread contention across DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Default Normalization Statistics (ImageNet RGB stats)
DEFAULT_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default Modality Subfolder Mapping
DEFAULT_MODALITY_DIRS = {
    "rgb": "IMAGE",
    "image": "IMAGE",
    "dtm": "DTM_NORM",
    "dtm_norm": "DTM_NORM",
    "slope": "SLOPE",
    "aspect": "ASPECT",
    "label": "LABEL",
    "mask": "LABEL",
}


def load_blacklist(blacklist_path: Optional[Union[str, Path]]) -> Set[str]:
    """Loads noisy tile stem identifiers from a blacklist text file."""
    if not blacklist_path:
        return set()
    path = Path(blacklist_path)
    if not path.is_file():
        print(f"[LandslideDatasetV2 WARN] Blacklist file '{path}' not found or is a directory. No tiles skipped.")
        return set()

    with open(path, "r", encoding="utf-8") as f:
        blacklist = set(
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        )
    return blacklist


class LandslideDatasetV2(Dataset):
    """
    Multimodal Landslide Semantic Segmentation Dataset V2.

    Args:
        data_dir: Path to dataset root or split directory (e.g. 'dataset/dataset_1V2').
        split: Dataset split ('train', 'val', 'test'). Default 'train'.
        img_size: Target resolution (Height, Width). Default (512, 512).
        channels: Sequence or comma-separated string of requested input channels.
                  Default ('rgb', 'dtm', 'slope') -> 5 total channels.
        apply_blacklist: Whether to filter out noisy tile stems from blacklist file. Default True.
        blacklist_path: Path to blacklist file. If None, auto-searches standard locations.
        is_training: Whether to apply data augmentations. If None, True for 'train', False otherwise.
        rgb_mean: Mean values for RGB normalization. Default ImageNet mean.
        rgb_std: Std values for RGB normalization. Default ImageNet std.
        ignore_rgb_black: If True, marks RGB(0,0,0) pixels in label mask as ignore_index. Default True.
        ignore_index: Sentinel integer label to ignore in loss and metrics. Default -100.
        return_concat: If True, returns concatenated tensor [C, H, W].
                       If False (default), returns (rgb, terrain, mask, sample_id).
        positive_aware_sampling: If True, centers random crop around landslide pixels when present.
        positive_sample_prob: Probability of centering crop on positive pixels. Default 0.5.
        crop_range: (min_size, max_size) for Random Crop. Default (448, 480).
        require_labels: Whether ground truth label masks must exist. Default True for train/val/test.
        modality_dir_map: Custom dictionary mapping channel name to directory name.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        img_size: Tuple[int, int] = (512, 512),
        channels: Union[Sequence[str], str] = ("rgb", "dtm", "slope"),
        apply_blacklist: bool = True,
        blacklist_path: Optional[Union[str, Path]] = None,
        is_training: Optional[bool] = None,
        rgb_mean: Sequence[float] = DEFAULT_RGB_MEAN,
        rgb_std: Sequence[float] = DEFAULT_RGB_STD,
        ignore_rgb_black: bool = True,
        ignore_index: int = -100,
        return_concat: bool = False,
        positive_aware_sampling: bool = True,
        positive_sample_prob: float = 0.5,
        crop_range: Tuple[int, int] = (448, 480),
        require_labels: Optional[bool] = None,
        modality_dir_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.split = str(split).lower()
        self.target_h, self.target_w = img_size
        self.is_training = (self.split == "train") if is_training is None else bool(is_training)

        # Support typo aliases from user prompts (apply_blaclist, blaclist_path)
        if "apply_blaclist" in kwargs:
            apply_blacklist = kwargs["apply_blaclist"]
        if "blaclist_path" in kwargs and blacklist_path is None:
            blacklist_path = kwargs["blaclist_path"]

        self.apply_blacklist = bool(apply_blacklist)
        self.ignore_rgb_black = bool(ignore_rgb_black)
        self.ignore_index = int(ignore_index)
        self.return_concat = bool(return_concat)
        self.positive_aware_sampling = bool(positive_aware_sampling)
        self.positive_sample_prob = float(positive_sample_prob)
        self.crop_min, self.crop_max = crop_range

        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)

        # Parse requested channels
        if isinstance(channels, str):
            self.channels = [c.strip().lower() for c in channels.split(",") if c.strip()]
        else:
            self.channels = [str(c).strip().lower() for c in channels]

        # Categorize channels into optical RGB vs terrain/auxiliary
        self.has_rgb = "rgb" in self.channels or "image" in self.channels
        self.rgb_channels_count = 3 if self.has_rgb else 0
        self.terrain_channels = [c for c in self.channels if c not in ("rgb", "image")]
        self.num_terrain_channels = len(self.terrain_channels)
        self.num_total_channels = self.rgb_channels_count + self.num_terrain_channels

        self.require_labels = (
            require_labels if require_labels is not None else (self.split in {"train", "val", "test"})
        )
        self._positive_cache: Dict[int, bool] = {}

        # Modality directory resolution map
        self.mod_dir_map = dict(DEFAULT_MODALITY_DIRS)
        if modality_dir_map:
            self.mod_dir_map.update(modality_dir_map)

        # Resolve split directory
        split_candidates = [
            self.data_dir / self.split,
            self.data_dir / self.split.upper(),
            self.data_dir,
        ]
        self.split_dir = None
        for cand in split_candidates:
            if cand.is_dir() and (cand / "IMAGE").exists() or (cand / "DTM_NORM").exists() or (cand / "DTM").exists():
                self.split_dir = cand
                break

        if self.split_dir is None:
            # Fallback: check if subdirectories directly exist under data_dir
            if (self.data_dir / "IMAGE").exists():
                self.split_dir = self.data_dir
            else:
                raise FileNotFoundError(
                    f"[LandslideDatasetV2] Could not resolve valid split directory for '{self.split}' under {self.data_dir}"
                )

        # Resolve blacklist file
        self.blacklist_path = self._resolve_blacklist_path(blacklist_path)
        self.blacklist = load_blacklist(self.blacklist_path) if self.apply_blacklist else set()

        # Locate modality folders
        self.channel_dirs = self._locate_modality_dirs()
        self.label_dir = self._locate_label_dir()

        # Index matched samples
        self.samples = self._index_dataset_samples()

        print(
            f"[LandslideDatasetV2] Split: {self.split.upper()} | Samples: {len(self.samples)} | "
            f"Channels: {self.channels} (Total={self.num_total_channels}) | "
            f"Blacklist Active: {self.apply_blacklist} ({len(self.blacklist)} stems registered) | "
            f"Dir: {self.split_dir}"
        )

    def _resolve_blacklist_path(self, user_path: Optional[Union[str, Path]]) -> Optional[Path]:
        """Resolves blacklist file path from user argument or default locations."""
        if user_path:
            p = Path(user_path)
            if p.is_file():
                return p.resolve()
            # Try relative to data_dir
            rel = self.data_dir / user_path
            if rel.is_file():
                return rel.resolve()

        # Check standard default candidate locations
        candidates = [
            self.split_dir / "black_list.txt",
            self.data_dir / "black_list.txt",
            Path("dataset/dataset_1V2/black_list.txt").resolve(),
            Path("dataset/black_list.txt").resolve(),
            Path("black_list.txt").resolve(),
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()

        return None

    def _locate_modality_dirs(self) -> Dict[str, Path]:
        """Finds disk directory for each requested input channel."""
        channel_dirs = {}
        for chan in self.channels:
            if chan in ("rgb", "image"):
                folder_names = [self.mod_dir_map.get("rgb", "IMAGE"), "IMAGE", "images", "image"]
            elif chan in ("dtm", "dtm_norm"):
                folder_names = [self.mod_dir_map.get("dtm", "DTM_NORM"), "DTM_NORM", "dtm_norm", "DTM", "dtm"]
            elif chan == "slope":
                folder_names = [self.mod_dir_map.get("slope", "SLOPE"), "SLOPE", "slope"]
            elif chan == "aspect":
                folder_names = [self.mod_dir_map.get("aspect", "ASPECT"), "ASPECT", "aspect"]
            else:
                folder_names = [self.mod_dir_map.get(chan, chan.upper()), chan.upper(), chan]

            found_dir = None
            for fname in folder_names:
                p = self.split_dir / fname
                if p.is_dir():
                    found_dir = p
                    break

            if found_dir is None:
                raise FileNotFoundError(
                    f"[LandslideDatasetV2] Folder for channel '{chan}' not found in {self.split_dir}. "
                    f"Searched: {folder_names}"
                )
            channel_dirs[chan] = found_dir
        return channel_dirs

    def _locate_label_dir(self) -> Optional[Path]:
        """Locates ground truth label directory."""
        label_candidates = [self.mod_dir_map.get("label", "LABEL"), "LABEL", "label", "masks", "mask"]
        for lname in label_candidates:
            p = self.split_dir / lname
            if p.is_dir():
                return p
        if self.require_labels:
            raise FileNotFoundError(
                f"[LandslideDatasetV2] Required LABEL folder not found in {self.split_dir} for split '{self.split}'."
            )
        return None

    def _index_dataset_samples(self) -> List[Dict[str, Union[str, Path, Dict[str, Path]]]]:
        """Indexes samples matching all requested modalities and labels, skipping blacklisted tiles."""
        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        # Collect stems per channel
        channel_files: Dict[str, Dict[str, Path]] = {}
        for chan, cdir in self.channel_dirs.items():
            channel_files[chan] = {
                f.stem: f for f in cdir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            }

        # Collect stems for label
        label_files: Dict[str, Path] = {}
        if self.label_dir and self.label_dir.is_dir():
            label_files = {
                f.stem: f for f in self.label_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            }

        # Find common stems across all required input channels
        common_stems = set.intersection(*[set(files.keys()) for files in channel_files.values()])

        if self.require_labels:
            common_stems = common_stems & set(label_files.keys())

        if not common_stems:
            raise RuntimeError(
                f"[LandslideDatasetV2] No matching samples found across channels {self.channels} in {self.split_dir}!"
            )

        sorted_stems = sorted(list(common_stems))
        samples = []
        skipped_blacklist_count = 0

        for stem in sorted_stems:
            if self.apply_blacklist and stem in self.blacklist:
                skipped_blacklist_count += 1
                continue

            sample_dict = {
                "stem": stem,
                "paths": {chan: channel_files[chan][stem] for chan in self.channels},
                "lbl_path": label_files.get(stem, None),
            }
            # Backwards-compatible convenience keys for scripts checking sample.get('dtm_path')
            if "dtm" in sample_dict["paths"]:
                sample_dict["dtm_path"] = sample_dict["paths"]["dtm"]
            elif "dtm_norm" in sample_dict["paths"]:
                sample_dict["dtm_path"] = sample_dict["paths"]["dtm_norm"]
            if "rgb" in sample_dict["paths"]:
                sample_dict["img_path"] = sample_dict["paths"]["rgb"]

            samples.append(sample_dict)

        if skipped_blacklist_count > 0:
            print(
                f"[LandslideDatasetV2] Blacklist filter active: skipped {skipped_blacklist_count} "
                f"noisy tiles ({len(samples)} valid samples retained)."
            )

        return samples

    # =========================================================================
    # I/O Readers
    # =========================================================================
    def _read_rgb(self, path: Path) -> np.ndarray:
        """Reads RGB image as 3-channel uint8 array in RGB color order."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            pil_img = Image.open(str(path)).convert("RGB")
            img = np.array(pil_img, dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img  # [H, W, 3], uint8

    def _read_float_raster(self, path: Path) -> np.ndarray:
        """Reads continuous terrain/topographic raster (e.g. DTM_NORM, SLOPE) as float32 [H, W]."""
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            arr = np.array(Image.open(str(path)), dtype=np.float32)
        else:
            arr = arr.astype(np.float32)

        if arr.ndim == 3:
            arr = arr[:, :, 0]

        # Clean any remaining NaNs / Infs with fast zero-replacement
        if np.isnan(arr).any() or np.isinf(arr).any():
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

        return arr  # [H, W], float32

    def _read_mask(self, path: Optional[Path], default_shape: Tuple[int, int]) -> np.ndarray:
        """Reads binary label mask (0: BG, 1: Landslide) as int64 [H, W]."""
        if path is None or not Path(path).exists():
            if self.require_labels:
                raise FileNotFoundError(f"Label mask not found at {path} while require_labels=True.")
            return np.zeros(default_shape, dtype=np.int64)

        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.array(Image.open(str(path)))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Standardize binary mask: 0=Background, >0=Landslide (handles 1, 255, 65535)
        binary_mask = (mask > 0).astype(np.int64)
        return binary_mask  # [H, W], int64

    # =========================================================================
    # Synchronous Spatial Augmentations (RGB + DTM + Slope + Label)
    # =========================================================================
    def _apply_spatial_augmentations(
        self,
        rgb: Optional[np.ndarray],
        terrains: List[np.ndarray],
        mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Applies coordinated spatial transformations simultaneously across RGB,
        all terrain channels, and ground truth mask to preserve exact pixel alignment.
        """
        h, w = mask.shape[:2]

        # ---------------------------------------------------------------------
        # 2.1 Random Rotate 90°, 180°, 270° (p=0.75)
        # Resolves topographic aspect blind spot (W/NW Void)
        # ---------------------------------------------------------------------
        if random.random() < 0.75:
            rot_k = random.choice([1, 2, 3])
            if rgb is not None:
                rgb = np.rot90(rgb, rot_k).copy()
            terrains = [np.rot90(t, rot_k).copy() for t in terrains]
            mask = np.rot90(mask, rot_k).copy()

        # ---------------------------------------------------------------------
        # 2.2 Random Horizontal & Vertical Flip (p=0.5 each)
        # Breaks linear flight-path bias
        # ---------------------------------------------------------------------
        if random.random() < 0.5:
            if rgb is not None:
                rgb = np.fliplr(rgb).copy()
            terrains = [np.fliplr(t).copy() for t in terrains]
            mask = np.fliplr(mask).copy()

        if random.random() < 0.5:
            if rgb is not None:
                rgb = np.flipud(rgb).copy()
            terrains = [np.flipud(t).copy() for t in terrains]
            mask = np.flipud(mask).copy()

        # ---------------------------------------------------------------------
        # 2.3 Random Crop & Resize (p=0.5)
        # Random Crop (448 to 480) -> Resize to target resolution (512, 512)
        # Disrupts the 128-px tile-overlap grid artifact
        # ---------------------------------------------------------------------
        if random.random() < 0.5:
            cur_h, cur_w = mask.shape[:2]
            # Ensure crop size is within bounds
            max_avail = min(cur_h, cur_w)
            c_min = min(self.crop_min, max_avail)
            c_max = min(self.crop_max, max_avail)
            if c_max > c_min:
                crop_h = random.randint(c_min, c_max)
                crop_w = random.randint(c_min, c_max)
            else:
                crop_h = crop_w = c_min

            has_pos = (mask == 1).any()
            if self.positive_aware_sampling and has_pos and (random.random() < self.positive_sample_prob):
                # Positive-aware crop: center around a known landslide pixel
                pos_ys, pos_xs = np.where(mask == 1)
                p_idx = random.randint(0, len(pos_ys) - 1)
                py, px = pos_ys[p_idx], pos_xs[p_idx]
                min_top = max(0, py - crop_h + 1)
                max_top = min(cur_h - crop_h, py)
                top = random.randint(min_top, max_top) if max_top >= min_top else 0
                min_left = max(0, px - crop_w + 1)
                max_left = min(cur_w - crop_w, px)
                left = random.randint(min_left, max_left) if max_left >= min_left else 0
            else:
                top = random.randint(0, max(0, cur_h - crop_h)) if cur_h > crop_h else 0
                left = random.randint(0, max(0, cur_w - crop_w)) if cur_w > crop_w else 0

            # Crop all
            if rgb is not None:
                rgb = rgb[top:top + crop_h, left:left + crop_w]
            terrains = [t[top:top + crop_h, left:left + crop_w] for t in terrains]
            mask = mask[top:top + crop_h, left:left + crop_w]

            # Resize back to target size (Bilinear for continuous rasters, Nearest for discrete mask)
            if rgb is not None:
                rgb = cv2.resize(rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            terrains = [
                cv2.resize(t, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
                for t in terrains
            ]
            mask = cv2.resize(
                mask.astype(np.float32),
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

        # ---------------------------------------------------------------------
        # 2.4 Elastic Distortion (p=0.2, alpha=1.0, sigma=50.0, alpha_affine=10.0)
        # Simulates non-rigid slope failure morphology
        # ---------------------------------------------------------------------
        if random.random() < 0.2:
            rgb, terrains, mask = self._apply_elastic_transform(
                rgb, terrains, mask, alpha=1.0, sigma=50.0, alpha_affine=10.0
            )

        return rgb, terrains, mask

    def _apply_elastic_transform(
        self,
        rgb: Optional[np.ndarray],
        terrains: List[np.ndarray],
        mask: np.ndarray,
        alpha: float = 1.0,
        sigma: float = 50.0,
        alpha_affine: float = 10.0,
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Fast coordinated elastic deformation with random affine displacement
        smoothed by Gaussian filtering.
        """
        h, w = mask.shape[:2]

        # 1. Random affine transformation matrix
        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        affine_mat = cv2.getAffineTransform(pts1, pts2)

        # 2. Base coordinate grid
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        affine_x = affine_mat[0, 0] * grid_x + affine_mat[0, 1] * grid_y + affine_mat[0, 2]
        affine_y = affine_mat[1, 0] * grid_x + affine_mat[1, 1] * grid_y + affine_mat[1, 2]

        # 3. Elastic displacement field with Gaussian blur (sigma=50)
        noise_x = (np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0)
        noise_y = (np.random.rand(h, w).astype(np.float32) * 2.0 - 1.0)
        dx = cv2.GaussianBlur(noise_x, (0, 0), sigmaX=sigma, sigmaY=sigma) * (alpha * 50.0)
        dy = cv2.GaussianBlur(noise_y, (0, 0), sigmaX=sigma, sigmaY=sigma) * (alpha * 50.0)

        map_x = (affine_x + dx).astype(np.float32)
        map_y = (affine_y + dy).astype(np.float32)

        # Remap continuous data with INTER_LINEAR & border reflect
        if rgb is not None:
            rgb = cv2.remap(
                rgb, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

        terrains_out = [
            cv2.remap(
                t, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            for t in terrains
        ]

        # Remap discrete mask with INTER_NEAREST & border reflect
        mask_out = cv2.remap(
            mask.astype(np.float32), map_x, map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101
        ).astype(np.int64)

        return rgb, terrains_out, mask_out

    # =========================================================================
    # Photometric Augmentations (EXCLUSIVE to RGB Channel)
    # =========================================================================
    def _apply_photometric_augmentations(self, rgb: np.ndarray) -> np.ndarray:
        """
        Applies color and texture augmentations EXCLUSIVELY to optical RGB.
        Never applied to physical DTM / Slope rasters.
        """
        # ---------------------------------------------------------------------
        # 2.5 Color Jitter / Random Brightness & Contrast (p=0.5)
        # Brightness +-0.2, Contrast +-0.3, Saturation +-0.2
        # Resolves optical luminance shift across validation corridors
        # ---------------------------------------------------------------------
        if random.random() < 0.5:
            # Contrast [0.7, 1.3], Brightness [-51, 51]
            contrast_factor = random.uniform(0.7, 1.3)
            brightness_offset = random.uniform(-51.0, 51.0)
            rgb = np.clip(
                contrast_factor * rgb.astype(np.float32) + brightness_offset, 0, 255
            ).astype(np.uint8)

            # Saturation scaling [0.8, 1.2] in HSV space
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            sat_factor = random.uniform(0.8, 1.2)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # ---------------------------------------------------------------------
        # 2.6 CLAHE (p=0.3, clip=2.0, tile=(8, 8))
        # Enhances ground texture and fractures in shadowed mountain flanks
        # ---------------------------------------------------------------------
        if random.random() < 0.3:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # ---------------------------------------------------------------------
        # 2.7 Gaussian Blur / Motion Blur (p=0.2, kernel 3x3 or 5x5)
        # Simulates drone platform motion vibration blur
        # ---------------------------------------------------------------------
        if random.random() < 0.2:
            k = random.choice([3, 5])
            if random.random() < 0.5:
                # Gaussian blur
                rgb = cv2.GaussianBlur(rgb, (k, k), 0)
            else:
                # Directional motion blur
                kernel = np.zeros((k, k), dtype=np.float32)
                if random.random() < 0.5:
                    kernel[int((k - 1) / 2), :] = np.ones(k, dtype=np.float32)
                else:
                    kernel[:, int((k - 1) / 2)] = np.ones(k, dtype=np.float32)
                kernel /= k
                rgb = cv2.filter2D(rgb, -1, kernel)

        return rgb

    # =========================================================================
    # Resizing & Spatial Standardization
    # =========================================================================
    def _resize_if_needed(
        self,
        rgb: Optional[np.ndarray],
        terrains: List[np.ndarray],
        mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], List[np.ndarray], np.ndarray]:
        """Resizes all modalities to (self.target_h, self.target_w) if dimensions differ."""
        h, w = mask.shape[:2]
        if (h, w) != (self.target_h, self.target_w):
            if rgb is not None:
                rgb = cv2.resize(rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            terrains = [
                cv2.resize(t, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
                for t in terrains
            ]
            mask = cv2.resize(
                mask.astype(np.float32),
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)
        return rgb, terrains, mask

    # =========================================================================
    # Dataset Accessors & Sampler Helper
    # =========================================================================
    def sample_has_positive(self, index: int) -> bool:
        """
        Determines whether the sample at index contains landslide foreground pixels.
        Results are cached to accelerate DataLoader WeightedRandomSampler initialization.
        """
        if index in self._positive_cache:
            return self._positive_cache[index]

        sample_meta = self.samples[index]
        lbl_path = sample_meta.get("lbl_path")
        if lbl_path is None or not Path(lbl_path).exists():
            self._positive_cache[index] = False
            return False

        mask = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            mask = np.array(Image.open(str(lbl_path)))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        has_pos = bool(np.any(mask > 0))
        self._positive_cache[index] = has_pos
        return has_pos

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        Tuple[torch.Tensor, torch.Tensor, str],
    ]:
        """
        Loads, transforms, and normalizes a multimodal sample.

        Returns:
            If return_concat == False (default for dual-stream FusionModel):
                rgb_tensor: float32 Tensor [3, H, W]
                terrain_tensor: float32 Tensor [K, H, W] (e.g. K=2 for DTM + Slope)
                mask_tensor: int64 Tensor [H, W] (0=Background, 1=Landslide, -100=Ignore)
                sample_id: str (Stem identifier)
            If return_concat == True:
                concat_tensor: float32 Tensor [C, H, W] (e.g. C=5 for RGB + DTM + Slope)
                mask_tensor: int64 Tensor [H, W]
                sample_id: str
        """
        sample_meta = self.samples[index]
        sample_id = sample_meta["stem"]
        paths = sample_meta["paths"]

        # 1. Read Raw Modalities
        rgb_raw = None
        if self.has_rgb:
            rgb_raw = self._read_rgb(paths["rgb"])

        terrains_raw = []
        for chan in self.terrain_channels:
            terrains_raw.append(self._read_float_raster(paths[chan]))

        ref_shape = rgb_raw.shape[:2] if rgb_raw is not None else terrains_raw[0].shape[:2]
        mask_raw = self._read_mask(sample_meta.get("lbl_path"), default_shape=ref_shape)

        # 2. Resize to target dimensions if needed
        rgb_cur, terrains_cur, mask_cur = self._resize_if_needed(rgb_raw, terrains_raw, mask_raw)

        # 3. Synchronous Spatial Augmentations (Training Only)
        if self.is_training:
            rgb_cur, terrains_cur, mask_cur = self._apply_spatial_augmentations(
                rgb_cur, terrains_cur, mask_cur
            )

        # 4. Ignore Mask for RGB(0,0,0) void/border pixels (detected BEFORE photometric jitter)
        if self.ignore_rgb_black and rgb_cur is not None:
            # Pixels where all 3 color channels are exactly 0
            is_black_void = (
                (rgb_cur[:, :, 0] == 0) &
                (rgb_cur[:, :, 1] == 0) &
                (rgb_cur[:, :, 2] == 0)
            )
            if np.any(is_black_void):
                mask_cur[is_black_void] = self.ignore_index

        # 5. Photometric Augmentations (Training Only, EXCLUSIVELY on RGB)
        if self.is_training and rgb_cur is not None:
            rgb_cur = self._apply_photometric_augmentations(rgb_cur)

        # 6. Normalize RGB -> float32 [3, H, W]
        rgb_tensor = None
        if rgb_cur is not None:
            rgb_norm = (rgb_cur.astype(np.float32) / 255.0 - self.rgb_mean) / self.rgb_std
            rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).contiguous().float()

        # 7. Stack Terrain channels -> float32 [K, H, W] (already normalized in [0.0, 1.0])
        terrain_tensor = None
        if terrains_cur:
            terrains_stacked = np.stack(terrains_cur, axis=0).astype(np.float32)
            terrain_tensor = torch.from_numpy(terrains_stacked).contiguous().float()

        # 8. Ground Truth Mask -> int64 [H, W]
        mask_tensor = torch.from_numpy(mask_cur).contiguous().long()

        # 9. Return formatting based on return_concat
        if self.return_concat:
            tensors_to_cat = []
            if rgb_tensor is not None:
                tensors_to_cat.append(rgb_tensor)
            if terrain_tensor is not None:
                tensors_to_cat.append(terrain_tensor)
            concat_tensor = torch.cat(tensors_to_cat, dim=0).contiguous()
            return concat_tensor, mask_tensor, sample_id

        # Standard dual-stream format for FAF FusionModel: (rgb, terrain, mask, sample_id)
        return rgb_tensor, terrain_tensor, mask_tensor, sample_id


def build_landslide_v2_dataloader(
    data_dir: Union[str, Path] = "./dataset/dataset_1V2",
    split: str = "train",
    batch_size: int = 16,
    channels: Sequence[str] = ("rgb", "dtm", "slope"),
    apply_blacklist: bool = True,
    blacklist_path: Optional[Union[str, Path]] = None,
    img_size: Tuple[int, int] = (512, 512),
    is_training: Optional[bool] = None,
    num_workers: int = 4,
    use_sampler: bool = False,
    ignore_rgb_black: bool = True,
    return_concat: bool = False,
    shuffle: Optional[bool] = None,
) -> torch.utils.data.DataLoader:
    """
    Convenience factory function to construct LandslideDatasetV2 and PyTorch DataLoader.
    """
    dataset = LandslideDatasetV2(
        data_dir=data_dir,
        split=split,
        img_size=img_size,
        channels=channels,
        apply_blacklist=apply_blacklist,
        blacklist_path=blacklist_path,
        is_training=is_training,
        ignore_rgb_black=ignore_rgb_black,
        return_concat=return_concat,
    )

    sampler = None
    is_train = dataset.is_training
    do_shuffle = (is_train if shuffle is None else shuffle)

    if is_train and use_sampler:
        sample_weights = [5.0 if dataset.sample_has_positive(i) else 1.0 for i in range(len(dataset))]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True
        )
        do_shuffle = False

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=do_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )
    return loader
