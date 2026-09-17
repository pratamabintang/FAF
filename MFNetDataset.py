"""
================================================================================
MFNet Multimodal Dataset Loader (RGB + Thermal / Infrared)
================================================================================
Dedicated PyTorch Dataset for Multimodal Urban Scene Semantic Segmentation (RGB-T).
Tailored specifically for dataset/dataset_2 (MFNet RGB-Thermal Benchmark).

Key Design & Engineering Principles:
  1. Pure PyTorch / OpenCV / PIL / NumPy implementation (No heavy external dependencies like albumentations).
  2. Native handling of 4-channel imagery:
     - Channels 0..2: Optical RGB (ImageNet normalized, float32 [3, H, W])
     - Channel 3: Thermal / Infrared (Empirical MFNet normalized, float32 [1, H, W])
  3. Ground-truth semantic segmentation masks (int64 [H, W], 9 classes: 0..8).
  4. On-the-fly virtual flip handling:
     - Automatically resolves '_flip' stems in train.txt by reading the base sample
       and applying horizontal flip, eliminating the need for offline file duplication.
  5. Coordinated spatial augmentations (synchronous horizontal flip, scale-crop, shift)
     preserving spatial alignment across RGB, Thermal, and Mask.
  6. Independent sensor augmentations:
     - Photometric adjustments (brightness, contrast, HSV jitter, blur) for RGB.
     - Thermal sensor noise and contrast scaling for Thermal.
  7. Seamless integration with FusionModelTrain.py, FusionModelRunDemo.py, and test suites.
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
from torch.utils.data import Dataset, DataLoader

# Prevent thread contention across DataLoader workers
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ==============================================================================
# MFNet Dataset Specification & Statistics
# ==============================================================================
MFNET_CLASSES = [
    "unlabeled",    # 0: Unlabeled / Background
    "car",          # 1: Car
    "person",       # 2: Person / Pedestrian
    "bike",         # 3: Bike / Cyclist
    "curve",        # 4: Curve
    "car_stop",     # 5: Car Stop
    "guardrail",    # 6: Guardrail
    "color_cone",   # 7: Color Cone
    "bump"          # 8: Speed Bump
]

# Normalization constants
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IR_MEAN  = np.array([0.3616], dtype=np.float32)
IR_STD   = np.array([0.0765], dtype=np.float32)

MFNET_PALETTE = np.array([
    [0, 0, 0],         # 0: Unlabeled (Black)
    [64, 0, 128],      # 1: Car
    [64, 64, 0],       # 2: Person
    [0, 128, 192],     # 3: Bike
    [0, 0, 192],       # 4: Curve
    [128, 128, 0],     # 5: Car Stop
    [64, 64, 128],     # 6: Guardrail
    [192, 128, 128],   # 7: Color Cone
    [192, 64, 0],      # 8: Bump
], dtype=np.uint8)


def get_mfnet_palette() -> np.ndarray:
    """Returns the standard 9-class RGB palette for MFNet."""
    return MFNET_PALETTE.copy()


class MFNetDataset(Dataset):
    """
    Multimodal MFNet RGB-Thermal Segmentation Dataset.

    Args:
        data_dir: Path to dataset root (e.g. 'dataset/dataset_2').
        split: Split name ('train', 'train_', 'val', 'test', 'test_day', 'test_night').
        have_label: Whether ground truth masks are expected/loaded.
        rgb_size: Target (Height, Width) for RGB image. Default (480, 640).
        ir_size: Target (Height, Width) for Thermal image. Default (480, 640).
        img_size: Optional unified (Height, Width) overriding rgb_size & ir_size.
        is_training: If True, applies data augmentations. Defaults to True for train splits.
        ignore_index: Sentinel label value to ignore in loss (-100 or 255).
        use_augmentation: Explicit toggle for data augmentations.
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        have_label: bool = True,
        rgb_size: Tuple[int, int] = (480, 640),
        ir_size: Tuple[int, int] = (480, 640),
        img_size: Optional[Tuple[int, int]] = None,
        is_training: Optional[bool] = None,
        ignore_index: int = -100,
        use_augmentation: Optional[bool] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.split = str(split).strip()
        self.have_label = have_label
        self.ignore_index = int(ignore_index)

        if img_size is not None:
            self.target_h, self.target_w = img_size
        else:
            self.target_h, self.target_w = rgb_size

        # Determine training mode
        is_train_split = self.split.startswith("train")
        if use_augmentation is not None:
            self.is_training = use_augmentation
        elif is_training is not None:
            self.is_training = is_training
        else:
            self.is_training = is_train_split

        self.img_dir = self.data_dir / "images"
        self.lbl_dir = self.data_dir / "labels"

        # Locate split text file
        split_file = self._resolve_split_file(self.split)
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        # Filter out blacklisted samples if blacklist exists
        blacklist_file = self.data_dir / "black_list.txt"
        blacklist = set()
        if blacklist_file.exists():
            with open(blacklist_file, "r", encoding="utf-8") as f:
                blacklist = {line.strip() for line in f if line.strip()}

        self.names: List[str] = [
            name for name in raw_lines
            if name not in blacklist and name.replace("_flip", "") not in blacklist
        ]

        if len(self.names) == 0:
            raise RuntimeError(f"No valid samples found in {split_file} after filtering!")

        n_filtered = len(raw_lines) - len(self.names)
        filter_msg = f" (filtered {n_filtered} via blacklist)" if n_filtered > 0 else ""
        print(f"[MFNetDataset] Split: {self.split} | Samples: {len(self.names)}{filter_msg} | Resolution: {self.target_h}x{self.target_w} | Augmentations: {self.is_training}")

    def _resolve_split_file(self, split: str) -> Path:
        """Finds matching split file name."""
        candidates = [
            self.data_dir / f"{split}.txt",
            self.data_dir / f"{split}",
        ]
        # Common aliases
        if split == "train_":
            candidates.insert(0, self.data_dir / "train_.txt")
            candidates.append(self.data_dir / "train.txt")
        elif split == "train":
            candidates.append(self.data_dir / "train_.txt")

        for cand in candidates:
            if cand.exists() and cand.is_file():
                return cand
        return self.data_dir / f"{split}.txt"

    def _load_image_and_mask(self, name: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Loads 4-channel image (RGB + Thermal) and corresponding ground-truth mask.
        Automatically handles virtual '_flip' samples on the fly.
        """
        is_flip = ("_flip" in name)
        base_name = name.replace("_flip", "")

        # 1. Resolve image path
        img_candidates = [
            self.img_dir / f"{name}.png",
            self.img_dir / f"{base_name}.png",
        ]
        img_path = None
        needs_flip = False

        for cand in img_candidates:
            if cand.exists():
                img_path = cand
                needs_flip = (is_flip and cand.stem == base_name)
                break

        if img_path is None:
            raise FileNotFoundError(f"Image not found for sample '{name}' in {self.img_dir}")

        # Read 4-channel image
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        is_bgr = True
        if img is None:
            # Fallback to PIL
            pil_img = Image.open(str(img_path))
            img = np.asarray(pil_img)
            is_bgr = False

        if img.ndim != 3 or img.shape[2] != 4:
            raise ValueError(
                f"{img_path}: expected MFNet image HxWx4, got {img.shape}"
            )

        if is_bgr:
            rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            thermal = img[:, :, 3]
        else:
            rgb = img[:, :, :3]
            thermal = img[:, :, 3]

        # 2. Resolve label mask path
        mask = None
        if self.have_label:
            lbl_candidates = [
                self.lbl_dir / f"{name}.png",
                self.lbl_dir / f"{base_name}.png",
            ]
            lbl_path = None
            for cand in lbl_candidates:
                if cand.exists():
                    lbl_path = cand
                    break

            if self.have_label and lbl_path is None:
                raise FileNotFoundError(
                    f"Missing MFNet label for sample: {name}"
                )

            lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
            if lbl is None:
                lbl = np.asarray(Image.open(str(lbl_path)))
            if lbl is None:
                raise IOError(f"Could not read label file: {lbl_path}")
            if lbl.ndim == 3:
                lbl = lbl[:, :, 0]
            mask = lbl.astype(np.int64)

        # Apply virtual horizontal flip if this is a _flip sample loaded from base
        if needs_flip:
            rgb = cv2.flip(rgb, 1)
            thermal = cv2.flip(thermal, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        return rgb, thermal, mask

    def _apply_spatial_augmentations(
        self,
        rgb: np.ndarray,
        thermal: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Synchronous spatial augmentations applied identically across RGB, Thermal, and Mask.
        """
        # 1. Random Horizontal Flip (in addition to offline virtual flips)
        if random.random() > 0.5:
            rgb = cv2.flip(rgb, 1)
            thermal = cv2.flip(thermal, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        # 2. Random Scale-Crop (85% to 100% of spatial dimensions)
        if random.random() < 0.5:
            h, w = thermal.shape[:2]
            scale = random.uniform(0.85, 1.0)
            crop_h, crop_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - crop_h) if h > crop_h else 0
            left = random.randint(0, w - crop_w) if w > crop_w else 0

            rgb = rgb[top:top+crop_h, left:left+crop_w]
            thermal = thermal[top:top+crop_h, left:left+crop_w]
            if mask is not None:
                mask = mask[top:top+crop_h, left:left+crop_w]

        # 3. Small Random Affine Shift (Translation within +/- 5%)
        if random.random() < 0.3:
            h, w = thermal.shape[:2]
            tx = random.uniform(-0.05, 0.05) * w
            ty = random.uniform(-0.05, 0.05) * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            rgb = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            thermal = cv2.warpAffine(thermal, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if mask is not None:
                mask = cv2.warpAffine(mask.astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_index).astype(np.int64)

        return rgb, thermal, mask

    def _apply_photometric_augmentations(self, rgb: np.ndarray) -> np.ndarray:
        """Photometric color augmentations applied ONLY to RGB optical imagery."""
        # Brightness & Contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-25, 25)
            rgb = np.clip(alpha * rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Color Jitter (HSV)
        if random.random() > 0.5:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Gaussian Blur
        if random.random() > 0.7:
            k = random.choice([3, 5])
            rgb = cv2.GaussianBlur(rgb, (k, k), 0)

        return rgb

    def _apply_thermal_augmentations(self, thermal: np.ndarray) -> np.ndarray:
        """Sensor-level augmentations applied ONLY to Thermal/IR imagery."""
        # Thermal Gain and Offset (sensor drift simulation)
        if random.random() > 0.5:
            t_alpha = random.uniform(0.85, 1.15)
            t_beta = random.uniform(-15, 15)
            thermal = np.clip(t_alpha * thermal.astype(np.float32) + t_beta, 0, 255).astype(np.uint8)

        # Thermal Sensor Noise
        if random.random() > 0.7:
            noise = np.random.normal(0, 4, thermal.shape)
            thermal = np.clip(thermal.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return thermal

    def _resize(
        self,
        rgb: np.ndarray,
        thermal: np.ndarray,
        mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Resizes all modalities to (self.target_h, self.target_w)."""
        h, w = thermal.shape[:2]
        if (h, w) != (self.target_h, self.target_w):
            rgb = cv2.resize(rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            thermal = cv2.resize(thermal, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            if mask is not None:
                mask = cv2.resize(
                    mask.astype(np.float32),
                    (self.target_w, self.target_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int64)

        return rgb, thermal, mask

    def is_night(self, index: int) -> bool:
        """Returns True if the sample is a night-time capture (stem contains 'N')."""
        name = self.names[index]
        return "N" in name

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        name = self.names[index]

        # 1. Load modalities
        rgb, thermal, mask = self._load_image_and_mask(name)

        # 2. Training augmentations
        if self.is_training:
            rgb, thermal, mask = self._apply_spatial_augmentations(rgb, thermal, mask)
            rgb = self._apply_photometric_augmentations(rgb)
            thermal = self._apply_thermal_augmentations(thermal)

        # 3. Resize to target resolution
        rgb, thermal, mask = self._resize(rgb, thermal, mask)

        # 4. Normalize RGB -> float32 [3, H, W]
        rgb_float = (rgb.astype(np.float32) / 255.0 - RGB_MEAN) / RGB_STD
        rgb_tensor = torch.from_numpy(rgb_float.transpose(2, 0, 1)).contiguous().float()

        # 5. Normalize Thermal -> float32 [1, H, W]
        thermal_float = (thermal.astype(np.float32) / 255.0 - IR_MEAN) / IR_STD
        thermal_tensor = torch.from_numpy(thermal_float[np.newaxis, :, :]).contiguous().float()

        # 6. Return sample
        if mask is not None:
            mask[mask == 255] = self.ignore_index
            mask_tensor = torch.from_numpy(mask).contiguous().long()
            return rgb_tensor, thermal_tensor, mask_tensor, name
        else:
            return rgb_tensor, thermal_tensor, name


def build_mfnet_dataloader(
    data_dir: Union[str, Path] = "./dataset/dataset_2",
    split: str = "train",
    batch_size: int = 4,
    img_size: Tuple[int, int] = (480, 640),
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
    have_label: bool = True,
    drop_last: bool = False,
) -> Tuple[MFNetDataset, DataLoader]:
    """
    Convenience factory function to build an MFNet DataLoader.

    Returns:
        (dataset, dataloader)
    """
    is_train = split.startswith("train")
    if shuffle is None:
        shuffle = is_train

    dataset = MFNetDataset(
        data_dir=data_dir,
        split=split,
        have_label=have_label,
        img_size=img_size,
        is_training=is_train
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last if is_train else False
    )

    return dataset, loader


if __name__ == "__main__":
    print("Testing MFNetDataset instantiation on dataset/dataset_2...")
    ds_path = "./dataset/dataset_2"
    if os.path.exists(ds_path):
        train_ds, train_ld = build_mfnet_dataloader(ds_path, split="train", batch_size=2, num_workers=0)
        print(f"Loaded {len(train_ds)} train samples.")
        sample_rgb, sample_ir, sample_mask, sample_id = train_ds[0]
        print(f"Sample '{sample_id}': RGB={sample_rgb.shape}, IR={sample_ir.shape}, Mask={sample_mask.shape}")
        batch_rgb, batch_ir, batch_masks, batch_ids = next(iter(train_ld))
        print(f"Batch: RGB={batch_rgb.shape}, IR={batch_ir.shape}, Masks={batch_masks.shape}, IDs={batch_ids}")
        print("MFNetDataset verified successfully!")
    else:
        print(f"Path {ds_path} not found.")
