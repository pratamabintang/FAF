"""
================================================================================
MFNet Dataset Loader (RGB + Thermal / Infrared)
================================================================================
Modernized native PyTorch implementation of the MFNet dataset loader (formerly in legacy/).
Provides full backwards compatibility as `FusionModelDataset` and `MFNetDataset`.
Eliminates any external dependency on 'albumentations'.
================================================================================
"""

from MFNetDataset import (
    MFNetDataset,
    MFNET_CLASSES,
    RGB_MEAN,
    RGB_STD,
    IR_MEAN,
    IR_STD,
    MFNET_PALETTE,
    get_mfnet_palette,
    build_mfnet_dataloader,
)

# Backwards compatibility alias
FusionModelDataset = MFNetDataset

__all__ = [
    "FusionModelDataset",
    "MFNetDataset",
    "MFNET_CLASSES",
    "RGB_MEAN",
    "RGB_STD",
    "IR_MEAN",
    "IR_STD",
    "MFNET_PALETTE",
    "get_mfnet_palette",
    "build_mfnet_dataloader",
]