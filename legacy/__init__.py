"""
================================================================================
Legacy Multimodal Semantic Segmentation Datasets and Utilities
================================================================================
Contains baseline RGB-Thermal datasets (MFNet, PST900) and TENT test-time adaptation.
Isolated from the primary RGB-DTM Landslide Detection pipeline.
================================================================================
"""

try:
    from .FusionModelDataset import FusionModelDataset
except ImportError:
    FusionModelDataset = None

try:
    from .PST900Dataset import PST900Dataset, get_pst900_palette
except ImportError:
    PST900Dataset = None
    get_pst900_palette = None

try:
    from .tent import Tent, collect_params, configure_model, softmax_entropy
except ImportError:
    Tent = None
