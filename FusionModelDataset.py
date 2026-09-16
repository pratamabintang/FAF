"""
================================================================================
Legacy Dataset Shim: MFNet (RGB-Thermal)
================================================================================
The MFNet dataset loader has been moved to `legacy/FusionModelDataset.py`.
This module forwards to the isolated legacy implementation for backwards-compatibility.
For Landslide RGB-DTM detection, use `LandslideDataset.py`.
================================================================================
"""

from legacy.FusionModelDataset import *