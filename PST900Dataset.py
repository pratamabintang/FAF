"""
================================================================================
Legacy Dataset Shim: PST900 (RGB-Thermal)
================================================================================
The PST900 dataset loader has been moved to `legacy/PST900Dataset.py`.
This module forwards to the isolated legacy implementation for backwards-compatibility.
For Landslide RGB-DTM detection, use `LandslideDataset.py`.
================================================================================
"""

from legacy.PST900Dataset import *
