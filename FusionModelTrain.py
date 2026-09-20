"""
================================================================================
FusionModelTrain.py (Backwards-compatibility Shim)
================================================================================
This script provides backward compatibility for workflows referencing FusionModelTrain.py.
All training logic has been modernized and relocated to `train.py`.
================================================================================
"""

import sys
from train import main

if __name__ == '__main__':
    print("[DEPRECATION NOTE] 'FusionModelTrain.py' has been modernized to 'train.py'. Redirecting execution...")
    main()
