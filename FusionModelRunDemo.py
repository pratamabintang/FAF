"""
================================================================================
FusionModelRunDemo.py (Backwards-compatibility Shim)
================================================================================
This script provides backward compatibility for workflows referencing FusionModelRunDemo.py.
All evaluation and inference logic has been modernized and relocated to `eval.py`.
================================================================================
"""

import sys
from eval import main

if __name__ == '__main__':
    print("[DEPRECATION NOTE] 'FusionModelRunDemo.py' has been modernized to 'eval.py'. Redirecting execution...")
    main()
