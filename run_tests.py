"""
================================================================================
Frequency-Aware Fusion (FAF) - Test Suite Runner
================================================================================
Runs all model, module, and dataset loader unit tests.

Usage:
  python run_tests.py
  python -m unittest discover -s tests
================================================================================
"""

import os
import sys
import unittest
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

def main():
    print("=" * 80)
    print("            RUNNING FREQUENCY-AWARE FUSION (FAF) UNIT TESTS                    ")
    print("=" * 80)

    # Discover and run tests in the 'tests' directory
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=test_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print(f"[TEST SUITE SUCCESS] Passed all {result.testsRun} unit test cases.")
        sys.exit(0)
    else:
        failures = len(result.failures)
        errors = len(result.errors)
        print(f"[TEST SUITE FAILED] Failures: {failures}, Errors: {errors} out of {result.testsRun} tests.")
        sys.exit(1)

if __name__ == '__main__':
    main()
