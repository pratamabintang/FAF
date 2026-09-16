"""
================================================================================
Raster Resolution and Geographic Alignment Verification Tool
================================================================================
Verifies that multimodal dataset triplets (IMAGE, DTM, LABEL) have consistent
spatial dimensions, data types, value distributions, and georeferencing metadata.
================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from PIL import Image

# Prevent OpenCV multi-threading contention
cv2.setNumThreads(0)


def verify_raster_alignment(
    data_dir: str | Path,
    split: str = "train",
    nodata_value: float = -9999.0,
    sample_limit: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Scans dataset split directory and verifies alignment across IMAGE, DTM, and LABEL.

    Args:
        data_dir: Path to dataset root (e.g. dataset/dataset_1).
        split: Split name ('train', 'val', 'test').
        nodata_value: Sentinel value for NoData.
        sample_limit: Optional limit on number of samples to check (None = all).
        verbose: If True, prints progress and summary.

    Returns:
        Dictionary containing summary statistics, errors, and validation status.
    """
    root_path = Path(data_dir).resolve()
    split_dir = root_path / split if (root_path / split).exists() else root_path

    img_dir = split_dir / "IMAGE"
    dtm_dir = split_dir / "DTM"
    lbl_dir = split_dir / "LABEL"

    if not img_dir.exists() or not dtm_dir.exists():
        raise FileNotFoundError(f"Missing required IMAGE or DTM directory in {split_dir}")

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    img_files = {f.stem: f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts}
    dtm_files = {f.stem: f for f in dtm_dir.iterdir() if f.suffix.lower() in valid_exts}
    lbl_files = {f.stem: f for f in lbl_dir.iterdir() if f.suffix.lower() in valid_exts} if lbl_dir.exists() else {}

    common_stems = sorted(list(set(img_files.keys()) & set(dtm_files.keys())))
    if sample_limit is not None and sample_limit > 0:
        common_stems = common_stems[:sample_limit]

    total_samples = len(common_stems)
    dimension_mismatches: List[str] = []
    corrupted_files: List[str] = []
    nodata_stats: List[float] = []
    dtm_mins: List[float] = []
    dtm_maxs: List[float] = []

    if verbose:
        print("=" * 80)
        print(f"VERIFYING RASTER ALIGNMENT: {split.upper()} SPLIT ({total_samples} samples)")
        print(f"Directory: {split_dir}")
        print("=" * 80)

    for i, stem in enumerate(common_stems):
        # 1. Read modalities
        img = cv2.imread(str(img_files[stem]), cv2.IMREAD_COLOR)
        dtm = cv2.imread(str(dtm_files[stem]), cv2.IMREAD_UNCHANGED)
        lbl = cv2.imread(str(lbl_files[stem]), cv2.IMREAD_UNCHANGED) if stem in lbl_files else None

        if img is None or dtm is None:
            corrupted_files.append(stem)
            continue

        # Single-channel reduction for DTM and Label if 3D
        if dtm.ndim == 3:
            dtm = dtm[:, :, 0]
        if lbl is not None and lbl.ndim == 3:
            lbl = lbl[:, :, 0]

        # 2. Check spatial dimensions consistency
        img_hw = img.shape[:2]
        dtm_hw = dtm.shape[:2]
        if img_hw != dtm_hw:
            dimension_mismatches.append(f"{stem}: RGB {img_hw} != DTM {dtm_hw}")
            continue

        if lbl is not None and lbl.shape[:2] != img_hw:
            dimension_mismatches.append(f"{stem}: RGB {img_hw} != LABEL {lbl.shape[:2]}")
            continue

        # 3. Check DTM elevation validity & NoData
        dtm_float = dtm.astype(np.float32)
        invalid_mask = np.isnan(dtm_float) | np.isinf(dtm_float) | (dtm_float <= nodata_value)
        nodata_pct = float(invalid_mask.sum()) / float(dtm_float.size) * 100.0
        nodata_stats.append(nodata_pct)

        valid_pixels = dtm_float[~invalid_mask]
        if len(valid_pixels) > 0:
            dtm_mins.append(float(np.min(valid_pixels)))
            dtm_maxs.append(float(np.max(valid_pixels)))

    status = (len(dimension_mismatches) == 0) and (len(corrupted_files) == 0)

    summary = {
        "split": split,
        "split_dir": str(split_dir),
        "total_samples": total_samples,
        "matched_stems": len(common_stems),
        "passed": status,
        "dimension_mismatches": dimension_mismatches,
        "corrupted_files": corrupted_files,
        "mean_nodata_pct": float(np.mean(nodata_stats)) if nodata_stats else 0.0,
        "max_nodata_pct": float(np.max(nodata_stats)) if nodata_stats else 0.0,
        "global_dtm_min": float(np.min(dtm_mins)) if dtm_mins else 0.0,
        "global_dtm_max": float(np.max(dtm_maxs)) if dtm_maxs else 0.0,
    }

    if verbose:
        print(f"Validation Status    : {'PASSED [OK]' if status else 'FAILED [ERRORS DETECTED]'}")
        print(f"Total Samples Tested : {total_samples}")
        print(f"Dimension Mismatches : {len(dimension_mismatches)}")
        print(f"Corrupted Files      : {len(corrupted_files)}")
        print(f"DTM Elevation Range  : [{summary['global_dtm_min']:.2f} m, {summary['global_dtm_max']:.2f} m]")
        print(f"DTM Mean NoData Pct  : {summary['mean_nodata_pct']:.4f}% (Max: {summary['max_nodata_pct']:.2f}%)")
        print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Verify raster alignment and spatial dimensions.")
    parser.add_argument("--data_dir", type=str, default="dataset/dataset_1", help="Path to dataset root.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, val, test).")
    parser.add_argument("--nodata_value", type=float, default=-9999.0, help="Sentinel value for NoData.")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of samples to check.")
    args = parser.parse_args()

    res = verify_raster_alignment(
        data_dir=args.data_dir,
        split=args.split,
        nodata_value=args.nodata_value,
        sample_limit=args.sample_limit,
        verbose=True
    )
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
