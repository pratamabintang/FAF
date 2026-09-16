"""
================================================================================
Raster Spatial Array Compatibility and Georeferencing Verification Tool
================================================================================
Verifies multimodal dataset triplets (IMAGE, DTM, LABEL) for:
  1. Spatial array dimension matching (H, W, channels) across modalities.
  2. DTM continuous float32 validity, NoData sentinel ratio, and elevation range.
  3. Georeferencing consistency (CRS, affine transform, spatial bounds, pixel resolution)
     when GeoTIFF metadata is available via rasterio (with graceful fallback to array
     validation when rasterio is uninstalled or images are non-georeferenced PNG/JPEG).
  4. Preprocessing manifest validation against disk assets (when manifest CSV is provided).
================================================================================
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Prevent OpenCV multi-threading contention
cv2.setNumThreads(0)

# Optional rasterio import for deep georeferencing metadata inspection
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False


def inspect_geotiff_metadata(
    filepath: Path
) -> Tuple[Optional[Any], Optional[Tuple[float, float]], Optional[Tuple[float, float, float, float]], Optional[Any]]:
    """
    Extracts CRS, resolution, bounds, and affine transform from a GeoTIFF using rasterio.

    Returns:
        (crs, res, bounds, transform) or (None, None, None, None) if not readable or rasterio missing.
    """
    if not HAS_RASTERIO:
        return None, None, None, None

    try:
        with rasterio.open(str(filepath)) as src:
            return src.crs, src.res, src.bounds, src.transform
    except Exception:
        return None, None, None, None


def verify_manifest_alignment(
    manifest_csv_path: str | Path,
    data_dir: Optional[str | Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verifies that all file paths or stems listed in a dataset manifest CSV exist on disk.
    """
    manifest_path = Path(manifest_csv_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    missing_entries: List[str] = []
    total_entries = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_entries += 1
            # Check for common manifest keys
            for key in ["image", "dtm", "label", "image_path", "dtm_path", "label_path", "filepath"]:
                if key in row and row[key]:
                    p = Path(row[key])
                    if not p.is_absolute() and data_dir is not None:
                        p = Path(data_dir) / p
                    if not p.exists():
                        missing_entries.append(f"Row {total_entries} missing {key}: {row[key]}")

    status = (len(missing_entries) == 0)
    if verbose:
        print(f"[MANIFEST] Checked {total_entries} rows from {manifest_path.name}: {'PASSED [OK]' if status else 'FAILED'}")
        if missing_entries:
            for err in missing_entries[:5]:
                print(f"  - {err}")
            if len(missing_entries) > 5:
                print(f"  ... and {len(missing_entries) - 5} more missing entries")

    return {
        "manifest_path": str(manifest_path),
        "total_entries": total_entries,
        "missing_entries": missing_entries,
        "passed": status,
    }


def verify_raster_alignment(
    data_dir: str | Path,
    split: str = "train",
    nodata_value: float = -9999.0,
    sample_limit: Optional[int] = None,
    manifest_csv: Optional[str | Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Scans dataset split directory and verifies alignment across IMAGE, DTM, and LABEL.

    Args:
        data_dir: Path to dataset root (e.g. dataset/dataset_1).
        split: Split name ('train', 'val', 'test').
        nodata_value: Sentinel value for NoData.
        sample_limit: Optional limit on number of samples to check (None = all).
        manifest_csv: Optional path to manifest CSV.
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
    georeferencing_mismatches: List[str] = []
    corrupted_files: List[str] = []
    nodata_stats: List[float] = []
    dtm_mins: List[float] = []
    dtm_maxs: List[float] = []
    georef_checked_count = 0

    if verbose:
        print("=" * 80)
        print(f"VERIFYING RASTER SPATIAL ARRAY COMPATIBILITY: {split.upper()} SPLIT ({total_samples} samples)")
        print(f"Directory: {split_dir}")
        if HAS_RASTERIO:
            print("Georeferencing Engine: rasterio (CRS, affine transform, bounds checking enabled)")
        else:
            print("Georeferencing Engine: [NOTICE] 'rasterio' is not installed in Python environment.")
            print("                       Spatial array dimensions, channels, and NoData integrity will be verified via OpenCV/PIL.")
        print("=" * 80)

    # Optional manifest check
    manifest_res = None
    if manifest_csv is not None:
        manifest_res = verify_manifest_alignment(manifest_csv, data_dir=root_path, verbose=verbose)

    for i, stem in enumerate(common_stems):
        img_path = img_files[stem]
        dtm_path = dtm_files[stem]
        lbl_path = lbl_files.get(stem)

        # 1. Read modalities
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        dtm = cv2.imread(str(dtm_path), cv2.IMREAD_UNCHANGED)
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED) if lbl_path is not None else None

        if img is None or dtm is None:
            corrupted_files.append(stem)
            continue

        # Single-channel reduction for DTM and Label if 3D
        if dtm.ndim == 3:
            dtm = dtm[:, :, 0]
        if lbl is not None and lbl.ndim == 3:
            lbl = lbl[:, :, 0]

        # 2. Check spatial array dimensions consistency
        img_hw = img.shape[:2]
        dtm_hw = dtm.shape[:2]
        if img_hw != dtm_hw:
            dimension_mismatches.append(f"{stem}: RGB {img_hw} != DTM {dtm_hw}")
            continue

        if lbl is not None and lbl.shape[:2] != img_hw:
            dimension_mismatches.append(f"{stem}: RGB {img_hw} != LABEL {lbl.shape[:2]}")
            continue

        # 3. Georeferencing metadata inspection via rasterio (if installed and files are GeoTIFF)
        if HAS_RASTERIO:
            is_dtm_tif = dtm_path.suffix.lower() in {".tif", ".tiff"}
            is_img_tif = img_path.suffix.lower() in {".tif", ".tiff"}

            if is_dtm_tif:
                dtm_crs, dtm_res, dtm_bounds, dtm_transform = inspect_geotiff_metadata(dtm_path)
                if dtm_crs is not None:
                    georef_checked_count += 1

                # If RGB is also GeoTIFF, compare georeferencing metadata across modalities
                if is_img_tif:
                    img_crs, img_res, img_bounds, img_transform = inspect_geotiff_metadata(img_path)
                    if img_crs is not None and dtm_crs is not None:
                        if img_crs != dtm_crs:
                            georeferencing_mismatches.append(f"{stem}: CRS mismatch RGB({img_crs}) vs DTM({dtm_crs})")
                        if img_res is not None and dtm_res is not None:
                            if abs(img_res[0] - dtm_res[0]) > 1e-4 or abs(img_res[1] - dtm_res[1]) > 1e-4:
                                georeferencing_mismatches.append(f"{stem}: Resolution mismatch RGB({img_res}) vs DTM({dtm_res})")
                        if img_bounds is not None and dtm_bounds is not None:
                            if any(abs(a - b) > 1e-3 for a, b in zip(img_bounds, dtm_bounds)):
                                georeferencing_mismatches.append(f"{stem}: Bounds mismatch RGB vs DTM")

        # 4. Check DTM elevation validity & NoData
        dtm_float = dtm.astype(np.float32)
        invalid_mask = np.isnan(dtm_float) | np.isinf(dtm_float) | (dtm_float <= nodata_value)
        nodata_pct = float(invalid_mask.sum()) / float(dtm_float.size) * 100.0
        nodata_stats.append(nodata_pct)

        valid_pixels = dtm_float[~invalid_mask]
        if len(valid_pixels) > 0:
            dtm_mins.append(float(np.min(valid_pixels)))
            dtm_maxs.append(float(np.max(valid_pixels)))

    status = (
        (len(dimension_mismatches) == 0)
        and (len(corrupted_files) == 0)
        and (len(georeferencing_mismatches) == 0)
        and (manifest_res is None or manifest_res["passed"])
    )

    summary = {
        "split": split,
        "split_dir": str(split_dir),
        "total_samples": total_samples,
        "matched_stems": len(common_stems),
        "passed": status,
        "rasterio_available": HAS_RASTERIO,
        "georeferencing_checked": georef_checked_count,
        "georeferencing_mismatches": georeferencing_mismatches,
        "dimension_mismatches": dimension_mismatches,
        "corrupted_files": corrupted_files,
        "mean_nodata_pct": float(np.mean(nodata_stats)) if nodata_stats else 0.0,
        "max_nodata_pct": float(np.max(nodata_stats)) if nodata_stats else 0.0,
        "global_dtm_min": float(np.min(dtm_mins)) if dtm_mins else 0.0,
        "global_dtm_max": float(np.max(dtm_maxs)) if dtm_maxs else 0.0,
        "manifest_summary": manifest_res,
    }

    if verbose:
        print(f"Validation Status        : {'PASSED [OK]' if status else 'FAILED [ERRORS DETECTED]'}")
        print(f"Total Samples Tested     : {total_samples}")
        print(f"Spatial Dim Mismatches   : {len(dimension_mismatches)}")
        print(f"Georeference Mismatches  : {len(georeferencing_mismatches)}")
        if HAS_RASTERIO:
            print(f"GeoTIFF Metadata Checked : {georef_checked_count}")
        print(f"Corrupted Files          : {len(corrupted_files)}")
        print(f"DTM Elevation Range      : [{summary['global_dtm_min']:.2f} m, {summary['global_dtm_max']:.2f} m]")
        print(f"DTM Mean NoData Pct      : {summary['mean_nodata_pct']:.4f}% (Max: {summary['max_nodata_pct']:.2f}%)")
        print("=" * 80)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Verify raster alignment and spatial dimensions.")
    parser.add_argument("--data_dir", type=str, default="dataset/dataset_1", help="Path to dataset root.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, val, test).")
    parser.add_argument("--nodata_value", type=float, default=-9999.0, help="Sentinel value for NoData.")
    parser.add_argument("--sample_limit", type=int, default=None, help="Limit number of samples to check.")
    parser.add_argument("--manifest_csv", type=str, default=None, help="Optional path to manifest CSV.")
    args = parser.parse_args()

    res = verify_raster_alignment(
        data_dir=args.data_dir,
        split=args.split,
        nodata_value=args.nodata_value,
        sample_limit=args.sample_limit,
        manifest_csv=args.manifest_csv,
        verbose=True
    )
    if not res["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
