"""
1. Penanganan DTM Void / NaN (imputasi dengan nilai median lokal, memastikan zero NaN).
2. Anti-aliasing DTM menggunakan filter Gaussian (sigma = 4.0 px) untuk mengeliminasi efek tangga 25cm.
3. Ekstraksi fitur fisik Slope (kemiringan lereng dalam derajat [0, 90] dinormalisasi ke [0.0, 1.0]).
4. Normalisasi relatif lokal elevasi (DTM_norm dalam rentang [0.0, 1.0]) untuk membuang domain shift +68.8m.
5. Standardisasi biner pada mask Label (uint8, {0, 255}).
6. Salin citra RGB bersih.
7. Menghasilkan metadata katalog baru (catalog_preprocessed.csv) lengkap dengan kolom is_blacklisted.
"""

import argparse
import os
import glob
import argparse
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter

def parse_args():
    parser = argparse.ArgumentParser(description="Pra-pemrosesan Dataset Landslide Multi-Modal")
    parser.add_argument("--input_dir", type=str, default="D:/Landslide/FAF(FrequencyAwareFusion)/dataset/dataset_1")
    parser.add_argument("--output_dir", type=str, default="D:/Landslide/FAF(FrequencyAwareFusion)/dataset/dataset_1V2")
    parser.add_argument("--blacklist_file", type=str, default="D:/Landslide/FAF(FrequencyAwareFusion)/dataset/black_list.txt")
    parser.add_argument("--skip_blacklisted", action="store_true", default=False, help="Jika diset, tile dalam blacklist tidak akan ditulis ke output_dir")
    parser.add_argument("--gsd", type=float, default=0.01953125, help="Resolusi spasial fisik per piksel dalam meter (default: 0.01953125) RGB")
    parser.add_argument("--gaussian_sigma", type=float, default=4.0, help="Sigma filter Gaussian untuk anti-aliasing DTM 25cm (default: 4.0)")
    return parser.parse_args()

def load_blacklist(blacklist_path):
    if not os.path.exists(blacklist_path):
        print(f"[WARN] File blacklist '{blacklist_path}' tidak ditemukan. Menggunakan blacklist kosong.")
        return set()
    with open(blacklist_path, "r", encoding="utf-8") as f:
        blacklist = set(line.strip() for line in f if line.strip() and not line.startswith("#"))
    return blacklist

def process_dataset(args):
    start_time = time.time()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    blacklist = load_blacklist(args.blacklist_file)

    print("=" * 80)
    print("PREPROCESSING DATASET LANDSLIDE MULTI-MODAL")
    print("=" * 80)
    print(f"Direktori Input        : {input_dir}")
    print(f"Direktori Output       : {output_dir}")
    print(f"Blacklist File         : {args.blacklist_file} ({len(blacklist)} tile terdaftar)")
    print(f"Skip Blacklisted Tiles : {args.skip_blacklisted}")
    print(f"GSD Fisik              : {args.gsd} m/pixel")
    print(f"Gaussian Sigma         : {args.gaussian_sigma} px (radius ~25 cm)")
    print("=" * 80)

    splits = ["train", "val", "test"]
    modalities = ["IMAGE", "DTM_NORM", "SLOPE", "LABEL"]

    # Buat struktur direktori target
    for s in splits:
        for m in modalities:
            os.makedirs(os.path.join(output_dir, s, m), exist_ok=True)

    # Muat metadata katalog lama jika ada untuk informasi chainage
    catalog_old_path = os.path.join(input_dir, "dataset_catalog.csv")
    old_catalog_map = {}
    if os.path.exists(catalog_old_path):
        old_df = pd.read_csv(catalog_old_path)
        for _, row in old_df.iterrows():
            old_catalog_map[row["tile_id"]] = row.to_dict()

    # Storage untuk katalog preprocessed
    processed_records = []

    total_scanned = 0
    total_written = 0
    total_skipped = 0

    for split in splits:
        dtm_files = sorted(glob.glob(os.path.join(input_dir, split, "DTM", "*.tif")))
        print(f"\nMemproses split '{split.upper()}' ({len(dtm_files)} tile)...")

        for dtm_in_path in dtm_files:
            total_scanned += 1
            filename = os.path.basename(dtm_in_path)
            tile_id = os.path.splitext(filename)[0]
            
            is_blacklisted = tile_id in blacklist
            
            # Jika opsi skip aktif dan tile ada di blacklist, lewati penulisan file
            if args.skip_blacklisted and is_blacklisted:
                total_skipped += 1
                continue

            img_in_path = os.path.join(input_dir, split, "IMAGE", f"{tile_id}.png")
            lbl_in_path = os.path.join(input_dir, split, "LABEL", f"{tile_id}.png")

            # -------------------------------------------------------------
            # 1. BACA CITRA RGB & SALIN BERSIH
            # -------------------------------------------------------------
            rgb_img = Image.open(img_in_path).convert("RGB")
            img_out_path = os.path.join(output_dir, split, "IMAGE", f"{tile_id}.png")
            rgb_img.save(img_out_path)

            # -------------------------------------------------------------
            # 2. BACA DTM, IMPUTASI NaN, GAUSSIAN FILTER
            # -------------------------------------------------------------
            dtm_arr = np.array(Image.open(dtm_in_path)).astype(np.float32)
            has_nan_initially = bool(np.isnan(dtm_arr).any())

            if has_nan_initially:
                if np.all(np.isnan(dtm_arr)):
                    dtm_arr = np.zeros_like(dtm_arr, dtype=np.float32)
                else:
                    median_val = float(np.nanmedian(dtm_arr))
                    dtm_arr = np.nan_to_num(dtm_arr, nan=median_val)

            # Gaussian Anti-Aliasing (skala 25cm)
            dtm_smooth = gaussian_filter(dtm_arr, sigma=args.gaussian_sigma)

            # -------------------------------------------------------------
            # 3. HITUNG PHYSICAL SLOPE
            # -------------------------------------------------------------
            grad_y, grad_x = np.gradient(dtm_smooth, args.gsd)
            slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
            slope_deg = np.rad2deg(slope_rad)
            slope_norm = np.clip(slope_deg / 90.0, 0.0, 1.0).astype(np.float32)

            slope_out_path = os.path.join(output_dir, split, "SLOPE", f"{tile_id}.tif")
            cv2.imwrite(slope_out_path, slope_norm)

            # -------------------------------------------------------------
            # 4. HITUNG DTM_NORM (LOCAL RELATIVE ELEVATION [0, 1])
            # -------------------------------------------------------------
            d_min = float(np.min(dtm_smooth))
            d_max = float(np.max(dtm_smooth))
            relief = d_max - d_min
            if relief > 1e-4:
                dtm_norm = ((dtm_smooth - d_min) / relief).astype(np.float32)
            else:
                dtm_norm = np.zeros_like(dtm_smooth, dtype=np.float32)

            dtm_norm_out_path = os.path.join(output_dir, split, "DTM_NORM", f"{tile_id}.tif")
            cv2.imwrite(dtm_norm_out_path, dtm_norm)

            # -------------------------------------------------------------
            # 5. BACA & STANDARISASI LABEL (UINT8 {0, 255})
            # -------------------------------------------------------------
            raw_lbl = np.array(Image.open(lbl_in_path))
            bin_mask = (raw_lbl > 0).astype(np.uint8) * 255
            lbl_out_path = os.path.join(output_dir, split, "LABEL", f"{tile_id}.png")
            Image.fromarray(bin_mask).save(lbl_out_path)

            total_written += 1

            # -------------------------------------------------------------
            # 6. CATAT METADATA
            # -------------------------------------------------------------
            ls_pixels = int((bin_mask > 0).sum())
            ls_pct = float(ls_pixels / bin_mask.size * 100)
            chainage_val = old_catalog_map.get(tile_id, {}).get("chainage", None)

            processed_records.append({
                "tile_id": tile_id,
                "split": split,
                "chainage": chainage_val,
                "img_rel_path": os.path.relpath(img_out_path, output_dir),
                "dtm_norm_rel_path": os.path.relpath(dtm_norm_out_path, output_dir),
                "slope_rel_path": os.path.relpath(slope_out_path, output_dir),
                "lbl_rel_path": os.path.relpath(lbl_out_path, output_dir),
                "has_landslide": ls_pixels > 0,
                "landslide_pixels": ls_pixels,
                "landslide_pct": ls_pct,
                "relief_local_m": round(relief, 3),
                "slope_mean_deg": round(float(np.mean(slope_deg)), 2),
                "had_nan_original": has_nan_initially,
                "is_blacklisted": is_blacklisted
            })

    # Simpan katalog CSV baru
    catalog_out_df = pd.DataFrame(processed_records)
    catalog_out_path = os.path.join(output_dir, "catalog_preprocessed.csv")
    catalog_out_df.to_csv(catalog_out_path, index=False)

    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("PREPROCESSING SELESAI DENGAN SUKSES!")
    print("=" * 80)
    print(f"Total tile dipindai      : {total_scanned:,}")
    print(f"Total tile ditulis       : {total_written:,}")
    print(f"Total tile di-skip       : {total_skipped:,}")
    print(f"Waktu eksekusi           : {elapsed:.2f} detik ({total_written/elapsed:.1f} tiles/detik)")
    print(f"Katalog baru tersimpan di: {catalog_out_path}")
    print("\nRingkasan per Split:")
    for s in splits:
        sub = catalog_out_df[catalog_out_df["split"] == s]
        pos_cnt = (sub["has_landslide"] == True).sum()
        neg_cnt = (sub["has_landslide"] == False).sum()
        blk_cnt = (sub["is_blacklisted"] == True).sum()
        print(f"  * {s.upper():<5}: Total {len(sub):>4} | Positif: {pos_cnt:>3} | Negatif: {neg_cnt:>3} | Blacklisted: {blk_cnt:>2}")

    print("\nCara Penggunaan Blacklist pada DataLoader PyTorch:")
    print("```python")
    print("with open('black_list.txt') as f:")
    print("    blacklist = set(line.strip() for line in f if line.strip() and not line.startswith('#'))")
    print("\n# Saat memuat list file:")
    print("file_list = [f for f in all_files if get_tile_id(f) not in blacklist]")
    print("```")
    print("=" * 80)

if __name__ == "__main__":
    args = parse_args()
    process_dataset(args)