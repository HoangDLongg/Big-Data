#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BƯỚC 2: TIỀN XỬ LÝ + KIỂM TRA LOGIC NHIỆT ĐỘ CHO MAPREDUCE K-MEANS
- Loại bỏ: min_temp > max_temp
- Loại bỏ: min_temp > 60°C
- Loại bỏ: |z-score| > 5σ
"""

import re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# === CẤU HÌNH ===
FEATURES = [
    "longitude", "latitude", "elevation",
    "max_temperature", "min_temperature",
    "precipitation", "wind", "relative_humidity"
]
SENTINELS = {-9999, -9999.0, "-9999"}
CLIP_BOUNDS = {
    "max_temperature": (-50, 60), "min_temperature": (-50, 60),
    "precipitation": (0, None), "wind": (0, None),
    "relative_humidity": (0, 100), "elevation": (None, None),
    "longitude": (-180, 180), "latitude": (-90, 90),
}
LOG1P_COLS = {"precipitation", "wind"}
CHUNK_SIZE = 200_000
ZSCORE_THRESHOLD = 5.0  # Loại nếu |z| > 5σ
# ==============================

def normalize_cols(cols):
    return [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in cols]

def clean_chunk(df):
    df.columns = normalize_cols(df.columns)
    df = df[[c for c in FEATURES if c in df.columns]]
    if df.empty: return df

    # 1. Thay sentinel
    df.replace(list(SENTINELS), np.nan, inplace=True)

    # 2. KIỂM TRA LOGIC NHIỆT ĐỘ
    if "min_temperature" in df.columns and "max_temperature" in df.columns:
        # Loại nếu min > max
        invalid_temp = df["min_temperature"] > df["max_temperature"]
        if invalid_temp.any():
            print(f"   [CẢNH BÁO] Loại {invalid_temp.sum():,} dòng: min_temp > max_temp")
            df = df[~invalid_temp]

    # 3. Clip giới hạn
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    # 4. log1p
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # 5. Loại NaN
    df.dropna(inplace=True)
    return df.astype("float32")

# PASS 1: Tính mean/std + đếm lỗi
def pass1_calculate_stats(input_dir, chunk_size):
    stats = {}
    total_invalid = 0
    files = sorted(Path(input_dir).rglob("*.csv"))
    print(f"[PASS 1] Tính mean/std + kiểm tra logic từ {len(files)} file...")

    for i, fp in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {fp.name}")
        for chunk in pd.read_csv(fp, chunksize=chunk_size, low_memory=False):
            original_rows = len(chunk)
            chunk = clean_chunk(chunk)
            if chunk.empty: continue

            # Đếm lỗi nhiệt độ
            dropped = original_rows - len(chunk)
            if dropped > 0:
                total_invalid += dropped

            for col in chunk.columns:
                x = chunk[col].astype("float64")
                n = len(x)
                if col not in stats:
                    stats[col] = {"count": 0, "sum": 0.0, "sumsq": 0.0}
                s = stats[col]
                s["count"] += n
                s["sum"] += x.sum()
                s["sumsq"] += (x * x).sum()

    print(f"   [TỔNG] Loại bỏ {total_invalid:,} dòng do lỗi nhiệt độ!")

    result = {}
    for col, s in stats.items():
        n = s["count"]
        mean = s["sum"] / n
        var = max((s["sumsq"] / n) - (mean ** 2), 0)
        std = max(var ** 0.5, 1e-8)
        result[col] = {"mean": mean, "std": std}
    return result

# PASS 2: Ghi .ml.csv + loại |z| > 5σ
def pass2_write_zscore(files, out_dir, stats, chunk_size):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stats_path = out_path / "_stats"
    stats_path.mkdir(exist_ok=True)

    # Lưu stats
    mean_df = pd.DataFrame({k: v["mean"] for k, v in stats.items()}, index=["mean"])
    std_df = pd.DataFrame({k: v["std"] for k, v in stats.items()}, index=["std"])
    mean_df.T.to_csv(stats_path / "mean.csv")
    std_df.T.to_csv(stats_path / "std.csv")

    print(f"[PASS 2] Ghi .ml.csv + loại |z| > {ZSCORE_THRESHOLD}σ...")
    total_outliers = 0
    for i, fp in enumerate(files, 1):
        out_file = out_path / f"{fp.stem}.ml.csv"
        header = True
        print(f"  [{i}/{len(files)}] -> {out_file.name}")
        for chunk in pd.read_csv(fp, chunksize=chunk_size, low_memory=False):
            chunk = clean_chunk(chunk)
            if chunk.empty: continue

            # Z-score
            z_chunk = chunk.copy()
            outlier_mask = pd.Series([False] * len(chunk))
            for col in chunk.columns:
                if col in stats:
                    mu, sigma = stats[col]["mean"], stats[col]["std"]
                    z = (chunk[col] - mu) / sigma
                    z_chunk[col] = z
                    outlier_mask |= (z.abs() > ZSCORE_THRESHOLD)

            # Loại outlier
            outliers = outlier_mask.sum()
            if outliers > 0:
                total_outliers += outliers
                z_chunk = z_chunk[~outlier_mask]

            mode = "w" if header else "a"
            z_chunk.to_csv(out_file, mode=mode, header=header, index=False)
            header = False

    print(f"   [TỔNG] Loại bỏ {total_outliers:,} dòng do |z| > {ZSCORE_THRESHOLD}σ!")

# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    files = sorted(Path(args.input).rglob("*.csv"))
    if not files:
        print("Không tìm thấy file!")
        return

    stats = pass1_calculate_stats(args.input, args.chunk)
    pass2_write_zscore(files, args.output, stats, args.chunk)
    print(f"\nHOÀN TẤT! Dữ liệu sạch 100% tại: {args.output}")

if __name__ == "__main__":
    main()