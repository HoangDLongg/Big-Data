#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess weather CSVs -> Parquet for ML / K-Means pipelines (HDFS-friendly)
- Tự động nhận dạng header (có khoảng trắng, hoa/thường)
- Clean dữ liệu: clip, log1p, scale RH nếu cần
- Hai lượt: tính thống kê toàn cục + ghi Parquet (Snappy)
"""

import os
import glob
import json
import math
from typing import Dict, List
import numpy as np
import pandas as pd

# =============== CONFIG ===============
INPUT_DIR = r"C:\Users\Administrator\Downloads\55020_2022-01-30-23-20-11"  # thư mục chứa .csv
OUTPUT_DIR = r"E:\BIGDATA\output_parquet_fixed"                            # thư mục xuất .parquet
CHUNK_ROWS = 2_000_000
BASE_NAME = "weather_clean_part"
KEEP_COLS = [
    "longitude", "latitude", "elevation",
    "max_temperature", "min_temperature",
    "precipitation", "wind", "relative_humidity"
]
LOG1P_COLS = {"precipitation", "wind"}
SENTINELS = {-9999, -999, -99}
CLIP_BOUNDS = {
    "max_temperature": (-50, 60),
    "min_temperature": (-50, 60),
    "precipitation": (0, None),
    "wind": (0, None),
    "relative_humidity": (None, None),
    "elevation": (None, None),
    "longitude": (-180, 180),
    "latitude": (-90, 90),
}
DTYPE_FLOAT = "float32"
WRITE_STANDARDIZED = True
# ======================================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def safe_read_csv(fp: str) -> pd.DataFrame:
    """Đọc CSV tự động phát hiện dấu phân cách và BOM"""
    try:
        return pd.read_csv(fp, engine="python", sep=None, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(fp, engine="python", encoding="utf-8", sep=",")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hoá header -> snake_case, bỏ khoảng trắng, lower"""
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean_frame(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    use = [c for c in keep_cols if c in df.columns]
    if not use:
        return df.iloc[0:0].copy()

    df = df[use].copy()

    # scale RH nếu max > 1.5 (thang 0-100)
    if "relative_humidity" in df.columns:
        if df["relative_humidity"].quantile(0.99) > 1.5:
            df["relative_humidity"] = df["relative_humidity"].clip(0, 100) / 100.0

    # thay sentinel -> NaN
    for s in SENTINELS:
        df.replace(s, np.nan, inplace=True)

    # clip trong khoảng
    for col, (lo, hi) in CLIP_BOUNDS.items():
        if col in df.columns:
            if lo is not None:
                df[col] = df[col].clip(lower=lo)
            if hi is not None:
                df[col] = df[col].clip(upper=hi)

    # log1p cho các cột lệch nhiều
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])

    # loại bỏ hàng NaN
    df.dropna(subset=use, inplace=True)

    # ép kiểu nhẹ
    for c in df.columns:
        df[c] = df[c].astype(DTYPE_FLOAT)
    return df


def update_running_stats(stats: Dict[str, Dict[str, float]], batch: pd.DataFrame):
    for col in batch.columns:
        x = batch[col].astype("float64")
        s = float(x.sum())
        ss = float((x * x).sum())
        n = int(x.shape[0])
        if col not in stats:
            stats[col] = {"sum": 0.0, "sumsq": 0.0, "count": 0, "min": float("+inf"), "max": float("-inf")}
        stats[col]["sum"] += s
        stats[col]["sumsq"] += ss
        stats[col]["count"] += n
        stats[col]["min"] = min(stats[col]["min"], float(x.min()))
        stats[col]["max"] = max(stats[col]["max"], float(x.max()))


def finalize_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for col, v in stats.items():
        n = max(v["count"], 1)
        mean = v["sum"] / n
        var = max(v["sumsq"] / n - mean * mean, 0.0)
        std = math.sqrt(var) if var > 0 else 1.0
        out[col] = {"mean": mean, "std": std, "min": v["min"], "max": v["max"], "count": n}
    return out


def standardize(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    z = df.copy()
    for col in z.columns:
        if col in stats:
            mu = stats[col]["mean"]
            sd = stats[col]["std"] if stats[col]["std"] != 0 else 1.0
            z[col] = (z[col].astype("float64") - mu) / sd
    return z.astype(DTYPE_FLOAT)


def write_parquet_chunks(df: pd.DataFrame, out_dir: str, base: str, part_idx_start: int = 0):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = df.shape[0]
    start = 0
    part = part_idx_start
    while start < rows:
        end = min(start + CHUNK_ROWS, rows)
        chunk = df.iloc[start:end]
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        out_path = os.path.join(out_dir, f"{base}-{part:05d}.parquet")
        pq.write_table(table, out_path, compression="snappy")
        print(f"Wrote {out_path} rows={chunk.shape[0]}")
        start = end
        part += 1
    return part


def main():
    ensure_dir(OUTPUT_DIR)
    files = list_csvs(INPUT_DIR)
    if not files:
        raise SystemExit(f"❌ Không tìm thấy file CSV trong {INPUT_DIR}")

    print(f"🧾 Tổng số file CSV: {len(files)}")

    stats_accum = {}
    for i, fp in enumerate(files, 1):
        try:
            df = safe_read_csv(fp)
            df = normalize_columns(df)
            df = clean_frame(df, KEEP_COLS)
            if df.empty:
                print(f"[WARN] Skip {fp}: sau clean không còn dữ liệu hợp lệ")
                continue
            update_running_stats(stats_accum, df)
        except Exception as e:
            print(f"[WARN] Skip {fp}: {e}")
        if i % 50 == 0:
            print(f"Pass1 processed {i}/{len(files)} files")

    stats = finalize_stats(stats_accum)
    stats_file = os.path.join(OUTPUT_DIR, "feature_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"💾 Đã lưu thống kê -> {stats_file}")

    # Ghi parquet
    next_part_idx = 0
    buffer = []
    rows_in_buf = 0

    def flush_buffer(buf: List[pd.DataFrame], idx: int) -> int:
        if not buf:
            return idx
        big = pd.concat(buf, ignore_index=True)
        idx = write_parquet_chunks(big, OUTPUT_DIR, f"{BASE_NAME}_raw", idx)
        if WRITE_STANDARDIZED:
            z = standardize(big, stats)
            idx = write_parquet_chunks(z, OUTPUT_DIR, f"{BASE_NAME}_z", idx)
        buf.clear()
        return idx

    for i, fp in enumerate(files, 1):
        try:
            df = safe_read_csv(fp)
            df = normalize_columns(df)
            df = clean_frame(df, KEEP_COLS)
            if df.empty:
                continue
            buffer.append(df)
            rows_in_buf += df.shape[0]
        except Exception as e:
            print(f"[WARN] Skip {fp}: {e}")

        if rows_in_buf >= CHUNK_ROWS:
            next_part_idx = flush_buffer(buffer, next_part_idx)
            rows_in_buf = 0

    flush_buffer(buffer, next_part_idx)
    print("✅ Hoàn tất tiền xử lý toàn bộ dữ liệu.")


if __name__ == "__main__":
    main()
