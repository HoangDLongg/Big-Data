# pyspark_kmeans_report_pipeline.py
# -*- coding: utf-8 -*-
"""
Huấn luyện K-Means (auto chọn k) + Xuất báo cáo chi tiết:
- Bảng silhouette theo k (CSV + PNG)
- Tâm cụm (centers) & phân bố số điểm mỗi cụm (CSV)
- Heatmap (ma trận) trung bình đặc trưng theo cụm (PNG)
- Scatter sơ đồ Lon-Lat (mẫu) tô màu theo cụm (PNG)
- Báo cáo HTML tổng hợp (link các ảnh & bảng)

Yêu cầu: pyspark, pandas, matplotlib (local vẽ ảnh). Không dùng seaborn.

Chạy:
  spark-submit --master local[*] pyspark_kmeans_report_pipeline.py
hoặc:
  spark-submit --master yarn pyspark_kmeans_report_pipeline.py
"""
import re
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # vẽ không cần GUI
import matplotlib.pyplot as plt
import traceback
import glob
import sys
import platform

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# ======== CONFIG ========
# Đường dẫn dữ liệu parquet (KHUYẾN NGHỊ dùng bản Z-score)
# Local Windows:
# Note: ensure the path points to your parquet folder. Add missing slash after BIGDATA.
INPUT_PATH = "file:///E:/BIGDATAoutput_parquet/weather_clean_part_raw-*.parquet"
# HDFS ví dụ:
# INPUT_PATH = "hdfs:///data/weather_kmeans/parquet/*_z-*.parquet"

# Danh sách đặc trưng (sau khi đã sanitize -> chữ thường_gạch_dưới)
FEATURES = [
    "longitude", "latitude", "elevation",
    "max_temperature", "min_temperature",
    "precipitation", "wind", "relative_humidity",
]

# Các k thử nghiệm (tự chọn)
K_CANDIDATES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
SEED = 42
MAX_ITER = 50

# Thư mục xuất báo cáo (local hoặc HDFS -> ở đây xuất local)
# Ensure directory exists or will be created by ensure_dir().
REPORT_DIR = r"E:\BIGDATA\output_parquet\kmeans_report"
# ========================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def sanitize(name: str) -> str:
    """Chuẩn hoá tên cột sang dạng an toàn cho Spark ML: chữ thường + gạch dưới."""
    name = (name or "").strip()
    name = name.replace("%", "pct").replace("/", "_per_")
    # thay mọi ký tự không thuộc [A-Za-z0-9_] bằng "_"
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # gộp nhiều "_" liên tiếp
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


def sanitize_df_columns(df):
    """Đổi tên tất cả cột theo sanitize(), trả về (df_renamed, mapping_old_new)."""
    old_cols = df.columns
    new_cols = [sanitize(c) for c in old_cols]
    mapping = dict(zip(old_cols, new_cols))
    for o, n in zip(old_cols, new_cols):
        if o != n:
            df = df.withColumnRenamed(o, n)
    return df, mapping


def plot_silhouette(scores_csv, out_png):
    df = pd.read_csv(scores_csv)
    plt.figure(figsize=(7, 5))
    plt.plot(df["k"], df["silhouette"], marker="o")
    plt.title("Silhouette score vs. k")
    plt.xlabel("k")
    plt.ylabel("silhouette")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_cluster_counts(counts_csv, out_png):
    df = pd.read_csv(counts_csv)
    plt.figure(figsize=(7, 5))
    plt.bar(df["cluster"].astype(str), df["count"])
    plt.title("Cluster sizes")
    plt.xlabel("cluster")
    plt.ylabel("count")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_feature_heatmap(means_csv, out_png):
    df = pd.read_csv(means_csv)
    # df: cluster, mean_<feature>...
    clusters = df["cluster"].astype(int).tolist()
    feat_cols = [c for c in df.columns if c.startswith("mean_")]
    mat = df[feat_cols].values  # shape [k, d]

    plt.figure(figsize=(max(8, len(feat_cols) * 0.9), max(5, len(clusters) * 0.6)))
    im = plt.imshow(mat, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(feat_cols)),
               [c.replace("mean_", "") for c in feat_cols],
               rotation=45, ha="right")
    plt.yticks(range(len(clusters)), [str(c) for c in clusters])

    plt.title("Feature means by cluster (z-score space)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_lonlat_scatter(sample_csv, out_png, max_points=200000):
    df = pd.read_csv(sample_csv)

    # Hỗ trợ cả tên chữ thường (sau sanitize) lẫn tên cũ viết hoa
    lon_col = "longitude" if "longitude" in df.columns else ("Longitude" if "Longitude" in df.columns else None)
    lat_col = "latitude" if "latitude" in df.columns else ("Latitude" if "Latitude" in df.columns else None)
    if lon_col is None or lat_col is None:
        raise ValueError(f"Không tìm thấy cột longitude/latitude trong {sample_csv}. Columns={list(df.columns)}")

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    plt.figure(figsize=(7, 6))
    plt.scatter(df[lon_col], df[lat_col], c=df["cluster"], s=2, alpha=0.6)
    plt.title("Clusters on Lon-Lat (sample)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def make_html_report(report_dir, best_k, best_silhouette, files):
    html_path = os.path.join(report_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>KMeans Report</title></head><body>")
        f.write(f"<h1>KMeans Report</h1>")
        f.write(f"<p><b>Best k</b>: {best_k} &nbsp; | &nbsp; <b>Silhouette</b>: {best_silhouette:.4f}</p>")

        # Silhouette
        f.write("<h2>Silhouette vs k</h2>")
        f.write(f"<p><img src='{os.path.basename(files['silhouette_png'])}' width='600'></p>")
        f.write(f"<p><a href='{os.path.basename(files['silhouette_csv'])}'>silhouette_per_k.csv</a></p>")

        # Counts
        f.write("<h2>Cluster sizes</h2>")
        f.write(f"<p><img src='{os.path.basename(files['counts_png'])}' width='600'></p>")
        f.write(f"<p><a href='{os.path.basename(files['counts_csv'])}'>cluster_counts.csv</a></p>")

        # Centers
        f.write("<h2>Cluster centers (vectors)</h2>")
        f.write(f"<p><a href='{os.path.basename(files['centers_csv'])}'>cluster_centers.csv</a></p>")

        # Means heatmap
        f.write("<h2>Feature means per cluster (z-score)</h2>")
        f.write(f"<p><img src='{os.path.basename(files['heatmap_png'])}' width='700'></p>")
        f.write(f"<p><a href='{os.path.basename(files['means_csv'])}'>cluster_feature_means.csv</a></p>")

        # Lon-Lat scatter
        f.write("<h2>Lon-Lat scatter (sample)</h2>")
        f.write(f"<p><img src='{os.path.basename(files['lonlat_png'])}' width='600'></p>")
        f.write(f"<p><a href='{os.path.basename(files['lonlat_csv'])}'>lonlat_sample.csv</a></p>")

        f.write("</body></html>")
    return html_path


def main():
    ensure_dir(REPORT_DIR)

    # Print quick environment info to help debug runtime issues
    print("Python:", sys.version.replace('\n', ' '))
    print("Platform:", platform.platform())

    spark = (
         SparkSession.builder
        .appName("WeatherKMeansReportPipeline")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.legacy.allowUnquotedCharacterInIdentifiers", "true")
        .getOrCreate()
    )

    # Check that input path has matching parquet files (if local file://)
    def list_input_files(path_pattern):
        if path_pattern.startswith("file://"):
            local_pattern = path_pattern.replace("file://", "")
            # On Windows a leading / may be present, strip it for glob
            if local_pattern.startswith('/') and os.name == 'nt' and re.match(r'^/[A-Za-z]:', local_pattern):
                local_pattern = local_pattern[1:]
            matches = glob.glob(local_pattern)
            return matches
        # For non-file paths we can't glob here; return empty to avoid false positives
        return []

    matches = list_input_files(INPUT_PATH)
    if matches:
        print(f"Found {len(matches):,} local files matching INPUT_PATH (showing up to 10):")
        for p in matches[:10]:
            print("  ", p)
    else:
        print("No local files matched INPUT_PATH pattern (or INPUT_PATH is non-file). INPUT_PATH=", INPUT_PATH)

    # Đọc parquet (lười thực thi – action sẽ diễn ra sau)
    df = spark.read.parquet(INPUT_PATH)

    # Sanitize tên cột NGAY LẬP TỨC, trước mọi thao tác khác
    df, col_map = sanitize_df_columns(df)
    print("Renamed columns mapping:\n", col_map)

    # Kiểm tra đầy đủ FEATURES
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        # Provide a more helpful error with available columns and suggestions
        avail = list(df.columns)
        raise ValueError(
            "Thiếu cột sau khi sanitize: {}\nSchema hiện tại: {}\n".format(missing, avail)
            + "Hãy kiểm tra tên cột gốc trong parquet hoặc cập nhật FEATURES list."
        )

    # Thông tin số dòng (trigger action lần đầu sau sanitize)
    print(f"Loaded data from {INPUT_PATH} with {df.count():,} rows.")

    # Vector hoá
    va = VectorAssembler(inputCols=FEATURES, outputCol="features", handleInvalid="skip")
    vdf = va.transform(df).select(*FEATURES, "features")

    evaluator = ClusteringEvaluator(featuresCol="features",
                                    predictionCol="prediction",
                                    metricName="silhouette")

    # Quét k
    sil_rows = []
    best_k, best_score, best_model = None, float("-inf"), None
    for k in K_CANDIDATES:
        try:
            km = KMeans(k=k, seed=SEED, maxIter=MAX_ITER, featuresCol="features", predictionCol="prediction")
            model = km.fit(vdf)
            score = evaluator.evaluate(model.transform(vdf))
            sil_rows.append({"k": k, "silhouette": float(score)})
            print(f"k={k} -> silhouette={score:.6f}")
            if score > best_score:
                best_k, best_score, best_model = k, score, model
        except Exception as e:
            # Log and continue with other k values
            print(f"Warning: failed to fit/evaluate KMeans for k={k}: {e}")
            sil_rows.append({"k": k, "silhouette": float('nan')})
            continue

    # Xuất silhouette per k
    silhouette_csv = os.path.join(REPORT_DIR, "silhouette_per_k.csv")
    pd.DataFrame(sil_rows).to_csv(silhouette_csv, index=False)
    silhouette_png = os.path.join(REPORT_DIR, "silhouette_per_k.png")
    plot_silhouette(silhouette_csv, silhouette_png)

    # Ensure we have a best model
    if best_model is None:
        raise RuntimeError("Không tìm thấy model hợp lệ trong các K_CANDIDATES. Kiểm tra dữ liệu và logs trước khi chạy lại.")

    # Dự đoán theo best model
    pred = best_model.transform(vdf).select(*FEATURES, "prediction")
    pred = pred.withColumnRenamed("prediction", "cluster")

    # Đếm mỗi cụm
    counts = pred.groupBy("cluster").count().orderBy("cluster")
    counts_pdf = counts.toPandas()
    counts_csv = os.path.join(REPORT_DIR, "cluster_counts.csv")
    counts_pdf.to_csv(counts_csv, index=False)
    counts_png = os.path.join(REPORT_DIR, "cluster_counts.png")
    plot_cluster_counts(counts_csv, counts_png)

    # Tâm cụm (vector centers)
    centers = best_model.clusterCenters()
    centers_pdf = pd.DataFrame(centers, columns=FEATURES)
    centers_pdf.insert(0, "cluster", range(len(centers)))
    centers_csv = os.path.join(REPORT_DIR, "cluster_centers.csv")
    centers_pdf.to_csv(centers_csv, index=False)

    # Trung bình đặc trưng theo cụm (Z-score space)
    means_expr = [F.mean(c).alias(f"mean_{c}") for c in FEATURES]
    means_df = pred.groupBy("cluster").agg(*means_expr).orderBy("cluster")
    means_csv = os.path.join(REPORT_DIR, "cluster_feature_means.csv")
    means_df.toPandas().to_csv(means_csv, index=False)

    # Heatmap
    heatmap_png = os.path.join(REPORT_DIR, "cluster_feature_means_heatmap.png")
    plot_feature_heatmap(means_csv, heatmap_png)

    # Scatter Lon-Lat mẫu
    sample_frac = 0.02
    sample_max = 300_000
    # LƯU Ý: dùng tên cột chữ thường (đã sanitize)
    sdf = pred.sample(False, sample_frac, seed=SEED).select("longitude", "latitude", "cluster").limit(sample_max)
    lonlat_csv = os.path.join(REPORT_DIR, "lonlat_sample.csv")
    sdf.toPandas().to_csv(lonlat_csv, index=False)
    lonlat_png = os.path.join(REPORT_DIR, "lonlat_scatter.png")
    plot_lonlat_scatter(lonlat_csv, lonlat_png, max_points=200000)

    files = {
        "silhouette_csv": silhouette_csv,
        "silhouette_png": silhouette_png,
        "counts_csv": counts_csv,
        "counts_png": counts_png,
        "centers_csv": centers_csv,
        "means_csv": means_csv,
        "heatmap_png": heatmap_png,
        "lonlat_csv": lonlat_csv,
        "lonlat_png": lonlat_png,
    }

    html_path = make_html_report(REPORT_DIR, best_k, best_score, files)
    print(f"\n✅ DONE. Best k={best_k}, silhouette={best_score:.4f}")
    print(f"📁 Report folder: {REPORT_DIR}")
    print(f"📄 HTML: {html_path}")

    spark.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Ensure report dir exists so we can write the traceback for debugging
        try:
            ensure_dir(REPORT_DIR)
        except Exception:
            pass
        tb = traceback.format_exc()
        err_path = os.path.join(REPORT_DIR, "error_traceback.log")
        try:
            with open(err_path, "w", encoding="utf-8") as ef:
                ef.write(tb)
        except Exception as e:
            print("Failed to write traceback to", err_path, "->", e)
        print("An exception occurred. Full traceback written to:", err_path)
        print(tb)
        # Exit with non-zero code for CI/automation
        sys.exit(1)
