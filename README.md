Dự án Phân cụm Dữ liệu Thời tiết (Big Data Weather Clustering)

Đây là một dự án phân tích dữ liệu lớn (Big Data) sử dụng các công cụ PySpark và Pandas để xử lý và phân cụm K-Means cho một bộ dữ liệu thời tiết 12GB.

Mục tiêu dự án

* **ETL (Extract, Transform, Load):** Xây dựng một quy trình tiền xử lý (preprocessing) hiệu quả để làm sạch, chuẩn hóa và chuyển đổi dữ liệu CSV thô sang định dạng Parquet tối ưu.
* **Phân cụm (Clustering):** Áp dụng thuật toán K-Means để khám phá các cụm (cluster) thời tiết đặc trưng trên toàn cầu.
* **Trực quan hóa (Visualization):** Trực quan hóa các tâm cụm (centroids) tìm được lên bản đồ thế giới bằng Folium.

---

##  Dữ liệu (Data)

Do giới hạn của GitHub (không cho phép file trên 100MB), bộ dữ liệu 12GB (.csv) không được lưu trữ trong kho chứa này. File `data.csv` đã được thêm vào `.gitignore`.

Để chạy dự án, bạn cần tải bộ dữ liệu và đặt vào thư mục phù hợp (ví dụ: `data/`).

* **Link tải dữ liệu (Spark):** `https://drive.google.com/file/d/1s-AQeVRUbJKurIcPerxnMz6gM2VczMjV/view?usp=sharing`
* **Link tải dữ liệu (Kmean - MapReduce):** `https://drive.google.com/file/d/1gPnUnPFG3volQRJdG1W6q4CJ6OCnms0Z/view?usp=sharing`
---

##  Công nghệ sử dụng

* **Ngôn ngữ:** Python 3
* **Big Data:** Apache Spark (qua PySpark)
* **Xử lý dữ liệu:** Pandas, NumPy
* **Trực quan hóa:** Folium (vẽ bản đồ), Matplotlib
* **Lưu trữ:** Parquet

---

##  Cách chạy dự án Spark-Kmeans

Quy trình được chia thành 2 bước chính, tương ứng với các script Python:

### 1. Tiền xử lý (Preprocessing)

Chạy script `weather_preprocess_full_local.py` (hoặc `weather_etl_full_report.py` nếu dùng Spark) để chuyển đổi CSV thô sang Parquet sạch và đã chuẩn hóa (Z-score).

```bash
python weather_preprocess_full_local.py
```
### 2. Huấn luyện mô hình (K-Means)

Sử dụng Spark để chạy K-Means trên dữ liệu Parquet đã tạo ở bước 1.

Script chính (`weather_etl_full_report.py`) sẽ tự động thử nhiều giá trị `k`, tìm ra `k` tốt nhất dựa trên chỉ số Silhouette, và tạo ra một báo cáo HTML chi tiết với các biểu đồ.

```bash
spark-submit weather_etl_full_report.py hdfs:///ten/folder/input hdfs:///ten/folder/output
```

##  Cách chạy dự án Kmeans - MapReduce(java)

Quy trình được chia thành 3 bước chính

### 1. Tiền xử lý (Preprocessing)
preprocess.py

Chạy script `preprocess.py` để tiền xử lý bao gồm làm sạch dữ liệu, chuẩn hóa z-score, kiểm tra logic của dữ liệu. 

```bash
python preprocess.py
```

### 2. Huấn luyện mô hình (K-Means)

Sử dụng kmeanfinalpromax.jar để huấn luyện bao gồm các file KCombiner.java KMapper.java KReducer.java PointWritable.java Main.java

```bash
hadoop jar E:\BIGDATA\kmeanfinalpromax.jar Main -in /user/hduser/weatherdata/data-kmeans -out /user/hduser/kmeans_sample400_k8 -k 8 -thresh 0.02 -maxloop 10 -NumReduceTask 1 -lines 200000
```
### 3. Trực quan hóa kết quả

Sau khi có file tâm cụm (centroids) từ bước 2 (bạn cần lấy kết quả từ HDFS về máy local và chuyển đổi sang định dạng JSON nếu cần), chạy script `plot_kmeans_map.py` để tạo bản đồ thế giới.

```bash
python plot_kmeans_map.py
```
