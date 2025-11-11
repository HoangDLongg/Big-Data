from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, expr, count, mean
from pyspark.ml.feature import VectorAssembler, StandardScaler
import sys

def main(input_path, output_path):
    spark = SparkSession.builder.appName("WeatherETLFullReport").getOrCreate()

    print("=== START WEATHER DATA PROCESSING ===")
    print(f"Reading data from: {input_path}")

    # 1️⃣ Read all CSV files
    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(input_path)
    )
    df = df.repartition(400) 
    total_rows_raw = df.count()
    total_cols = len(df.columns)
    print(f" Loaded {total_rows_raw:,} rows, {total_cols} columns\n")

    # 2️⃣ Replace error code -9999 with null
    df = df.replace(-9999, None)

    # 3️⃣ Count missing values before cleaning
    print("=== Missing values before cleaning ===")
    null_before = df.select([
        count(when(col(c).isNull(), c)).alias(c) for c in df.columns
    ])
    null_before.show(truncate=False)

    # 4️⃣ Drop duplicates
    df = df.dropDuplicates()

    # 5️⃣ Filter logic: Max >= Min
    if "Max Temperature" in df.columns and "Min Temperature" in df.columns:
        df = df.filter(col("Max Temperature") >= col("Min Temperature"))

    # 6️⃣ Fill missing numeric values with mean
    numeric_cols = [c for (c, t) in df.dtypes if t in ["double", "int", "float"]]
    fill_report = []
    for c in numeric_cols:
        avg_value = df.select(mean(col(c))).collect()[0][0]
        if avg_value is not None:
            count_missing = df.filter(col(c).isNull()).count()
            if count_missing > 0:
                fill_report.append((c, count_missing, float(avg_value)))
                df = df.na.fill({c: avg_value})

    # 7️⃣ Remove outliers (1st–99th percentile)
    outlier_report = []
    for c in ["Max Temperature", "Min Temperature", "Precipitation", "Wind", "Relative Humidity"]:
        if c in df.columns:
            q_low, q_high = df.approxQuantile(c, [0.01, 0.99], 0.05)  
            before_count = df.count()
            df = df.filter((col(c) >= q_low) & (col(c) <= q_high))
            after_count = df.count()
            removed = before_count - after_count
            outlier_report.append((c, q_low, q_high, removed))

    # 8️⃣ Add average temperature column
    if "Max Temperature" in df.columns and "Min Temperature" in df.columns:
        df = df.withColumn("AvgTemp", expr("(`Max Temperature` + `Min Temperature`) / 2"))

    # 9️⃣ Normalize data (standardization)
    selected_features = ["Max Temperature", "Min Temperature", "Precipitation", "Wind", "Relative Humidity"]
    available_features = [c for c in selected_features if c in df.columns]
    assembler = VectorAssembler(inputCols=available_features, outputCol="features_raw")
    scaled_data = assembler.transform(df)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    scaled_model = scaler.fit(scaled_data)
    final_df = scaled_model.transform(scaled_data)

    # 🔟 Save cleaned data
    final_df.write.mode("overwrite").option("header", True).csv(output_path + "/clean_data")

    # 🧾 11️⃣ Create full processing report
    total_rows_final = final_df.count()

    print("\n ==== DATA CLEANING SUMMARY ==== ")
    print(f"Total raw rows: {total_rows_raw:,}")
    print(f"Total final rows: {total_rows_final:,}")
    print(f"Removed rows (duplicates, outliers, etc.): {total_rows_raw - total_rows_final:,}")
    print(f"Percentage removed: {((total_rows_raw - total_rows_final)/total_rows_raw)*100:.2f}%\n")

    print("🧮 Columns with missing values filled:")
    for (c, n, v) in fill_report:
        print(f"  - {c}: {n} values replaced with mean = {v:.3f}")

    print("\n Columns filtered for outliers:")
    for (c, q1, q2, removed) in outlier_report:
        print(f"  - {c}: [{q1:.2f}, {q2:.2f}] → {removed} rows removed")

    # 12️⃣ Write report to CSV
    report_data = []
    for (c, n, v) in fill_report:
        report_data.append(("fill_missing", c, n, v, None, None))
    for (c, q1, q2, removed) in outlier_report:
        report_data.append(("outlier_filter", c, None, None, q1, q2))

    report_df = spark.createDataFrame(report_data, ["step", "column", "affected_values", "fill_mean", "q_low", "q_high"])
    report_df.write.mode("overwrite").option("header", True).csv(output_path + "/etl_report")

    print("\n Detailed reports saved at:")
    print(f"   - {output_path}/etl_report/")
    print(f"   - {output_path}/clean_data/")
    print(" ETL PROCESS COMPLETED SUCCESSFULLY ")

    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit weather_etl_full_report.py <input_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
