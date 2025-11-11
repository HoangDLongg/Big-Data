#!/usr/bin/env python3
import os
import json
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

FEATURES = [
    "longitude", "latitude", "elevation",
    "max_temperature", "min_temperature",
    "precipitation", "wind", "relative_humidity"
]

spark = (SparkSession.builder
         .appName("GetCentersOnly")
         .config("spark.driver.memory", "6g")
         .config("spark.executor.memory", "6g")
         .getOrCreate())

print("Reading Parquet files from output_parquet_fixed...")
df = spark.read.parquet("output_parquet_fixed/*.parquet")
df = df.select(*FEATURES).na.drop()

# Use 10% data for speed (still accurate centers)
df = df.sample(False, 0.1).cache()
print(f"LOADED {df.count():,} points (10% data) -> TRAINING...")

assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
kmeans = KMeans().setK(8).setSeed(42).setFeaturesCol("features")
pipeline = Pipeline(stages=[assembler, kmeans])

print("TRAINING KMeans k=8...")
model = pipeline.fit(df)

print("EXTRACTING CLUSTER CENTERS...")
centers = model.stages[-1].clusterCenters()
result = [
    {"cluster": i, **dict(zip(FEATURES, map(float, c)))}
    for i, c in enumerate(centers)
]

os.makedirs("centers_result", exist_ok=True)
with open("centers_result/centers_k8.json", "w") as f:
    json.dump(result, f, indent=2)

print("DONE! Centers saved to: centers_result/centers_k8.json")
spark.stop()