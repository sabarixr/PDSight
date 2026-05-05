from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml import PipelineModel
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf
import numpy as np

spark = SparkSession.builder \
    .appName("Inference") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

df = spark.read.parquet("hdfs://localhost:9000/pds/output/anomalies")

pipeline = PipelineModel.load("hdfs://localhost:9000/pds/models/preprocessing_pipeline")
kmeans = KMeansModel.load("hdfs://localhost:9000/pds/models/kmeans_model")
gbt = GBTRegressionModel.load("hdfs://localhost:9000/pds/models/gbt_model")

df = df.drop("raw_features", "features", "cluster", "prediction")

df = pipeline.transform(df)

df = kmeans.transform(df)
df = df.withColumnRenamed("prediction", "cluster")

centers = kmeans.clusterCenters()

def dist(v, cid):
    return float(np.linalg.norm(np.array(v) - centers[cid]))

dist_udf = udf(dist, FloatType())

df = df.withColumn("anomaly_score", dist_udf("features", "cluster"))

threshold = df.approxQuantile("anomaly_score", [0.95], 0.01)[0]

df = df.withColumn(
    "is_anomaly",
    (F.col("anomaly_score") > threshold).cast("int")
)

df = df.drop("prediction")

df = gbt.transform(df)

df.select(
    "state",
    "district",
    "month",
    "cluster",
    "prediction",
    "anomaly_score",
    "is_anomaly"
).write.mode("overwrite").parquet(
    "hdfs://localhost:9000/pds/output/final_results"
)

print("FINAL RESULTS CREATED")