from spark_session import get_spark
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline

spark = get_spark()

path = "hdfs://localhost:9000/pds/data/pds_synthetic.csv"

numeric_cols = [
    "allocation","offtake","leakage_pct",
    "offtake_pct","per_hh_offtake",
    "price_index","rainfall","transport_cost","storage_loss",
    "hh_count","month"
]

def load_and_clean():
    df = spark.read.csv(path, header=True, inferSchema=True)
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))
    df = df.dropna(subset=numeric_cols)
    for c in numeric_cols:
        df = df.filter(
            (~F.isnan(F.col(c))) &
            (F.col(c) != float("inf")) &
            (F.col(c) != float("-inf"))
        )
    return df

df = load_and_clean()

df = df.sample(0.25, seed=42).repartition(6).persist()

df = df.withColumn("leakage_ratio", F.col("leakage_pct") / 100)
df = df.withColumn("demand_pressure", F.col("offtake") / F.col("hh_count"))

feature_cols = [
    "allocation","offtake","leakage_pct",
    "offtake_pct","per_hh_offtake",
    "price_index","rainfall","transport_cost","storage_loss"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
scaler = StandardScaler(inputCol="raw_features", outputCol="features")

df = assembler.transform(df)
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)

models = {
    "KMeans": KMeans(k=6, maxIter=8, featuresCol="features"),
    "BisectingKMeans": BisectingKMeans(k=6, maxIter=8, featuresCol="features")
}

evaluator = ClusteringEvaluator(featuresCol="features")

best_model = None
best_score = -1
best_name = None

for name, model in models.items():
    m = model.fit(df)
    pred = m.transform(df)
    score = evaluator.evaluate(pred)
    print(f"{name} silhouette score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = m
        best_name = name

print(f"Best anomaly model: {best_name}")

df = best_model.transform(df).withColumnRenamed("prediction", "cluster")

forecast_features = ["month", "hh_count", "price_index", "rainfall"]

assembler2 = VectorAssembler(inputCols=forecast_features, outputCol="forecast_features")
df = assembler2.transform(df)

train = df.filter(F.col("month") <= 80)
test  = df.filter(F.col("month") > 80)

models_reg = {
    "GBT": GBTRegressor(featuresCol="forecast_features", labelCol="offtake", maxIter=20),
    "RandomForest": RandomForestRegressor(featuresCol="forecast_features", labelCol="offtake", numTrees=20)
}

evaluator_reg = RegressionEvaluator(labelCol="offtake", predictionCol="prediction", metricName="rmse")

best_reg_model = None
best_rmse = float("inf")
best_reg_name = None

for name, model in models_reg.items():
    m = model.fit(train)
    preds = m.transform(test)
    rmse = evaluator_reg.evaluate(preds)
    print(f"{name} RMSE: {rmse:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_reg_model = m
        best_reg_name = name

print(f"Best forecast model: {best_reg_name}")

predictions = best_reg_model.transform(test)

df.write.mode("overwrite").parquet("hdfs://localhost:9000/pds/output/anomalies")

predictions.select(
    "state","month","offtake","prediction"
).write.mode("overwrite").parquet("hdfs://localhost:9000/pds/output/forecast")

best_model.write().overwrite().save("hdfs://localhost:9000/pds/models/kmeans_model")
best_reg_model.write().overwrite().save("hdfs://localhost:9000/pds/models/gbt_model")

df_pipeline = load_and_clean()

pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(df_pipeline)

pipeline_model.write().overwrite().save(
    "hdfs://localhost:9000/pds/models/preprocessing_pipeline"
)

print("MODELS SAVED")
print("DONE")