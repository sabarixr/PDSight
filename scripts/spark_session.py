from pyspark.sql import SparkSession

def get_spark():
    return SparkSession.builder \
        .appName("PDS_Project") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()