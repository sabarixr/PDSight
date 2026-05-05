import os

# path to your dataset
local_file = "data/synthetic/pds_synthetic.csv"

# HDFS destination
hdfs_path = "/pds/data/pds_synthetic.csv"

# create directory
os.system("hdfs dfs -mkdir -p /pds/data")

# upload file
os.system(f"hdfs dfs -put -f {local_file} {hdfs_path}")

print("✅ Uploaded to HDFS")
