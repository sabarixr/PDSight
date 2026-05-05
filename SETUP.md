# PDS Anomaly & Forecast System — Setup Guide

Full pipeline: Hadoop (HDFS) → PySpark ML → Inference → Streamlit dashboard

Covers **Ubuntu 20.04/22.04** and **Arch Linux**.

---

## Requirements

| Component | Version           |
| --------- | ----------------- |
| Java      | 11                |
| Python    | 3.10+             |
| Hadoop    | 3.3.6             |
| RAM       | 16 GB recommended |
| Disk      | ~5 GB free        |

---

## 1. System Dependencies

### Ubuntu

```bash
sudo apt update
sudo apt install -y openjdk-11-jdk python3.10 python3.10-venv python3-pip git openssh-server
```

### Arch Linux

```bash
sudo pacman -S jdk11-openjdk python python-pip git openssh
```

Verify:

```bash
java -version        # should say openjdk 11
python3 --version    # 3.10+
```

---

## 2. SSH Localhost (Required for Hadoop)

Hadoop single-node mode requires passwordless SSH to localhost. Skip this and `start-dfs.sh` will hang.

```bash
ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 0600 ~/.ssh/authorized_keys
```

Enable and start SSH:

### Ubuntu

```bash
sudo systemctl enable ssh
sudo systemctl start ssh
```

### Arch Linux

```bash
sudo systemctl enable sshd
sudo systemctl start sshd
```

Test it:

```bash
ssh localhost   # should connect without password prompt
```

---

## 3. Set JAVA_HOME

### Ubuntu

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Arch Linux

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

> If you use zsh/fish, apply the same exports to `~/.zshrc` or `~/.config/fish/config.fish`.

---

## 4. Install Hadoop

```bash
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
tar -xzf hadoop-3.3.6.tar.gz
mv hadoop-3.3.6 ~/hadoop
```

Add to `~/.bashrc`:

```bash
export HADOOP_HOME=$HOME/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
```

```bash
source ~/.bashrc
```

---

## 5. Configure Hadoop

### Tell Hadoop where Java is

Edit `$HADOOP_HOME/etc/hadoop/hadoop-env.sh` and set:

```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64   # Ubuntu
# or
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk          # Arch
```

> This is separate from your shell's JAVA_HOME. Without this, `start-dfs.sh` fails even if your terminal has it set.

---

### core-site.xml

```bash
nano $HADOOP_HOME/etc/hadoop/core-site.xml
```

```xml
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
```

---

### hdfs-site.xml

```bash
nano $HADOOP_HOME/etc/hadoop/hdfs-site.xml
```

```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file:///home/${user.name}/hadoopdata/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file:///home/${user.name}/hadoopdata/datanode</value>
  </property>
</configuration>
```

---

### mapred-site.xml

```bash
nano $HADOOP_HOME/etc/hadoop/mapred-site.xml
```

```xml
<configuration>
  <property>
    <name>mapreduce.framework.name</name>
    <value>yarn</value>
  </property>
</configuration>
```

---

### yarn-site.xml

```bash
nano $HADOOP_HOME/etc/hadoop/yarn-site.xml
```

```xml
<configuration>
  <property>
    <name>yarn.nodemanager.aux-services</name>
    <value>mapreduce_shuffle</value>
  </property>
</configuration>
```

---

## 6. Format HDFS and Start

Only format once. Re-formatting wipes all data.

```bash
hdfs namenode -format
start-dfs.sh
```

Verify with:

```bash
jps
```

Expected output:

```
NameNode
DataNode
SecondaryNameNode
```

If any are missing, check logs at `$HADOOP_HOME/logs/`.

Web UI: http://localhost:9870

---

## 7. Python Environment

```bash
python3 -m venv pyspark-env
source pyspark-env/bin/activate

pip install pyspark pandas numpy pyarrow streamlit plotly
```

---

## 8. Project Structure

```
project/
├── scripts/
│   ├── spark_session.py
│   └── train_and_compare.py
├── inference.py
├── dashboard.py
├── india_states.geojson
└── data/
    └── pds_synthetic.csv
```

---

## 9. Upload Data to HDFS

```bash
hdfs dfs -mkdir -p /pds/data
hdfs dfs -put data/pds_synthetic.csv /pds/data/
```

Verify:

```bash
hdfs dfs -ls /pds/data
```

---

## 10. Train Models

```bash
python scripts/train_and_compare.py
```

Trains clustering + regression models and saves results to HDFS under `/pds/output/`.

---

## 11. Run Inference

```bash
python inference.py
```

Generates:

```
/pds/output/final_results
/pds/output/forecast
```

---

## 12. Run Dashboard

```bash
streamlit run dashboard.py
```

Open: http://localhost:8501

---

## 13. GeoJSON

Place `india_states.geojson` in the project root. The file must have `properties.NAME_1` as the state name field. If state names don't match, the heatmap will silently drop those states.

---

## Architecture

```
HDFS (Storage)
     ↓
Spark ML (Training + Inference)
     ↓
Parquet Output (/pds/output/)
     ↓
Streamlit Dashboard
     ↓
India Heatmap + Analytics
```

---

## Troubleshooting

**`start-dfs.sh` hangs or fails**

- SSH localhost not set up → redo step 2
- `JAVA_HOME` not set inside `hadoop-env.sh` → redo step 5
- Run `ssh localhost` manually and check if it connects

**`DataNode` not in `jps`**

- Usually a clusterID mismatch from re-formatting. Fix:
  ```bash
  stop-dfs.sh
  rm -rf ~/hadoopdata
  hdfs namenode -format
  start-dfs.sh
  ```

**HDFS path not found**

```bash
hdfs dfs -ls /
```

- If this fails, HDFS isn't running. Check `$HADOOP_HOME/logs/hadoop-*-namenode-*.log`

**Empty or broken dashboard**

- Re-run `inference.py` and confirm `/pds/output/final_results` exists:
  ```bash
  hdfs dfs -ls /pds/output/
  ```

**Map not showing states**

- State name mismatch between your data and GeoJSON `NAME_1` values
- Print unique state names from both and compare

**Spark can't connect to HDFS**

- Ensure `fs.defaultFS` in `core-site.xml` matches what your Spark session uses
- Pass it explicitly if needed:
  ```python
  spark = SparkSession.builder \
      .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
      .getOrCreate()
  ```
