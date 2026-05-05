import streamlit as st
import pandas as pd
import subprocess
import os
import shutil
import json
import plotly.express as px

st.set_page_config(page_title="PDS Dashboard", layout="wide")

HDFS_ANOM = "hdfs://localhost:9000/pds/output/final_results"
HDFS_FORE = "hdfs://localhost:9000/pds/output/forecast"

LOCAL_ANOM = "/tmp/final_results"
LOCAL_FORE = "/tmp/forecast"

def fetch_from_hdfs(hdfs_path, local_path):
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    subprocess.run(
        ["hdfs", "dfs", "-get", "-f", hdfs_path, local_path],
        check=True
    )

@st.cache_data(show_spinner=False)
def load_data():
    fetch_from_hdfs(HDFS_ANOM, LOCAL_ANOM)
    fetch_from_hdfs(HDFS_FORE, LOCAL_FORE)

    df_anom = pd.read_parquet(
        LOCAL_ANOM,
        columns=["state","district","month","anomaly_score","is_anomaly","prediction"]
    )

    df_fore = pd.read_parquet(
        LOCAL_FORE,
        columns=["state","month","offtake","prediction"]
    )

    df_anom["state"] = df_anom["state"].str.strip()
    df_fore["state"] = df_fore["state"].str.strip()

    df_anom = df_anom.sample(min(len(df_anom), 5000), random_state=42)
    df_fore = df_fore.sample(min(len(df_fore), 5000), random_state=42)

    return df_anom, df_fore

df_anom, df_fore = load_data()

st.title("PDS Anomaly & Forecast Dashboard")

# Sidebar
states = sorted(df_anom["state"].dropna().unique().tolist())
state = st.sidebar.selectbox("Select State", ["All"] + states)

if state != "All":
    df_anom = df_anom[df_anom["state"] == state]
    df_fore = df_fore[df_fore["state"] == state]

# -------------------------
# METRICS
# -------------------------
colA, colB, colC = st.columns(3)

colA.metric("Records", len(df_anom))
colB.metric("Anomalies", int(df_anom["is_anomaly"].sum()))
colC.metric("Avg Risk", round(df_anom["anomaly_score"].mean(), 2))

# -------------------------
# TOP + BAR
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Anomalies")
    top = df_anom.nlargest(20, "anomaly_score")
    st.dataframe(top, use_container_width=True)

with col2:
    st.subheader("Anomalies by State")
    agg = df_anom.groupby("state")["is_anomaly"].sum().sort_values(ascending=False)
    st.bar_chart(agg)

# -------------------------
# FORECAST
# -------------------------
st.subheader("Forecast vs Actual")

if not df_fore.empty:
    chart_df = (
        df_fore.groupby("month")[["offtake","prediction"]]
        .mean()
        .reset_index()
        .sort_values("month")
    )

    st.line_chart(chart_df.set_index("month"))
else:
    st.warning("No forecast data available")

# -------------------------
# RISK DISTRICTS
# -------------------------
st.subheader("Top Risk Districts")

risk = (
    df_anom.groupby("district")["anomaly_score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(risk)

st.subheader("India Risk Map")

with open("india_states.geojson") as f:
    india_geo = json.load(f)

# 🔥 normalize names
df_anom["state"] = df_anom["state"].astype(str).str.strip()

# 🔥 FIX mismatches (CRITICAL)
state_map = {
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
    "Jammu & Kashmir": "Jammu and Kashmir",
    "NCT of Delhi": "Delhi",
    "Uttarakhand": "Uttaranchal",   # depends on geojson version
    "Odisha": "Orissa"              # some geojsons use old name
}

df_anom["state"] = df_anom["state"].replace(state_map)

# 🔥 aggregate
heat_df = (
    df_anom.groupby("state")["anomaly_score"]
    .mean()
    .reset_index()
)

# 🔥 get geo states
geo_states = {
    f["properties"]["NAME_1"]
    for f in india_geo["features"]
}

# 🔥 keep only matching states
heat_df = heat_df[heat_df["state"].isin(geo_states)]

# 🧠 DEBUG (optional)
# st.write("Data:", sorted(df_anom["state"].unique()))
# st.write("Geo:", sorted(geo_states))

if heat_df.empty:
    st.error("No matching states after mapping")
else:
    fig = px.choropleth(
        heat_df,
        geojson=india_geo,
        locations="state",
        featureidkey="properties.NAME_1",
        color="anomaly_score",
        color_continuous_scale="Reds",
    )

    fig.update_geos(fitbounds="locations", visible=False)

    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    st.plotly_chart(fig, use_container_width=True)