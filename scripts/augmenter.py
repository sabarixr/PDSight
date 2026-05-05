import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

np.random.seed(42)

# =========================
# 1. LOAD DATA
# =========================
df1 = pd.read_csv("./RS_Session_256_AU_404_7.csv")
df2 = pd.read_csv("./RS_Session_257_AU_2316_1.csv")

# =========================
# 2. CLEAN FUNCTION
# =========================
def clean_rs(df):
    df.columns = df.columns.str.strip()

    rename_map = {}
    for col in df.columns:
        if "States" in col:
            rename_map[col] = "state"
        if "2017-18" in col and "Allocation" in col:
            rename_map[col] = "alloc_2017"
        if "2017-18" in col and "Offtake" in col:
            rename_map[col] = "off_2017"
        if "2018-19" in col and "Allocation" in col:
            rename_map[col] = "alloc_2018"
        if "2018-19" in col and "Offtake" in col:
            rename_map[col] = "off_2018"
        if "2019-20" in col and "Allocation" in col:
            rename_map[col] = "alloc_2019"
        if "2019-20" in col and "Offtake" in col:
            rename_map[col] = "off_2019"

    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[~df["state"].str.contains("Total", case=False, na=False)]
    df = df.dropna(subset=["state"])

    return df


df1 = clean_rs(df1)
df2 = clean_rs(df2)

# =========================
# 3. MERGE & DEDUPLICATE
# =========================
common_cols = list(set(df1.columns).intersection(set(df2.columns)))
df1 = df1[common_cols]
df2 = df2[common_cols]

df_rs = pd.concat([df1, df2], ignore_index=True)
df_rs = df_rs.drop_duplicates(subset=["state"])

# =========================
# 4. WIDE → LONG
# =========================
records = []

for _, row in df_rs.iterrows():
    for year in ["2017", "2018", "2019"]:
        records.append({
            "state": row["state"],
            "year": int(year),
            "allocation": row.get(f"alloc_{year}", 0),
            "offtake": row.get(f"off_{year}", 0)
        })

df_long = pd.DataFrame(records)
df_long = df_long[(df_long["allocation"] > 0) & (df_long["offtake"] > 0)]

# =========================
# 5. LOAD HCES
# =========================
hces = pd.read_excel("./HCES_2022-23_Households.xlsx", skiprows=3)
hces.columns = ['state', 'sample_rural', 'sample_urban', 'est_rural', 'est_urban']
hces['hh_total'] = hces['est_rural'] + hces['est_urban']

df_long = df_long.merge(hces[['state', 'hh_total']], on="state", how="left")
df_long['hh_total'] = df_long['hh_total'].fillna(50000)

# =========================
# 6. SCALE SETTINGS
# =========================
DISTRICTS_PER_STATE = 200   # bumped: more unique district-level variation
MONTHS = 180               # bumped: 15 years of monthly data (2019–2034)
COMMODITIES = ["Rice", "Wheat", "Sugar", "Dal", "Kerosene", "Edible Oil"]  # more commodities

# =========================
# 7. PER-STATE WORKER FUNCTION
# =========================
def process_state(row_tuple, districts=DISTRICTS_PER_STATE, months=MONTHS, commodities=COMMODITIES):
    """
    Generate synthetic rows for a single state-year entry.
    Each call is fully independent — safe for multiprocessing.
    """
    # Re-seed per process so results are reproducible but not correlated
    idx, row = row_tuple
    rng = np.random.default_rng(seed=idx)

    state        = row['state']
    yearly_alloc = row['allocation']
    yearly_off   = row['offtake']

    base_alloc = yearly_alloc / districts / 12
    base_off   = yearly_off   / districts / 12
    efficiency = base_off / (base_alloc + 1e-6)
    hh         = row['hh_total'] / districts

    rows = []
    for d in range(districts):
        district = f"{state[:3].upper()}_D{d+1:02d}"

        for m in range(1, months + 1):
            year = 2019 + (m - 1) // 12

            for commodity in commodities:
                noise    = rng.normal(1.0, 0.08)
                seasonal = 1 + 0.1 * np.sin(2 * np.pi * m / 12)
                alloc    = abs(base_alloc * noise * seasonal)

                is_fraud = rng.random() < 0.07

                if is_fraud:
                    fraud_type = rng.choice(["over_withdrawal", "diversion", "ghost"])

                    if fraud_type == "over_withdrawal":
                        off = alloc * rng.uniform(1.1, 1.5)
                    elif fraud_type == "diversion":
                        off = alloc * rng.uniform(0.2, 0.5)
                    else:
                        off = 0 if m % 3 != 0 else alloc * 1.8
                else:
                    off = alloc * efficiency * rng.normal(1.0, 0.05)
                    fraud_type = "none"

                rows.append({
                    "state":          state,
                    "district":       district,
                    "year":           year,
                    "month":          m,
                    "commodity":      commodity,
                    "allocation":     round(alloc, 3),
                    "offtake":        round(max(off, 0), 3),
                    "hh_count":       int(hh),
                    "fraud_type":     fraud_type,
                    "is_fraud":       int(is_fraud),
                    "price_index":    round(rng.uniform(80, 120), 2),
                    "rainfall":       round(rng.uniform(0, 300), 2),
                    "transport_cost": round(rng.uniform(5, 50), 2),
                    "storage_loss":   round(rng.uniform(0, 10), 2),
                })

    return rows


# =========================
# 8. MULTIPROCESSED GENERATION
# =========================
if __name__ == "__main__":

    state_rows = list(df_long.iterrows())   # list of (idx, Series)
    n_workers  = max(1, mp.cpu_count() - 1)

    print(f"Spawning {n_workers} workers for {len(state_rows)} state-year rows …")

    all_records = []

    with mp.Pool(processes=n_workers) as pool:
        # imap_unordered streams results as they finish → low peak memory
        for chunk in tqdm(
            pool.imap_unordered(process_state, state_rows, chunksize=1),
            total=len(state_rows),
            desc="Generating",
            unit="state-yr",
            colour="green",
            dynamic_ncols=True,
        ):
            all_records.extend(chunk)

    print(f"\nTotal raw rows generated: {len(all_records):,}")

    # =========================
    # 9. BUILD DATAFRAME + FEATURES
    # =========================
    print("Building DataFrame and derived features …")
    df_synth = pd.DataFrame(all_records)

    df_synth["leakage_pct"]    = ((df_synth["allocation"] - df_synth["offtake"]) / df_synth["allocation"] * 100).round(2)
    df_synth["offtake_pct"]    = (df_synth["offtake"] / df_synth["allocation"] * 100).round(2)
    df_synth["per_hh_offtake"] = (df_synth["offtake"] / df_synth["hh_count"] * 1000).round(4)
    df_synth["over_withdrawal"] = (df_synth["offtake"] > df_synth["allocation"]).astype(int)

    # =========================
    # 10. SET OUTPUT PATH
    # =========================
    file_path = ".pds_synthetic.csv"
    TARGET_MB = 500
    print(f"Target size: {TARGET_MB} MB — all data is unique (no duplication)")

    # =========================
    # 11. SINGLE CSV WRITE WITH PROGRESS BAR
    # =========================
    print(f"\nWriting CSV → {file_path} …")

    CHUNK = 500_000   # rows per chunk written

    total_rows = len(df_synth)
    header_written = False

    with tqdm(total=total_rows, desc="Writing CSV", unit="rows", unit_scale=True, colour="cyan") as pbar:
        for start in range(0, total_rows, CHUNK):
            chunk_df = df_synth.iloc[start : start + CHUNK]
            chunk_df.to_csv(
                file_path,
                mode   = "w" if not header_written else "a",
                header = not header_written,
                index  = False,
            )
            header_written = True
            pbar.update(len(chunk_df))

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\n✅  DONE — {file_path}")
    print(f"   Rows  : {total_rows:,}")
    print(f"   Size  : {size_mb:.2f} MB")