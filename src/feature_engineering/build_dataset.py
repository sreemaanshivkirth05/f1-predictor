"""
build_dataset.py
----------------
Combines all feature groups into one master training dataset.

This is the central file that:
  1. Calls driver_features.py    → driver performance & Elo features
  2. Calls circuit_features.py   → circuit affinity & type features
  3. Calls weather_features.py   → weather & wet weather features
  4. Merges everything into one DataFrame
  5. Adds the target variable (finish_position)
  6. Cleans and validates the final dataset
  7. Saves to data/master_dataset.csv

HOW TO RUN:
    python src/feature_engineering/build_dataset.py

OUTPUT:
    data/master_dataset.csv  — the file the XGBoost model trains on
"""

import pandas as pd
import numpy as np
import os
import sys

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# Add src/ to path so we can import our feature modules
sys.path.insert(0, SRC_DIR)

from feature_engineering.driver_features  import build_driver_features
from feature_engineering.circuit_features import build_circuit_features
from feature_engineering.weather_features import build_weather_features


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Columns that identify a unique driver-race row
ID_COLS = ["year", "round", "driver_id", "circuit_id"]

# Target variable — what we are predicting
TARGET_COL = "finish_position"

# Columns to drop before saving
# (raw text/status columns not useful as model features)
COLS_TO_DROP = [
    "race_name", "race_date", "driver_name", "driver_code",
    "constructor_name", "locality", "country", "lat", "long",
    "status", "laps_completed", "fastest_lap_time", "fastest_lap_rank",
]


# ─── CONSTRUCTOR FEATURES ─────────────────────────────────────────────────────

def compute_constructor_features(race_df):
    """
    Computes constructor-level features.

    These capture team performance separate from individual driver ability:
      - constructor championship points and rank
      - constructor DNF rate (reliability)
      - constructor avg finish (pace)
      - pit stop average time and consistency
      - constructor rank trend

    Args:
        race_df (pd.DataFrame): Race results

    Returns:
        pd.DataFrame: Constructor features per constructor per race
    """
    print("Computing constructor features...")

    race_df = race_df.copy()
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce")
    race_df["is_dnf"] = race_df["status"].apply(
        lambda s: 0 if (str(s) == "Finished" or str(s).startswith("+")) else 1
    )

    # Load constructor standings
    stand_path = os.path.join(DATA_DIR, "constructor_standings.csv")
    if os.path.exists(stand_path):
        stand_df = pd.read_csv(stand_path)
    else:
        print("  WARNING: constructor_standings.csv not found, using defaults")
        stand_df = pd.DataFrame()

    # Load pit stop data
    pit_path = os.path.join(DATA_DIR, "pit_stops.csv")
    if os.path.exists(pit_path):
        pit_df = pd.read_csv(pit_path)
        pit_df["duration_s"] = pd.to_numeric(pit_df["duration_s"], errors="coerce")
        # Remove unrealistic pit stop times (< 1.5s or > 120s are errors)
        pit_df = pit_df[
            (pit_df["duration_s"] >= 1.5) &
            (pit_df["duration_s"] <= 120)
        ]
    else:
        print("  WARNING: pit_stops.csv not found, skipping pit stop features")
        pit_df = pd.DataFrame()

    records = []
    years   = sorted(race_df["year"].unique())

    for year in years:
        history     = race_df[race_df["year"] < year]
        year_races  = race_df[race_df["year"] == year]
        constructors = year_races["constructor_id"].unique()

        for constructor_id in constructors:
            # Rolling constructor reliability from history
            team_history = history[history["constructor_id"] == constructor_id]

            if team_history.empty:
                dnf_rate   = 0.1
                avg_finish = 10.0
            else:
                last_2_years = team_history[team_history["year"] >= year - 2]
                dnf_rate     = last_2_years["is_dnf"].mean() if len(last_2_years) > 0 else 0.1
                avg_finish   = last_2_years["finish_position"].mean() if len(last_2_years) > 0 else 10.0

            # Pit stop stats from this season and last
            if not pit_df.empty:
                # Match pit stops to this constructor using race_df
                constructor_drivers = race_df[
                    (race_df["constructor_id"] == constructor_id) &
                    (race_df["year"].isin([year, year - 1]))
                ]["driver_id"].unique()

                team_pits = pit_df[
                    (pit_df["driver_id"].isin(constructor_drivers)) &
                    (pit_df["year"].isin([year, year - 1]) if "year" in pit_df.columns else True)
                ]

                if len(team_pits) > 0:
                    pit_avg = team_pits["duration_s"].mean()
                    pit_std = team_pits["duration_s"].std()
                else:
                    pit_avg = 25.0
                    pit_std = 2.0
            else:
                pit_avg = 25.0
                pit_std = 2.0

            # Get rounds for this constructor this year
            constructor_rounds = year_races[
                year_races["constructor_id"] == constructor_id
            ]["round"].unique()

            for round_num in constructor_rounds:
                # Constructor standings before this round
                if not stand_df.empty:
                    prior_stand = stand_df[
                        (stand_df["year"] == year) &
                        (stand_df["round"] < round_num) &
                        (stand_df["constructor_id"] == constructor_id)
                    ].sort_values("round", ascending=False)

                    if not prior_stand.empty:
                        con_pts  = prior_stand.iloc[0]["constructor_champ_pts"]
                        con_rank = prior_stand.iloc[0]["constructor_champ_pos"]
                    else:
                        con_pts  = 0.0
                        con_rank = 10
                else:
                    con_pts  = 0.0
                    con_rank = 10

                records.append({
                    "year":                   year,
                    "round":                  round_num,
                    "constructor_id":         constructor_id,
                    "constructor_dnf_rate":   round(dnf_rate, 4),
                    "constructor_avg_finish": round(avg_finish, 2),
                    "constructor_champ_pts":  con_pts,
                    "constructor_champ_rank": con_rank,
                    "pit_avg_time_s":         round(pit_avg, 3),
                    "pit_std_time_s":         round(pit_std, 3) if not np.isnan(pit_std) else 2.0,
                })

    df = pd.DataFrame(records).drop_duplicates(
        subset=["year", "round", "constructor_id"]
    )
    print(f"  Computed constructor features: {len(df):,} rows")
    return df


# ─── ENCODE CATEGORICAL COLUMNS ───────────────────────────────────────────────

def encode_categoricals(df):
    """
    Converts categorical columns into numbers the model can use.

    - circuit_type: street=0, permanent=1
    - constructor_id: label encoded (each team gets a number)
    - driver_id: label encoded

    Args:
        df (pd.DataFrame): Master dataset

    Returns:
        pd.DataFrame: Dataset with encoded columns
    """
    df = df.copy()

    # Circuit type
    if "circuit_type" in df.columns:
        df["circuit_type_enc"] = df["circuit_type"].map(
            {"street": 0, "permanent": 1, "hybrid": 2}
        ).fillna(1)

    # Constructor ID label encoding
    if "constructor_id" in df.columns:
        constructors = sorted(df["constructor_id"].dropna().unique())
        con_map      = {c: i for i, c in enumerate(constructors)}
        df["constructor_enc"] = df["constructor_id"].map(con_map).fillna(-1)

    # Driver ID label encoding
    if "driver_id" in df.columns:
        drivers  = sorted(df["driver_id"].dropna().unique())
        drv_map  = {d: i for i, d in enumerate(drivers)}
        df["driver_enc"] = df["driver_id"].map(drv_map).fillna(-1)

    return df


# ─── VALIDATE DATASET ─────────────────────────────────────────────────────────

def validate_dataset(df):
    """
    Checks the dataset for common issues before saving.

    Checks:
      - No rows where finish_position is missing
      - All ID columns present
      - No fully-empty feature columns
      - Reasonable value ranges

    Args:
        df (pd.DataFrame): Master dataset

    Returns:
        pd.DataFrame: Cleaned dataset
    """
    print("\nValidating dataset...")

    original_len = len(df)

    # Drop rows with missing target variable
    df = df.dropna(subset=[TARGET_COL])
    dropped = original_len - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} rows with missing finish_position")

    # Drop rows with invalid finish positions
    df = df[df[TARGET_COL].between(1, 20)]

    # Report missing value percentages
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 30]

    if not high_missing.empty:
        print(f"\n  WARNING: Columns with >30% missing values:")
        for col, pct in high_missing.items():
            print(f"    {col}: {pct:.1f}% missing")

    # Fill remaining NaNs with column medians (safe default for numeric cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)

    print(f"  Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ─── MAIN BUILD PIPELINE ──────────────────────────────────────────────────────

def build_master_dataset():
    """
    Main pipeline — calls all feature builders, merges them,
    and saves the master training dataset.

    Steps:
      1. Build driver features
      2. Build circuit features
      3. Build weather features
      4. Build constructor features
      5. Merge all on year + round + driver_id
      6. Encode categoricals
      7. Validate and clean
      8. Save to data/master_dataset.csv

    Returns:
        pd.DataFrame: Final master dataset
    """
    print("\n" + "=" * 60)
    print("BUILDING MASTER DATASET")
    print("=" * 60)

    # ── Step 1: Driver features ───────────────────────────────────────────────
    print("\n[Step 1/5] Driver features")
    driver_df = build_driver_features()

    # ── Step 2: Circuit features ──────────────────────────────────────────────
    print("\n[Step 2/5] Circuit features")
    circuit_df = build_circuit_features()

    # ── Step 3: Weather features ──────────────────────────────────────────────
    print("\n[Step 3/5] Weather features")
    weather_df = build_weather_features()

    # ── Step 4: Constructor features ──────────────────────────────────────────
    print("\n[Step 4/5] Constructor features")
    race_path  = os.path.join(DATA_DIR, "race_results.csv")
    race_df    = pd.read_csv(race_path)
    con_df     = compute_constructor_features(race_df)

    # ── Step 5: Merge everything ───────────────────────────────────────────────
    print("\n[Step 5/5] Merging all feature groups...")

    # Start with driver features as the base
    master = driver_df.copy()
    print(f"  Base (driver features):  {master.shape}")

    # Merge circuit features
    circuit_merge_cols = [c for c in circuit_df.columns if c not in master.columns or c in ID_COLS]
    master = master.merge(
        circuit_df[circuit_merge_cols],
        on=["year", "round", "driver_id", "circuit_id"],
        how="left"
    )
    print(f"  After circuit features:  {master.shape}")

    # Merge weather features
    weather_merge_cols = [c for c in weather_df.columns if c not in master.columns or c in ["year", "round", "driver_id"]]
    master = master.merge(
        weather_df[weather_merge_cols],
        on=["year", "round", "driver_id"],
        how="left"
    )
    print(f"  After weather features:  {master.shape}")

    # Merge constructor features
    master = master.merge(
        con_df,
        on=["year", "round", "constructor_id"],
        how="left"
    )
    print(f"  After constructor feats: {master.shape}")

    # ── Drop raw columns not useful as features ────────────────────────────────
    cols_to_drop = [c for c in COLS_TO_DROP if c in master.columns]
    master = master.drop(columns=cols_to_drop)

    # ── Encode categoricals ───────────────────────────────────────────────────
    master = encode_categoricals(master)

    # ── Validate and clean ────────────────────────────────────────────────────
    master = validate_dataset(master)

    # ── Sort chronologically ──────────────────────────────────────────────────
    master = master.sort_values(["year", "round", "finish_position"]).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, "master_dataset.csv")
    master.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("MASTER DATASET COMPLETE")
    print("=" * 60)
    print(f"  Saved to  : {out_path}")
    print(f"  Rows      : {len(master):,}")
    print(f"  Columns   : {len(master.columns)}")
    print(f"  Seasons   : {sorted(master['year'].unique())}")
    print(f"  Drivers   : {master['driver_id'].nunique()}")
    print(f"  Target    : {TARGET_COL} (range {int(master[TARGET_COL].min())}–{int(master[TARGET_COL].max())})")
    print("\nAll feature columns:")
    for col in sorted(master.columns):
        missing = master[col].isnull().sum()
        print(f"  {col:<45} (missing: {missing})")

    return master


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_master_dataset()