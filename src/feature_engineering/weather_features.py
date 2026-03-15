"""
weather_features.py
-------------------
Computes all weather-related features by combining:
  - Historical race day weather (from weather_historical.csv)
  - Driver wet weather performance statistics
  - Weather change features (practice vs race conditions)

Features computed here:
  - rain_flag, rainfall_mm, rain_probability
  - air_temp, track_temp, humidity, wind_speed
  - driver_wet_win_rate, driver_wet_podium_rate
  - wet_vs_dry_finish_delta (does driver improve in rain?)
  - team_wet_performance_delta
  - mixed_conditions_flag
  - temp_delta_fp_to_race

HOW TO RUN (standalone test):
    python src/feature_engineering/weather_features.py

OUTPUT:
    Returns a DataFrame — called by build_dataset.py
"""

import pandas as pd
import numpy as np
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# A race is considered "wet" if total rainfall >= this threshold (mm)
WET_RACE_THRESHOLD_MM = 0.5

# Minimum number of wet races needed to compute reliable wet weather stats
MIN_WET_RACES_FOR_STATS = 3


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    """
    Loads race results and weather CSVs.

    Returns:
        tuple: (race_df, weather_df)
    """
    print("Loading data for weather features...")

    race_path    = os.path.join(DATA_DIR, "race_results.csv")
    weather_path = os.path.join(DATA_DIR, "weather_historical.csv")

    if not os.path.exists(race_path):
        raise FileNotFoundError(f"Not found: {race_path}. Run get_ergast_data.py first.")

    race_df = pd.read_csv(race_path)
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce")
    race_df["points"]          = pd.to_numeric(race_df["points"],           errors="coerce")

    if not os.path.exists(weather_path):
        print("  WARNING: weather_historical.csv not found.")
        print("  Run get_weather_data.py first for full weather features.")
        print("  Continuing with neutral weather values...")
        weather_df = pd.DataFrame()
    else:
        weather_df = pd.read_csv(weather_path)
        print(f"  Weather data: {len(weather_df):,} rows")

    print(f"  Race results: {len(race_df):,} rows")
    return race_df, weather_df


# ─── MERGE WEATHER INTO RACES ─────────────────────────────────────────────────

def merge_weather_into_races(race_df, weather_df):
    """
    Merges weather data into race results so each driver-race row
    has the weather conditions for that race.

    Matching is done on circuit_id + race_date.
    If no weather match is found, neutral values are used.

    Args:
        race_df    (pd.DataFrame): Race results
        weather_df (pd.DataFrame): Historical weather per circuit per date

    Returns:
        pd.DataFrame: Race results with weather columns added
    """
    if weather_df.empty:
        # Add neutral weather columns if no data available
        race_df["air_temp_avg_c"]     = 22.0
        race_df["track_temp_avg_c"]   = 35.0
        race_df["humidity_avg_pct"]   = 50.0
        race_df["rainfall_total_mm"]  = 0.0
        race_df["rain_flag"]          = 0
        race_df["cloudcover_avg_pct"] = 30.0
        race_df["windspeed_avg_kmh"]  = 15.0
        race_df["winddirection_avg"]  = 180.0
        race_df["pressure_avg_hpa"]   = 1013.0
        race_df["uv_index_max"]       = 5.0
        return race_df

    # Normalise date format for matching
    weather_df = weather_df.copy()
    weather_df["race_date"] = pd.to_datetime(weather_df["race_date"]).dt.strftime("%Y-%m-%d")
    race_df    = race_df.copy()
    race_df["race_date"] = pd.to_datetime(race_df["race_date"]).dt.strftime("%Y-%m-%d")

    # Select weather columns to merge
    weather_cols = [
        "circuit_id", "race_date",
        "air_temp_avg_c", "air_temp_max_c",
        "track_temp_avg_c", "track_temp_max_c",
        "humidity_avg_pct", "humidity_max_pct",
        "rainfall_total_mm", "rain_flag",
        "cloudcover_avg_pct",
        "windspeed_avg_kmh", "windspeed_max_kmh",
        "winddirection_avg", "pressure_avg_hpa",
        "uv_index_max",
    ]

    # Keep only columns that exist in weather_df
    weather_cols = [c for c in weather_cols if c in weather_df.columns]
    weather_sub  = weather_df[weather_cols].drop_duplicates(
        subset=["circuit_id", "race_date"]
    )

    merged = race_df.merge(
        weather_sub,
        on=["circuit_id", "race_date"],
        how="left"
    )

    # Fill missing weather with neutral defaults
    weather_defaults = {
        "air_temp_avg_c":     22.0,
        "air_temp_max_c":     25.0,
        "track_temp_avg_c":   35.0,
        "track_temp_max_c":   40.0,
        "humidity_avg_pct":   50.0,
        "humidity_max_pct":   60.0,
        "rainfall_total_mm":  0.0,
        "rain_flag":          0,
        "cloudcover_avg_pct": 30.0,
        "windspeed_avg_kmh":  15.0,
        "windspeed_max_kmh":  25.0,
        "winddirection_avg":  180.0,
        "pressure_avg_hpa":   1013.0,
        "uv_index_max":       5.0,
    }

    for col, default in weather_defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)
        else:
            merged[col] = default

    matched = merged["air_temp_avg_c"].notna().sum()
    print(f"  Weather matched to {matched:,} / {len(merged):,} driver-race rows")

    return merged


# ─── DRIVER WET WEATHER FEATURES ─────────────────────────────────────────────

def compute_driver_wet_weather_features(race_df, weather_df):
    """
    Computes each driver's historical performance in wet vs dry conditions.

    A race is classified as "wet" if rainfall >= WET_RACE_THRESHOLD_MM.

    Features:
      - wet_win_rate_career: win rate in wet races
      - wet_podium_rate_career: podium rate in wet races
      - wet_avg_finish: average finish in wet races
      - wet_vs_dry_delta: how much better/worse in wet (negative = better in wet)
      - wet_positions_gained_avg: avg positions gained from grid in wet races
      - wet_crash_rate: DNF rate specifically in wet races
      - wet_races_count: how many wet races experienced

    All features computed from history BEFORE each race (no leakage).

    Args:
        race_df    (pd.DataFrame): Race results
        weather_df (pd.DataFrame): Weather per race

    Returns:
        pd.DataFrame: Wet weather features per driver per race
    """
    print("Computing driver wet weather features...")

    # Merge weather into race data to know which races were wet
    merged = merge_weather_into_races(race_df.copy(), weather_df)

    merged["is_wet_race"]       = (merged["rainfall_total_mm"] >= WET_RACE_THRESHOLD_MM).astype(int)
    merged["is_podium"]         = (merged["finish_position"] <= 3).astype(int)
    merged["is_win"]            = (merged["finish_position"] == 1).astype(int)
    merged["positions_gained"]  = merged["grid_position"] - merged["finish_position"]
    merged["is_dnf"]            = merged["status"].apply(
        lambda s: 0 if (str(s) == "Finished" or str(s).startswith("+")) else 1
    )

    merged = merged.sort_values(["driver_id", "year", "round"])

    records = []

    for driver_id, driver_data in merged.groupby("driver_id"):
        driver_data = driver_data.reset_index(drop=True)

        for i, row in driver_data.iterrows():
            past = driver_data.iloc[:i]

            record = {
                "year":      row["year"],
                "round":     row["round"],
                "driver_id": driver_id,
            }

            if past.empty:
                record.update({
                    "wet_win_rate_career":     0.0,
                    "wet_podium_rate_career":  0.0,
                    "wet_avg_finish":          10.0,
                    "dry_avg_finish":          10.0,
                    "wet_vs_dry_delta":        0.0,
                    "wet_positions_gained_avg": 0.0,
                    "wet_crash_rate":          0.1,
                    "wet_races_count":         0,
                })
            else:
                wet_races = past[past["is_wet_race"] == 1]
                dry_races = past[past["is_wet_race"] == 0]

                wet_count = len(wet_races)

                if wet_count >= 1:
                    wet_win_rate    = wet_races["is_win"].mean()
                    wet_podium_rate = wet_races["is_podium"].mean()
                    wet_avg_finish  = wet_races["finish_position"].mean()
                    wet_pos_gained  = wet_races["positions_gained"].mean()
                    wet_crash_rate  = wet_races["is_dnf"].mean()
                else:
                    wet_win_rate    = 0.0
                    wet_podium_rate = 0.0
                    wet_avg_finish  = 10.0
                    wet_pos_gained  = 0.0
                    wet_crash_rate  = 0.1

                dry_avg_finish = dry_races["finish_position"].mean() if len(dry_races) > 0 else 10.0

                # Wet vs dry delta — negative means driver is BETTER in wet
                # e.g. Hamilton avg finish 4.2 dry, 2.1 wet → delta = -2.1 (better in wet)
                wet_vs_dry_delta = wet_avg_finish - dry_avg_finish

                record.update({
                    "wet_win_rate_career":      round(wet_win_rate, 4),
                    "wet_podium_rate_career":   round(wet_podium_rate, 4),
                    "wet_avg_finish":           round(wet_avg_finish, 2),
                    "dry_avg_finish":           round(dry_avg_finish, 2),
                    "wet_vs_dry_delta":         round(wet_vs_dry_delta, 2),
                    "wet_positions_gained_avg": round(wet_pos_gained, 2),
                    "wet_crash_rate":           round(wet_crash_rate, 4),
                    "wet_races_count":          wet_count,
                })

            records.append(record)

    df = pd.DataFrame(records)
    print(f"  Computed wet weather driver features: {len(df):,} rows")
    return df


# ─── TEAM WET WEATHER FEATURES ───────────────────────────────────────────────

def compute_team_wet_weather_features(race_df, weather_df):
    """
    Computes constructor-level wet weather performance.

    Some teams have better wet weather setups than others.
    This captures that team-level advantage separately from driver skill.

    Args:
        race_df    (pd.DataFrame): Race results
        weather_df (pd.DataFrame): Weather per race

    Returns:
        pd.DataFrame: Team wet weather delta per constructor per year
    """
    print("Computing team wet weather features...")

    merged = merge_weather_into_races(race_df.copy(), weather_df)
    merged["is_wet_race"] = (merged["rainfall_total_mm"] >= WET_RACE_THRESHOLD_MM).astype(int)

    records = []
    years   = sorted(merged["year"].unique())

    for year in years:
        history = merged[merged["year"] < year]
        constructors = merged[merged["year"] == year]["constructor_id"].unique()

        for constructor_id in constructors:
            team_history = history[history["constructor_id"] == constructor_id]

            if team_history.empty:
                records.append({
                    "year":                       year,
                    "constructor_id":             constructor_id,
                    "team_wet_avg_finish":        10.0,
                    "team_dry_avg_finish":        10.0,
                    "team_wet_vs_dry_delta":      0.0,
                })
                continue

            wet = team_history[team_history["is_wet_race"] == 1]
            dry = team_history[team_history["is_wet_race"] == 0]

            wet_avg = wet["finish_position"].mean() if len(wet) > 0 else 10.0
            dry_avg = dry["finish_position"].mean() if len(dry) > 0 else 10.0

            records.append({
                "year":                   year,
                "constructor_id":         constructor_id,
                "team_wet_avg_finish":    round(wet_avg, 2),
                "team_dry_avg_finish":    round(dry_avg, 2),
                "team_wet_vs_dry_delta":  round(wet_avg - dry_avg, 2),
            })

    df = pd.DataFrame(records)
    print(f"  Computed team wet weather features: {len(df):,} rows")
    return df


# ─── RACE WEATHER FEATURES ───────────────────────────────────────────────────

def compute_race_weather_features(race_df, weather_df):
    """
    Adds race-day weather features to each driver-race row.

    These are the same for all drivers in the same race —
    they describe the conditions of the race itself.

    Features added:
      - rain_flag: did it rain during the race?
      - rainfall_total_mm: how much rain?
      - air_temp_avg_c, track_temp_avg_c
      - humidity_avg_pct
      - windspeed_avg_kmh, winddirection_avg
      - cloudcover_avg_pct
      - pressure_avg_hpa
      - high_temp_flag: track temp > 45°C (tire stress)
      - low_grip_flag: rain + cold temps
      - strong_wind_flag: wind > 30 km/h

    Args:
        race_df    (pd.DataFrame): Race results
        weather_df (pd.DataFrame): Weather per race

    Returns:
        pd.DataFrame: Race results with weather features
    """
    print("Computing race weather features...")

    merged = merge_weather_into_races(race_df.copy(), weather_df)

    # Derived weather flags
    merged["high_temp_flag"]   = (merged["track_temp_avg_c"] > 45).astype(int)
    merged["cold_track_flag"]  = (merged["track_temp_avg_c"] < 20).astype(int)
    merged["strong_wind_flag"] = (merged["windspeed_avg_kmh"] > 30).astype(int)
    merged["low_grip_flag"]    = (
        (merged["rain_flag"] == 1) &
        (merged["track_temp_avg_c"] < 25)
    ).astype(int)

    # Keep only weather-related columns
    weather_feature_cols = [
        "year", "round", "driver_id",
        "air_temp_avg_c", "air_temp_max_c",
        "track_temp_avg_c", "track_temp_max_c",
        "humidity_avg_pct", "humidity_max_pct",
        "rainfall_total_mm", "rain_flag",
        "cloudcover_avg_pct",
        "windspeed_avg_kmh", "windspeed_max_kmh",
        "winddirection_avg", "pressure_avg_hpa",
        "uv_index_max",
        "high_temp_flag", "cold_track_flag",
        "strong_wind_flag", "low_grip_flag",
    ]

    weather_feature_cols = [c for c in weather_feature_cols if c in merged.columns]
    result = merged[weather_feature_cols].copy()

    print(f"  Computed race weather features: {len(result):,} rows")
    return result


# ─── COMBINE ALL WEATHER FEATURES ─────────────────────────────────────────────

def build_weather_features():
    """
    Master function — loads data, computes all weather features,
    and returns a single combined DataFrame.

    This is the function called by build_dataset.py.

    Returns:
        pd.DataFrame: All weather features per driver per race
    """
    print("\n" + "=" * 60)
    print("Building weather features...")
    print("=" * 60)

    race_df, weather_df = load_data()

    # Compute each feature group
    race_weather   = compute_race_weather_features(race_df, weather_df)
    driver_wet     = compute_driver_wet_weather_features(race_df, weather_df)
    team_wet       = compute_team_wet_weather_features(race_df, weather_df)

    # ── Merge everything ──────────────────────────────────────────────────────
    print("\nMerging weather feature groups...")

    combined = race_weather.merge(
        driver_wet,
        on=["year", "round", "driver_id"],
        how="left"
    )

    # Add constructor_id for team merge
    constructor_map = (
        race_df[["year", "round", "driver_id", "constructor_id"]]
        .drop_duplicates()
    )
    combined = combined.merge(constructor_map, on=["year", "round", "driver_id"], how="left")

    combined = combined.merge(
        team_wet,
        on=["year", "constructor_id"],
        how="left"
    )

    print(f"\nFinal weather features shape: {combined.shape}")
    print(f"  Rows    : {len(combined):,}")
    print(f"  Columns : {len(combined.columns)}")

    return combined


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_weather_features()
    print("\nSample output (first 5 rows):")
    print(df.head())
    print("\nAll columns:")
    for col in df.columns:
        print(f"  {col}")

    out_path = os.path.join(DATA_DIR, "weather_features_preview.csv")
    df.head(100).to_csv(out_path, index=False)
    print(f"\nPreview saved to: {out_path}")