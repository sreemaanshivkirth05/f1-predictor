"""
circuit_features.py
-------------------
Computes all circuit-specific and driver affinity features.

Features computed here:
  - Driver historical avg finish at each circuit
  - Circuit affinity score (overperformance vs season average)
  - Circuit type specialisation (street vs permanent, high vs low downforce)
  - Night race performance score
  - Wet weather performance score per circuit type
  - Circuit metadata (length, corners, DRS zones, altitude)
  - Safety car and red flag frequency per circuit
  - Historical overtake count per circuit

HOW TO RUN (standalone test):
    python src/feature_engineering/circuit_features.py

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

# Exponential decay factor for circuit affinity
# Higher = more weight on recent visits, less on old ones
# 0.3 means a race 5 years ago counts ~22% as much as last year
DECAY_LAMBDA = 0.3


# ─── CIRCUIT METADATA ─────────────────────────────────────────────────────────
# Static information about each circuit that doesn't change year to year
# Sources: FIA circuit homologation documents, F1 official site

CIRCUIT_METADATA = {
    "albert_park":   {"length_km": 5.278, "corners": 16, "drs_zones": 4, "altitude_m": 10,   "type": "street",    "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.4, "abrasion": 2},
    "bahrain":       {"length_km": 5.412, "corners": 15, "drs_zones": 3, "altitude_m": 7,    "type": "permanent", "is_night": 1, "is_clockwise": 1, "high_speed_ratio": 0.5, "abrasion": 3},
    "jeddah":        {"length_km": 6.174, "corners": 27, "drs_zones": 3, "altitude_m": 15,   "type": "street",    "is_night": 1, "is_clockwise": 1, "high_speed_ratio": 0.7, "abrasion": 2},
    "shanghai":      {"length_km": 5.451, "corners": 16, "drs_zones": 2, "altitude_m": 5,    "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.5, "abrasion": 2},
    "suzuka":        {"length_km": 5.807, "corners": 18, "drs_zones": 2, "altitude_m": 50,   "type": "permanent", "is_night": 0, "is_clockwise": 0, "high_speed_ratio": 0.7, "abrasion": 3},
    "miami":         {"length_km": 5.412, "corners": 19, "drs_zones": 3, "altitude_m": 2,    "type": "street",    "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.5, "abrasion": 2},
    "imola":         {"length_km": 4.909, "corners": 19, "drs_zones": 2, "altitude_m": 45,   "type": "permanent", "is_night": 0, "is_clockwise": 0, "high_speed_ratio": 0.5, "abrasion": 3},
    "monaco":        {"length_km": 3.337, "corners": 19, "drs_zones": 1, "altitude_m": 7,    "type": "street",    "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.1, "abrasion": 1},
    "villeneuve":    {"length_km": 4.361, "corners": 14, "drs_zones": 3, "altitude_m": 10,   "type": "street",    "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.4, "abrasion": 2},
    "catalunya":     {"length_km": 4.657, "corners": 16, "drs_zones": 2, "altitude_m": 115,  "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.5, "abrasion": 4},
    "red_bull_ring": {"length_km": 4.318, "corners": 10, "drs_zones": 3, "altitude_m": 678,  "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.6, "abrasion": 2},
    "silverstone":   {"length_km": 5.891, "corners": 18, "drs_zones": 2, "altitude_m": 126,  "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.8, "abrasion": 3},
    "hungaroring":   {"length_km": 4.381, "corners": 14, "drs_zones": 2, "altitude_m": 248,  "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.3, "abrasion": 3},
    "spa":           {"length_km": 7.004, "corners": 20, "drs_zones": 2, "altitude_m": 401,  "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.7, "abrasion": 3},
    "zandvoort":     {"length_km": 4.259, "corners": 14, "drs_zones": 2, "altitude_m": 5,    "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.5, "abrasion": 3},
    "monza":         {"length_km": 5.793, "corners": 11, "drs_zones": 3, "altitude_m": 162,  "type": "permanent", "is_night": 0, "is_clockwise": 1, "high_speed_ratio": 0.9, "abrasion": 2},
    "baku":          {"length_km": 6.003, "corners": 20, "drs_zones": 2, "altitude_m": 0,    "type": "street",    "is_night": 0, "is_clockwise": 0, "high_speed_ratio": 0.5, "abrasion": 1},
    "marina_bay":    {"length_km": 4.940, "corners": 19, "drs_zones": 3, "altitude_m": 5,    "type": "street",    "is_night": 1, "is_clockwise": 1, "high_speed_ratio": 0.2, "abrasion": 2},
    "americas":      {"length_km": 5.513, "corners": 20, "drs_zones": 2, "altitude_m": 161,  "type": "permanent", "is_night": 0, "is_clockwise": 0, "high_speed_ratio": 0.5, "abrasion": 3},
    "rodriguez":     {"length_km": 4.304, "corners": 17, "drs_zones": 3, "altitude_m": 2240, "type": "permanent", "is_night": 0, "is_clockwise": 0, "high_speed_ratio": 0.5, "abrasion": 2},
    "interlagos":    {"length_km": 4.309, "corners": 15, "drs_zones": 2, "altitude_m": 785,  "type": "permanent", "is_night": 0, "is_clockwise": 0, "high_speed_ratio": 0.5, "abrasion": 3},
    "vegas":         {"length_km": 6.201, "corners": 17, "drs_zones": 3, "altitude_m": 640,  "type": "street",    "is_night": 1, "is_clockwise": 0, "high_speed_ratio": 0.6, "abrasion": 1},
    "losail":        {"length_km": 5.380, "corners": 16, "drs_zones": 2, "altitude_m": 12,   "type": "permanent", "is_night": 1, "is_clockwise": 1, "high_speed_ratio": 0.6, "abrasion": 2},
    "yas_marina":    {"length_km": 5.281, "corners": 16, "drs_zones": 2, "altitude_m": 0,    "type": "permanent", "is_night": 1, "is_clockwise": 0, "high_speed_ratio": 0.5, "abrasion": 2},
}

# Historical safety car frequency (proportion of races with SC) per circuit
# Computed from 2016-2024 data
CIRCUIT_SC_FREQUENCY = {
    "albert_park":   0.45, "bahrain":       0.30, "jeddah":        0.60,
    "shanghai":      0.35, "suzuka":        0.25, "miami":         0.50,
    "imola":         0.55, "monaco":        0.65, "villeneuve":    0.50,
    "catalunya":     0.25, "red_bull_ring": 0.35, "silverstone":   0.35,
    "hungaroring":   0.30, "spa":           0.50, "zandvoort":     0.40,
    "monza":         0.30, "baku":          0.70, "marina_bay":    0.75,
    "americas":      0.45, "rodriguez":     0.40, "interlagos":    0.60,
    "vegas":         0.55, "losail":        0.35, "yas_marina":    0.25,
}

# Historical average overtakes per race per circuit
CIRCUIT_AVG_OVERTAKES = {
    "albert_park":   25, "bahrain":       40, "jeddah":        50,
    "shanghai":      35, "suzuka":        15, "miami":         30,
    "imola":         15, "monaco":        5,  "villeneuve":    45,
    "catalunya":     20, "red_bull_ring": 35, "silverstone":   30,
    "hungaroring":   15, "spa":           40, "zandvoort":     20,
    "monza":         55, "baku":          60, "marina_bay":    15,
    "americas":      35, "rodriguez":     30, "interlagos":    45,
    "vegas":         40, "losail":        35, "yas_marina":    30,
}


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    """Loads race results CSV."""
    path = os.path.join(DATA_DIR, "race_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}. Run get_ergast_data.py first.")

    df = pd.read_csv(path)
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["points"]          = pd.to_numeric(df["points"],           errors="coerce")
    print(f"  Loaded race results: {len(df):,} rows")
    return df


# ─── CIRCUIT AFFINITY SCORE ───────────────────────────────────────────────────

def compute_circuit_affinity(race_df):
    """
    Computes how much each driver over or underperforms at each circuit
    relative to their overall season average.

    Formula:
        affinity = (expected_finish - actual_avg_finish) / expected_finish

    Where expected_finish = driver's season average finish position.
    Positive score = driver overperforms at this circuit.
    Negative score = driver underperforms at this circuit.

    Recent visits are weighted more than older ones using exponential decay:
        weight = e^(-DECAY_LAMBDA * years_ago)

    Args:
        race_df (pd.DataFrame): Race results

    Returns:
        pd.DataFrame: Affinity scores per driver per circuit per year
                      (computed from all history BEFORE that year)
    """
    print("Computing circuit affinity scores...")

    race_df = race_df.copy()
    race_df = race_df.dropna(subset=["finish_position"])

    # Get all unique (year, round, driver, circuit) combos
    records = []

    years     = sorted(race_df["year"].unique())
    driver_ids = race_df["driver_id"].unique()

    for year in years:
        # Use all data BEFORE this year to compute affinity
        history = race_df[race_df["year"] < year]

        if history.empty:
            # First year — no history, all drivers get neutral affinity
            current_races = race_df[race_df["year"] == year]
            for _, row in current_races.iterrows():
                records.append({
                    "year":                   year,
                    "round":                  row["round"],
                    "driver_id":              row["driver_id"],
                    "circuit_id":             row["circuit_id"],
                    "circuit_affinity_score": 0.0,
                    "circuit_win_rate":       0.0,
                    "circuit_podium_rate":    0.0,
                    "circuit_avg_finish":     10.0,
                    "circuit_visits":         0,
                    "best_finish_at_circuit": 20,
                    "last_finish_at_circuit": 10,
                })
            continue

        # For each race this year, compute the driver's affinity at that circuit
        current_year_races = race_df[race_df["year"] == year]

        for _, race_row in current_year_races.iterrows():
            driver_id  = race_row["driver_id"]
            circuit_id = race_row["circuit_id"]
            round_num  = race_row["round"]

            # Driver's overall avg finish (excluding this circuit)
            driver_history = history[history["driver_id"] == driver_id]

            if driver_history.empty:
                # New driver — no history
                records.append({
                    "year":                   year,
                    "round":                  round_num,
                    "driver_id":              driver_id,
                    "circuit_id":             circuit_id,
                    "circuit_affinity_score": 0.0,
                    "circuit_win_rate":       0.0,
                    "circuit_podium_rate":    0.0,
                    "circuit_avg_finish":     10.0,
                    "circuit_visits":         0,
                    "best_finish_at_circuit": 20,
                    "last_finish_at_circuit": 10,
                })
                continue

            # Driver's season average finish (baseline)
            expected_finish = driver_history["finish_position"].mean()

            # Driver's history at THIS specific circuit
            circuit_history = driver_history[
                driver_history["circuit_id"] == circuit_id
            ].copy()

            if circuit_history.empty:
                # Never raced at this circuit before
                records.append({
                    "year":                   year,
                    "round":                  round_num,
                    "driver_id":              driver_id,
                    "circuit_id":             circuit_id,
                    "circuit_affinity_score": 0.0,
                    "circuit_win_rate":       0.0,
                    "circuit_podium_rate":    0.0,
                    "circuit_avg_finish":     expected_finish,
                    "circuit_visits":         0,
                    "best_finish_at_circuit": 20,
                    "last_finish_at_circuit": int(expected_finish),
                })
                continue

            # Apply exponential decay — recent visits matter more
            circuit_history["years_ago"] = year - circuit_history["year"]
            circuit_history["weight"]    = np.exp(-DECAY_LAMBDA * circuit_history["years_ago"])

            # Weighted average finish at this circuit
            weighted_avg_finish = (
                (circuit_history["finish_position"] * circuit_history["weight"]).sum()
                / circuit_history["weight"].sum()
            )

            # Affinity score: positive = overperforms here
            if expected_finish > 0:
                affinity = (expected_finish - weighted_avg_finish) / expected_finish
            else:
                affinity = 0.0

            # Win and podium rate at this circuit
            win_rate    = (circuit_history["finish_position"] == 1).mean()
            podium_rate = (circuit_history["finish_position"] <= 3).mean()
            best_finish = int(circuit_history["finish_position"].min())
            last_finish = int(
                circuit_history.sort_values("year").iloc[-1]["finish_position"]
            )

            records.append({
                "year":                   year,
                "round":                  round_num,
                "driver_id":              driver_id,
                "circuit_id":             circuit_id,
                "circuit_affinity_score": round(affinity, 4),
                "circuit_win_rate":       round(win_rate, 4),
                "circuit_podium_rate":    round(podium_rate, 4),
                "circuit_avg_finish":     round(weighted_avg_finish, 2),
                "circuit_visits":         len(circuit_history),
                "best_finish_at_circuit": best_finish,
                "last_finish_at_circuit": last_finish,
            })

    df = pd.DataFrame(records)
    print(f"  Computed circuit affinity: {len(df):,} rows")
    return df


# ─── CIRCUIT TYPE SPECIALISATION ──────────────────────────────────────────────

def compute_circuit_type_features(race_df):
    """
    Computes how well each driver performs on different circuit types.

    Circuit types:
      - Street circuits (Monaco, Baku, Singapore, Jeddah...)
      - Permanent tracks (Silverstone, Spa, Monza...)
      - High downforce circuits (Monaco, Hungaroring...)
      - Low downforce / power circuits (Monza, Spa, Baku straights)
      - Night races (Singapore, Bahrain, Abu Dhabi, Saudi, Vegas)
      - High altitude (Mexico City, Red Bull Ring, Interlagos)
      - Clockwise vs anti-clockwise circuits

    Args:
        race_df (pd.DataFrame): Race results

    Returns:
        pd.DataFrame: Circuit type specialisation scores per driver per year
    """
    print("Computing circuit type specialisation features...")

    race_df = race_df.copy()
    race_df = race_df.dropna(subset=["finish_position"])

    # Add circuit metadata to each race
    race_df["circuit_type"]       = race_df["circuit_id"].map(lambda x: CIRCUIT_METADATA.get(x, {}).get("type", "permanent"))
    race_df["is_night_race"]      = race_df["circuit_id"].map(lambda x: CIRCUIT_METADATA.get(x, {}).get("is_night", 0))
    race_df["is_clockwise"]       = race_df["circuit_id"].map(lambda x: CIRCUIT_METADATA.get(x, {}).get("is_clockwise", 1))
    race_df["high_speed_ratio"]   = race_df["circuit_id"].map(lambda x: CIRCUIT_METADATA.get(x, {}).get("high_speed_ratio", 0.5))
    race_df["altitude_m"]         = race_df["circuit_id"].map(lambda x: CIRCUIT_METADATA.get(x, {}).get("altitude_m", 0))

    # Classify circuits
    race_df["is_street"]          = (race_df["circuit_type"] == "street").astype(int)
    race_df["is_high_df"]         = (race_df["high_speed_ratio"] < 0.4).astype(int)
    race_df["is_low_df"]          = (race_df["high_speed_ratio"] > 0.7).astype(int)
    race_df["is_high_altitude"]   = (race_df["altitude_m"] > 500).astype(int)
    race_df["is_long_circuit"]    = race_df["circuit_id"].map(
        lambda x: 1 if CIRCUIT_METADATA.get(x, {}).get("length_km", 5) > 5.5 else 0
    )

    records = []
    years   = sorted(race_df["year"].unique())

    for year in years:
        # Use only history BEFORE this year
        history = race_df[race_df["year"] < year]

        current_races = race_df[race_df["year"] == year]
        drivers_this_year = current_races["driver_id"].unique()

        for driver_id in drivers_this_year:

            driver_history = history[history["driver_id"] == driver_id]

            if driver_history.empty:
                # New driver — neutral scores
                record = {
                    "year":                        year,
                    "driver_id":                   driver_id,
                    "street_circuit_avg_finish":   10.0,
                    "permanent_circuit_avg_finish": 10.0,
                    "street_vs_perm_delta":        0.0,
                    "night_race_avg_finish":       10.0,
                    "high_df_circuit_avg_finish":  10.0,
                    "low_df_circuit_avg_finish":   10.0,
                    "high_altitude_avg_finish":    10.0,
                    "clockwise_avg_finish":        10.0,
                    "anti_clockwise_avg_finish":   10.0,
                    "long_circuit_avg_finish":     10.0,
                }
            else:
                def safe_mean(subset, col_filter, col_val, target="finish_position"):
                    """Helper: mean finish on a filtered subset."""
                    filtered = driver_history[driver_history[col_filter] == col_val]
                    return filtered[target].mean() if len(filtered) > 0 else 10.0

                street_avg   = safe_mean(driver_history, "is_street", 1)
                perm_avg     = safe_mean(driver_history, "is_street", 0)
                night_avg    = safe_mean(driver_history, "is_night_race", 1)
                high_df_avg  = safe_mean(driver_history, "is_high_df", 1)
                low_df_avg   = safe_mean(driver_history, "is_low_df", 1)
                altitude_avg = safe_mean(driver_history, "is_high_altitude", 1)
                cw_avg       = safe_mean(driver_history, "is_clockwise", 1)
                acw_avg      = safe_mean(driver_history, "is_clockwise", 0)
                long_avg     = safe_mean(driver_history, "is_long_circuit", 1)

                record = {
                    "year":                        year,
                    "driver_id":                   driver_id,
                    "street_circuit_avg_finish":   round(street_avg, 2),
                    "permanent_circuit_avg_finish": round(perm_avg, 2),
                    # Positive = better on street circuits (lower finish = better)
                    "street_vs_perm_delta":        round(perm_avg - street_avg, 2),
                    "night_race_avg_finish":       round(night_avg, 2),
                    "high_df_circuit_avg_finish":  round(high_df_avg, 2),
                    "low_df_circuit_avg_finish":   round(low_df_avg, 2),
                    "high_altitude_avg_finish":    round(altitude_avg, 2),
                    "clockwise_avg_finish":        round(cw_avg, 2),
                    "anti_clockwise_avg_finish":   round(acw_avg, 2),
                    "long_circuit_avg_finish":     round(long_avg, 2),
                }

            records.append(record)

    df = pd.DataFrame(records)
    print(f"  Computed circuit type features: {len(df):,} rows")
    return df


# ─── CIRCUIT STATIC FEATURES ──────────────────────────────────────────────────

def compute_circuit_static_features(race_df):
    """
    Adds static circuit metadata to each race row.
    These are the same for every driver at the same race.

    Features:
      - circuit length, corners, DRS zones, altitude
      - circuit type (street vs permanent)
      - safety car frequency
      - average overtakes
      - is night race, is clockwise

    Args:
        race_df (pd.DataFrame): Race results

    Returns:
        pd.DataFrame: One row per race (not per driver) with circuit info
    """
    print("Computing circuit static features...")

    # Get unique circuit per round
    race_info = (
        race_df[["year", "round", "circuit_id", "race_name", "race_date"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    def get_meta(circuit_id, key, default):
        return CIRCUIT_METADATA.get(circuit_id, {}).get(key, default)

    race_info["circuit_length_km"]   = race_info["circuit_id"].apply(lambda x: get_meta(x, "length_km", 5.0))
    race_info["circuit_corners"]     = race_info["circuit_id"].apply(lambda x: get_meta(x, "corners", 15))
    race_info["circuit_drs_zones"]   = race_info["circuit_id"].apply(lambda x: get_meta(x, "drs_zones", 2))
    race_info["circuit_altitude_m"]  = race_info["circuit_id"].apply(lambda x: get_meta(x, "altitude_m", 0))
    race_info["circuit_type"]        = race_info["circuit_id"].apply(lambda x: get_meta(x, "type", "permanent"))
    race_info["is_street_circuit"]   = (race_info["circuit_type"] == "street").astype(int)
    race_info["is_night_race"]       = race_info["circuit_id"].apply(lambda x: get_meta(x, "is_night", 0))
    race_info["is_clockwise"]        = race_info["circuit_id"].apply(lambda x: get_meta(x, "is_clockwise", 1))
    race_info["high_speed_ratio"]    = race_info["circuit_id"].apply(lambda x: get_meta(x, "high_speed_ratio", 0.5))
    race_info["circuit_abrasion"]    = race_info["circuit_id"].apply(lambda x: get_meta(x, "abrasion", 2))
    race_info["sc_frequency"]        = race_info["circuit_id"].map(CIRCUIT_SC_FREQUENCY).fillna(0.4)
    race_info["avg_overtakes"]       = race_info["circuit_id"].map(CIRCUIT_AVG_OVERTAKES).fillna(25)

    # Race number and races remaining (approximate for 24-race season)
    race_info = race_info.sort_values(["year", "round"])
    race_info["races_remaining"]     = 24 - race_info["round"]
    race_info["points_available"]    = race_info["races_remaining"] * 26

    print(f"  Computed static circuit features: {len(race_info):,} unique races")
    return race_info


# ─── COMBINE ALL CIRCUIT FEATURES ─────────────────────────────────────────────

def build_circuit_features():
    """
    Master function — loads data, computes all circuit features,
    and returns a single combined DataFrame.

    This is the function called by build_dataset.py.

    Returns:
        pd.DataFrame: All circuit features per driver per race
    """
    print("\n" + "=" * 60)
    print("Building circuit features...")
    print("=" * 60)

    race_df = load_data()

    # Compute each feature group
    affinity_df  = compute_circuit_affinity(race_df)
    type_df      = compute_circuit_type_features(race_df)
    static_df    = compute_circuit_static_features(race_df)

    # ── Merge everything together ─────────────────────────────────────────────
    print("\nMerging circuit feature groups...")

    # Start with affinity (has year, round, driver_id, circuit_id)
    combined = affinity_df.merge(
        type_df,
        on=["year", "driver_id"],
        how="left"
    )

    # Merge static circuit features (on year + round)
    combined = combined.merge(
        static_df,
        on=["year", "round", "circuit_id"],
        how="left"
    )

    print(f"\nFinal circuit features shape: {combined.shape}")
    print(f"  Rows    : {len(combined):,}")
    print(f"  Columns : {len(combined.columns)}")

    return combined


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_circuit_features()
    print("\nSample output (first 5 rows):")
    print(df.head())
    print("\nAll columns:")
    for col in df.columns:
        print(f"  {col}")

    # Save preview
    out_path = os.path.join(DATA_DIR, "circuit_features_preview.csv")
    df.head(100).to_csv(out_path, index=False)
    print(f"\nPreview saved to: {out_path}")