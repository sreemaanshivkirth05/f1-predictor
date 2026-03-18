"""
preseason_features.py
---------------------
Computes all pre-season features that are fixed before Race 1
and carry through the entire season.

Features:
  - Pre-season testing lap times and reliability
  - Testing long run pace vs field
  - Car launch date (early = more preparation time)
  - Regulation change magnitude (0-3 scale)
  - Power unit generation (new vs carry-over)
  - Driver team switch flag and rookie flag
  - Seasons with current team (familiarity)
  - Pre-season betting odds (championship market)
  - Prior season momentum (final 5 races)
  - F2 champion promotion flag

HOW TO RUN:
    python src/feature_engineering/preseason_features.py

OUTPUT:
    data/preseason_features.csv  — one row per driver per season
"""

import pandas as pd
import numpy as np
import os
import json

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# ─── STATIC SEASON DATA ───────────────────────────────────────────────────────
# This data is manually maintained each season
# Update before each new season starts

REGULATION_CHANGES = {
    2016: 1,   # New PU regulations
    2017: 2,   # Wider cars, more downforce
    2018: 0,   # Minor changes
    2019: 0,   # Minor changes
    2020: 0,   # Minor changes (COVID season)
    2021: 1,   # Aerodynamic changes
    2022: 3,   # Complete regulation overhaul (ground effect)
    2023: 1,   # Minor updates
    2024: 1,   # Minor updates
    2025: 1,   # Minor updates
    2026: 3,   # Complete overhaul (new PU + aero rules)
}

# Power unit suppliers per constructor per season
PU_SUPPLIERS = {
    2026: {
        "mercedes":         "Mercedes",
        "ferrari":          "Ferrari",
        "red_bull":         "Honda",
        "mclaren":          "Mercedes",
        "alpine":           "Renault",
        "aston_martin":     "Honda",
        "williams":         "Mercedes",
        "rb":               "Honda",
        "haas":             "Ferrari",
        "sauber":           "Audi",
        "cadillac":         "Ferrari",
    },
    2025: {
        "mercedes":         "Mercedes",
        "ferrari":          "Ferrari",
        "red_bull":         "Honda",
        "mclaren":          "Mercedes",
        "alpine":           "Renault",
        "aston_martin":     "Honda",
        "williams":         "Mercedes",
        "rb":               "Honda",
        "haas":             "Ferrari",
        "kick_sauber":      "Ferrari",
    },
}

# New power unit generation flags (1 = brand new PU spec this season)
NEW_PU_GENERATION = {
    2026: 1,   # Completely new hybrid regulations
    2025: 0,
    2024: 0,
    2023: 0,
    2022: 0,
    2021: 0,
    2020: 0,
    2019: 0,
    2018: 0,
    2017: 0,
    2016: 1,   # New PU regs
}

# Pre-season betting odds (championship win probability)
# Source: approximate pre-season market odds
# Update each season before Race 1
PRESEASON_ODDS = {
    2026: {
        "max_verstappen":  0.28,
        "norris":          0.18,
        "leclerc":         0.14,
        "russell":         0.12,
        "hamilton":        0.10,
        "piastri":         0.07,
        "antonelli":       0.04,
        "sainz":           0.03,
        "alonso":          0.01,
        "bearman":         0.005,
        "hulkenberg":      0.005,
        "gasly":           0.003,
        "albon":           0.003,
        "lawson":          0.003,
        "hadjar":          0.002,
        "colapinto":       0.002,
        "bortoleto":       0.001,
        "ocon":            0.001,
        "stroll":          0.001,
        "perez":           0.001,
    },
    2025: {
        "max_verstappen":  0.42,
        "norris":          0.22,
        "leclerc":         0.12,
        "russell":         0.08,
        "hamilton":        0.06,
        "piastri":         0.04,
        "sainz":           0.02,
        "alonso":          0.01,
        "antonelli":       0.01,
    },
    2024: {
        "max_verstappen":  0.55,
        "norris":          0.12,
        "leclerc":         0.10,
        "hamilton":        0.09,
        "sainz":           0.06,
        "russell":         0.04,
        "piastri":         0.03,
        "alonso":          0.01,
    },
}

# F2 champions promoted to F1 (gets a rookie bonus flag)
F2_CHAMPION_PROMOTIONS = {
    2026: ["bortoleto"],   # Gabriel Bortoleto F2 2024 champion
    2025: ["antonelli"],
    2024: [],
    2023: ["sargeant"],
    2022: ["zhou"],
}

# Car launch dates (day of year — earlier = more preparation)
# 365 = Dec 31, 1 = Jan 1. Typical range is 35-60 (Feb).
CAR_LAUNCH_DOY = {
    2026: {
        "mercedes": 42, "ferrari": 40, "red_bull": 45,
        "mclaren": 38,  "alpine": 50,  "aston_martin": 48,
        "williams": 52, "rb": 55,      "haas": 58,
        "sauber": 60,   "cadillac": 62,
    },
}


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    """Loads race results and FastF1 testing data."""
    race_path = os.path.join(DATA_DIR, "race_results.csv")
    if not os.path.exists(race_path):
        raise FileNotFoundError("race_results.csv not found")

    race_df = pd.read_csv(race_path)
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce")
    race_df["points"]          = pd.to_numeric(race_df["points"],           errors="coerce")

    # FastF1 testing data
    test_path = os.path.join(DATA_DIR, "fastf1_testing_laps.csv")
    test_df   = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()

    print(f"  Race results   : {len(race_df):,} rows")
    print(f"  Testing laps   : {len(test_df):,} rows")

    return race_df, test_df


# ─── TESTING FEATURES ─────────────────────────────────────────────────────────

def compute_testing_features(test_df, race_df):
    """
    Computes pre-season testing features from FastF1 testing sessions.

    Features:
      - fastest_test_lap_s: fastest lap in testing (seconds)
      - test_gap_to_fastest: gap to fastest car in testing
      - test_total_laps: total laps completed (reliability)
      - test_long_run_pace: avg pace on stints > 10 laps
      - test_short_run_pace: avg pace on hot laps (stints < 5 laps)
      - test_laps_vs_field: laps completed vs field average
      - test_reliability_score: 1 - (stoppages / total_days)

    Returns:
        pd.DataFrame: Testing features per driver per year
    """
    print("Computing pre-season testing features...")

    records = []
    years   = sorted(race_df["year"].unique())

    for year in years:
        drivers_this_year = race_df[
            race_df["year"] == year
        ]["driver_id"].unique()

        if test_df.empty or "year" not in test_df.columns:
            # No testing data — use neutral values
            for driver_id in drivers_this_year:
                records.append({
                    "year":                    year,
                    "driver_id":               driver_id,
                    "fastest_test_lap_s":      0.0,
                    "test_gap_to_fastest_s":   0.0,
                    "test_total_laps":         0,
                    "test_long_run_pace_s":    0.0,
                    "test_short_run_pace_s":   0.0,
                    "test_laps_vs_field_pct":  0.0,
                    "test_reliability_score":  0.5,
                    "test_data_available":     0,
                })
            continue

        year_test = test_df[test_df["year"] == year].copy()

        if year_test.empty:
            for driver_id in drivers_this_year:
                records.append({
                    "year": year, "driver_id": driver_id,
                    "fastest_test_lap_s": 0.0, "test_gap_to_fastest_s": 0.0,
                    "test_total_laps": 0, "test_long_run_pace_s": 0.0,
                    "test_short_run_pace_s": 0.0, "test_laps_vs_field_pct": 0.0,
                    "test_reliability_score": 0.5, "test_data_available": 0,
                })
            continue

        drv_col = "Driver" if "Driver" in year_test.columns else "driver_id"
        year_test["LapTime"] = pd.to_numeric(year_test.get("LapTime", pd.Series()), errors="coerce")

        # Filter valid laps (not pit/outlaps)
        valid = year_test.dropna(subset=["LapTime"])
        valid = valid[valid["LapTime"] > 60]  # Remove unrealistic laps

        # Field-level stats
        fastest_overall = valid["LapTime"].min() if len(valid) > 0 else 0
        field_avg_laps  = valid.groupby(drv_col).size().mean() if len(valid) > 0 else 0

        for driver_id in drivers_this_year:
            # Map driver_id to test driver code
            driver_name = race_df[
                (race_df["year"] == year) &
                (race_df["driver_id"] == driver_id)
            ]["driver_name"].values

            drv_test = pd.DataFrame()
            if len(driver_name) > 0:
                # Try matching by surname
                surname = driver_name[0].split()[-1].upper()
                drv_test = valid[
                    valid[drv_col].str.upper().str.contains(surname[:3], na=False)
                ]

            if drv_test.empty:
                records.append({
                    "year": year, "driver_id": driver_id,
                    "fastest_test_lap_s": 0.0, "test_gap_to_fastest_s": 0.0,
                    "test_total_laps": 0, "test_long_run_pace_s": 0.0,
                    "test_short_run_pace_s": 0.0, "test_laps_vs_field_pct": 0.0,
                    "test_reliability_score": 0.5, "test_data_available": 0,
                })
                continue

            fastest_lap = drv_test["LapTime"].min()
            total_laps  = len(drv_test)

            # Long run pace (stints > 10 laps)
            long_run_laps = []
            if "Stint" in drv_test.columns:
                for stint in drv_test["Stint"].unique():
                    s = drv_test[drv_test["Stint"] == stint]
                    if len(s) >= 10:
                        # Remove first 2 and last 1 (in/out laps)
                        core = s.iloc[2:-1]["LapTime"].values
                        long_run_laps.extend(core)

            long_run_pace = np.median(long_run_laps) if long_run_laps else fastest_lap * 1.03

            # Short run pace (qualifying sim — stints < 5 laps)
            short_run_laps = []
            if "Stint" in drv_test.columns:
                for stint in drv_test["Stint"].unique():
                    s = drv_test[drv_test["Stint"] == stint]
                    if len(s) <= 4:
                        short_run_laps.extend(s["LapTime"].values)

            short_run_pace = np.min(short_run_laps) if short_run_laps else fastest_lap

            gap_to_fastest      = fastest_lap - fastest_overall if fastest_overall > 0 else 0
            laps_vs_field_pct   = (total_laps / field_avg_laps - 1) * 100 if field_avg_laps > 0 else 0
            reliability_score   = min(1.0, total_laps / 200)  # 200+ laps = full reliability

            records.append({
                "year":                   year,
                "driver_id":              driver_id,
                "fastest_test_lap_s":     round(fastest_lap, 3),
                "test_gap_to_fastest_s":  round(gap_to_fastest, 3),
                "test_total_laps":        total_laps,
                "test_long_run_pace_s":   round(long_run_pace, 3),
                "test_short_run_pace_s":  round(short_run_pace, 3),
                "test_laps_vs_field_pct": round(laps_vs_field_pct, 1),
                "test_reliability_score": round(reliability_score, 3),
                "test_data_available":    1,
            })

    df = pd.DataFrame(records)
    print(f"  Testing features: {len(df):,} rows")
    return df


# ─── DRIVER CONTEXT FEATURES ──────────────────────────────────────────────────

def compute_driver_context_features(race_df):
    """
    Computes driver contextual pre-season features.

    Features:
      - is_rookie: first full season in F1
      - is_f2_champion: promoted as reigning F2 champion
      - team_switch: moved to a different team
      - seasons_in_team: familiarity with current team
      - driver_age: age at season start
      - career_wins: total career wins entering season
      - career_championships: total titles won

    Returns:
        pd.DataFrame: Driver context per driver per year
    """
    print("Computing driver context features...")

    records = []
    years   = sorted(race_df["year"].unique())

    for year in years:
        current_drivers = race_df[race_df["year"] == year].drop_duplicates("driver_id")

        for _, row in current_drivers.iterrows():
            driver_id      = row["driver_id"]
            constructor_id = row.get("constructor_id", "")

            # Career history before this season
            career_hist = race_df[
                (race_df["driver_id"] == driver_id) &
                (race_df["year"] < year)
            ]

            is_rookie       = int(career_hist.empty)
            career_wins     = int((career_hist["finish_position"] == 1).sum()) if not career_hist.empty else 0
            career_podiums  = int((career_hist["finish_position"] <= 3).sum()) if not career_hist.empty else 0
            career_races    = len(career_hist["round"].unique()) if not career_hist.empty else 0

            # Team switch detection
            if not career_hist.empty:
                last_team = career_hist.sort_values("year").iloc[-1]["constructor_id"]
                team_switch = int(last_team != constructor_id)
            else:
                team_switch = 0

            # Seasons with current team
            team_hist = career_hist[career_hist["constructor_id"] == constructor_id] if not career_hist.empty else pd.DataFrame()
            seasons_in_team = team_hist["year"].nunique() if not team_hist.empty else 0

            # F2 champion promotion
            is_f2_champ = int(driver_id in F2_CHAMPION_PROMOTIONS.get(year, []))

            # Pre-season odds
            year_odds = PRESEASON_ODDS.get(year, {})
            odds      = year_odds.get(driver_id, 0.001)
            # Normalise
            total_odds = sum(year_odds.values()) if year_odds else 1
            preseason_champ_prob = odds / total_odds if total_odds > 0 else 0.05

            # Regulation change magnitude for this season
            reg_change = REGULATION_CHANGES.get(year, 0)

            # New PU generation
            new_pu = NEW_PU_GENERATION.get(year, 0)

            # PU supplier
            constructor_lower = str(constructor_id).lower()
            pu_map            = PU_SUPPLIERS.get(year, {})
            pu_supplier       = pu_map.get(constructor_lower, "Unknown")
            is_mercedes_pu    = int("mercedes" in pu_supplier.lower())
            is_ferrari_pu     = int("ferrari"  in pu_supplier.lower())
            is_honda_pu       = int("honda"    in pu_supplier.lower())

            # Car launch date (earliness = more testing time)
            launch_doy = CAR_LAUNCH_DOY.get(year, {}).get(constructor_lower, 50)
            launch_earliness = max(0, 70 - launch_doy) / 40  # 0-1, 1 = earliest

            # Prior season momentum (final 5 races)
            if not career_hist.empty:
                last_year     = year - 1
                last_yr_races = career_hist[career_hist["year"] == last_year]
                if not last_yr_races.empty:
                    max_round    = last_yr_races["round"].max()
                    final_5      = last_yr_races[last_yr_races["round"] > max_round - 5]
                    final_5_pts  = float(final_5["points"].sum())
                    final_5_avg  = float(final_5["finish_position"].mean())
                else:
                    final_5_pts = 0.0
                    final_5_avg = 10.0

                last_yr_pos = career_hist[career_hist["year"] == last_year]["finish_position"]
                prior_avg_finish = float(last_yr_pos.mean()) if len(last_yr_pos) > 0 else 10.0
            else:
                final_5_pts      = 0.0
                final_5_avg      = 10.0
                prior_avg_finish = 10.0

            records.append({
                "year":                   year,
                "driver_id":              driver_id,
                "is_rookie":              is_rookie,
                "is_f2_champion":         is_f2_champ,
                "team_switch":            team_switch,
                "seasons_in_team":        seasons_in_team,
                "career_wins":            career_wins,
                "career_podiums":         career_podiums,
                "career_races":           career_races,
                "preseason_champ_prob":   round(preseason_champ_prob, 4),
                "regulation_change_mag":  reg_change,
                "new_pu_generation":      new_pu,
                "is_mercedes_pu":         is_mercedes_pu,
                "is_ferrari_pu":          is_ferrari_pu,
                "is_honda_pu":            is_honda_pu,
                "launch_earliness_score": round(launch_earliness, 3),
                "prior_season_final5_pts": final_5_pts,
                "prior_season_final5_avg": round(final_5_avg, 2),
                "prior_season_avg_finish": round(prior_avg_finish, 2),
            })

    df = pd.DataFrame(records)
    print(f"  Driver context features: {len(df):,} rows")
    return df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def build_preseason_features():
    """
    Builds all pre-season features and returns combined DataFrame.
    Called by build_dataset.py.

    Returns:
        pd.DataFrame: Pre-season features per driver per year
    """
    print("\n" + "=" * 60)
    print("Building pre-season features...")
    print("=" * 60)

    race_df, test_df = load_data()

    testing_df = compute_testing_features(test_df, race_df)
    context_df = compute_driver_context_features(race_df)

    combined = context_df.merge(testing_df, on=["year","driver_id"], how="left")

    # Fill missing testing data with neutral values
    test_cols = [c for c in combined.columns if c.startswith("test_")]
    combined[test_cols] = combined[test_cols].fillna(0)

    print(f"\nPre-season features: {combined.shape}")
    print(f"  Rows    : {len(combined):,}")
    print(f"  Columns : {len(combined.columns)}")

    # Save
    out_path = os.path.join(DATA_DIR, "preseason_features.csv")
    combined.to_csv(out_path, index=False)
    print(f"  Saved   : {out_path}")

    return combined


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_preseason_features()
    print("\nAll columns:")
    for col in sorted(df.columns):
        print(f"  {col}")