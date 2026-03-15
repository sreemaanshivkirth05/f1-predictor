"""
driver_features.py
------------------
Computes all driver performance features from the raw CSV files.

Features computed here:
  - Rolling average points (last 3, 5, 10 races)
  - Points gap to championship leader
  - DNF rate (total, mechanical, crash)
  - Qualifying vs finish position delta
  - Podium rate, win rate, points finish rate
  - Head-to-head vs teammate (qualifying and race)
  - Consistency score (standard deviation of finishes)
  - Elo rating (dynamic driver strength)
  - Prior season stats carried forward

HOW TO RUN (standalone test):
    python src/feature_engineering/driver_features.py

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

# DNF statuses from Jolpica API
# These are used to classify whether a DNF was mechanical or a crash
MECHANICAL_DNF_STATUSES = [
    "Engine", "Gearbox", "Transmission", "Hydraulics", "Electrical",
    "Brakes", "Suspension", "Wheel", "Tyre", "Mechanical", "Power Unit",
    "Oil pressure", "Water pressure", "Fuel pressure", "Turbo", "Exhaust",
    "Overheating", "Fire", "Battery", "ERS", "MGU-H", "MGU-K",
]

CRASH_DNF_STATUSES = [
    "Accident", "Collision", "Spun off", "Retired", "Damage",
    "Collision damage", "Fatal accident", "Withdrew",
]

# Elo rating starting value for all drivers
ELO_START = 1500

# Elo K-factor — how quickly ratings update after each race
ELO_K = 32


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    """
    Loads all required CSV files from the data/ folder.

    Returns:
        tuple: (race_results_df, qualifying_df, driver_standings_df)
    """
    print("Loading raw data files...")

    race_path  = os.path.join(DATA_DIR, "race_results.csv")
    quali_path = os.path.join(DATA_DIR, "qualifying_results.csv")
    stand_path = os.path.join(DATA_DIR, "driver_standings.csv")

    for path in [race_path, quali_path, stand_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Please run get_ergast_data.py first."
            )

    race_df  = pd.read_csv(race_path)
    quali_df = pd.read_csv(quali_path)
    stand_df = pd.read_csv(stand_path)

    # Convert numeric columns
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce")
    race_df["grid_position"]   = pd.to_numeric(race_df["grid_position"],   errors="coerce")
    race_df["points"]          = pd.to_numeric(race_df["points"],           errors="coerce")
    quali_df["quali_position"] = pd.to_numeric(quali_df["quali_position"],  errors="coerce")

    print(f"  Race results : {len(race_df):,} rows")
    print(f"  Qualifying   : {len(quali_df):,} rows")
    print(f"  Standings    : {len(stand_df):,} rows")

    return race_df, quali_df, stand_df


# ─── DNF CLASSIFICATION ───────────────────────────────────────────────────────

def classify_dnf(status):
    """
    Classifies a race status string into: finished, mechanical_dnf, crash_dnf.

    Args:
        status (str): Status string from Jolpica API e.g. "Finished", "Engine"

    Returns:
        str: "finished", "mechanical_dnf", or "crash_dnf"
    """
    if pd.isna(status):
        return "finished"

    status_str = str(status).strip()

    # "Finished" or "+X Laps" means they finished
    if status_str == "Finished" or status_str.startswith("+"):
        return "finished"

    # Check mechanical failure keywords
    for keyword in MECHANICAL_DNF_STATUSES:
        if keyword.lower() in status_str.lower():
            return "mechanical_dnf"

    # Check crash keywords
    for keyword in CRASH_DNF_STATUSES:
        if keyword.lower() in status_str.lower():
            return "crash_dnf"

    # Default unknown DNF to mechanical
    return "mechanical_dnf"


# ─── ELO RATING ───────────────────────────────────────────────────────────────

def compute_elo_ratings(race_df):
    """
    Computes Elo ratings for all drivers across all races chronologically.

    How Elo works for F1:
      - Each driver starts at 1500
      - After each race, all pairs of drivers are compared
      - A driver who finishes ahead of a higher-rated driver gains more points
      - A driver who finishes behind a lower-rated driver loses more points
      - This creates a dynamic strength rating that updates every race

    Args:
        race_df (pd.DataFrame): Race results with finish_position, driver_id

    Returns:
        pd.DataFrame: One row per driver per race with their Elo rating
                      BEFORE that race (so we don't leak future info)
    """
    print("Computing Elo ratings...")

    # Sort races chronologically
    races_sorted = (
        race_df[["year", "round", "race_name", "race_date", "driver_id", "finish_position"]]
        .dropna(subset=["finish_position"])
        .sort_values(["year", "round"])
    )

    # Current Elo for each driver (updated as we process each race)
    elo_current = {}

    # Storage: elo rating BEFORE each race (this is the feature we use)
    elo_records = []

    # Process each race in order
    for (year, round_num), race_group in races_sorted.groupby(["year", "round"]):

        # Record each driver's Elo BEFORE this race
        for _, row in race_group.iterrows():
            driver_id = row["driver_id"]
            elo_before = elo_current.get(driver_id, ELO_START)

            elo_records.append({
                "year":       year,
                "round":      round_num,
                "driver_id":  driver_id,
                "elo_before": elo_before,
            })

        # Now update Elo based on this race's results
        drivers_in_race = race_group["driver_id"].tolist()
        positions       = dict(zip(race_group["driver_id"], race_group["finish_position"]))

        # Compare every pair of drivers
        for i, driver_a in enumerate(drivers_in_race):
            for driver_b in drivers_in_race[i+1:]:

                pos_a = positions[driver_a]
                pos_b = positions[driver_b]

                elo_a = elo_current.get(driver_a, ELO_START)
                elo_b = elo_current.get(driver_b, ELO_START)

                # Expected score for driver A (probability A finishes ahead of B)
                expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
                expected_b = 1 - expected_a

                # Actual score: 1 if finished ahead, 0 if behind
                actual_a = 1 if pos_a < pos_b else 0
                actual_b = 1 - actual_a

                # Update ratings
                elo_current[driver_a] = elo_a + ELO_K * (actual_a - expected_a)
                elo_current[driver_b] = elo_b + ELO_K * (actual_b - expected_b)

    elo_df = pd.DataFrame(elo_records)
    print(f"  Computed Elo for {elo_df['driver_id'].nunique()} drivers")
    return elo_df


# ─── ROLLING FEATURES ─────────────────────────────────────────────────────────

def compute_rolling_features(race_df):
    """
    Computes rolling/cumulative features per driver per race.

    Rolling features use only PAST races (no current race info leakage).
    For example, rolling_avg_pts_5 at Race 8 = avg of races 3,4,5,6,7.

    Features computed:
      - rolling_avg_pts_3, _5, _10
      - rolling_dnf_rate (last 5 races)
      - rolling_podium_rate (last 5 races)
      - rolling_avg_finish (last 5 races)
      - finish_std_dev (consistency — last 10 races)
      - positions_gained_avg (last 5 races)
      - cumulative win rate this season

    Args:
        race_df (pd.DataFrame): Race results

    Returns:
        pd.DataFrame: Rolling features, one row per driver per race
    """
    print("Computing rolling features...")

    # Add DNF classification column
    race_df = race_df.copy()
    race_df["dnf_type"] = race_df["status"].apply(classify_dnf)
    race_df["is_dnf"]           = (race_df["dnf_type"] != "finished").astype(int)
    race_df["is_mechanical_dnf"] = (race_df["dnf_type"] == "mechanical_dnf").astype(int)
    race_df["is_crash_dnf"]     = (race_df["dnf_type"] == "crash_dnf").astype(int)
    race_df["is_podium"]        = (race_df["finish_position"] <= 3).astype(int)
    race_df["is_win"]           = (race_df["finish_position"] == 1).astype(int)
    race_df["is_points_finish"] = (race_df["finish_position"] <= 10).astype(int)
    race_df["positions_gained"] = race_df["grid_position"] - race_df["finish_position"]

    # Sort chronologically per driver
    race_df = race_df.sort_values(["driver_id", "year", "round"])

    rolling_records = []

    for driver_id, driver_races in race_df.groupby("driver_id"):
        driver_races = driver_races.reset_index(drop=True)

        for i, row in driver_races.iterrows():
            # Only use PAST races (rows before this one)
            past = driver_races.iloc[:i]

            record = {
                "year":      row["year"],
                "round":     row["round"],
                "driver_id": driver_id,
            }

            if len(past) == 0:
                # First race — no history, use neutral defaults
                record.update({
                    "rolling_avg_pts_3":     0.0,
                    "rolling_avg_pts_5":     0.0,
                    "rolling_avg_pts_10":    0.0,
                    "rolling_dnf_rate_5":    0.0,
                    "rolling_mech_dnf_rate": 0.0,
                    "rolling_crash_rate":    0.0,
                    "rolling_podium_rate_5": 0.0,
                    "rolling_win_rate_5":    0.0,
                    "rolling_avg_finish_5":  10.0,
                    "finish_std_dev_10":     0.0,
                    "positions_gained_avg":  0.0,
                    "season_wins":           0,
                    "season_podiums":        0,
                    "season_points_total":   0.0,
                    "races_completed":       0,
                })
            else:
                # Rolling windows
                last_3  = past.tail(3)
                last_5  = past.tail(5)
                last_10 = past.tail(10)

                record.update({
                    "rolling_avg_pts_3":     last_3["points"].mean(),
                    "rolling_avg_pts_5":     last_5["points"].mean(),
                    "rolling_avg_pts_10":    last_10["points"].mean(),
                    "rolling_dnf_rate_5":    last_5["is_dnf"].mean(),
                    "rolling_mech_dnf_rate": last_5["is_mechanical_dnf"].mean(),
                    "rolling_crash_rate":    last_5["is_crash_dnf"].mean(),
                    "rolling_podium_rate_5": last_5["is_podium"].mean(),
                    "rolling_win_rate_5":    last_5["is_win"].mean(),
                    "rolling_avg_finish_5":  last_5["finish_position"].mean(),
                    "finish_std_dev_10":     last_10["finish_position"].std(ddof=0),
                    "positions_gained_avg":  last_5["positions_gained"].mean(),

                    # Season totals up to this race (not including this race)
                    "season_wins":           past[past["year"] == row["year"]]["is_win"].sum(),
                    "season_podiums":        past[past["year"] == row["year"]]["is_podium"].sum(),
                    "season_points_total":   past[past["year"] == row["year"]]["points"].sum(),
                    "races_completed":       len(past[past["year"] == row["year"]]),
                })

            rolling_records.append(record)

    df = pd.DataFrame(rolling_records)
    print(f"  Computed rolling features: {len(df):,} rows")
    return df


# ─── QUALIFYING FEATURES ──────────────────────────────────────────────────────

def compute_qualifying_features(race_df, quali_df):
    """
    Computes qualifying-based features per driver per race.

    Features:
      - qualifying position (from quali_df)
      - quali vs finish delta (how much do they gain/lose from grid?)
      - rolling avg qualifying position (last 5 races)
      - teammate qualifying delta

    Args:
        race_df  (pd.DataFrame): Race results
        quali_df (pd.DataFrame): Qualifying results

    Returns:
        pd.DataFrame: Qualifying features per driver per race
    """
    print("Computing qualifying features...")

    # Merge qualifying position into race results
    merged = race_df.merge(
        quali_df[["year", "round", "driver_id", "quali_position"]],
        on=["year", "round", "driver_id"],
        how="left"
    )

    merged["quali_vs_finish_delta"] = merged["quali_position"] - merged["finish_position"]

    # Sort chronologically per driver for rolling calculations
    merged = merged.sort_values(["driver_id", "year", "round"])

    quali_records = []

    for driver_id, driver_data in merged.groupby("driver_id"):
        driver_data = driver_data.reset_index(drop=True)

        for i, row in driver_data.iterrows():
            past = driver_data.iloc[:i]

            record = {
                "year":      row["year"],
                "round":     row["round"],
                "driver_id": driver_id,
                "quali_position": row.get("quali_position", np.nan),
            }

            if len(past) >= 1:
                last_5 = past.tail(5)
                record["rolling_avg_quali_pos_5"] = last_5["quali_position"].mean()
                record["rolling_avg_quali_delta_5"] = last_5["quali_vs_finish_delta"].mean()
            else:
                record["rolling_avg_quali_pos_5"]   = np.nan
                record["rolling_avg_quali_delta_5"]  = 0.0

            quali_records.append(record)

    quali_features_df = pd.DataFrame(quali_records)

    # ── Teammate qualifying delta ──────────────────────────────────────────────
    # For each race, compare each driver to their teammate in qualifying
    teammate_deltas = []

    for (year, round_num), race_group in merged.groupby(["year", "round"]):
        # Get constructor for each driver in this race
        constructor_map = dict(zip(
            race_group["driver_id"],
            race_group["constructor_id"]
        ))
        quali_map = dict(zip(
            race_group["driver_id"],
            race_group["quali_position"]
        ))

        for driver_id, constructor_id in constructor_map.items():
            # Find teammate (same constructor, different driver)
            teammates = [
                d for d, c in constructor_map.items()
                if c == constructor_id and d != driver_id
            ]

            if teammates:
                teammate_id   = teammates[0]
                my_quali      = quali_map.get(driver_id, np.nan)
                teammate_quali = quali_map.get(teammate_id, np.nan)

                if pd.notna(my_quali) and pd.notna(teammate_quali):
                    # Positive = I qualified ahead of teammate
                    delta = teammate_quali - my_quali
                else:
                    delta = np.nan
            else:
                delta = np.nan

            teammate_deltas.append({
                "year":                    year,
                "round":                   round_num,
                "driver_id":               driver_id,
                "teammate_quali_delta":    delta,
            })

    teammate_df = pd.DataFrame(teammate_deltas)

    # Merge teammate delta into qualifying features
    quali_features_df = quali_features_df.merge(
        teammate_df,
        on=["year", "round", "driver_id"],
        how="left"
    )

    print(f"  Computed qualifying features: {len(quali_features_df):,} rows")
    return quali_features_df


# ─── CHAMPIONSHIP CONTEXT FEATURES ────────────────────────────────────────────

def compute_championship_features(stand_df):
    """
    Computes championship context features from the standings CSV.

    Features:
      - championship position before each race
      - points gap to leader
      - points gap to P2 and P3
      - championship position trend (improving or dropping?)
      - races remaining in season
      - points mathematically available

    Args:
        stand_df (pd.DataFrame): Driver standings per round

    Returns:
        pd.DataFrame: Championship features per driver per race
    """
    print("Computing championship context features...")

    stand_df = stand_df.copy()
    stand_df = stand_df.sort_values(["year", "round", "championship_pos"])

    champ_records = []

    for (year, round_num), round_group in stand_df.groupby(["year", "round"]):

        # Leader's points for this round
        leader_pts = round_group["championship_pts"].max()

        # Total rounds in this season (estimate from max round seen so far)
        max_round_seen = stand_df[stand_df["year"] == year]["round"].max()

        # Approximate remaining races (F1 seasons have ~24 races in 2026)
        total_rounds    = max(max_round_seen, 24)
        races_remaining = total_rounds - round_num

        # Max points per race = 26 (25 for win + 1 for fastest lap)
        points_available = races_remaining * 26

        for _, row in round_group.iterrows():
            driver_id   = row["driver_id"]
            driver_pts  = row["championship_pts"]
            champ_pos   = row["championship_pos"]

            # Gap to leader
            gap_to_leader = leader_pts - driver_pts

            # Gap to P2 (relevant for the leader)
            p2_pts = round_group[round_group["championship_pos"] == 2]["championship_pts"].values
            gap_to_p2 = driver_pts - p2_pts[0] if len(p2_pts) > 0 else 0

            # Can this driver still mathematically win?
            can_win_title = int(gap_to_leader <= points_available)

            champ_records.append({
                "year":                  year,
                "round":                 round_num,
                "driver_id":             driver_id,
                "championship_pos":      champ_pos,
                "championship_pts":      driver_pts,
                "gap_to_leader":         gap_to_leader,
                "gap_to_p2":             gap_to_p2,
                "races_remaining":       races_remaining,
                "points_available":      points_available,
                "can_win_title":         can_win_title,
                "wins_this_season":      row.get("wins", 0),
            })

    # Add position trend (are they moving up or down in standings?)
    champ_df = pd.DataFrame(champ_records)
    champ_df = champ_df.sort_values(["driver_id", "year", "round"])

    champ_df["champ_pos_prev_3"] = (
        champ_df.groupby(["driver_id", "year"])["championship_pos"]
        .shift(3)
    )
    champ_df["championship_trend"] = (
        champ_df["champ_pos_prev_3"] - champ_df["championship_pos"]
    )
    # Positive trend = moving UP in standings (better)
    champ_df = champ_df.drop(columns=["champ_pos_prev_3"])

    print(f"  Computed championship features: {len(champ_df):,} rows")
    return champ_df


# ─── PRIOR SEASON FEATURES ────────────────────────────────────────────────────

def compute_prior_season_features(race_df, stand_df):
    """
    Computes carry-forward features from the previous season.

    These are used at the start of each season before any races
    have happened, providing baseline expectations for each driver.

    Features:
      - prior_season_champ_pos
      - prior_season_points
      - prior_season_win_rate
      - prior_season_dnf_rate
      - prior_season_avg_finish
      - prior_season_final_5_pts (momentum going into new season)

    Args:
        race_df  (pd.DataFrame): Race results
        stand_df (pd.DataFrame): Driver standings

    Returns:
        pd.DataFrame: Prior season features per driver per year
    """
    print("Computing prior season features...")

    race_df  = race_df.copy()
    stand_df = stand_df.copy()

    # Add DNF flag
    race_df["is_dnf"] = race_df["status"].apply(
        lambda s: 0 if (str(s) == "Finished" or str(s).startswith("+")) else 1
    )

    prior_records = []

    years = sorted(race_df["year"].unique())

    for year in years:
        prior_year = year - 1

        if prior_year not in race_df["year"].values:
            # No prior year data — use neutral defaults
            drivers_this_year = race_df[race_df["year"] == year]["driver_id"].unique()

            for driver_id in drivers_this_year:
                prior_records.append({
                    "year":                       year,
                    "driver_id":                  driver_id,
                    "prior_season_champ_pos":     10,
                    "prior_season_champ_pts":     0.0,
                    "prior_season_win_rate":      0.0,
                    "prior_season_dnf_rate":      0.1,
                    "prior_season_avg_finish":    10.0,
                    "prior_season_podium_rate":   0.0,
                    "prior_season_final_5_pts":   0.0,
                    "is_new_to_team":             1,
                })
            continue

        prior_race_df  = race_df[race_df["year"] == prior_year]
        prior_stand_df = stand_df[stand_df["year"] == prior_year]

        # Final standings from prior year
        final_standings = (
            prior_stand_df
            .sort_values("round", ascending=False)
            .drop_duplicates("driver_id")
            [["driver_id", "championship_pos", "championship_pts"]]
        )

        # Per-driver stats from prior year
        prior_stats = prior_race_df.groupby("driver_id").agg(
            total_races    = ("round", "count"),
            total_wins     = ("finish_position", lambda x: (x == 1).sum()),
            total_podiums  = ("finish_position", lambda x: (x <= 3).sum()),
            total_dnfs     = ("is_dnf", "sum"),
            avg_finish     = ("finish_position", "mean"),
        ).reset_index()

        prior_stats["win_rate"]    = prior_stats["total_wins"]   / prior_stats["total_races"]
        prior_stats["podium_rate"] = prior_stats["total_podiums"] / prior_stats["total_races"]
        prior_stats["dnf_rate"]    = prior_stats["total_dnfs"]    / prior_stats["total_races"]

        # Final 5 races points (momentum)
        max_round = prior_race_df["round"].max()
        final_5   = prior_race_df[prior_race_df["round"] > max_round - 5]
        final_5_pts = final_5.groupby("driver_id")["points"].sum().reset_index()
        final_5_pts.columns = ["driver_id", "prior_season_final_5_pts"]

        # This year's drivers and their teams
        current_year_teams = (
            race_df[race_df["year"] == year]
            .drop_duplicates("driver_id")
            [["driver_id", "constructor_id"]]
        )
        prior_year_teams = (
            prior_race_df
            .drop_duplicates("driver_id")
            [["driver_id", "constructor_id"]]
        )

        for driver_id in race_df[race_df["year"] == year]["driver_id"].unique():

            # Get final championship position
            final = final_standings[final_standings["driver_id"] == driver_id]
            champ_pos  = final["championship_pos"].values[0]  if len(final) > 0 else 15
            champ_pts  = final["championship_pts"].values[0]  if len(final) > 0 else 0.0

            # Get prior year stats
            stats = prior_stats[prior_stats["driver_id"] == driver_id]
            win_rate    = stats["win_rate"].values[0]    if len(stats) > 0 else 0.0
            podium_rate = stats["podium_rate"].values[0] if len(stats) > 0 else 0.0
            dnf_rate    = stats["dnf_rate"].values[0]    if len(stats) > 0 else 0.1
            avg_finish  = stats["avg_finish"].values[0]  if len(stats) > 0 else 10.0

            # Final 5 races points
            f5 = final_5_pts[final_5_pts["driver_id"] == driver_id]
            final_5_pts_val = f5["prior_season_final_5_pts"].values[0] if len(f5) > 0 else 0.0

            # Did driver switch teams?
            curr_team = current_year_teams[current_year_teams["driver_id"] == driver_id]["constructor_id"].values
            prev_team = prior_year_teams[prior_year_teams["driver_id"] == driver_id]["constructor_id"].values
            switched_team = 1 if (len(curr_team) > 0 and len(prev_team) > 0 and curr_team[0] != prev_team[0]) else 0

            prior_records.append({
                "year":                       year,
                "driver_id":                  driver_id,
                "prior_season_champ_pos":     champ_pos,
                "prior_season_champ_pts":     champ_pts,
                "prior_season_win_rate":      win_rate,
                "prior_season_podium_rate":   podium_rate,
                "prior_season_dnf_rate":      dnf_rate,
                "prior_season_avg_finish":    avg_finish,
                "prior_season_final_5_pts":   final_5_pts_val,
                "is_new_to_team":             switched_team,
            })

    df = pd.DataFrame(prior_records)
    print(f"  Computed prior season features: {len(df):,} rows")
    return df


# ─── COMBINE ALL DRIVER FEATURES ──────────────────────────────────────────────

def build_driver_features():
    """
    Master function — loads data, computes all driver features,
    and returns a single combined DataFrame.

    This is the function called by build_dataset.py.

    Returns:
        pd.DataFrame: All driver features, one row per driver per race
    """
    print("\n" + "=" * 60)
    print("Building driver features...")
    print("=" * 60)

    # Load raw data
    race_df, quali_df, stand_df = load_data()

    # Compute each feature group
    elo_df     = compute_elo_ratings(race_df)
    rolling_df = compute_rolling_features(race_df)
    quali_feat = compute_qualifying_features(race_df, quali_df)
    champ_feat = compute_championship_features(stand_df)
    prior_feat = compute_prior_season_features(race_df, stand_df)

    # ── Merge everything together ─────────────────────────────────────────────
    print("\nMerging all driver feature groups...")

    base = race_df[["year", "round", "driver_id", "driver_name",
                     "constructor_id", "constructor_name",
                     "circuit_id", "race_date",
                     "finish_position", "grid_position", "points", "status"]].copy()

    base = base.merge(elo_df,     on=["year", "round", "driver_id"], how="left")
    base = base.merge(rolling_df, on=["year", "round", "driver_id"], how="left")
    base = base.merge(quali_feat, on=["year", "round", "driver_id"], how="left")
    base = base.merge(champ_feat, on=["year", "round", "driver_id"], how="left")
    base = base.merge(prior_feat, on=["year", "driver_id"],          how="left")

    print(f"\nFinal driver features shape: {base.shape}")
    print(f"  Rows    : {len(base):,}")
    print(f"  Columns : {len(base.columns)}")
    print(f"  Drivers : {base['driver_id'].nunique()}")
    print(f"  Seasons : {sorted(base['year'].unique())}")

    return base


# ─── ENTRY POINT (for standalone testing) ─────────────────────────────────────

if __name__ == "__main__":
    df = build_driver_features()
    print("\nSample output (first 5 rows):")
    print(df.head())
    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")

    # Save a preview
    out_path = os.path.join(DATA_DIR, "driver_features_preview.csv")
    df.head(100).to_csv(out_path, index=False)
    print(f"\nPreview saved to: {out_path}")