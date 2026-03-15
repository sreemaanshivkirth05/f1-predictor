"""
elo_rating.py
-------------
Computes dynamic Elo / Glicko-style driver strength ratings.

How Elo works for F1:
  - Every driver starts at 1500 points
  - After each race, all pairs of drivers are compared
  - Beat a higher-rated driver = gain more points
  - Lose to a lower-rated driver = lose more points
  - Ratings update after every single race

This gives us a continuously updated "true strength" score
that reacts faster than season standings alone.

The Elo rating is used as:
  1. A feature in the XGBoost model
  2. A weight in the Monte Carlo simulator
  3. A standalone championship predictor

HOW TO RUN:
    python src/models/elo_rating.py

OUTPUT:
    data/elo_ratings.csv        — full rating history
    data/elo_current.csv        — latest rating per driver
"""

import pandas as pd
import numpy as np
import os

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

ELO_START = 1500   # Starting rating for all drivers
ELO_K     = 32     # How quickly ratings update (higher = faster)
ELO_D     = 400    # Scaling factor (standard chess value)

# New drivers start slightly below average to reflect uncertainty
NEW_DRIVER_START = 1450

# Ratings decay slightly between seasons to prevent stale ratings
# 0.9 means ratings move 10% back toward 1500 each new season
SEASON_DECAY = 0.9


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_race_results():
    """
    Loads race results from data/race_results.csv.

    Returns:
        pd.DataFrame: Race results sorted chronologically
    """
    path = os.path.join(DATA_DIR, "race_results.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"race_results.csv not found at {path}\n"
            "Please run get_ergast_data.py first."
        )

    df = pd.read_csv(path)
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df = df.dropna(subset=["finish_position"])
    df = df.sort_values(["year", "round", "finish_position"])

    print(f"Loaded {len(df):,} race results for Elo computation")
    return df


# ─── ELO UPDATE FORMULA ───────────────────────────────────────────────────────

def expected_score(rating_a, rating_b):
    """
    Computes the expected score for driver A against driver B.

    This is the probability that driver A finishes ahead of driver B
    based purely on their ratings.

    Formula: E_A = 1 / (1 + 10^((R_B - R_A) / D))

    Args:
        rating_a (float): Elo rating of driver A
        rating_b (float): Elo rating of driver B

    Returns:
        float: Probability (0-1) that A finishes ahead of B
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / ELO_D))


def update_elo(rating_a, rating_b, a_finished_ahead):
    """
    Updates Elo ratings for one head-to-head comparison.

    Args:
        rating_a         (float): Current rating of driver A
        rating_b         (float): Current rating of driver B
        a_finished_ahead (bool):  True if A finished ahead of B

    Returns:
        tuple: (new_rating_a, new_rating_b)
    """
    exp_a  = expected_score(rating_a, rating_b)
    exp_b  = 1.0 - exp_a

    actual_a = 1.0 if a_finished_ahead else 0.0
    actual_b = 1.0 - actual_a

    new_rating_a = rating_a + ELO_K * (actual_a - exp_a)
    new_rating_b = rating_b + ELO_K * (actual_b - exp_b)

    return new_rating_a, new_rating_b


# ─── COMPUTE ELO HISTORY ──────────────────────────────────────────────────────

def compute_elo_history(race_df):
    """
    Computes the full Elo rating history for all drivers.

    Processes every race chronologically:
      1. Records each driver's rating BEFORE the race
      2. Updates ratings based on all pairwise comparisons in the race
      3. Applies season decay between seasons

    Args:
        race_df (pd.DataFrame): Race results sorted by year and round

    Returns:
        pd.DataFrame: Elo rating before each race per driver
    """
    print("Computing Elo rating history...")

    # Current ratings — updated as we process each race
    ratings = {}

    # Records: Elo rating BEFORE each race (used as a feature)
    history = []

    years      = sorted(race_df["year"].unique())
    prev_year  = None

    for year in years:
        year_df = race_df[race_df["year"] == year]
        rounds  = sorted(year_df["round"].unique())

        # Apply season decay between years
        # This moves all ratings slightly back toward 1500
        # so stale ratings from old seasons don't dominate
        if prev_year is not None:
            for driver_id in list(ratings.keys()):
                ratings[driver_id] = (
                    ELO_START + SEASON_DECAY * (ratings[driver_id] - ELO_START)
                )

        for round_num in rounds:
            race = year_df[year_df["round"] == round_num].copy()
            race = race.sort_values("finish_position")

            drivers   = race["driver_id"].tolist()
            positions = dict(zip(race["driver_id"], race["finish_position"]))

            # Record rating BEFORE this race
            for driver_id in drivers:
                if driver_id not in ratings:
                    ratings[driver_id] = NEW_DRIVER_START

                history.append({
                    "year":       year,
                    "round":      round_num,
                    "driver_id":  driver_id,
                    "elo_before": round(ratings[driver_id], 2),
                })

            # Update ratings from all pairwise comparisons
            for i in range(len(drivers)):
                for j in range(i + 1, len(drivers)):
                    driver_a = drivers[i]
                    driver_b = drivers[j]

                    pos_a = positions[driver_a]
                    pos_b = positions[driver_b]

                    a_ahead = pos_a < pos_b  # Lower position = better finish

                    new_a, new_b = update_elo(
                        ratings[driver_a],
                        ratings[driver_b],
                        a_ahead
                    )

                    ratings[driver_a] = new_a
                    ratings[driver_b] = new_b

        prev_year = year

    history_df = pd.DataFrame(history)
    print(f"  Computed {len(history_df):,} Elo records "
          f"for {history_df['driver_id'].nunique()} drivers")

    return history_df, ratings


# ─── CURRENT RATINGS ──────────────────────────────────────────────────────────

def get_current_ratings(ratings_dict, race_df):
    """
    Builds a DataFrame of the most recent Elo rating per driver.
    Also includes driver name and constructor from the most recent race.

    Args:
        ratings_dict (dict):         driver_id → current Elo rating
        race_df      (pd.DataFrame): Race results (for name lookup)

    Returns:
        pd.DataFrame: Current ratings sorted highest to lowest
    """
    # Get most recent constructor and name per driver
    latest = (
        race_df.sort_values(["year", "round"])
        .groupby("driver_id")
        .last()
        .reset_index()
        [["driver_id", "driver_name", "constructor_id", "constructor_name"]]
    )

    rows = []
    for driver_id, rating in ratings_dict.items():
        info = latest[latest["driver_id"] == driver_id]

        rows.append({
            "driver_id":        driver_id,
            "driver_name":      info["driver_name"].values[0]      if len(info) > 0 else driver_id,
            "constructor_id":   info["constructor_id"].values[0]   if len(info) > 0 else "",
            "constructor_name": info["constructor_name"].values[0] if len(info) > 0 else "",
            "elo_rating":       round(rating, 2),
        })

    current_df = (
        pd.DataFrame(rows)
        .sort_values("elo_rating", ascending=False)
        .reset_index(drop=True)
    )

    current_df["elo_rank"] = current_df.index + 1

    return current_df


# ─── ELO TREND ────────────────────────────────────────────────────────────────

def compute_elo_trend(history_df):
    """
    Computes how a driver's Elo rating has changed recently.

    Trend features:
      - elo_change_last_race: rating change from previous race
      - elo_change_last_5:    total change over last 5 races
      - elo_trend_direction:  +1 improving, -1 declining, 0 stable

    Args:
        history_df (pd.DataFrame): Full Elo history

    Returns:
        pd.DataFrame: History with trend columns added
    """
    history_df = history_df.sort_values(
        ["driver_id", "year", "round"]
    ).copy()

    history_df["elo_change_last_race"] = (
        history_df.groupby("driver_id")["elo_before"].diff()
    )

    history_df["elo_change_last_5"] = (
        history_df.groupby("driver_id")["elo_before"]
        .transform(lambda x: x.diff(5))
    )

    history_df["elo_trend_direction"] = np.sign(
        history_df["elo_change_last_race"].fillna(0)
    ).astype(int)

    return history_df


# ─── CHAMPIONSHIP WIN PROBABILITY FROM ELO ────────────────────────────────────

def elo_championship_probability(current_ratings_df):
    """
    Estimates championship win probability from current Elo ratings.

    Uses a simple softmax over the active drivers' ratings.
    This is a rough estimate — the Monte Carlo simulation is more accurate
    but Elo probability is fast and useful for a quick overview.

    Args:
        current_ratings_df (pd.DataFrame): Current Elo ratings

    Returns:
        pd.DataFrame: Championship win probability per driver
    """
    ratings = current_ratings_df["elo_rating"].values

    # Softmax with temperature scaling
    # Temperature = 200 makes the distribution less extreme
    temp        = 200.0
    exp_ratings = np.exp((ratings - ratings.max()) / temp)
    probs       = exp_ratings / exp_ratings.sum()

    result = current_ratings_df[["driver_id", "driver_name",
                                  "constructor_name", "elo_rating"]].copy()
    result["elo_champ_probability"] = np.round(probs * 100, 2)

    return result.sort_values("elo_champ_probability", ascending=False)


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Full Elo pipeline:
      1. Load race results
      2. Compute full Elo history
      3. Add trend features
      4. Get current ratings
      5. Compute championship probabilities
      6. Save all outputs
    """
    print("\n" + "=" * 60)
    print("F1 ELO RATING SYSTEM")
    print("=" * 60)

    # Load data
    race_df = load_race_results()

    # Compute history
    history_df, current_ratings = compute_elo_history(race_df)

    # Add trend features
    history_df = compute_elo_trend(history_df)

    # Current standings
    current_df = get_current_ratings(current_ratings, race_df)

    # Championship probabilities
    champ_df = elo_championship_probability(current_df)

    # ── Print current standings ───────────────────────────────────────────────
    print("\nCurrent Elo Ratings (Top 20):")
    print("-" * 55)
    print(f"  {'Rank':<5} {'Driver':<25} {'Team':<20} {'Elo':>6}")
    print("-" * 55)
    for _, row in current_df.head(20).iterrows():
        print(f"  {int(row['elo_rank']):<5} "
              f"{row['driver_name']:<25} "
              f"{str(row['constructor_name']):<20} "
              f"{row['elo_rating']:>6.1f}")

    print("\nChampionship Win Probability (Elo-based, Top 10):")
    print("-" * 45)
    for _, row in champ_df.head(10).iterrows():
        bar = "█" * int(row["elo_champ_probability"] / 5)
        print(f"  {row['driver_name']:<25} {row['elo_champ_probability']:>5.1f}%  {bar}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    hist_path    = os.path.join(DATA_DIR, "elo_ratings.csv")
    current_path = os.path.join(DATA_DIR, "elo_current.csv")
    champ_path   = os.path.join(DATA_DIR, "elo_championship.csv")

    history_df.to_csv(hist_path,    index=False)
    current_df.to_csv(current_path, index=False)
    champ_df.to_csv(champ_path,     index=False)

    print(f"\nSaved:")
    print(f"  {hist_path}    ({len(history_df):,} rows)")
    print(f"  {current_path}")
    print(f"  {champ_path}")

    return history_df, current_df, champ_df


# ─── HELPER: LOAD CURRENT ELO ─────────────────────────────────────────────────

def load_current_elo():
    """
    Loads the most recent Elo ratings from disk.
    Called by monte_carlo.py and the Streamlit app.

    Returns:
        pd.DataFrame: Current Elo ratings per driver
    """
    path = os.path.join(DATA_DIR, "elo_current.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"elo_current.csv not found at {path}\n"
            "Please run elo_rating.py first."
        )

    return pd.read_csv(path)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    history_df, current_df, champ_df = run_full_pipeline()