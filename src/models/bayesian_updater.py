"""
bayesian_updater.py
-------------------
Updates championship win probabilities after each race using
Bayesian inference.

How it works:
  - Prior: pre-season championship probabilities (from betting odds or Elo)
  - Likelihood: how consistent is this race result with a title contender?
  - Posterior: updated probability after observing the race result

This gives us a mathematically sound way to update predictions
after every race without overreacting to a single result.

The Bayesian updater runs alongside the Monte Carlo simulator.
The ensemble.py file combines both into the final prediction.

HOW TO RUN:
    python src/models/bayesian_updater.py

OUTPUT:
    data/bayesian_probabilities.csv  — updated championship odds
    data/bayesian_history.csv        — probability evolution over season
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CURRENT_YEAR = 2026
TOTAL_RACES  = 24

# F1 points system
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6:  8, 7:  6, 8:  4, 9:  2, 10: 1,
}

# How strongly a single race result updates the probability
# Higher = faster reaction to results (range 0.1 to 1.0)
# 0.3 means a win adds 30% to likelihood, a DNF subtracts 30%
UPDATE_STRENGTH = 0.3

# Pre-season betting odds for 2026 championship
# These are the PRIOR probabilities before any races
# Format: driver_id → implied probability (sums to ~1.0 after normalisation)
# Source: approximate pre-season betting market odds
PRESEASON_ODDS = {
    "max_verstappen":  0.28,
    "norris":          0.18,
    "leclerc":         0.14,
    "russell":         0.12,
    "hamilton":        0.10,
    "piastri":         0.07,
    "antonelli":       0.04,
    "sainz":           0.03,
    "alonso":          0.01,
    "bearman":         0.01,
    "hulkenberg":      0.005,
    "gasly":           0.005,
    "albon":           0.003,
    "lawson":          0.003,
    "hadjar":          0.002,
    "colapinto":       0.002,
    "bortoleto":       0.001,
    "ocon":            0.001,
    "stroll":          0.001,
    "perez":           0.001,
    "bottas":          0.001,
    "arvid_lindblad":  0.001,
}


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    """
    Loads race results and Elo ratings for Bayesian updating.

    Returns:
        tuple: (race_df, elo_df)
    """
    race_path = os.path.join(DATA_DIR, "race_results.csv")
    elo_path  = os.path.join(DATA_DIR, "elo_current.csv")

    race_df = pd.read_csv(race_path)
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce")
    race_df["points"]          = pd.to_numeric(race_df["points"],           errors="coerce")

    elo_df = pd.read_csv(elo_path) if os.path.exists(elo_path) else pd.DataFrame()

    print(f"  Race results : {len(race_df):,} rows")
    print(f"  Elo ratings  : {len(elo_df)} drivers")

    return race_df, elo_df


# ─── PRIOR PROBABILITIES ──────────────────────────────────────────────────────

def get_prior_probabilities(drivers, elo_df):
    """
    Builds the prior probability distribution for championship win.

    Uses pre-season betting odds as the primary source.
    For drivers not in the odds (e.g. new drivers), falls back
    to Elo-based probabilities.

    Args:
        drivers (list):          List of driver_ids in the season
        elo_df  (pd.DataFrame):  Current Elo ratings

    Returns:
        dict: driver_id → prior probability (sums to 1.0)
    """
    prior = {}

    for driver in drivers:
        if driver in PRESEASON_ODDS:
            prior[driver] = PRESEASON_ODDS[driver]
        else:
            # Fall back to small equal share for unknown drivers
            prior[driver] = 0.001

    # Normalise so probabilities sum to 1.0
    total = sum(prior.values())
    prior = {d: p / total for d, p in prior.items()}

    return prior


# ─── LIKELIHOOD FUNCTION ──────────────────────────────────────────────────────

def compute_likelihood(race_result, driver, current_points,
                        leader_points, races_remaining):
    """
    Computes how likely this race result is IF this driver wins the championship.

    The likelihood is higher when:
      - Driver won or finished on the podium
      - Driver is currently leading or close to the leader
      - There are many races remaining (more recovery opportunity)

    The likelihood is lower when:
      - Driver DNF'd or finished poorly
      - Driver is far behind the leader with few races remaining

    Args:
        race_result      (int):   Driver's finishing position this race
        driver           (str):   Driver ID
        current_points   (float): Driver's total points so far
        leader_points    (float): Championship leader's points
        races_remaining  (int):   Races left in the season

    Returns:
        float: Likelihood multiplier (> 1 if result helps, < 1 if it hurts)
    """
    # Base likelihood from race result
    if race_result == 1:
        result_factor = 1.0 + UPDATE_STRENGTH * 1.5   # Win = big boost
    elif race_result <= 3:
        result_factor = 1.0 + UPDATE_STRENGTH          # Podium = boost
    elif race_result <= 6:
        result_factor = 1.0 + UPDATE_STRENGTH * 0.3   # Top 6 = small boost
    elif race_result <= 10:
        result_factor = 1.0                             # Points finish = neutral
    elif race_result <= 15:
        result_factor = 1.0 - UPDATE_STRENGTH * 0.3   # Out of points = penalty
    else:
        result_factor = 1.0 - UPDATE_STRENGTH          # DNF / last = big penalty

    # Points gap factor
    # Being far behind the leader with few races left = unlikely to win
    points_gap = max(0, leader_points - current_points)
    max_possible_gain = races_remaining * 26  # 25 points + fastest lap each race

    if max_possible_gain > 0:
        gap_ratio = points_gap / max_possible_gain
        # If gap is > 100% of remaining points, mathematically cannot win
        if gap_ratio > 1.0:
            return 0.001  # Near-zero but not exactly zero
        gap_factor = 1.0 - 0.5 * gap_ratio
    else:
        gap_factor = 1.0

    return result_factor * gap_factor


# ─── BAYESIAN UPDATE ──────────────────────────────────────────────────────────

def bayesian_update(prior, race_results_df, round_num,
                    current_points_dict, races_remaining):
    """
    Performs one Bayesian update after a single race.

    Formula:
        posterior(driver) ∝ prior(driver) × likelihood(race_result | driver wins)

    After computing all posteriors, we normalise so they sum to 1.

    Args:
        prior               (dict): driver_id → current probability
        race_results_df     (pd.DataFrame): Results of this specific race
        round_num           (int):  Race round number
        current_points_dict (dict): driver_id → points before this race
        races_remaining     (int):  Races left after this one

    Returns:
        dict: driver_id → updated (posterior) probability
    """
    # Find the leader's points
    leader_points = max(current_points_dict.values()) if current_points_dict else 0

    posterior = {}

    for driver, prior_prob in prior.items():
        # Get this driver's result in this race
        result_row = race_results_df[race_results_df["driver_id"] == driver]

        if result_row.empty:
            # Driver not in this race (e.g. substituted) — neutral update
            posterior[driver] = prior_prob
            continue

        finish_pos     = result_row["finish_position"].values[0]
        driver_points  = current_points_dict.get(driver, 0)

        likelihood = compute_likelihood(
            finish_pos,
            driver,
            driver_points,
            leader_points,
            races_remaining
        )

        posterior[driver] = prior_prob * likelihood

    # Normalise
    total = sum(posterior.values())
    if total > 0:
        posterior = {d: p / total for d, p in posterior.items()}
    else:
        # Fallback if all likelihoods collapse
        n = len(posterior)
        posterior = {d: 1.0 / n for d in posterior}

    return posterior


# ─── PROCESS FULL SEASON ──────────────────────────────────────────────────────

def process_season(race_df, elo_df, year=CURRENT_YEAR):
    """
    Processes all completed races in the current season,
    updating Bayesian probabilities after each one.

    Args:
        race_df  (pd.DataFrame): Race results
        elo_df   (pd.DataFrame): Elo ratings
        year     (int):          Season year

    Returns:
        tuple: (final_probs_df, history_df)
    """
    print(f"\nProcessing {year} season Bayesian updates...")

    # Get this season's races
    season_df = race_df[race_df["year"] == year].copy()

    if season_df.empty:
        print(f"  No {year} race data found")
        return pd.DataFrame(), pd.DataFrame()

    rounds     = sorted(season_df["round"].unique())
    drivers    = season_df["driver_id"].unique().tolist()
    n_drivers  = len(drivers)

    print(f"  Drivers: {n_drivers}")
    print(f"  Rounds completed: {len(rounds)}")

    # Initialise prior
    prior = get_prior_probabilities(drivers, elo_df)

    # Track cumulative points
    cum_points = {d: 0.0 for d in drivers}

    # History of probability evolution
    history = []

    # Save initial (pre-season) probabilities
    for driver in drivers:
        name = season_df[season_df["driver_id"] == driver]["driver_name"].values
        team = season_df[season_df["driver_id"] == driver]["constructor_name"].values

        history.append({
            "round":       0,
            "race_name":   "Pre-season",
            "driver_id":   driver,
            "driver_name": name[0] if len(name) > 0 else driver,
            "constructor": team[0] if len(team) > 0 else "",
            "probability": round(prior[driver] * 100, 3),
            "cum_points":  0,
        })

    # Process each completed race
    for round_num in rounds:
        race_results = season_df[season_df["round"] == round_num].copy()
        race_name    = race_results["race_name"].dropna().values[0] if len(race_results) > 0 and race_results["race_name"].notna().any() else f"Round {round_num}"
        races_remaining = TOTAL_RACES - round_num

        # Update cumulative points first
        for _, row in race_results.iterrows():
            driver = row["driver_id"]
            points = row.get("points", 0)
            if pd.notna(points):
                cum_points[driver] = cum_points.get(driver, 0) + float(points)

        # Bayesian update
        posterior = bayesian_update(
            prior,
            race_results,
            round_num,
            cum_points.copy(),
            races_remaining
        )

        # Save this round's probabilities to history
        for driver in drivers:
            name = race_results[race_results["driver_id"] == driver]["driver_name"].values
            team = race_results[race_results["driver_id"] == driver]["constructor_name"].values

            history.append({
                "round":       round_num,
                "race_name":   race_name,
                "driver_id":   driver,
                "driver_name": name[0] if len(name) > 0 else driver,
                "constructor": team[0] if len(team) > 0 else "",
                "probability": round(posterior[driver] * 100, 3),
                "cum_points":  round(cum_points.get(driver, 0), 1),
            })

        rn = str(race_name)[:30] if race_name and str(race_name) != "nan" else f"Round {round_num}"
        leader = max(posterior, key=posterior.get)
        print(f"  Round {round_num:>2} ({rn:<30}) — Leader: {leader} ({posterior[leader]*100:.1f}%)")

        # Prior for next race = posterior from this race
        prior = posterior

    # Build final probabilities DataFrame
    final_history = [h for h in history if h["round"] == max(rounds)]
    final_df = pd.DataFrame(final_history).sort_values(
        "probability", ascending=False
    ).reset_index(drop=True)
    final_df["rank"] = final_df.index + 1

    history_df = pd.DataFrame(history)

    return final_df, history_df


# ─── PRINT RESULTS ────────────────────────────────────────────────────────────

def print_results(final_df):
    """Prints the Bayesian championship probabilities."""
    print("\n" + "=" * 60)
    print("BAYESIAN CHAMPIONSHIP PROBABILITIES")
    print("=" * 60)
    print(f"  {'Rank':<5} {'Driver':<25} {'Team':<22} {'Pts':>5} {'Prob':>7}")
    print("-" * 60)

    for _, row in final_df.head(10).iterrows():
        bar = "█" * int(row["probability"] / 3)
        print(f"  {int(row['rank']):<5} "
              f"{row['driver_name']:<25} "
              f"{str(row['constructor']):<22} "
              f"{int(row['cum_points']):>5} "
              f"{row['probability']:>6.1f}%  {bar}")

    print("=" * 60)


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Full Bayesian update pipeline:
      1. Load race results and Elo ratings
      2. Initialise prior from pre-season betting odds
      3. Update after each completed race
      4. Save final probabilities and history
    """
    print("\n" + "=" * 60)
    print("F1 BAYESIAN CHAMPIONSHIP UPDATER")
    print("=" * 60)

    race_df, elo_df = load_data()

    final_df, history_df = process_season(race_df, elo_df)

    if final_df.empty:
        print("No completed races to process")
        return pd.DataFrame(), pd.DataFrame()

    print_results(final_df)

    # Save outputs
    final_path   = os.path.join(DATA_DIR, "bayesian_probabilities.csv")
    history_path = os.path.join(DATA_DIR, "bayesian_history.csv")

    final_df.to_csv(final_path,   index=False)
    history_df.to_csv(history_path, index=False)

    print(f"\nSaved:")
    print(f"  {final_path}")
    print(f"  {history_path}")

    return final_df, history_df


# ─── HELPER: LOAD PROBABILITIES ───────────────────────────────────────────────

def load_bayesian_probabilities():
    """
    Loads saved Bayesian probabilities.
    Called by ensemble.py and the Streamlit app.

    Returns:
        pd.DataFrame: Current Bayesian championship probabilities
    """
    path = os.path.join(DATA_DIR, "bayesian_probabilities.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"bayesian_probabilities.csv not found.\n"
            "Please run bayesian_updater.py first."
        )

    return pd.read_csv(path)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    final_df, history_df = run_full_pipeline()