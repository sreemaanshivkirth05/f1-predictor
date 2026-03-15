"""
ensemble.py
-----------
Combines Monte Carlo, Bayesian, and Elo predictions into one
final championship probability using a weighted ensemble.

Why ensemble?
  - Monte Carlo is best late in the season (fewer races = less uncertainty)
  - Bayesian is best early (pre-season odds carry useful information)
  - Elo provides a smooth baseline that reacts to form

The weights shift automatically as the season progresses:
  - Race 1:  Monte Carlo 40%, Bayesian 45%, Elo 15%
  - Race 12: Monte Carlo 55%, Bayesian 35%, Elo 10%
  - Race 24: Monte Carlo 70%, Bayesian 25%, Elo 5%

HOW TO RUN:
    python src/models/ensemble.py

OUTPUT:
    data/final_predictions.csv  — the main output used by the website
"""

import pandas as pd
import numpy as np
import os
import sys
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


# ─── LOAD ALL MODEL OUTPUTS ───────────────────────────────────────────────────

def load_all_predictions():
    """
    Loads the output from all three models.

    Returns:
        tuple: (monte_carlo_df, bayesian_df, elo_df)
                Each has driver_id and a probability column
    """
    print("Loading model outputs...")

    # Monte Carlo
    mc_path = os.path.join(DATA_DIR, "championship_probabilities.csv")
    if os.path.exists(mc_path):
        mc_df = pd.read_csv(mc_path)
        print(f"  Monte Carlo   : {len(mc_df)} drivers")
    else:
        print("  Monte Carlo   : NOT FOUND — run monte_carlo.py first")
        mc_df = pd.DataFrame()

    # Bayesian
    bayes_path = os.path.join(DATA_DIR, "bayesian_probabilities.csv")
    if os.path.exists(bayes_path):
        bayes_df = pd.read_csv(bayes_path)
        print(f"  Bayesian      : {len(bayes_df)} drivers")
    else:
        print("  Bayesian      : NOT FOUND — run bayesian_updater.py first")
        bayes_df = pd.DataFrame()

    # Elo
    elo_path = os.path.join(DATA_DIR, "elo_championship.csv")
    if os.path.exists(elo_path):
        elo_df = pd.read_csv(elo_path)
        print(f"  Elo           : {len(elo_df)} drivers")
    else:
        print("  Elo           : NOT FOUND — run elo_rating.py first")
        elo_df = pd.DataFrame()

    return mc_df, bayes_df, elo_df


# ─── COMPUTE ENSEMBLE WEIGHTS ─────────────────────────────────────────────────

def get_ensemble_weights(races_completed):
    """
    Returns the ensemble weights for each model based on
    how many races have been completed.

    Early season: trust Bayesian more (pre-season odds are informative)
    Late season:  trust Monte Carlo more (actual results dominate)

    Args:
        races_completed (int): Number of races completed so far

    Returns:
        dict: model → weight (sums to 1.0)
    """
    progress = races_completed / TOTAL_RACES  # 0.0 to 1.0

    # Monte Carlo weight increases linearly from 0.40 to 0.70
    mc_weight    = 0.40 + 0.30 * progress

    # Bayesian weight decreases from 0.45 to 0.25
    bayes_weight = 0.45 - 0.20 * progress

    # Elo weight decreases from 0.15 to 0.05
    elo_weight   = 0.15 - 0.10 * progress

    # Normalise to ensure they sum to exactly 1.0
    total = mc_weight + bayes_weight + elo_weight
    mc_weight    /= total
    bayes_weight /= total
    elo_weight   /= total

    return {
        "monte_carlo": round(mc_weight, 3),
        "bayesian":    round(bayes_weight, 3),
        "elo":         round(elo_weight, 3),
    }


# ─── BUILD ENSEMBLE ───────────────────────────────────────────────────────────

def build_ensemble(mc_df, bayes_df, elo_df, races_completed):
    """
    Combines the three model predictions into one final probability.

    Steps:
      1. Align all three DataFrames on driver_id
      2. Fill missing drivers with zero probability
      3. Apply ensemble weights
      4. Normalise final probabilities to sum to 100%

    Args:
        mc_df            (pd.DataFrame): Monte Carlo probabilities
        bayes_df         (pd.DataFrame): Bayesian probabilities
        elo_df           (pd.DataFrame): Elo probabilities
        races_completed  (int):          Races done so far

    Returns:
        pd.DataFrame: Final ensemble predictions
    """
    weights = get_ensemble_weights(races_completed)

    print(f"\nEnsemble weights (after {races_completed} races):")
    print(f"  Monte Carlo : {weights['monte_carlo']:.1%}")
    print(f"  Bayesian    : {weights['bayesian']:.1%}")
    print(f"  Elo         : {weights['elo']:.1%}")

    # ── Extract probability columns ───────────────────────────────────────────

    # Monte Carlo
    if not mc_df.empty:
        mc_probs = mc_df[["driver_id", "driver_name", "constructor_name",
                           "current_points", "championship_prob_pct"]].copy()
        mc_probs.columns = ["driver_id", "driver_name", "constructor_name",
                             "current_points", "mc_prob"]
    else:
        mc_probs = pd.DataFrame(columns=["driver_id", "driver_name",
                                          "constructor_name", "current_points", "mc_prob"])

    # Bayesian
    if not bayes_df.empty:
        bayes_probs = bayes_df[["driver_id", "probability", "cum_points"]].copy()
        bayes_probs.columns = ["driver_id", "bayes_prob", "bayes_points"]
    else:
        bayes_probs = pd.DataFrame(columns=["driver_id", "bayes_prob", "bayes_points"])

    # Elo
    if not elo_df.empty:
        elo_probs = elo_df[["driver_id", "elo_champ_probability"]].copy()
        elo_probs.columns = ["driver_id", "elo_prob"]
    else:
        elo_probs = pd.DataFrame(columns=["driver_id", "elo_prob"])

    # ── Merge all on driver_id ────────────────────────────────────────────────
    if mc_probs.empty and bayes_probs.empty and elo_probs.empty:
        print("ERROR: No model outputs found. Run all three models first.")
        return pd.DataFrame()

    # Start with whichever has the most drivers
    if not mc_probs.empty:
        combined = mc_probs.copy()
    elif not bayes_probs.empty:
        combined = bayes_probs[["driver_id"]].copy()
        combined["driver_name"]      = combined["driver_id"]
        combined["constructor_name"] = ""
        combined["current_points"]   = 0
        combined["mc_prob"]          = 0.0
    else:
        combined = elo_probs[["driver_id"]].copy()
        combined["driver_name"]      = combined["driver_id"]
        combined["constructor_name"] = ""
        combined["current_points"]   = 0
        combined["mc_prob"]          = 0.0

    # Merge Bayesian
    if not bayes_probs.empty:
        combined = combined.merge(
            bayes_probs[["driver_id", "bayes_prob"]],
            on="driver_id", how="left"
        )
    else:
        combined["bayes_prob"] = 0.0

    # Merge Elo
    if not elo_probs.empty:
        combined = combined.merge(
            elo_probs[["driver_id", "elo_prob"]],
            on="driver_id", how="left"
        )
    else:
        combined["elo_prob"] = 0.0

    # Fill missing values with 0
    combined["mc_prob"]    = combined["mc_prob"].fillna(0.0)
    combined["bayes_prob"] = combined["bayes_prob"].fillna(0.0)
    combined["elo_prob"]   = combined["elo_prob"].fillna(0.0)

    # ── Compute weighted ensemble probability ─────────────────────────────────
    combined["final_prob"] = (
        weights["monte_carlo"] * combined["mc_prob"]  +
        weights["bayesian"]    * combined["bayes_prob"] +
        weights["elo"]         * combined["elo_prob"]
    )

    # Normalise to sum to 100%
    total = combined["final_prob"].sum()
    if total > 0:
        combined["final_prob"] = combined["final_prob"] / total * 100

    # Round all probability columns
    for col in ["mc_prob", "bayes_prob", "elo_prob", "final_prob"]:
        combined[col] = combined[col].round(2)

    # Sort by final probability
    combined = combined.sort_values("final_prob", ascending=False).reset_index(drop=True)
    combined["rank"] = combined.index + 1

    # Add metadata
    combined["races_completed"]  = races_completed
    combined["races_remaining"]  = TOTAL_RACES - races_completed
    combined["updated_at"]       = datetime.now().strftime("%Y-%m-%d %H:%M")

    return combined


# ─── PRINT RESULTS ────────────────────────────────────────────────────────────

def print_results(combined):
    """Prints the final ensemble championship probabilities."""

    print("\n" + "=" * 72)
    print("FINAL CHAMPIONSHIP PREDICTIONS — ENSEMBLE MODEL")
    print("=" * 72)
    print(f"  {'Rank':<5} {'Driver':<25} {'Team':<20} "
          f"{'Pts':>5} {'MC':>6} {'Bayes':>7} {'Elo':>6} {'FINAL':>7}")
    print("-" * 72)

    for _, row in combined.head(12).iterrows():
        bar = "█" * int(row["final_prob"] / 2)
        print(f"  {int(row['rank']):<5} "
              f"{str(row['driver_name']):<25} "
              f"{str(row['constructor_name']):<20} "
              f"{int(row['current_points']):>5} "
              f"{row['mc_prob']:>5.1f}% "
              f"{row['bayes_prob']:>6.1f}% "
              f"{row['elo_prob']:>5.1f}% "
              f"{row['final_prob']:>6.1f}%  {bar}")

    print("=" * 72)
    print(f"\n  Based on: Monte Carlo {combined['races_completed'].iloc[0]} races | "
          f"Bayesian prior + {combined['races_completed'].iloc[0]} race updates | "
          f"Elo from {combined['races_completed'].iloc[0]} races")
    print(f"  Updated: {combined['updated_at'].iloc[0]}")


# ─── GET RACES COMPLETED ──────────────────────────────────────────────────────

def get_races_completed():
    """
    Reads race_results.csv to find how many 2026 races have been completed.

    Returns:
        int: Number of completed races
    """
    race_path = os.path.join(DATA_DIR, "race_results.csv")
    if not os.path.exists(race_path):
        return 0

    race_df = pd.read_csv(race_path)
    current = race_df[race_df["year"] == CURRENT_YEAR]

    if current.empty:
        return 0

    return int(current["round"].max())


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Full ensemble pipeline:
      1. Load all three model outputs
      2. Compute season-aware ensemble weights
      3. Build combined prediction
      4. Print and save final results
    """
    print("\n" + "=" * 60)
    print("F1 ENSEMBLE CHAMPIONSHIP PREDICTOR")
    print("=" * 60)

    # Load model outputs
    mc_df, bayes_df, elo_df = load_all_predictions()

    # How many races done?
    races_completed = get_races_completed()
    print(f"\nRaces completed: {races_completed} / {TOTAL_RACES}")

    # Build ensemble
    combined = build_ensemble(mc_df, bayes_df, elo_df, races_completed)

    if combined.empty:
        print("Could not build ensemble — check that all models have been run")
        return pd.DataFrame()

    # Print results
    print_results(combined)

    # Save
    out_path = os.path.join(DATA_DIR, "final_predictions.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return combined


# ─── HELPER: LOAD FINAL PREDICTIONS ──────────────────────────────────────────

def load_final_predictions():
    """
    Loads saved final predictions.
    Called by the Streamlit app.

    Returns:
        pd.DataFrame: Final ensemble championship predictions
    """
    path = os.path.join(DATA_DIR, "final_predictions.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"final_predictions.csv not found.\n"
            "Please run ensemble.py first."
        )

    return pd.read_csv(path)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    combined = run_full_pipeline()