"""
monte_carlo.py
--------------
Simulates the remaining F1 season 10,000 times to compute
championship win probabilities for each driver.

How it works:
  1. Locks in actual results for races already completed
  2. For remaining races, samples finishing positions from
     the XGBoost model's probability distributions
  3. Injects random noise for DNFs, safety cars, weather shifts
  4. Counts championship points after all 24 races
  5. Repeats 10,000 times
  6. Championship win probability = how often each driver wins

HOW TO RUN:
    python src/models/monte_carlo.py

REQUIREMENTS:
    Requires xgboost_model.pkl and elo_current.csv to exist.
    Run xgboost_model.py and elo_rating.py first.

OUTPUT:
    data/championship_probabilities.csv
    data/monte_carlo_results.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

sys.path.insert(0, SRC_DIR)


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Number of simulations to run
# 10,000 gives stable probabilities — more = more accurate but slower
N_SIMULATIONS = 10_000

# F1 points system (P1 to P20)
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6:  8, 7:  6, 8:  4, 9:  2, 10: 1,
}

# Bonus point for fastest lap (only if finishing in top 10)
FASTEST_LAP_POINT = 1

# DNF probability per driver per race (historical average ~3.5%)
BASE_DNF_PROB = 0.035

# Safety car probability per race (historical ~38%)
SAFETY_CAR_PROB = 0.38

# Probability of significant rain during race
BASE_RAIN_PROB = 0.15

# How much rain reshuffles the grid (std dev of position change)
RAIN_SHUFFLE_STD = 3.0

# Current season
CURRENT_YEAR = 2026

# Total races in the season
TOTAL_RACES = 24


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_all_data():
    """
    Loads all required data for the Monte Carlo simulation.

    Returns:
        tuple: (race_results_df, elo_df, model, feature_cols, master_df)
    """
    print("Loading data for Monte Carlo simulation...")

    # Race results (to know what has already happened)
    race_path = os.path.join(DATA_DIR, "race_results.csv")
    race_df   = pd.read_csv(race_path)
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce")
    race_df["points"]          = pd.to_numeric(race_df["points"], errors="coerce")
    print(f"  Race results   : {len(race_df):,} rows")

    # Current Elo ratings
    elo_path = os.path.join(DATA_DIR, "elo_current.csv")
    elo_df   = pd.read_csv(elo_path)
    print(f"  Elo ratings    : {len(elo_df)} drivers")

    # XGBoost model
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        model        = saved["model"]
        feature_cols = saved["feature_cols"]
        print(f"  XGBoost model  : loaded ({len(feature_cols)} features)")
    else:
        model        = None
        feature_cols = None
        print("  XGBoost model  : NOT FOUND — using Elo-only simulation")

    # Master dataset (for feature lookup for upcoming races)
    master_path = os.path.join(DATA_DIR, "master_dataset.csv")
    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path)
        print(f"  Master dataset : {len(master_df):,} rows")
    else:
        master_df = pd.DataFrame()
        print("  Master dataset : not found")

    return race_df, elo_df, model, feature_cols, master_df


# ─── CURRENT STANDINGS ────────────────────────────────────────────────────────

def get_current_standings(race_df, year=CURRENT_YEAR):
    """
    Computes actual points standings so far in the current season.

    Args:
        race_df (pd.DataFrame): Race results
        year    (int):          Current season year

    Returns:
        tuple: (standings_df, last_completed_round)
    """
    current = race_df[race_df["year"] == year].copy()

    if current.empty:
        print(f"  No {year} race data found — starting from Round 1")
        return pd.DataFrame(), 0

    # Sum actual points per driver
    standings = (
        current.groupby(["driver_id", "driver_name", "constructor_id", "constructor_name"])
        ["points"]
        .sum()
        .reset_index()
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )

    standings["position"] = standings.index + 1
    last_round = current["round"].max()

    print(f"  Current standings: {len(standings)} drivers, {last_round} races completed")
    return standings, last_round


# ─── RACE CALENDAR ────────────────────────────────────────────────────────────

def get_remaining_races(race_df, last_completed_round, year=CURRENT_YEAR):
    """
    Returns the list of remaining races in the season.

    Args:
        race_df               (pd.DataFrame): Race results
        last_completed_round  (int):          Last round already completed
        year                  (int):          Season year

    Returns:
        list: Round numbers for remaining races
    """
    remaining = list(range(last_completed_round + 1, TOTAL_RACES + 1))
    print(f"  Remaining races: {len(remaining)} "
          f"(rounds {last_completed_round + 1} to {TOTAL_RACES})")
    return remaining


# ─── DRIVER WIN PROBABILITIES ─────────────────────────────────────────────────

def get_driver_win_probs(drivers, elo_df, model, feature_cols,
                          master_df, round_num, year=CURRENT_YEAR):
    """
    Gets win probability for each driver at a specific race.

    Priority order:
      1. Use XGBoost model if available and features exist for this race
      2. Fall back to Elo-based probabilities

    Args:
        drivers      (list):           List of driver_ids in this race
        elo_df       (pd.DataFrame):   Current Elo ratings
        model        (xgb.XGBRanker):  Trained XGBoost model or None
        feature_cols (list):           XGBoost feature columns
        master_df    (pd.DataFrame):   Master feature dataset
        round_num    (int):            Race round number
        year         (int):            Season year

    Returns:
        dict: driver_id → win probability (sums to 1.0)
    """
    # ── Try XGBoost first ─────────────────────────────────────────────────────
    if model is not None and feature_cols is not None and not master_df.empty:
        race_features = master_df[
            (master_df["year"] == year) &
            (master_df["round"] == round_num) &
            (master_df["driver_id"].isin(drivers))
        ]

        if len(race_features) == len(drivers):
            # All drivers have features — use XGBoost
            try:
                X      = race_features[feature_cols].values
                scores = model.predict(X)
                driver_ids = race_features["driver_id"].tolist()

                # Softmax to get probabilities
                exp_s  = np.exp(scores - scores.max())
                probs  = exp_s / exp_s.sum()

                return dict(zip(driver_ids, probs))
            except Exception:
                pass  # Fall through to Elo

    # ── Fall back to Elo-based probabilities ──────────────────────────────────
    elo_map = dict(zip(elo_df["driver_id"], elo_df["elo_rating"]))

    ratings = np.array([
        elo_map.get(d, 1500) for d in drivers
    ])

    # Softmax with temperature
    temp      = 150.0
    exp_r     = np.exp((ratings - ratings.max()) / temp)
    probs     = exp_r / exp_r.sum()

    return dict(zip(drivers, probs))


# ─── SIMULATE ONE RACE ────────────────────────────────────────────────────────

def simulate_one_race(drivers, win_probs, wet_weather_ratings=None,
                       rain_prob=BASE_RAIN_PROB):
    """
    Simulates a single race result by sampling from win probabilities
    with random noise for DNFs, safety cars, and weather.

    Args:
        drivers             (list):  Driver IDs in this race
        win_probs           (dict):  driver_id → win probability
        wet_weather_ratings (dict):  driver_id → wet weather score (optional)
        rain_prob           (float): Probability of rain in this race

    Returns:
        dict: driver_id → finishing position (1 = winner)
              DNF drivers get position 20+
    """
    n_drivers = len(drivers)
    probs     = np.array([win_probs.get(d, 1.0 / n_drivers) for d in drivers])

    # ── Rain adjustment ───────────────────────────────────────────────────────
    raining = np.random.random() < rain_prob

    if raining and wet_weather_ratings is not None:
        # Boost drivers with good wet weather ratings
        # Wet weather score ranges from -1 (bad in wet) to +1 (great in wet)
        wet_boost = np.array([
            wet_weather_ratings.get(d, 0.0) for d in drivers
        ])
        # Convert to a multiplier: score 1.0 → multiply by 1.5
        wet_multiplier = 1.0 + 0.5 * wet_boost
        probs = probs * wet_multiplier
        probs = probs / probs.sum()  # Re-normalise

    # ── DNF injection ─────────────────────────────────────────────────────────
    # Each driver independently has a chance of DNF
    dnf_mask = np.random.random(n_drivers) < BASE_DNF_PROB

    # ── Safety car shuffle ────────────────────────────────────────────────────
    safety_car = np.random.random() < SAFETY_CAR_PROB

    if safety_car:
        # Safety car partially randomises the grid
        # Add random noise to probabilities
        noise = np.abs(np.random.normal(0, 0.1, n_drivers))
        probs = probs + noise
        probs = probs / probs.sum()

    # ── Sample finishing order ────────────────────────────────────────────────
    # Sample without replacement — each driver gets a unique position
    # Higher probability = more likely to be sampled first = higher finish
    sampled_order = np.random.choice(
        n_drivers,
        size=n_drivers,
        replace=False,
        p=probs
    )

    # Assign finishing positions
    positions = {}
    finish_pos = 1

    for idx in sampled_order:
        driver = drivers[idx]
        if dnf_mask[idx]:
            # DNF — assign position 20+ (won't score points)
            positions[driver] = 20 + idx
        else:
            positions[driver] = finish_pos
            finish_pos += 1

    return positions


# ─── COMPUTE POINTS FROM RESULT ───────────────────────────────────────────────

def compute_race_points(positions):
    """
    Converts finishing positions to championship points.

    Args:
        positions (dict): driver_id → finishing position

    Returns:
        dict: driver_id → points scored
    """
    points = {}
    finishers = sorted(
        [(d, p) for d, p in positions.items() if p <= 20],
        key=lambda x: x[1]
    )

    for driver, pos in finishers:
        race_pts = POINTS_SYSTEM.get(pos, 0)

        # Fastest lap bonus (randomly assign to a top-10 finisher)
        if pos <= 10 and np.random.random() < (1.0 / 10):
            race_pts += FASTEST_LAP_POINT

        points[driver] = race_pts

    return points


# ─── RUN SIMULATION ───────────────────────────────────────────────────────────

def run_simulation(race_df, elo_df, model, feature_cols,
                   master_df, year=CURRENT_YEAR):
    """
    Runs N_SIMULATIONS full season simulations.

    For each simulation:
      1. Start with actual points from completed races
      2. Simulate each remaining race
      3. Add up total points
      4. Record who won the championship

    Args:
        race_df      (pd.DataFrame):   Race results
        elo_df       (pd.DataFrame):   Current Elo ratings
        model        (xgb.XGBRanker):  XGBoost model or None
        feature_cols (list):           Feature columns
        master_df    (pd.DataFrame):   Master dataset
        year         (int):            Season year

    Returns:
        tuple: (championship_wins_dict, all_simulations_df)
    """
    print(f"\nRunning {N_SIMULATIONS:,} season simulations...")

    # Get current standings
    standings, last_round = get_current_standings(race_df, year)
    remaining_rounds      = get_remaining_races(race_df, last_round, year)

    # Get active drivers for this season
    if not standings.empty:
        drivers = standings["driver_id"].tolist()
        actual_points = dict(zip(standings["driver_id"], standings["points"]))
    else:
        # Use all drivers from master dataset or Elo
        drivers = elo_df["driver_id"].tolist()[:20]
        actual_points = {d: 0 for d in drivers}

    print(f"  Active drivers : {len(drivers)}")
    print(f"  Starting points: {actual_points}")

    # Build wet weather ratings per driver
    wet_ratings = {}
    if not master_df.empty and "wet_vs_dry_delta" in master_df.columns:
        latest_wet = (
            master_df.sort_values(["year", "round"])
            .groupby("driver_id")
            .last()
            .reset_index()
            [["driver_id", "wet_vs_dry_delta"]]
        )
        # Invert: negative delta means better in wet (lower finish pos = better)
        for _, row in latest_wet.iterrows():
            wet_ratings[row["driver_id"]] = -row["wet_vs_dry_delta"] / 10.0

    # Championship win counter
    champ_wins = {d: 0 for d in drivers}

    # Storage for all simulation results
    sim_records = []

    # ── Run simulations ───────────────────────────────────────────────────────
    for sim_num in tqdm(range(N_SIMULATIONS), desc="Simulating", unit="sim"):
        # Start with actual earned points
        sim_points = actual_points.copy()

        # Simulate each remaining race
        for round_num in remaining_rounds:
            # Get win probabilities for this race
            win_probs = get_driver_win_probs(
                drivers, elo_df, model, feature_cols,
                master_df, round_num, year
            )

            # Simulate the race
            positions = simulate_one_race(
                drivers, win_probs, wet_ratings
            )

            # Add points
            race_pts = compute_race_points(positions)
            for driver in drivers:
                sim_points[driver] = sim_points.get(driver, 0) + race_pts.get(driver, 0)

        # Find champion (highest points, ties broken by wins — simplified here)
        champion = max(sim_points, key=sim_points.get)
        champ_wins[champion] += 1

        # Save summary of this simulation
        if sim_num < 1000:  # Only save first 1000 for file size
            sim_records.append({
                "sim_num":   sim_num,
                "champion":  champion,
                "max_points": sim_points[champion],
            })

    # ── Compute probabilities ─────────────────────────────────────────────────
    total_sims = N_SIMULATIONS

    champ_probs = []
    for driver in drivers:
        wins    = champ_wins[driver]
        prob    = wins / total_sims * 100

        # Get driver name and team
        info = standings[standings["driver_id"] == driver] if not standings.empty else pd.DataFrame()
        if info.empty:
            elo_info = elo_df[elo_df["driver_id"] == driver]
            name = elo_info["driver_name"].values[0] if len(elo_info) > 0 else driver
            team = elo_info["constructor_name"].values[0] if len(elo_info) > 0 else ""
        else:
            name = info["driver_name"].values[0]
            team = info["constructor_name"].values[0]

        champ_probs.append({
            "driver_id":            driver,
            "driver_name":          name,
            "constructor_name":     team,
            "current_points":       actual_points.get(driver, 0),
            "championship_wins":    wins,
            "championship_prob_pct": round(prob, 2),
        })

    champ_df = (
        pd.DataFrame(champ_probs)
        .sort_values("championship_prob_pct", ascending=False)
        .reset_index(drop=True)
    )

    champ_df["rank"] = champ_df.index + 1

    sim_df = pd.DataFrame(sim_records)

    return champ_df, sim_df


# ─── PRINT RESULTS ────────────────────────────────────────────────────────────

def print_results(champ_df):
    """Prints the championship probability table to the terminal."""

    print("\n" + "=" * 65)
    print("CHAMPIONSHIP WIN PROBABILITIES")
    print(f"Based on {N_SIMULATIONS:,} Monte Carlo simulations")
    print("=" * 65)
    print(f"  {'Rank':<5} {'Driver':<25} {'Team':<22} {'Pts':>5} {'Prob':>6}")
    print("-" * 65)

    for _, row in champ_df.head(15).iterrows():
        bar  = "█" * int(row["championship_prob_pct"] / 2)
        print(f"  {int(row['rank']):<5} "
              f"{row['driver_name']:<25} "
              f"{str(row['constructor_name']):<22} "
              f"{int(row['current_points']):>5} "
              f"{row['championship_prob_pct']:>5.1f}% "
              f"{bar}")

    print("=" * 65)


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Full Monte Carlo pipeline:
      1. Load all data
      2. Run 10,000 season simulations
      3. Print championship probabilities
      4. Save results to CSV
    """
    print("\n" + "=" * 60)
    print("F1 MONTE CARLO CHAMPIONSHIP SIMULATOR")
    print("=" * 60)

    # Load data
    race_df, elo_df, model, feature_cols, master_df = load_all_data()

    # Run simulation
    champ_df, sim_df = run_simulation(
        race_df, elo_df, model, feature_cols, master_df
    )

    # Print results
    print_results(champ_df)

    # Save outputs
    champ_path = os.path.join(DATA_DIR, "championship_probabilities.csv")
    sim_path   = os.path.join(DATA_DIR, "monte_carlo_results.csv")

    champ_df.to_csv(champ_path, index=False)
    sim_df.to_csv(sim_path,     index=False)

    print(f"\nSaved:")
    print(f"  {champ_path}")
    print(f"  {sim_path}")

    return champ_df, sim_df


# ─── HELPER: LOAD PROBABILITIES ───────────────────────────────────────────────

def load_championship_probabilities():
    """
    Loads saved championship probabilities.
    Called by the Streamlit app.

    Returns:
        pd.DataFrame: Championship probabilities per driver
    """
    path = os.path.join(DATA_DIR, "championship_probabilities.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"championship_probabilities.csv not found.\n"
            "Please run monte_carlo.py first."
        )

    return pd.read_csv(path)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bar...")
        os.system("pip install tqdm")
        from tqdm import tqdm

    champ_df, sim_df = run_full_pipeline()