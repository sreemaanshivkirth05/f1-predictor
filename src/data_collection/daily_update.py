"""
daily_update.py
---------------
Automatically updates all datasets and re-runs all models daily.

What this does every time it runs:
  1. Pulls latest race results from Jolpica (only new races since last run)
  2. Updates weather forecast for upcoming races
  3. Saves actual weather for past races
  4. Re-computes Elo ratings
  5. Re-runs Bayesian updater
  6. Re-runs Monte Carlo simulation
  7. Re-runs ensemble to generate final predictions
  8. Rebuilds master dataset if new race data was found

Smart updating — only rebuilds what has changed:
  - If no new race happened since last run → skip heavy steps
  - If a new race result appeared → rebuild everything
  - Weather always updated (forecasts change daily)

HOW TO RUN MANUALLY:
    python src/data_collection/daily_update.py

HOW TO AUTOMATE (Task Scheduler — Windows):
    Program : C:\f1-prediction-system\venv\Scripts\python.exe
    Arguments: src/data_collection/daily_update.py
    Start in : C:\f1-prediction-system
    Schedule : Daily at 8:00 AM

    After a race weekend, also schedule for:
      Sunday  23:00  (race night — results available)
      Monday  08:00  (morning refresh)
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# Path to python in venv
PYTHON = sys.executable

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

CURRENT_YEAR = 2026

# Scripts to run in order
SCRIPTS = {
    "weather_update":   os.path.join(SRC_DIR, "data_collection", "update_weather.py"),
    "elo":              os.path.join(SRC_DIR, "models", "elo_rating.py"),
    "bayesian":         os.path.join(SRC_DIR, "models", "bayesian_updater.py"),
    "monte_carlo":      os.path.join(SRC_DIR, "models", "monte_carlo.py"),
    "ensemble":         os.path.join(SRC_DIR, "models", "ensemble.py"),
    "race_data":        os.path.join(SRC_DIR, "data_collection", "get_ergast_data.py"),
    "build_dataset":    os.path.join(SRC_DIR, "feature_engineering", "build_dataset.py"),
    "xgboost":          os.path.join(SRC_DIR, "models", "xgboost_model.py"),
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "  ✓", "WARN": "  ⚠", "ERROR": "  ✗", "STEP": "\n►"}.get(level, "  ")
    print(f"{prefix} [{timestamp}] {msg}")


def run_script(script_path, label):
    """Runs a Python script and returns True if it succeeded."""
    if not os.path.exists(script_path):
        log(f"{label} script not found: {script_path}", "WARN")
        return False

    log(f"Running {label}...", "STEP")
    result = subprocess.run(
        [PYTHON, script_path],
        capture_output=False,
        cwd=PROJECT_ROOT,
    )

    if result.returncode == 0:
        log(f"{label} completed successfully", "INFO")
        return True
    else:
        log(f"{label} failed with return code {result.returncode}", "ERROR")
        return False


def get_last_known_round():
    """Returns the last completed round from race_results.csv."""
    path = os.path.join(DATA_DIR, "race_results.csv")
    if not os.path.exists(path):
        return 0, 0

    df = pd.read_csv(path)
    current = df[df["year"] == CURRENT_YEAR]
    if current.empty:
        return CURRENT_YEAR, 0

    return CURRENT_YEAR, int(current["round"].max())


def check_for_new_race():
    """
    Checks Jolpica API to see if a new race result is available
    that we don't have in our local data yet.

    Returns:
        bool: True if a new race result was found
    """
    import requests

    try:
        url      = f"http://api.jolpi.ca/ergast/f1/{CURRENT_YEAR}/results.json?limit=1000"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        api_rounds = len(races)

        _, local_rounds = get_last_known_round()

        if api_rounds > local_rounds:
            log(f"New race found! API has {api_rounds} rounds, we have {local_rounds}", "INFO")
            return True
        else:
            log(f"No new race. API: {api_rounds} rounds, Local: {local_rounds} rounds", "INFO")
            return False

    except Exception as e:
        log(f"Could not check for new race: {e}", "WARN")
        return False


def save_update_log(new_race_found, steps_run):
    """Saves a log of what was updated."""
    log_path = os.path.join(DATA_DIR, "update_log.csv")

    entry = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "new_race_found": new_race_found,
        "steps_run":      ", ".join(steps_run),
    }

    if os.path.exists(log_path):
        log_df = pd.read_csv(log_path)
        log_df = pd.concat([log_df, pd.DataFrame([entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([entry])

    # Keep last 100 entries
    log_df = log_df.tail(100)
    log_df.to_csv(log_path, index=False)


# ─── MAIN UPDATE PIPELINE ─────────────────────────────────────────────────────

def run_daily_update():
    """
    Smart daily update:

    Every day:
      1. Update weather forecast (always — forecasts change daily)
      2. Check if a new race happened

    If new race found:
      3. Pull new race data from Jolpica
      4. Rebuild master dataset
      5. Retrain XGBoost model
      6. Recompute Elo ratings
      7. Run Bayesian updater
      8. Run Monte Carlo simulation
      9. Run ensemble → update final_predictions.csv

    If no new race:
      3. Recompute Elo (fast — no retraining needed)
      4. Run Bayesian updater
      5. Run Monte Carlo simulation
      6. Run ensemble → update final_predictions.csv
    """
    print("\n" + "=" * 60)
    print("F1 DAILY UPDATE PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Python : {PYTHON}")
    print(f"Root   : {PROJECT_ROOT}")
    print("=" * 60)

    steps_run = []

    # ── Step 1: Always update weather ─────────────────────────────────────────
    log("STEP 1: Weather update", "STEP")
    if run_script(SCRIPTS["weather_update"], "Weather updater"):
        steps_run.append("weather")

    # ── Step 2: Check for new race ────────────────────────────────────────────
    log("STEP 2: Checking for new race results", "STEP")
    new_race_found = check_for_new_race()

    # ── Step 3: If new race — pull data and retrain ───────────────────────────
    if new_race_found:
        log("New race found — running full update pipeline", "INFO")

        log("STEP 3a: Pulling new race data", "STEP")
        if run_script(SCRIPTS["race_data"], "Race data (Jolpica)"):
            steps_run.append("race_data")

        log("STEP 3b: Rebuilding master dataset", "STEP")
        if run_script(SCRIPTS["build_dataset"], "Feature engineering"):
            steps_run.append("build_dataset")

        log("STEP 3c: Retraining XGBoost model", "STEP")
        if run_script(SCRIPTS["xgboost"], "XGBoost model"):
            steps_run.append("xgboost")

    else:
        log("No new race — skipping data pull and model retraining", "INFO")
        log("(XGBoost retrained only when new race results appear)", "INFO")

    # ── Step 4: Always update Elo, Bayesian, Monte Carlo, Ensemble ────────────
    log("STEP 4: Updating Elo ratings", "STEP")
    if run_script(SCRIPTS["elo"], "Elo ratings"):
        steps_run.append("elo")

    log("STEP 5: Running Bayesian updater", "STEP")
    if run_script(SCRIPTS["bayesian"], "Bayesian updater"):
        steps_run.append("bayesian")

    log("STEP 6: Running Monte Carlo simulation", "STEP")
    if run_script(SCRIPTS["monte_carlo"], "Monte Carlo"):
        steps_run.append("monte_carlo")

    log("STEP 7: Running ensemble", "STEP")
    if run_script(SCRIPTS["ensemble"], "Ensemble"):
        steps_run.append("ensemble")

    # ── Summary ───────────────────────────────────────────────────────────────
    save_update_log(new_race_found, steps_run)

    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)
    print(f"Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"New race   : {'YES' if new_race_found else 'No'}")
    print(f"Steps run  : {', '.join(steps_run)}")
    print(f"Output     : data/final_predictions.csv")
    print("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_daily_update()