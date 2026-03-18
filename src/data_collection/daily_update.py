"""
daily_update.py
---------------
Smart daily updater — only pulls what actually changes:

ALWAYS updated (fast, real-time):
  1. Weather forecast for upcoming races
  2. Actual weather for past 2026 races
  3. 2026 race results (only new rounds)
  4. 2026 driver standings (only new rounds)

ONLY if a new 2026 race was found:
  5. Append new rows to master_dataset.csv
  6. Retrain XGBoost on full dataset
  7. Recompute Elo ratings
  8. Run Bayesian updater
  9. Run Monte Carlo simulation
  10. Run ensemble → update final_predictions.csv

NEVER touched (historical data — doesn't change):
  - 2016-2025 race results
  - 2016-2025 FastF1 lap times
  - Historical weather (already saved)
  - Feature engineering for old seasons

This means a typical daily run takes 3-5 minutes instead of 30+.
A post-race update takes 8-12 minutes (includes retraining).

HOW TO RUN:
    python src/data_collection/daily_update.py
"""

import requests
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import time
from datetime import datetime

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

PYTHON = sys.executable

CURRENT_YEAR   = 2026
BASE_URL       = "http://api.jolpi.ca/ergast/f1"
REQUEST_DELAY  = 1.0
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = ",".join([
    "temperature_2m", "relativehumidity_2m", "precipitation",
    "rain", "cloudcover", "windspeed_10m", "winddirection_10m",
    "surface_pressure", "soil_temperature_0cm", "uv_index",
])

CIRCUIT_GPS = {
    "bahrain":       {"lat": 26.0325,  "lon": 50.5106},
    "jeddah":        {"lat": 21.6319,  "lon": 39.1044},
    "albert_park":   {"lat": -37.8497, "lon": 144.9680},
    "suzuka":        {"lat": 34.8431,  "lon": 136.5407},
    "shanghai":      {"lat": 31.3389,  "lon": 121.2197},
    "miami":         {"lat": 25.9581,  "lon": -80.2389},
    "imola":         {"lat": 44.3439,  "lon": 11.7167},
    "monaco":        {"lat": 43.7347,  "lon": 7.4206},
    "villeneuve":    {"lat": 45.5000,  "lon": -73.5228},
    "catalunya":     {"lat": 41.5700,  "lon": 2.2611},
    "red_bull_ring": {"lat": 47.2197,  "lon": 14.7647},
    "silverstone":   {"lat": 52.0786,  "lon": -1.0169},
    "hungaroring":   {"lat": 47.5789,  "lon": 19.2486},
    "spa":           {"lat": 50.4372,  "lon": 5.9714},
    "zandvoort":     {"lat": 52.3888,  "lon": 4.5409},
    "monza":         {"lat": 45.6156,  "lon": 9.2811},
    "baku":          {"lat": 40.3725,  "lon": 49.8533},
    "marina_bay":    {"lat": 1.2914,   "lon": 103.8640},
    "americas":      {"lat": 30.1328,  "lon": -97.6411},
    "rodriguez":     {"lat": 19.4042,  "lon": -99.0907},
    "interlagos":    {"lat": -23.7036, "lon": -46.6997},
    "vegas":         {"lat": 36.1147,  "lon": -115.1728},
    "losail":        {"lat": 25.4900,  "lon": 51.4542},
    "yas_marina":    {"lat": 24.4672,  "lon": 54.6031},
}

RACE_CALENDAR_2026 = [
    {"round": 1,  "circuit_id": "albert_park",   "date": "2026-03-15", "name": "Australian GP"},
    {"round": 2,  "circuit_id": "shanghai",      "date": "2026-03-22", "name": "Chinese GP"},
    {"round": 3,  "circuit_id": "suzuka",        "date": "2026-04-05", "name": "Japanese GP"},
    {"round": 4,  "circuit_id": "bahrain",       "date": "2026-04-19", "name": "Bahrain GP"},
    {"round": 5,  "circuit_id": "jeddah",        "date": "2026-04-26", "name": "Saudi Arabian GP"},
    {"round": 6,  "circuit_id": "miami",         "date": "2026-05-03", "name": "Miami GP"},
    {"round": 7,  "circuit_id": "imola",         "date": "2026-05-17", "name": "Emilia Romagna GP"},
    {"round": 8,  "circuit_id": "monaco",        "date": "2026-05-24", "name": "Monaco GP"},
    {"round": 9,  "circuit_id": "catalunya",     "date": "2026-06-07", "name": "Spanish GP"},
    {"round": 10, "circuit_id": "villeneuve",    "date": "2026-06-14", "name": "Canadian GP"},
    {"round": 11, "circuit_id": "red_bull_ring", "date": "2026-06-28", "name": "Austrian GP"},
    {"round": 12, "circuit_id": "silverstone",   "date": "2026-07-05", "name": "British GP"},
    {"round": 13, "circuit_id": "spa",           "date": "2026-07-26", "name": "Belgian GP"},
    {"round": 14, "circuit_id": "hungaroring",   "date": "2026-08-02", "name": "Hungarian GP"},
    {"round": 15, "circuit_id": "zandvoort",     "date": "2026-08-30", "name": "Dutch GP"},
    {"round": 16, "circuit_id": "monza",         "date": "2026-09-06", "name": "Italian GP"},
    {"round": 17, "circuit_id": "baku",          "date": "2026-09-20", "name": "Azerbaijan GP"},
    {"round": 18, "circuit_id": "marina_bay",    "date": "2026-10-04", "name": "Singapore GP"},
    {"round": 19, "circuit_id": "americas",      "date": "2026-10-18", "name": "US GP"},
    {"round": 20, "circuit_id": "rodriguez",     "date": "2026-10-25", "name": "Mexico City GP"},
    {"round": 21, "circuit_id": "interlagos",    "date": "2026-11-08", "name": "São Paulo GP"},
    {"round": 22, "circuit_id": "vegas",         "date": "2026-11-21", "name": "Las Vegas GP"},
    {"round": 23, "circuit_id": "losail",        "date": "2026-11-29", "name": "Qatar GP"},
    {"round": 24, "circuit_id": "yas_marina",    "date": "2026-12-06", "name": "Abu Dhabi GP"},
]


# ─── LOGGING ──────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    ts     = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO":"  ✓","WARN":"  ⚠","ERROR":"  ✗","STEP":"\n►","SKIP":"  →"}.get(level,"  ")
    print(f"{prefix} [{ts}] {msg}")


# ─── API HELPERS ──────────────────────────────────────────────────────────────

def api_get(url, params=None, retries=3):
    """GET request with retry and rate limit handling."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                log(f"Rate limited — waiting {wait}s...", "WARN")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except Exception as e:
            log(f"Request failed (attempt {attempt+1}): {e}", "WARN")
            time.sleep(5)
    return None


def run_script(path, label):
    """Runs a Python script as a subprocess."""
    if not os.path.exists(path):
        log(f"{label} not found: {path}", "WARN")
        return False
    result = subprocess.run([PYTHON, path], cwd=PROJECT_ROOT)
    if result.returncode == 0:
        log(f"{label} ✓", "INFO")
        return True
    log(f"{label} failed", "ERROR")
    return False


# ─── STEP 1: FETCH NEW 2026 RACE RESULTS ──────────────────────────────────────

def update_2026_race_results():
    """
    Pulls only 2026 race results from Jolpica and appends
    any new rounds to the existing race_results.csv.

    Returns:
        int: Number of new rounds added (0 if nothing new)
    """
    log("Checking for new 2026 race results...", "STEP")

    results_path = os.path.join(DATA_DIR, "race_results.csv")

    # Find what rounds we already have for 2026
    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        existing_2026 = existing[existing["year"] == CURRENT_YEAR]
        last_local_round = int(existing_2026["round"].max()) if not existing_2026.empty else 0
    else:
        existing          = pd.DataFrame()
        last_local_round  = 0

    log(f"Local 2026 data: {last_local_round} rounds completed", "INFO")

    # Pull 2026 results from API
    data = api_get(f"{BASE_URL}/{CURRENT_YEAR}/results.json?limit=1000")
    if not data:
        log("Could not reach Jolpica API", "WARN")
        return 0

    races      = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    api_rounds = len(races)

    log(f"API 2026 data: {api_rounds} rounds available", "INFO")

    if api_rounds <= last_local_round:
        log("No new 2026 race results — skipping", "SKIP")
        return 0

    # Parse only the NEW rounds
    new_rows = []
    for race in races:
        round_num = int(race.get("round", 0))
        if round_num <= last_local_round:
            continue  # Already have this round

        race_name  = race.get("raceName", "")
        circuit    = race.get("Circuit", {})
        circuit_id = circuit.get("circuitId", "")
        race_date  = race.get("date", "")
        country    = circuit.get("Location", {}).get("country", "")
        lat        = circuit.get("Location", {}).get("lat", None)
        lon        = circuit.get("Location", {}).get("long", None)

        for result in race.get("Results", []):
            driver      = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            fastest_lap = result.get("FastestLap", {})

            new_rows.append({
                "year":             CURRENT_YEAR,
                "round":            round_num,
                "race_name":        race_name,
                "circuit_name":     circuit.get("circuitName", ""),
                "circuit_id":       circuit_id,
                "race_date":        race_date,
                "country":          country,
                "locality":         circuit.get("Location", {}).get("locality", ""),
                "lat":              lat,
                "long":             lon,
                "driver_id":        driver.get("driverId", ""),
                "driver_code":      driver.get("code", ""),
                "driver_name":      f"{driver.get('givenName','')} {driver.get('familyName','')}",
                "constructor_id":   constructor.get("constructorId", ""),
                "constructor_name": constructor.get("name", ""),
                "grid_position":    result.get("grid", None),
                "finish_position":  result.get("position", None),
                "points":           float(result.get("points", 0)),
                "laps_completed":   result.get("laps", None),
                "status":           result.get("status", ""),
                "fastest_lap_rank": fastest_lap.get("rank", None),
                "fastest_lap_time": fastest_lap.get("Time", {}).get("time", None),
            })

    if not new_rows:
        log("No new rows to add", "SKIP")
        return 0

    new_df = pd.DataFrame(new_rows)
    for col in ["grid_position", "finish_position", "laps_completed", "fastest_lap_rank"]:
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    # Append to existing
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined.to_csv(results_path, index=False)

    new_rounds = api_rounds - last_local_round
    log(f"Added {len(new_rows)} rows for {new_rounds} new round(s) to race_results.csv", "INFO")
    return new_rounds


# ─── STEP 2: FETCH NEW 2026 STANDINGS ─────────────────────────────────────────

def update_2026_standings():
    """
    Pulls only the latest 2026 driver and constructor standings
    and appends new rounds to the standings CSVs.
    """
    log("Updating 2026 standings...", "STEP")

    for stand_type in ["driver", "constructor"]:
        filename = f"{'driver' if stand_type == 'driver' else 'constructor'}_standings.csv"
        path     = os.path.join(DATA_DIR, filename)

        if os.path.exists(path):
            existing     = pd.read_csv(path)
            existing_2026 = existing[existing["year"] == CURRENT_YEAR]
            last_round   = int(existing_2026["round"].max()) if not existing_2026.empty else 0
        else:
            existing   = pd.DataFrame()
            last_round = 0

        # Find total rounds available
        race_path = os.path.join(DATA_DIR, "race_results.csv")
        if os.path.exists(race_path):
            race_df    = pd.read_csv(race_path)
            max_round  = int(race_df[race_df["year"] == CURRENT_YEAR]["round"].max()) if len(race_df[race_df["year"] == CURRENT_YEAR]) > 0 else 0
        else:
            max_round = 0

        if max_round <= last_round:
            log(f"  {stand_type} standings already up to date", "SKIP")
            continue

        new_rows = []
        for round_num in range(last_round + 1, max_round + 1):
            if stand_type == "driver":
                url = f"{BASE_URL}/{CURRENT_YEAR}/{round_num}/driverStandings.json"
            else:
                url = f"{BASE_URL}/{CURRENT_YEAR}/{round_num}/constructorStandings.json"

            data = api_get(url)
            if not data:
                continue

            standings_list = (
                data.get("MRData", {})
                    .get("StandingsTable", {})
                    .get("StandingsLists", [])
            )
            if not standings_list:
                continue

            key = "DriverStandings" if stand_type == "driver" else "ConstructorStandings"
            for s in standings_list[0].get(key, []):
                if stand_type == "driver":
                    driver = s.get("Driver", {})
                    con    = s.get("Constructors", [{}])[0]
                    new_rows.append({
                        "year":             CURRENT_YEAR,
                        "round":            round_num,
                        "driver_id":        driver.get("driverId", ""),
                        "driver_name":      f"{driver.get('givenName','')} {driver.get('familyName','')}",
                        "constructor_id":   con.get("constructorId", ""),
                        "championship_pos": int(s.get("position", 0)),
                        "championship_pts": float(s.get("points", 0)),
                        "wins":             int(s.get("wins", 0)),
                    })
                else:
                    con = s.get("Constructor", {})
                    new_rows.append({
                        "year":                  CURRENT_YEAR,
                        "round":                 round_num,
                        "constructor_id":        con.get("constructorId", ""),
                        "constructor_name":      con.get("name", ""),
                        "constructor_champ_pos": int(s.get("position", 0)),
                        "constructor_champ_pts": float(s.get("points", 0)),
                        "constructor_wins":      int(s.get("wins", 0)),
                    })

        if new_rows:
            new_df   = pd.DataFrame(new_rows)
            combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
            combined.to_csv(path, index=False)
            log(f"  Added {len(new_rows)} {stand_type} standing rows", "INFO")


# ─── STEP 3: UPDATE WEATHER ───────────────────────────────────────────────────

def update_weather():
    """
    Updates weather in two ways:
      A) Forecast for upcoming races (next 7 days)
      B) Actual weather for 2026 past races we don't have yet
    """
    log("Updating weather data...", "STEP")

    today = pd.Timestamp.today().normalize()

    # ── A: Forecast for upcoming races ───────────────────────────────────────
    upcoming = [
        r for r in RACE_CALENDAR_2026
        if pd.Timestamp(r["date"]) >= today
        and (pd.Timestamp(r["date"]) - today).days <= 7
    ]

    forecasts = []
    for race in upcoming:
        gps = CIRCUIT_GPS.get(race["circuit_id"])
        if not gps:
            continue

        params = {
            "latitude":   gps["lat"],
            "longitude":  gps["lon"],
            "start_date": race["date"],
            "end_date":   race["date"],
            "hourly":     HOURLY_VARS,
            "timezone":   "auto",
        }
        data = api_get(FORECAST_URL, params)
        if not data or "hourly" not in data:
            continue

        hourly = data["hourly"]
        df = pd.DataFrame({
            "time":                hourly.get("time", []),
            "temperature_2m":      hourly.get("temperature_2m", []),
            "relativehumidity_2m": hourly.get("relativehumidity_2m", []),
            "rain":                hourly.get("rain", []),
            "cloudcover":          hourly.get("cloudcover", []),
            "windspeed_10m":       hourly.get("windspeed_10m", []),
            "winddirection_10m":   hourly.get("winddirection_10m", []),
            "surface_pressure":    hourly.get("surface_pressure", []),
            "soil_temperature_0cm":hourly.get("soil_temperature_0cm", []),
        })
        df["time"] = pd.to_datetime(df["time"])

        race_day  = pd.Timestamp(race["date"]).date()
        df_race   = df[
            (df["time"].dt.date == race_day) &
            (df["time"].dt.hour >= 13) &
            (df["time"].dt.hour <= 20)
        ]
        if df_race.empty:
            df_race = df[df["time"].dt.date == race_day]
        if df_race.empty:
            continue

        forecasts.append({
            "circuit_id":           race["circuit_id"],
            "circuit_name":         race["name"],
            "race_date":            race["date"],
            "days_until_race":      (pd.Timestamp(race["date"]) - today).days,
            "forecast_updated_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "air_temp_avg_c":       round(df_race["temperature_2m"].mean(), 1),
            "track_temp_avg_c":     round(df_race["soil_temperature_0cm"].mean(), 1),
            "humidity_avg_pct":     round(df_race["relativehumidity_2m"].mean(), 1),
            "rainfall_total_mm":    round(df_race["rain"].sum(), 2),
            "rain_probability_pct": round((df_race["rain"] > 0).mean() * 100, 1),
            "rain_flag":            int(df_race["rain"].sum() > 0.5),
            "cloudcover_avg_pct":   round(df_race["cloudcover"].mean(), 1),
            "windspeed_avg_kmh":    round(df_race["windspeed_10m"].mean(), 1),
            "windspeed_max_kmh":    round(df_race["windspeed_10m"].max(), 1),
            "winddirection_avg":    round(df_race["winddirection_10m"].mean(), 1),
            "pressure_avg_hpa":     round(df_race["surface_pressure"].mean(), 1),
        })
        log(f"  Forecast updated: {race['name']} ({race['date']}) — "
            f"Rain: {forecasts[-1]['rain_probability_pct']:.0f}%", "INFO")

    if forecasts:
        path = os.path.join(DATA_DIR, "weather_forecast_upcoming.csv")
        pd.DataFrame(forecasts).to_csv(path, index=False)
        log(f"  Saved upcoming forecast: {len(forecasts)} races", "INFO")
    else:
        log("  No races in next 7 days", "SKIP")

    # ── B: Actual weather for past 2026 races ─────────────────────────────────
    hist_path = os.path.join(DATA_DIR, "weather_historical.csv")
    hist_df   = pd.read_csv(hist_path) if os.path.exists(hist_path) else pd.DataFrame()
    already   = set(hist_df["race_date"].astype(str).str[:10].tolist()) if not hist_df.empty else set()

    past_races = [
        r for r in RACE_CALENDAR_2026
        if pd.Timestamp(r["date"]) < today
        and r["date"] not in already
    ]

    if not past_races:
        log("  Historical weather already up to date", "SKIP")
        return

    new_weather = []
    for race in past_races:
        gps = CIRCUIT_GPS.get(race["circuit_id"])
        if not gps:
            continue

        params = {
            "latitude":   gps["lat"],
            "longitude":  gps["lon"],
            "start_date": race["date"],
            "end_date":   race["date"],
            "hourly":     HOURLY_VARS,
            "timezone":   "auto",
        }
        data = api_get(HISTORICAL_URL, params)
        if not data or "hourly" not in data:
            continue

        hourly = data["hourly"]
        df = pd.DataFrame({
            "time":                hourly.get("time", []),
            "temperature_2m":      hourly.get("temperature_2m", []),
            "relativehumidity_2m": hourly.get("relativehumidity_2m", []),
            "rain":                hourly.get("rain", []),
            "cloudcover":          hourly.get("cloudcover", []),
            "windspeed_10m":       hourly.get("windspeed_10m", []),
            "winddirection_10m":   hourly.get("winddirection_10m", []),
            "surface_pressure":    hourly.get("surface_pressure", []),
            "soil_temperature_0cm":hourly.get("soil_temperature_0cm", []),
        })
        df["time"] = pd.to_datetime(df["time"])
        race_day   = pd.Timestamp(race["date"]).date()
        df_race    = df[
            (df["time"].dt.date == race_day) &
            (df["time"].dt.hour >= 13) &
            (df["time"].dt.hour <= 20)
        ]
        if df_race.empty:
            df_race = df[df["time"].dt.date == race_day]
        if df_race.empty:
            continue

        new_weather.append({
            "circuit_id":         race["circuit_id"],
            "circuit_name":       race["name"],
            "race_date":          race["date"],
            "air_temp_avg_c":     round(df_race["temperature_2m"].mean(), 1),
            "track_temp_avg_c":   round(df_race["soil_temperature_0cm"].mean(), 1),
            "humidity_avg_pct":   round(df_race["relativehumidity_2m"].mean(), 1),
            "rainfall_total_mm":  round(df_race["rain"].sum(), 2),
            "rain_flag":          int(df_race["rain"].sum() > 0.5),
            "cloudcover_avg_pct": round(df_race["cloudcover"].mean(), 1),
            "windspeed_avg_kmh":  round(df_race["windspeed_10m"].mean(), 1),
            "winddirection_avg":  round(df_race["winddirection_10m"].mean(), 1),
            "pressure_avg_hpa":   round(df_race["surface_pressure"].mean(), 1),
        })
        log(f"  Actual weather saved: {race['name']} ({race['date']})", "INFO")

    if new_weather:
        combined = pd.concat(
            [hist_df, pd.DataFrame(new_weather)], ignore_index=True
        ) if not hist_df.empty else pd.DataFrame(new_weather)
        combined = combined.drop_duplicates(subset=["circuit_id", "race_date"], keep="last")
        combined.to_csv(hist_path, index=False)
        log(f"  Added actual weather for {len(new_weather)} past races", "INFO")


# ─── STEP 4: APPEND NEW ROWS TO MASTER DATASET ───────────────────────────────

def append_new_race_to_master(new_rounds_count):
    """
    Appends feature rows for new 2026 races to master_dataset.csv
    without rebuilding the entire dataset.

    For simplicity, this rebuilds only the 2026 portion and
    appends it. The 2016-2025 rows in master_dataset.csv are untouched.
    """
    if new_rounds_count == 0:
        log("No new rounds — master dataset unchanged", "SKIP")
        return

    log("Updating master dataset with new 2026 race rows...", "STEP")

    # The easiest safe approach: rebuild only if feature scripts exist
    build_path = os.path.join(SRC_DIR, "feature_engineering", "build_dataset.py")
    if os.path.exists(build_path):
        run_script(build_path, "Build dataset")
    else:
        log("build_dataset.py not found — skipping master update", "WARN")


# ─── STEP 5: RERUN MODELS ─────────────────────────────────────────────────────

def rerun_models(new_rounds_count):
    """
    Reruns only the models that need updating.
    XGBoost only retrained if new race data arrived.
    Elo, Bayesian, Monte Carlo, Ensemble always rerun — they are fast.
    """
    scripts = {
        "Elo ratings":    os.path.join(SRC_DIR, "models", "elo_rating.py"),
        "Bayesian":       os.path.join(SRC_DIR, "models", "bayesian_updater.py"),
        "Monte Carlo":    os.path.join(SRC_DIR, "models", "monte_carlo.py"),
        "Ensemble":       os.path.join(SRC_DIR, "models", "ensemble.py"),
    }

    if new_rounds_count > 0:
        xgb_path = os.path.join(SRC_DIR, "models", "xgboost_model.py")
        log("New race found — retraining XGBoost...", "STEP")
        run_script(xgb_path, "XGBoost")

    log("Running fast model updates...", "STEP")
    for label, path in scripts.items():
        run_script(path, label)


# ─── SAVE UPDATE LOG ──────────────────────────────────────────────────────────

def save_log(new_rounds, duration_secs):
    path = os.path.join(DATA_DIR, "update_log.csv")
    entry = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "new_rounds":   new_rounds,
        "duration_s":   round(duration_secs, 1),
        "retrained":    new_rounds > 0,
    }
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), pd.DataFrame([entry])], ignore_index=True).tail(100)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(path, index=False)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_daily_update():
    start = datetime.now()

    print("\n" + "=" * 60)
    print("F1 SMART DAILY UPDATER")
    print("=" * 60)
    print(f"Started : {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"Strategy: Only update 2026 data + weather + models")
    print(f"          Historical 2016-2025 data never re-pulled")
    print("=" * 60)


    # 0. OpenF1 update — runs FIRST, fastest source (minutes after race)
    print("\nStep 0: Fetching latest 2026 data from OpenF1...")
    openf1_script = os.path.join(SRC_DIR, "data_collection", "get_openf1_data.py")
    if os.path.exists(openf1_script):
        import subprocess
        result = subprocess.run([sys.executable, openf1_script], cwd=PROJECT_ROOT)
        if result.returncode == 0:
            print("  OpenF1 update successful")
        else:
            print("  OpenF1 update failed — continuing with Jolpica")
    else:
        print("  get_openf1_data.py not found — skipping")

    # 1. Pull new 2026 race results
    new_rounds = update_2026_race_results()

    # 2. Pull new 2026 standings
    update_2026_standings()

    # 3. Update weather (forecast + actual for past races)
    update_weather()

    # 4. Append new rows to master dataset
    append_new_race_to_master(new_rounds)

    # 5. Rerun models
    rerun_models(new_rounds)

    # 6. Save log
    duration = (datetime.now() - start).total_seconds()
    save_log(new_rounds, duration)

    print("\n" + "=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)
    print(f"Finished    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Duration    : {duration:.0f} seconds")
    print(f"New rounds  : {new_rounds}")
    print(f"XGBoost     : {'Retrained' if new_rounds > 0 else 'Skipped (no new race)'}")
    print(f"Output      : data/final_predictions.csv")
    print("=" * 60)


if __name__ == "__main__":
    run_daily_update()