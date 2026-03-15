"""
get_weather_data.py
-------------------
Pulls historical and forecast weather data for every F1 circuit
using the Open-Meteo API (completely free, no API key needed).

HOW TO RUN:
    python src/data_collection/get_weather_data.py

REQUIREMENTS:
    pip install requests pandas

OUTPUT:
    Saves data/weather_historical.csv and data/weather_forecast.csv
"""

import requests
import pandas as pd
import time
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Finds the project root no matter where you run the script from
# e.g. file is at: C:/f1-prediction-system/src/data_collection/get_weather_data.py
# PROJECT_ROOT  = C:/f1-prediction-system
THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data")

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"
REQUEST_DELAY  = 1.0

HOURLY_VARS = ",".join([
    "temperature_2m",
    "relativehumidity_2m",
    "precipitation",
    "rain",
    "cloudcover",
    "windspeed_10m",
    "winddirection_10m",
    "surface_pressure",
    "soil_temperature_0cm",
    "uv_index",
])


# ─── F1 CIRCUIT GPS COORDINATES ───────────────────────────────────────────────

CIRCUITS = {
    "bahrain":       {"name": "Bahrain Grand Prix",        "lat": 26.0325,  "lon": 50.5106},
    "jeddah":        {"name": "Saudi Arabian Grand Prix",  "lat": 21.6319,  "lon": 39.1044},
    "albert_park":   {"name": "Australian Grand Prix",     "lat": -37.8497, "lon": 144.9680},
    "suzuka":        {"name": "Japanese Grand Prix",       "lat": 34.8431,  "lon": 136.5407},
    "shanghai":      {"name": "Chinese Grand Prix",        "lat": 31.3389,  "lon": 121.2197},
    "miami":         {"name": "Miami Grand Prix",          "lat": 25.9581,  "lon": -80.2389},
    "imola":         {"name": "Emilia Romagna Grand Prix", "lat": 44.3439,  "lon": 11.7167},
    "monaco":        {"name": "Monaco Grand Prix",         "lat": 43.7347,  "lon": 7.4206},
    "villeneuve":    {"name": "Canadian Grand Prix",       "lat": 45.5000,  "lon": -73.5228},
    "catalunya":     {"name": "Spanish Grand Prix",        "lat": 41.5700,  "lon": 2.2611},
    "red_bull_ring": {"name": "Austrian Grand Prix",       "lat": 47.2197,  "lon": 14.7647},
    "silverstone":   {"name": "British Grand Prix",        "lat": 52.0786,  "lon": -1.0169},
    "hungaroring":   {"name": "Hungarian Grand Prix",      "lat": 47.5789,  "lon": 19.2486},
    "spa":           {"name": "Belgian Grand Prix",        "lat": 50.4372,  "lon": 5.9714},
    "zandvoort":     {"name": "Dutch Grand Prix",          "lat": 52.3888,  "lon": 4.5409},
    "monza":         {"name": "Italian Grand Prix",        "lat": 45.6156,  "lon": 9.2811},
    "baku":          {"name": "Azerbaijan Grand Prix",     "lat": 40.3725,  "lon": 49.8533},
    "marina_bay":    {"name": "Singapore Grand Prix",      "lat": 1.2914,   "lon": 103.8640},
    "americas":      {"name": "United States Grand Prix",  "lat": 30.1328,  "lon": -97.6411},
    "rodriguez":     {"name": "Mexico City Grand Prix",    "lat": 19.4042,  "lon": -99.0907},
    "interlagos":    {"name": "São Paulo Grand Prix",      "lat": -23.7036, "lon": -46.6997},
    "vegas":         {"name": "Las Vegas Grand Prix",      "lat": 36.1147,  "lon": -115.1728},
    "losail":        {"name": "Qatar Grand Prix",          "lat": 25.4900,  "lon": 51.4542},
    "yas_marina":    {"name": "Abu Dhabi Grand Prix",      "lat": 24.4672,  "lon": 54.6031},
}


# ─── HELPER ───────────────────────────────────────────────────────────────────

def make_request(url, params, retries=3):
    """Makes a GET request with automatic retry on rate limit or failure."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"    Request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(5)

    return None


def parse_race_day_weather(data, circuit_id, circuit_name, race_date):
    """
    Parses Open-Meteo hourly JSON into a single-row race-day summary.
    Filters to race window hours (13:00-20:00) and computes averages/totals.
    """
    if not data or "hourly" not in data:
        return pd.DataFrame()

    hourly = data["hourly"]

    df = pd.DataFrame({
        "time":                 hourly.get("time", []),
        "temperature_2m":       hourly.get("temperature_2m", []),
        "relativehumidity_2m":  hourly.get("relativehumidity_2m", []),
        "precipitation":        hourly.get("precipitation", []),
        "rain":                 hourly.get("rain", []),
        "cloudcover":           hourly.get("cloudcover", []),
        "windspeed_10m":        hourly.get("windspeed_10m", []),
        "winddirection_10m":    hourly.get("winddirection_10m", []),
        "surface_pressure":     hourly.get("surface_pressure", []),
        "soil_temperature_0cm": hourly.get("soil_temperature_0cm", []),
        "uv_index":             hourly.get("uv_index", []),
    })

    df["time"] = pd.to_datetime(df["time"])

    race_day = pd.to_datetime(race_date).date()
    df_day   = df[df["time"].dt.date == race_day]

    df_race = df_day[
        (df_day["time"].dt.hour >= 13) &
        (df_day["time"].dt.hour <= 20)
    ]

    if df_race.empty:
        df_race = df_day

    if df_race.empty:
        return pd.DataFrame()

    summary = {
        "circuit_id":         circuit_id,
        "circuit_name":       circuit_name,
        "race_date":          race_date,
        "air_temp_avg_c":     round(df_race["temperature_2m"].mean(), 2),
        "air_temp_max_c":     round(df_race["temperature_2m"].max(), 2),
        "track_temp_avg_c":   round(df_race["soil_temperature_0cm"].mean(), 2),
        "track_temp_max_c":   round(df_race["soil_temperature_0cm"].max(), 2),
        "humidity_avg_pct":   round(df_race["relativehumidity_2m"].mean(), 2),
        "humidity_max_pct":   round(df_race["relativehumidity_2m"].max(), 2),
        "rainfall_total_mm":  round(df_race["rain"].sum(), 3),
        "rain_flag":          int(df_race["rain"].sum() > 0.5),
        "cloudcover_avg_pct": round(df_race["cloudcover"].mean(), 2),
        "windspeed_avg_kmh":  round(df_race["windspeed_10m"].mean(), 2),
        "windspeed_max_kmh":  round(df_race["windspeed_10m"].max(), 2),
        "winddirection_avg":  round(df_race["winddirection_10m"].mean(), 2),
        "pressure_avg_hpa":   round(df_race["surface_pressure"].mean(), 2),
        "uv_index_max":       round(df_race["uv_index"].max(), 2) if df_race["uv_index"].notna().any() else None,
    }

    return pd.DataFrame([summary])


# ─── HISTORICAL WEATHER ───────────────────────────────────────────────────────

def fetch_historical_weather(race_df):
    """
    Pulls actual recorded weather for every past race.
    Uses race_results.csv to get all circuit_id + race_date combinations.
    """
    print("Fetching historical weather for all past races...")

    races = (
        race_df[["race_date", "circuit_id", "circuit_name"]]
        .drop_duplicates()
        .sort_values("race_date")
        .reset_index(drop=True)
    )

    today = pd.Timestamp.today().normalize()
    races = races[pd.to_datetime(races["race_date"]) < today].reset_index(drop=True)

    print(f"  Found {len(races)} past races to fetch weather for")

    all_weather = []

    for i, row in races.iterrows():
        race_date    = str(row["race_date"])[:10]
        circuit_id   = row["circuit_id"]
        circuit_name = row.get("circuit_name", circuit_id)

        circuit_info = CIRCUITS.get(circuit_id)

        if not circuit_info:
            for key in CIRCUITS:
                if key in circuit_id or circuit_id in key:
                    circuit_info = CIRCUITS[key]
                    break

        if not circuit_info:
            print(f"  [{i+1}/{len(races)}] {circuit_id} — no GPS coords found, skipping")
            continue

        params = {
            "latitude":   circuit_info["lat"],
            "longitude":  circuit_info["lon"],
            "start_date": race_date,
            "end_date":   race_date,
            "hourly":     HOURLY_VARS,
            "timezone":   "auto",
        }

        data = make_request(HISTORICAL_URL, params)

        if data:
            weather_row = parse_race_day_weather(data, circuit_id, circuit_name, race_date)
            if not weather_row.empty:
                all_weather.append(weather_row)
                print(f"  [{i+1}/{len(races)}] {race_date} {circuit_id} OK")
            else:
                print(f"  [{i+1}/{len(races)}] {race_date} {circuit_id} — no data in window")
        else:
            print(f"  [{i+1}/{len(races)}] {race_date} {circuit_id} — request failed")

    if not all_weather:
        print("  No historical weather data collected")
        return pd.DataFrame()

    df = pd.concat(all_weather, ignore_index=True)
    print(f"\n  Collected weather for {len(df)} races")
    return df


# ─── FORECAST WEATHER ─────────────────────────────────────────────────────────

def fetch_forecast_weather():
    """Pulls 7-day weather forecast for all F1 circuits."""
    print("Fetching 7-day weather forecast for all circuits...")

    today    = pd.Timestamp.today().normalize()
    week_end = today + pd.Timedelta(days=7)

    all_forecasts = []

    for circuit_id, info in CIRCUITS.items():
        params = {
            "latitude":   info["lat"],
            "longitude":  info["lon"],
            "start_date": today.strftime("%Y-%m-%d"),
            "end_date":   week_end.strftime("%Y-%m-%d"),
            "hourly":     HOURLY_VARS,
            "timezone":   "auto",
        }

        data = make_request(FORECAST_URL, params)

        if not data or "hourly" not in data:
            continue

        hourly = data["hourly"]

        df = pd.DataFrame({
            "time":                 hourly.get("time", []),
            "temperature_2m":       hourly.get("temperature_2m", []),
            "relativehumidity_2m":  hourly.get("relativehumidity_2m", []),
            "rain":                 hourly.get("rain", []),
            "cloudcover":           hourly.get("cloudcover", []),
            "windspeed_10m":        hourly.get("windspeed_10m", []),
            "winddirection_10m":    hourly.get("winddirection_10m", []),
            "surface_pressure":     hourly.get("surface_pressure", []),
            "soil_temperature_0cm": hourly.get("soil_temperature_0cm", []),
        })

        df["time"]          = pd.to_datetime(df["time"])
        df["circuit_id"]    = circuit_id
        df["circuit_name"]  = info["name"]
        df["forecast_date"] = today.strftime("%Y-%m-%d")

        all_forecasts.append(df)
        print(f"  {circuit_id} OK")

    if not all_forecasts:
        return pd.DataFrame()

    df = pd.concat(all_forecasts, ignore_index=True)
    print(f"\n  Total: {len(df):,} hourly rows across {df['circuit_id'].nunique()} circuits")
    return df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nF1 Weather Data Collection")
    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data folder  : {OUTPUT_DIR}")
    print("API          : Open-Meteo (free, no key needed)")
    print("=" * 60)

    # Load race calendar
    race_results_path = os.path.join(OUTPUT_DIR, "race_results.csv")
    print(f"\nLooking for: {race_results_path}")

    if not os.path.exists(race_results_path):
        print(f"\nERROR: race_results.csv not found!")
        print("Please run get_ergast_data.py first.")
        return

    race_df = pd.read_csv(race_results_path)
    print(f"Loaded {len(race_df):,} rows, {race_df['year'].nunique()} seasons")
    print(f"Circuits in data: {sorted(race_df['circuit_id'].unique())}")

    # Historical weather
    print()
    historical_df = fetch_historical_weather(race_df)

    if not historical_df.empty:
        path = os.path.join(OUTPUT_DIR, "weather_historical.csv")
        historical_df.to_csv(path, index=False)
        print(f"\nSaved: {path}  ({len(historical_df):,} rows)")

    # Forecast weather
    print()
    forecast_df = fetch_forecast_weather()

    if not forecast_df.empty:
        path = os.path.join(OUTPUT_DIR, "weather_forecast.csv")
        forecast_df.to_csv(path, index=False)
        print(f"Saved: {path}  ({len(forecast_df):,} rows)")

    print("\nWeather data collection complete!")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_pipeline()