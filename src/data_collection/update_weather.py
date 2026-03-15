"""
update_weather.py
-----------------
Automatically keeps weather data fresh by:
  1. Checking if there is an upcoming race within the next 7 days
  2. If yes — pulls the latest forecast and updates weather_forecast.csv
  3. For past races — replaces forecast with actual recorded weather
  4. Flags if weather changed significantly (triggers model re-run)

HOW TO RUN MANUALLY:
    python src/data_collection/update_weather.py

HOW TO AUTOMATE (run every day at 6am):
    Windows  — Task Scheduler → run this script daily
    Mac/Linux — add to crontab:  0 6 * * * python /path/to/update_weather.py

BEST PRACTICE:
    - Run daily during race weekends (Thursday to Sunday)
    - Run once daily during the rest of the season
    - Always run manually on race morning for the most accurate prediction
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data")

HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"
REQUEST_DELAY  = 1.0

# If rain probability changes by more than this %, flag a significant change
SIGNIFICANT_RAIN_CHANGE_PCT = 20.0

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

# ─── F1 2026 RACE CALENDAR ────────────────────────────────────────────────────
# Update this each season with the actual race dates and circuits

RACE_CALENDAR_2026 = [
    {"round": 1,  "name": "Australian Grand Prix",      "circuit_id": "albert_park",   "date": "2026-03-15", "lat": -37.8497, "lon": 144.9680},
    {"round": 2,  "name": "Chinese Grand Prix",         "circuit_id": "shanghai",      "date": "2026-03-22", "lat": 31.3389,  "lon": 121.2197},
    {"round": 3,  "name": "Japanese Grand Prix",        "circuit_id": "suzuka",        "date": "2026-04-05", "lat": 34.8431,  "lon": 136.5407},
    {"round": 4,  "name": "Bahrain Grand Prix",         "circuit_id": "bahrain",       "date": "2026-04-19", "lat": 26.0325,  "lon": 50.5106},
    {"round": 5,  "name": "Saudi Arabian Grand Prix",   "circuit_id": "jeddah",        "date": "2026-04-26", "lat": 21.6319,  "lon": 39.1044},
    {"round": 6,  "name": "Miami Grand Prix",           "circuit_id": "miami",         "date": "2026-05-03", "lat": 25.9581,  "lon": -80.2389},
    {"round": 7,  "name": "Emilia Romagna Grand Prix",  "circuit_id": "imola",         "date": "2026-05-17", "lat": 44.3439,  "lon": 11.7167},
    {"round": 8,  "name": "Monaco Grand Prix",          "circuit_id": "monaco",        "date": "2026-05-24", "lat": 43.7347,  "lon": 7.4206},
    {"round": 9,  "name": "Spanish Grand Prix",         "circuit_id": "catalunya",     "date": "2026-06-07", "lat": 41.5700,  "lon": 2.2611},
    {"round": 10, "name": "Canadian Grand Prix",        "circuit_id": "villeneuve",    "date": "2026-06-14", "lat": 45.5000,  "lon": -73.5228},
    {"round": 11, "name": "Austrian Grand Prix",        "circuit_id": "red_bull_ring", "date": "2026-06-28", "lat": 47.2197,  "lon": 14.7647},
    {"round": 12, "name": "British Grand Prix",         "circuit_id": "silverstone",   "date": "2026-07-05", "lat": 52.0786,  "lon": -1.0169},
    {"round": 13, "name": "Belgian Grand Prix",         "circuit_id": "spa",           "date": "2026-07-26", "lat": 50.4372,  "lon": 5.9714},
    {"round": 14, "name": "Hungarian Grand Prix",       "circuit_id": "hungaroring",   "date": "2026-08-02", "lat": 47.5789,  "lon": 19.2486},
    {"round": 15, "name": "Dutch Grand Prix",           "circuit_id": "zandvoort",     "date": "2026-08-30", "lat": 52.3888,  "lon": 4.5409},
    {"round": 16, "name": "Italian Grand Prix",         "circuit_id": "monza",         "date": "2026-09-06", "lat": 45.6156,  "lon": 9.2811},
    {"round": 17, "name": "Azerbaijan Grand Prix",      "circuit_id": "baku",          "date": "2026-09-20", "lat": 40.3725,  "lon": 49.8533},
    {"round": 18, "name": "Singapore Grand Prix",       "circuit_id": "marina_bay",    "date": "2026-10-04", "lat": 1.2914,   "lon": 103.8640},
    {"round": 19, "name": "United States Grand Prix",   "circuit_id": "americas",      "date": "2026-10-18", "lat": 30.1328,  "lon": -97.6411},
    {"round": 20, "name": "Mexico City Grand Prix",     "circuit_id": "rodriguez",     "date": "2026-10-25", "lat": 19.4042,  "lon": -99.0907},
    {"round": 21, "name": "São Paulo Grand Prix",       "circuit_id": "interlagos",    "date": "2026-11-08", "lat": -23.7036, "lon": -46.6997},
    {"round": 22, "name": "Las Vegas Grand Prix",       "circuit_id": "vegas",         "date": "2026-11-21", "lat": 36.1147,  "lon": -115.1728},
    {"round": 23, "name": "Qatar Grand Prix",           "circuit_id": "losail",        "date": "2026-11-29", "lat": 25.4900,  "lon": 51.4542},
    {"round": 24, "name": "Abu Dhabi Grand Prix",       "circuit_id": "yas_marina",    "date": "2026-12-06", "lat": 24.4672,  "lon": 54.6031},
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def make_request(url, params, retries=3):
    """Makes a GET request with automatic retry on rate limit."""
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"  Request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(5)

    return None


def get_upcoming_races(days_ahead=7):
    """
    Returns races happening within the next N days.

    Args:
        days_ahead (int): How many days to look ahead

    Returns:
        list: Race dicts for upcoming races
    """
    today    = pd.Timestamp.today().normalize()
    deadline = today + pd.Timedelta(days=days_ahead)

    upcoming = []
    for race in RACE_CALENDAR_2026:
        race_date = pd.Timestamp(race["date"])
        if today <= race_date <= deadline:
            days_until = (race_date - today).days
            race["days_until"] = days_until
            upcoming.append(race)

    return upcoming


def get_past_races_without_actual_weather():
    """
    Returns past races that still have forecast data instead of actual data.
    These need to be updated with real recorded weather.

    Returns:
        list: Race dicts for past races needing weather update
    """
    today = pd.Timestamp.today().normalize()

    # Load existing historical weather
    hist_path = os.path.join(OUTPUT_DIR, "weather_historical.csv")
    if os.path.exists(hist_path):
        hist_df      = pd.read_csv(hist_path)
        already_have = set(hist_df["race_date"].astype(str).str[:10].tolist())
    else:
        already_have = set()

    past_missing = []
    for race in RACE_CALENDAR_2026:
        race_date = pd.Timestamp(race["date"])
        # Race is in the past AND we don't have actual weather yet
        if race_date < today and race["date"] not in already_have:
            past_missing.append(race)

    return past_missing


# ─── FETCH FORECAST ───────────────────────────────────────────────────────────

def fetch_race_forecast(race):
    """
    Pulls the latest weather forecast for a specific race.

    Args:
        race (dict): Race info with lat, lon, date, name

    Returns:
        dict: Weather summary for that race day, or None if failed
    """
    race_date = race["date"]
    lat       = race["lat"]
    lon       = race["lon"]

    # Open-Meteo forecast goes up to 16 days ahead
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": race_date,
        "end_date":   race_date,
        "hourly":     HOURLY_VARS,
        "timezone":   "auto",
    }

    data = make_request(FORECAST_URL, params)

    if not data or "hourly" not in data:
        return None

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
        "uv_index":             hourly.get("uv_index", []),
    })

    df["time"] = pd.to_datetime(df["time"])

    race_day = pd.to_datetime(race_date).date()
    df_day   = df[df["time"].dt.date == race_day]
    df_race  = df_day[
        (df_day["time"].dt.hour >= 13) &
        (df_day["time"].dt.hour <= 20)
    ]

    if df_race.empty:
        df_race = df_day

    if df_race.empty:
        return None

    return {
        "circuit_id":          race["circuit_id"],
        "circuit_name":        race["name"],
        "race_date":           race_date,
        "days_until_race":     race.get("days_until", 0),
        "forecast_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "air_temp_avg_c":      round(df_race["temperature_2m"].mean(), 2),
        "air_temp_max_c":      round(df_race["temperature_2m"].max(), 2),
        "track_temp_avg_c":    round(df_race["soil_temperature_0cm"].mean(), 2),
        "humidity_avg_pct":    round(df_race["relativehumidity_2m"].mean(), 2),
        "rainfall_total_mm":   round(df_race["rain"].sum(), 3),
        "rain_probability_pct": round(
            (df_race["rain"] > 0).sum() / len(df_race) * 100, 1
        ),
        "rain_flag":           int(df_race["rain"].sum() > 0.5),
        "cloudcover_avg_pct":  round(df_race["cloudcover"].mean(), 2),
        "windspeed_avg_kmh":   round(df_race["windspeed_10m"].mean(), 2),
        "windspeed_max_kmh":   round(df_race["windspeed_10m"].max(), 2),
        "winddirection_avg":   round(df_race["winddirection_10m"].mean(), 2),
        "pressure_avg_hpa":    round(df_race["surface_pressure"].mean(), 2),
    }


# ─── FETCH ACTUAL WEATHER ─────────────────────────────────────────────────────

def fetch_actual_weather(race):
    """
    Pulls actual recorded weather for a past race from the historical API.
    This replaces any forecast data with ground truth.

    Args:
        race (dict): Race info with lat, lon, date, name

    Returns:
        dict: Actual weather summary, or None if failed
    """
    race_date = race["date"]

    params = {
        "latitude":   race["lat"],
        "longitude":  race["lon"],
        "start_date": race_date,
        "end_date":   race_date,
        "hourly":     HOURLY_VARS,
        "timezone":   "auto",
    }

    data = make_request(HISTORICAL_URL, params)

    if not data or "hourly" not in data:
        return None

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
        "uv_index":             hourly.get("uv_index", []),
    })

    df["time"] = pd.to_datetime(df["time"])

    race_day = pd.to_datetime(race_date).date()
    df_day   = df[df["time"].dt.date == race_day]
    df_race  = df_day[
        (df_day["time"].dt.hour >= 13) &
        (df_day["time"].dt.hour <= 20)
    ]

    if df_race.empty:
        df_race = df_day

    if df_race.empty:
        return None

    return {
        "circuit_id":         race["circuit_id"],
        "circuit_name":       race["name"],
        "race_date":          race_date,
        "data_type":          "actual",    # marks this as real data not forecast
        "air_temp_avg_c":     round(df_race["temperature_2m"].mean(), 2),
        "air_temp_max_c":     round(df_race["temperature_2m"].max(), 2),
        "track_temp_avg_c":   round(df_race["soil_temperature_0cm"].mean(), 2),
        "humidity_avg_pct":   round(df_race["relativehumidity_2m"].mean(), 2),
        "rainfall_total_mm":  round(df_race["rain"].sum(), 3),
        "rain_flag":          int(df_race["rain"].sum() > 0.5),
        "cloudcover_avg_pct": round(df_race["cloudcover"].mean(), 2),
        "windspeed_avg_kmh":  round(df_race["windspeed_10m"].mean(), 2),
        "windspeed_max_kmh":  round(df_race["windspeed_10m"].max(), 2),
        "winddirection_avg":  round(df_race["winddirection_10m"].mean(), 2),
        "pressure_avg_hpa":   round(df_race["surface_pressure"].mean(), 2),
    }


# ─── CHECK FOR SIGNIFICANT CHANGE ─────────────────────────────────────────────

def check_significant_change(old_forecast_df, new_forecast):
    """
    Compares old and new forecast to detect significant weather changes.
    If rain probability changes by more than 20%, flags a model re-run.

    Args:
        old_forecast_df (pd.DataFrame): Previous forecast data
        new_forecast    (dict):         New forecast summary

    Returns:
        bool: True if significant change detected
    """
    circuit_id = new_forecast["circuit_id"]
    race_date  = new_forecast["race_date"]

    old = old_forecast_df[
        (old_forecast_df["circuit_id"] == circuit_id) &
        (old_forecast_df["race_date"]  == race_date)
    ]

    if old.empty:
        return False  # No previous forecast to compare

    old_rain = old["rain_probability_pct"].values[0] if "rain_probability_pct" in old.columns else 0
    new_rain = new_forecast.get("rain_probability_pct", 0)

    change = abs(new_rain - old_rain)

    if change >= SIGNIFICANT_RAIN_CHANGE_PCT:
        print(f"  SIGNIFICANT CHANGE DETECTED for {circuit_id}:")
        print(f"    Rain probability: {old_rain:.0f}% → {new_rain:.0f}% (change: {change:.0f}%)")
        print(f"    → Model should be re-run with updated weather features")
        return True

    return False


# ─── MAIN UPDATE PIPELINE ─────────────────────────────────────────────────────

def run_update():
    """
    Main daily update routine:
      1. Updates forecasts for upcoming races
      2. Saves actual weather for past races
      3. Reports any significant changes
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    today = pd.Timestamp.today().normalize()

    print("\nF1 Weather Updater")
    print("=" * 60)
    print(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Today     : {today.strftime('%Y-%m-%d')}")
    print("=" * 60)

    significant_changes = []

    # ── Step 1: Update forecasts for upcoming races ───────────────────────────
    upcoming = get_upcoming_races(days_ahead=7)

    if upcoming:
        print(f"\nFound {len(upcoming)} upcoming race(s) within 7 days:")
        for r in upcoming:
            print(f"  Round {r['round']}: {r['name']} — {r['date']} ({r['days_until']} days away)")

        # Load old forecast to compare
        forecast_path = os.path.join(OUTPUT_DIR, "weather_forecast_upcoming.csv")
        old_forecast_df = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else pd.DataFrame()

        new_forecasts = []
        for race in upcoming:
            print(f"\nUpdating forecast for {race['name']} ({race['date']})...")
            forecast = fetch_race_forecast(race)

            if forecast:
                new_forecasts.append(forecast)
                print(f"  Air temp    : {forecast['air_temp_avg_c']}°C avg")
                print(f"  Track temp  : {forecast['track_temp_avg_c']}°C avg")
                print(f"  Rain prob   : {forecast.get('rain_probability_pct', 0):.0f}%")
                print(f"  Rain flag   : {'YES' if forecast['rain_flag'] else 'NO'}")
                print(f"  Wind        : {forecast['windspeed_avg_kmh']} km/h avg")

                # Check if this is a significant change
                if not old_forecast_df.empty:
                    changed = check_significant_change(old_forecast_df, forecast)
                    if changed:
                        significant_changes.append(race["name"])
            else:
                print(f"  Failed to fetch forecast")

        if new_forecasts:
            new_df = pd.DataFrame(new_forecasts)
            new_df.to_csv(forecast_path, index=False)
            print(f"\nSaved updated forecast: {forecast_path}")

    else:
        print("\nNo races in the next 7 days — no forecast update needed")

    # ── Step 2: Fetch actual weather for past races ───────────────────────────
    past_missing = get_past_races_without_actual_weather()

    if past_missing:
        print(f"\nFound {len(past_missing)} past race(s) missing actual weather:")
        for r in past_missing:
            print(f"  {r['date']}: {r['name']}")

        actual_weather_rows = []
        for race in past_missing:
            print(f"\nFetching actual weather for {race['name']} ({race['date']})...")
            actual = fetch_actual_weather(race)

            if actual:
                actual_weather_rows.append(actual)
                print(f"  Rain flag: {'YES' if actual['rain_flag'] else 'NO'}")
                print(f"  Temp: {actual['air_temp_avg_c']}°C")
            else:
                # Historical API may not have data for very recent races yet
                print(f"  Not available yet (race may be too recent)")

        if actual_weather_rows:
            hist_path = os.path.join(OUTPUT_DIR, "weather_historical.csv")

            if os.path.exists(hist_path):
                existing = pd.read_csv(hist_path)
                new_rows = pd.DataFrame(actual_weather_rows)
                combined = pd.concat([existing, new_rows], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["circuit_id", "race_date"], keep="last"
                )
            else:
                combined = pd.DataFrame(actual_weather_rows)

            combined.to_csv(hist_path, index=False)
            print(f"\nUpdated historical weather: {hist_path} ({len(combined)} rows)")
    else:
        print("\nAll past races already have actual weather data")

    # ── Step 3: Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Update complete!")

    if significant_changes:
        print(f"\nSIGNIFICANT WEATHER CHANGES DETECTED:")
        for race_name in significant_changes:
            print(f"  {race_name}")
        print("\nRecommendation: Re-run the Monte Carlo simulation")
        print("  python models/monte_carlo.py")
    else:
        print("No significant weather changes — model re-run not needed")

    print("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_update()