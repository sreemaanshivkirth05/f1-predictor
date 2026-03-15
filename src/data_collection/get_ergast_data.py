"""
get_ergast_data.py
------------------
Pulls F1 race data from the Jolpica API (the free replacement for Ergast).
This file fetches:
  - Race results (finishing positions, points, status)
  - Driver standings (championship points per round)
  - Constructor standings (team points per round)
  - Qualifying results (grid positions)
  - Pit stop data
  - Lap times (summary level)

HOW TO RUN:
    python src/data_collection/get_ergast_data.py

OUTPUT:
    Saves CSV files into the data/ folder.
"""

import requests
import pandas as pd
import time
import os

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Base URL for the Jolpica API (free, no API key needed)
BASE_URL = "http://api.jolpi.ca/ergast/f1"

# Years of data to pull (we go back to 2010 for a good training set)
START_YEAR = 2016
END_YEAR   = 2026

# Where to save the output CSV files
OUTPUT_DIR = "data"

# How long to wait between API requests (be polite to the free API)
REQUEST_DELAY = 1.0  # seconds — increased to avoid 429 rate limit errors


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────

def make_request(url, retries=5):
    """
    Makes a GET request to the Jolpica API and returns the JSON response.

    Handles 429 Too Many Requests errors by waiting and retrying automatically.
    The free Jolpica API has a rate limit — if we hit it, we wait longer and try again.

    Args:
        url     (str): The full API URL to request
        retries (int): How many times to retry on failure before giving up

    Returns:
        dict: The JSON response, or None if all retries failed
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)

            # 429 = Too Many Requests — wait and retry
            if response.status_code == 429:
                wait = 10 * (attempt + 1)  # Wait 10s, then 20s, then 30s...
                print(f"  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            time.sleep(REQUEST_DELAY)  # Always wait between successful requests
            return response.json()

        except requests.exceptions.RequestException as e:
            wait = 5 * (attempt + 1)
            print(f"  Request failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(wait)

    print(f"  Giving up on: {url}")
    return None


def ensure_output_dir():
    """Creates the data/ folder if it doesn't already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory ready: {OUTPUT_DIR}/")


# ─── RACE RESULTS ─────────────────────────────────────────────────────────────

def fetch_race_results(year):
    """
    Fetches all race results for a given season.

    For each race in the season, we get:
      - Driver name and code
      - Constructor (team) name
      - Starting grid position
      - Finishing position
      - Points scored
      - Race status (Finished, Retired, Accident, etc.)
      - Fastest lap flag

    Args:
        year (int): The season year (e.g. 2024)

    Returns:
        pd.DataFrame: One row per driver per race
    """
    print(f"  Fetching race results for {year}...")

    url = f"{BASE_URL}/{year}/results.json?limit=1000"
    data = make_request(url)

    if not data:
        print(f"  No data returned for {year}")
        return pd.DataFrame()

    rows = []

    # Navigate the nested JSON structure from Jolpica
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

    for race in races:
        round_num   = int(race.get("round", 0))
        race_name   = race.get("raceName", "")
        circuit     = race.get("Circuit", {}).get("circuitName", "")
        circuit_id  = race.get("Circuit", {}).get("circuitId", "")
        race_date   = race.get("date", "")
        country     = race.get("Circuit", {}).get("Location", {}).get("country", "")
        locality    = race.get("Circuit", {}).get("Location", {}).get("locality", "")
        lat         = race.get("Circuit", {}).get("Location", {}).get("lat", None)
        long        = race.get("Circuit", {}).get("Location", {}).get("long", None)

        for result in race.get("Results", []):
            driver      = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            fastest_lap = result.get("FastestLap", {})

            rows.append({
                "year":               year,
                "round":              round_num,
                "race_name":          race_name,
                "circuit_name":       circuit,
                "circuit_id":         circuit_id,
                "race_date":          race_date,
                "country":            country,
                "locality":           locality,
                "lat":                lat,
                "long":               long,
                "driver_id":          driver.get("driverId", ""),
                "driver_code":        driver.get("code", ""),
                "driver_name":        f"{driver.get('givenName','')} {driver.get('familyName','')}",
                "constructor_id":     constructor.get("constructorId", ""),
                "constructor_name":   constructor.get("name", ""),
                "grid_position":      result.get("grid", None),
                "finish_position":    result.get("position", None),
                "points":             float(result.get("points", 0)),
                "laps_completed":     result.get("laps", None),
                "status":             result.get("status", ""),
                "fastest_lap_rank":   fastest_lap.get("rank", None),
                "fastest_lap_time":   fastest_lap.get("Time", {}).get("time", None),
            })

    df = pd.DataFrame(rows)

    # Convert position columns to numeric (they come as strings from the API)
    for col in ["grid_position", "finish_position", "laps_completed", "fastest_lap_rank"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"    Got {len(df)} driver-race rows for {year}")
    return df


# ─── QUALIFYING RESULTS ───────────────────────────────────────────────────────

def fetch_qualifying(year):
    """
    Fetches qualifying results for each race in a season.

    Returns Q1, Q2, Q3 lap times and the final qualifying position for each driver.

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per driver per qualifying session
    """
    print(f"  Fetching qualifying results for {year}...")

    url = f"{BASE_URL}/{year}/qualifying.json?limit=1000"
    data = make_request(url)

    if not data:
        return pd.DataFrame()

    rows = []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

    for race in races:
        round_num  = int(race.get("round", 0))
        race_name  = race.get("raceName", "")
        circuit_id = race.get("Circuit", {}).get("circuitId", "")

        for result in race.get("QualifyingResults", []):
            driver      = result.get("Driver", {})
            constructor = result.get("Constructor", {})

            rows.append({
                "year":             year,
                "round":            round_num,
                "race_name":        race_name,
                "circuit_id":       circuit_id,
                "driver_id":        driver.get("driverId", ""),
                "driver_code":      driver.get("code", ""),
                "constructor_id":   constructor.get("constructorId", ""),
                "quali_position":   result.get("position", None),
                "q1_time":          result.get("Q1", None),
                "q2_time":          result.get("Q2", None),
                "q3_time":          result.get("Q3", None),
            })

    df = pd.DataFrame(rows)
    df["quali_position"] = pd.to_numeric(df["quali_position"], errors="coerce")

    print(f"    Got {len(df)} qualifying rows for {year}")
    return df


# ─── DRIVER STANDINGS ─────────────────────────────────────────────────────────

def fetch_driver_standings(year):
    """
    Fetches the driver championship standings after each round.

    This gives us the cumulative points and position after every race —
    essential for computing rolling averages and points gaps.

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per driver per round showing standings
    """
    print(f"  Fetching driver standings for {year}...")

    # First find out how many rounds there are in this season
    url = f"{BASE_URL}/{year}/races.json"
    data = make_request(url)

    if not data:
        return pd.DataFrame()

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    total_rounds = len(races)

    rows = []

    for round_num in range(1, total_rounds + 1):
        url = f"{BASE_URL}/{year}/{round_num}/driverStandings.json"
        data = make_request(url)

        if not data:
            continue

        standings_list = (
            data.get("MRData", {})
                .get("StandingsTable", {})
                .get("StandingsLists", [])
        )

        if not standings_list:
            continue

        for standing in standings_list[0].get("DriverStandings", []):
            driver      = standing.get("Driver", {})
            constructor = standing.get("Constructors", [{}])[0]

            rows.append({
                "year":               year,
                "round":              round_num,
                "driver_id":          driver.get("driverId", ""),
                "driver_name":        f"{driver.get('givenName','')} {driver.get('familyName','')}",
                "constructor_id":     constructor.get("constructorId", ""),
                "championship_pos":   int(standing.get("position", 0)),
                "championship_pts":   float(standing.get("points", 0)),
                "wins":               int(standing.get("wins", 0)),
            })

    df = pd.DataFrame(rows)
    print(f"    Got {len(df)} standing rows for {year} ({total_rounds} rounds)")
    return df


# ─── CONSTRUCTOR STANDINGS ────────────────────────────────────────────────────

def fetch_constructor_standings(year):
    """
    Fetches constructor (team) championship standings after each round.

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per constructor per round
    """
    print(f"  Fetching constructor standings for {year}...")

    url = f"{BASE_URL}/{year}/races.json"
    data = make_request(url)

    if not data:
        return pd.DataFrame()

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    total_rounds = len(races)

    rows = []

    for round_num in range(1, total_rounds + 1):
        url = f"{BASE_URL}/{year}/{round_num}/constructorStandings.json"
        data = make_request(url)

        if not data:
            continue

        standings_list = (
            data.get("MRData", {})
                .get("StandingsTable", {})
                .get("StandingsLists", [])
        )

        if not standings_list:
            continue

        for standing in standings_list[0].get("ConstructorStandings", []):
            constructor = standing.get("Constructor", {})

            rows.append({
                "year":                   year,
                "round":                  round_num,
                "constructor_id":         constructor.get("constructorId", ""),
                "constructor_name":       constructor.get("name", ""),
                "constructor_champ_pos":  int(standing.get("position", 0)),
                "constructor_champ_pts":  float(standing.get("points", 0)),
                "constructor_wins":       int(standing.get("wins", 0)),
            })

    df = pd.DataFrame(rows)
    print(f"    Got {len(df)} constructor standing rows for {year}")
    return df


# ─── PIT STOPS ────────────────────────────────────────────────────────────────

def fetch_pit_stops(year):
    """
    Fetches pit stop timing data for all races in a season.

    Gives us stop number, lap number, and duration for each pit stop.
    Used to compute avg pit stop time and team pit crew speed rank.

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per pit stop
    """
    print(f"  Fetching pit stops for {year}...")

    # Get total rounds first
    url = f"{BASE_URL}/{year}/races.json"
    data = make_request(url)

    if not data:
        return pd.DataFrame()

    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    total_rounds = len(races)

    rows = []

    for round_num in range(1, total_rounds + 1):
        url = f"{BASE_URL}/{year}/{round_num}/pitstops.json?limit=100"
        data = make_request(url)

        if not data:
            continue

        race_table = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

        if not race_table:
            continue

        race      = race_table[0]
        race_name = race.get("raceName", "")

        for stop in race.get("PitStops", []):
            rows.append({
                "year":        year,
                "round":       round_num,
                "race_name":   race_name,
                "driver_id":   stop.get("driverId", ""),
                "stop_number": int(stop.get("stop", 0)),
                "lap":         int(stop.get("lap", 0)),
                "duration_s":  stop.get("duration", None),  # e.g. "23.456"
            })

    df = pd.DataFrame(rows)

    # Only convert duration_s if we actually got some rows
    if not df.empty and "duration_s" in df.columns:
        df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    elif df.empty:
        # Return empty dataframe with correct columns so the rest of the pipeline doesn't break
        df = pd.DataFrame(columns=["year", "round", "race_name", "driver_id",
                                    "stop_number", "lap", "duration_s"])

    print(f"    Got {len(df)} pit stop rows for {year}")
    return df


# ─── SPRINT RESULTS ───────────────────────────────────────────────────────────

def fetch_sprint_results(year):
    """
    Fetches sprint race results where applicable.
    Not all rounds have sprints — the API returns empty for non-sprint rounds.

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per driver per sprint race
    """
    print(f"  Fetching sprint results for {year}...")

    url = f"{BASE_URL}/{year}/sprint.json?limit=200"
    data = make_request(url)

    if not data:
        return pd.DataFrame()

    rows = []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])

    for race in races:
        round_num = int(race.get("round", 0))
        race_name = race.get("raceName", "")

        for result in race.get("SprintResults", []):
            driver      = result.get("Driver", {})
            constructor = result.get("Constructor", {})

            rows.append({
                "year":             year,
                "round":            round_num,
                "race_name":        race_name,
                "driver_id":        driver.get("driverId", ""),
                "driver_code":      driver.get("code", ""),
                "constructor_id":   constructor.get("constructorId", ""),
                "sprint_position":  result.get("position", None),
                "sprint_points":    float(result.get("points", 0)),
                "sprint_status":    result.get("status", ""),
            })

    df = pd.DataFrame(rows)
    if not df.empty and "sprint_position" in df.columns:
        df["sprint_position"] = pd.to_numeric(df["sprint_position"], errors="coerce")
    elif df.empty:
        df = pd.DataFrame(columns=["year", "round", "race_name", "driver_id",
                                    "driver_code", "constructor_id", "sprint_position",
                                    "sprint_points", "sprint_status"])

    print(f"    Got {len(df)} sprint rows for {year}")
    return df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Runs the complete data collection pipeline for all years.

    For each year from START_YEAR to END_YEAR:
      1. Fetches race results
      2. Fetches qualifying results
      3. Fetches driver standings (per round)
      4. Fetches constructor standings (per round)
      5. Fetches pit stop data
      6. Fetches sprint results

    Saves one combined CSV per data type into the data/ folder.
    """
    ensure_output_dir()

    # Storage lists — we collect all years then combine into one CSV
    all_results      = []
    all_qualifying   = []
    all_drv_standing = []
    all_con_standing = []
    all_pit_stops    = []
    all_sprints      = []

    years = list(range(START_YEAR, END_YEAR + 1))
    print(f"\nPulling data for {len(years)} seasons: {START_YEAR} to {END_YEAR}")
    print("=" * 60)

    for year in years:
        print(f"\n[{year}]")

        results      = fetch_race_results(year)
        qualifying   = fetch_qualifying(year)
        drv_standing = fetch_driver_standings(year)
        con_standing = fetch_constructor_standings(year)
        pit_stops    = fetch_pit_stops(year)
        sprints      = fetch_sprint_results(year)

        if not results.empty:      all_results.append(results)
        if not qualifying.empty:   all_qualifying.append(qualifying)
        if not drv_standing.empty: all_drv_standing.append(drv_standing)
        if not con_standing.empty: all_con_standing.append(con_standing)
        if not pit_stops.empty:    all_pit_stops.append(pit_stops)
        if not sprints.empty:      all_sprints.append(sprints)

    # ── Combine and save ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving CSV files...")

    datasets = {
        "race_results.csv":           all_results,
        "qualifying_results.csv":     all_qualifying,
        "driver_standings.csv":       all_drv_standing,
        "constructor_standings.csv":  all_con_standing,
        "pit_stops.csv":              all_pit_stops,
        "sprint_results.csv":         all_sprints,
    }

    for filename, data_list in datasets.items():
        if data_list:
            df = pd.concat(data_list, ignore_index=True)
            path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(path, index=False)
            print(f"  Saved {path}  ({len(df):,} rows)")
        else:
            print(f"  Skipped {filename} — no data collected")

    print("\nData collection complete!")
    print(f"All files saved to: {OUTPUT_DIR}/")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_pipeline()