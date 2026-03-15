"""
get_fastf1_data.py
------------------
Pulls lap times, sector times, tire data, and practice session data
from the official F1 timing feed using the FastF1 Python library.

This file fetches:
  - Race lap times per driver
  - Sector 1 / 2 / 3 split times
  - Tire compound and age per stint
  - Practice session (FP1, FP2, FP3) lap times
  - Pre-season testing session lap times
  - Weather data per lap (track temp, air temp, rain)

HOW TO RUN:
    python src/data_collection/get_fastf1_data.py

FIRST TIME WARNING:
    FastF1 downloads data from the F1 servers and caches it locally.
    The first run takes 1-2 hours depending on your internet speed.
    After that, data is cached and loads in seconds.

OUTPUT:
    Saves CSV files into the data/ folder.
"""

import fastf1
import pandas as pd
import os
import warnings

# Suppress minor warnings from FastF1
warnings.filterwarnings("ignore")

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Years to pull FastF1 data for
# Note: FastF1 only has reliable data from 2018 onwards
START_YEAR = 2018
END_YEAR   = 2026

# Where to save output CSV files
OUTPUT_DIR = "data"

# Where FastF1 stores its local cache (avoids re-downloading data)
CACHE_DIR  = "data/fastf1_cache"


# ─── SETUP ────────────────────────────────────────────────────────────────────

def setup():
    """
    Creates output folders and enables the FastF1 cache.

    The cache is important — without it FastF1 re-downloads everything
    from the F1 servers every time you run the script. With the cache,
    data loads instantly after the first download.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Tell FastF1 where to store cached data
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f"FastF1 cache enabled at: {CACHE_DIR}")
    print(f"Output directory ready:  {OUTPUT_DIR}/")


# ─── RACE LAP TIMES ───────────────────────────────────────────────────────────

def fetch_race_laps(year):
    """
    Fetches lap-by-lap data for every race in a season.

    For each lap we get:
      - Lap time
      - Sector 1, 2, 3 times
      - Tire compound and age
      - Whether it was a pit lap
      - Track status (yellow flag, safety car, etc.)
      - Speed trap (fastest speed on that lap)

    This is used to compute:
      - Long run race pace (avg lap time over stints)
      - Tire degradation rate
      - Sector specialisation scores

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per lap per driver per race
    """
    print(f"  Fetching race laps for {year}...")

    # Get the schedule for this year so we know all the race names
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"    Could not get schedule for {year}: {e}")
        return pd.DataFrame()

    all_laps = []

    for _, event in schedule.iterrows():
        round_num  = event.get("RoundNumber", 0)
        event_name = event.get("EventName", "")

        # Skip rounds that haven't happened yet (future races have no data)
        if round_num == 0:
            continue

        print(f"    Round {round_num}: {event_name}", end=" ... ")

        try:
            # Load the race session
            session = fastf1.get_session(year, round_num, "R")
            session.load(telemetry=False, weather=True, messages=False)

            laps = session.laps.copy()

            if laps.empty:
                print("no data")
                continue

            # Add identifiers so we know which race this lap belongs to
            laps["year"]       = year
            laps["round"]      = round_num
            laps["event_name"] = event_name

            # Select only the columns we need (keeps file size manageable)
            cols_to_keep = [
                "year", "round", "event_name",
                "Driver", "Team",
                "LapNumber", "LapTime",
                "Sector1Time", "Sector2Time", "Sector3Time",
                "Compound", "TyreLife", "FreshTyre",
                "Stint", "PitOutTime", "PitInTime",
                "TrackStatus", "IsPersonalBest",
                "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                "AirTemp", "TrackTemp", "Humidity",
                "Rainfall", "WindSpeed", "WindDirection",
            ]

            # Only keep columns that actually exist in this session's data
            cols_to_keep = [c for c in cols_to_keep if c in laps.columns]
            laps = laps[cols_to_keep].copy()

            # Convert timedelta columns to seconds (easier to work with)
            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                if col in laps.columns:
                    laps[col] = laps[col].dt.total_seconds()

            all_laps.append(laps)
            print(f"{len(laps)} laps")

        except Exception as e:
            print(f"failed ({e})")
            continue

    if not all_laps:
        print(f"    No race lap data collected for {year}")
        return pd.DataFrame()

    df = pd.concat(all_laps, ignore_index=True)
    print(f"  Total race laps for {year}: {len(df):,}")
    return df


# ─── PRACTICE SESSION LAPS ────────────────────────────────────────────────────

def fetch_practice_laps(year):
    """
    Fetches FP1, FP2, and FP3 lap data for every race weekend.

    Practice data is extremely valuable because:
      - FP2 long run pace is the best pre-race predictor of race pace
      - FP3 single lap pace predicts qualifying performance
      - Practice shows tire behavior before the race

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per lap per driver per practice session
    """
    print(f"  Fetching practice laps for {year}...")

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"    Could not get schedule for {year}: {e}")
        return pd.DataFrame()

    all_laps = []

    # Practice session identifiers
    practice_sessions = ["FP1", "FP2", "FP3"]

    for _, event in schedule.iterrows():
        round_num  = event.get("RoundNumber", 0)
        event_name = event.get("EventName", "")

        if round_num == 0:
            continue

        for session_name in practice_sessions:
            try:
                session = fastf1.get_session(year, round_num, session_name)
                session.load(telemetry=False, weather=True, messages=False)

                laps = session.laps.copy()

                if laps.empty:
                    continue

                laps["year"]         = year
                laps["round"]        = round_num
                laps["event_name"]   = event_name
                laps["session_type"] = session_name

                cols_to_keep = [
                    "year", "round", "event_name", "session_type",
                    "Driver", "Team",
                    "LapNumber", "LapTime",
                    "Sector1Time", "Sector2Time", "Sector3Time",
                    "Compound", "TyreLife", "FreshTyre",
                    "SpeedFL", "SpeedST",
                    "AirTemp", "TrackTemp", "Humidity", "Rainfall",
                ]

                cols_to_keep = [c for c in cols_to_keep if c in laps.columns]
                laps = laps[cols_to_keep].copy()

                for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                    if col in laps.columns:
                        laps[col] = laps[col].dt.total_seconds()

                all_laps.append(laps)

            except Exception:
                # Not all sessions exist for all events (e.g. sprint weekends have no FP2)
                continue

    if not all_laps:
        print(f"    No practice data collected for {year}")
        return pd.DataFrame()

    df = pd.concat(all_laps, ignore_index=True)
    print(f"  Total practice laps for {year}: {len(df):,}")
    return df


# ─── PRE-SEASON TESTING ───────────────────────────────────────────────────────

def fetch_testing_laps(year):
    """
    Fetches pre-season testing session lap data.

    Testing data is one of the most valuable pre-season signals:
      - Long run pace (stints > 10 laps) shows true race pace
      - Total laps completed shows car reliability
      - Fastest lap gap to the field shows car performance

    FastF1 has testing data from 2019 onwards.
    Testing sessions are labelled as 'Testing' in the schedule.

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per lap per driver per testing day
    """
    print(f"  Fetching testing laps for {year}...")

    if year < 2019:
        print(f"    Testing data not available before 2019, skipping {year}")
        return pd.DataFrame()

    all_laps = []

    # FastF1 testing sessions: test number (1) and day (1, 2, 3)
    for test_num in [1]:           # Usually 1 pre-season test
        for day in [1, 2, 3]:      # 3 days of testing
            try:
                session = fastf1.get_testing_session(year, test_num, day)
                session.load(telemetry=False, weather=True, messages=False)

                laps = session.laps.copy()

                if laps.empty:
                    continue

                laps["year"]         = year
                laps["test_number"]  = test_num
                laps["test_day"]     = day
                laps["session_type"] = "Testing"

                cols_to_keep = [
                    "year", "test_number", "test_day", "session_type",
                    "Driver", "Team",
                    "LapNumber", "LapTime",
                    "Sector1Time", "Sector2Time", "Sector3Time",
                    "Compound", "TyreLife", "FreshTyre",
                    "Stint", "SpeedFL",
                    "AirTemp", "TrackTemp",
                ]

                cols_to_keep = [c for c in cols_to_keep if c in laps.columns]
                laps = laps[cols_to_keep].copy()

                for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                    if col in laps.columns:
                        laps[col] = laps[col].dt.total_seconds()

                all_laps.append(laps)
                print(f"    Test {test_num} Day {day}: {len(laps)} laps")

            except Exception as e:
                # Some years/days may not exist
                continue

    if not all_laps:
        print(f"    No testing data found for {year}")
        return pd.DataFrame()

    df = pd.concat(all_laps, ignore_index=True)
    print(f"  Total testing laps for {year}: {len(df):,}")
    return df


# ─── QUALIFYING LAP TIMES ─────────────────────────────────────────────────────

def fetch_qualifying_laps(year):
    """
    Fetches qualifying session lap data including Q1, Q2, Q3 laps.

    This gives us precise lap times for qualifying including:
      - Exact sector times for each qualifying lap
      - Which tyre compound was used in qualifying
      - Speed trap measurements

    Args:
        year (int): The season year

    Returns:
        pd.DataFrame: One row per lap per driver per qualifying session
    """
    print(f"  Fetching qualifying laps for {year}...")

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"    Could not get schedule for {year}: {e}")
        return pd.DataFrame()

    all_laps = []

    for _, event in schedule.iterrows():
        round_num  = event.get("RoundNumber", 0)
        event_name = event.get("EventName", "")

        if round_num == 0:
            continue

        try:
            session = fastf1.get_session(year, round_num, "Q")
            session.load(telemetry=False, weather=False, messages=False)

            laps = session.laps.copy()

            if laps.empty:
                continue

            laps["year"]       = year
            laps["round"]      = round_num
            laps["event_name"] = event_name

            cols_to_keep = [
                "year", "round", "event_name",
                "Driver", "Team",
                "LapNumber", "LapTime",
                "Sector1Time", "Sector2Time", "Sector3Time",
                "Compound", "TyreLife",
                "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
                "IsPersonalBest",
            ]

            cols_to_keep = [c for c in cols_to_keep if c in laps.columns]
            laps = laps[cols_to_keep].copy()

            for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
                if col in laps.columns:
                    laps[col] = laps[col].dt.total_seconds()

            all_laps.append(laps)

        except Exception as e:
            continue

    if not all_laps:
        print(f"    No qualifying lap data for {year}")
        return pd.DataFrame()

    df = pd.concat(all_laps, ignore_index=True)
    print(f"  Total qualifying laps for {year}: {len(df):,}")
    return df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Runs the complete FastF1 data collection pipeline.

    For each year from START_YEAR to END_YEAR:
      1. Race lap times (lap-by-lap for every race)
      2. Practice session laps (FP1, FP2, FP3)
      3. Qualifying laps (Q1, Q2, Q3)
      4. Pre-season testing laps

    Saves one CSV per data type into the data/ folder.

    NOTE: First run is slow (1-2 hours) as FastF1 downloads from F1 servers.
          Subsequent runs are fast as data is cached locally.
    """
    setup()

    all_race_laps  = []
    all_prac_laps  = []
    all_quali_laps = []
    all_test_laps  = []

    years = list(range(START_YEAR, END_YEAR + 1))
    print(f"\nPulling FastF1 data for {len(years)} seasons: {START_YEAR} to {END_YEAR}")
    print("=" * 60)
    print("NOTE: First run downloads from F1 servers — this takes 1-2 hours.")
    print("      Subsequent runs load from local cache and take seconds.")
    print("=" * 60)

    for year in years:
        print(f"\n[{year}]")

        race_laps  = fetch_race_laps(year)
        prac_laps  = fetch_practice_laps(year)
        quali_laps = fetch_qualifying_laps(year)
        test_laps  = fetch_testing_laps(year)

        if not race_laps.empty:  all_race_laps.append(race_laps)
        if not prac_laps.empty:  all_prac_laps.append(prac_laps)
        if not quali_laps.empty: all_quali_laps.append(quali_laps)
        if not test_laps.empty:  all_test_laps.append(test_laps)

    # ── Combine and save ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving CSV files...")

    datasets = {
        "fastf1_race_laps.csv":    all_race_laps,
        "fastf1_practice_laps.csv": all_prac_laps,
        "fastf1_quali_laps.csv":   all_quali_laps,
        "fastf1_testing_laps.csv": all_test_laps,
    }

    for filename, data_list in datasets.items():
        if data_list:
            df = pd.concat(data_list, ignore_index=True)
            path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(path, index=False)
            print(f"  Saved {path}  ({len(df):,} rows)")
        else:
            print(f"  Skipped {filename} — no data collected")

    print("\nFastF1 data collection complete!")
    print(f"All files saved to: {OUTPUT_DIR}/")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_pipeline()