"""
get_openf1_data.py
------------------
Fetches 2026 F1 race results from OpenF1 API.
OpenF1 updates within MINUTES of each session ending — much faster
than Jolpica which takes 24-48 hours.

Strategy:
  - OpenF1 handles 2023+ data (fast updates)
  - Jolpica handles 2016-2022 historical data (already collected)
  - This script MERGES OpenF1 2026 data into existing race_results.csv

HOW TO RUN:
    python src/data_collection/get_openf1_data.py

OUTPUT:
    Updates data/race_results.csv with latest 2026 results
    Updates data/qualifying_results.csv with latest 2026 quali
    Saves data/openf1_positions.csv (live position data)

API: https://api.openf1.org/v1/
No API key needed. Free tier: 3 req/sec, 30 req/min.
"""

import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

BASE_URL     = "https://api.openf1.org/v1"
CURRENT_YEAR = 2026
REQUEST_DELAY = 0.4  # Stay within 3 req/sec limit

# ─── DRIVER NUMBER TO ID MAPPING ──────────────────────────────────────────────
# OpenF1 uses driver numbers, Jolpica uses driver IDs
# This maps 2026 driver numbers to our driver_id format

# 2026 driver numbers — verified from OpenF1 live data
DRIVER_NUMBER_MAP = {
    1:  "norris",           # Lando Norris - McLaren
    3:  "max_verstappen",   # Max Verstappen - Red Bull
    4:  "norris",           # backup
    5:  "bortoleto",        # Gabriel Bortoleto - Audi
    6:  "hadjar",           # Isack Hadjar - Red Bull
    10: "gasly",            # Pierre Gasly - Alpine
    11: "perez",            # Sergio Perez - Cadillac
    12: "antonelli",        # Andrea Kimi Antonelli - Mercedes
    14: "alonso",           # Fernando Alonso - Aston Martin
    16: "leclerc",          # Charles Leclerc - Ferrari
    18: "stroll",           # Lance Stroll - Aston Martin
    23: "albon",            # Alexander Albon - Williams
    27: "hulkenberg",       # Nico Hulkenberg - Audi
    30: "lawson",           # Liam Lawson - Racing Bulls
    31: "ocon",             # Esteban Ocon - Haas
    41: "arvid_lindblad",   # Arvid Lindblad - Racing Bulls
    43: "colapinto",        # Franco Colapinto - Alpine
    44: "hamilton",         # Lewis Hamilton - Ferrari
    55: "sainz",            # Carlos Sainz - Williams
    63: "russell",          # George Russell - Mercedes
    77: "bottas",           # Valtteri Bottas - Cadillac
    81: "piastri",          # Oscar Piastri - McLaren
    87: "bearman",          # Oliver Bearman - Haas
}


# Race name mapping — round number → proper race name
RACE_NAMES_2026 = {
    1: "Australian Grand Prix",    2: "Chinese Grand Prix",
    3: "Japanese Grand Prix",      4: "Bahrain Grand Prix",
    5: "Saudi Arabian Grand Prix", 6: "Miami Grand Prix",
    7: "Emilia Romagna Grand Prix",8: "Monaco Grand Prix",
    9: "Spanish Grand Prix",       10:"Canadian Grand Prix",
    11:"Austrian Grand Prix",      12:"British Grand Prix",
    13:"Belgian Grand Prix",       14:"Hungarian Grand Prix",
    15:"Dutch Grand Prix",         16:"Italian Grand Prix",
    17:"Azerbaijan Grand Prix",    18:"Singapore Grand Prix",
    19:"United States Grand Prix", 20:"Mexico City Grand Prix",
    21:"São Paulo Grand Prix",     22:"Las Vegas Grand Prix",
    23:"Qatar Grand Prix",         24:"Abu Dhabi Grand Prix",
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def api_get(endpoint, params=None, retries=3):
    """GET request to OpenF1 API with retry."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.json()
        except Exception as e:
            print(f"  Request failed ({attempt+1}/{retries}): {e}")
            time.sleep(2)
    return []


def get_driver_id(driver_number, name_first="", name_last=""):
    """
    Converts OpenF1 driver number to our driver_id format.
    Falls back to name-based lookup if number not in map.
    """
    # Try number map first
    if driver_number in DRIVER_NUMBER_MAP:
        return DRIVER_NUMBER_MAP[driver_number]

    # Fall back to name — convert "Kimi Antonelli" → "antonelli"
    if name_last:
        return name_last.lower().replace(" ", "_").replace("-", "_")

    return f"driver_{driver_number}"


# ─── STEP 1: GET ALL 2026 SESSIONS ────────────────────────────────────────────

def get_sessions_2026():
    """
    Fetches all 2026 F1 sessions from OpenF1.

    Returns:
        pd.DataFrame: All sessions with session_key, type, meeting info
    """
    print("Fetching 2026 sessions from OpenF1...")

    data = api_get("sessions", {"year": CURRENT_YEAR})

    if not data:
        print("  No sessions found")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    print(f"  Found {len(df)} sessions in 2026")
    return df


# ─── STEP 2: GET RACE RESULTS FROM POSITIONS ──────────────────────────────────

def get_race_result_for_session(session_key, session_info):
    """
    Gets final race result for one session using OpenF1 position data.

    OpenF1 doesn't have a dedicated "results" endpoint — instead we:
    1. Get the final position of each driver (last position update)
    2. Get driver info (name, number, team)
    3. Combine into a race result

    Args:
        session_key  (int):           OpenF1 session key
        session_info (pd.Series):     Row from sessions DataFrame

    Returns:
        list: Race result rows in our format
    """
    print(f"  Fetching result for session {session_key} ({session_info.get('session_name','?')})...")

    # Get final driver positions
    positions = api_get("position", {
        "session_key": session_key,
    })

    if not positions:
        return []

    pos_df = pd.DataFrame(positions)

    # Get the LAST position update per driver = final finishing position
    if "date" in pos_df.columns:
        pos_df["date"] = pd.to_datetime(pos_df["date"], errors="coerce")
        final_pos = (
            pos_df.sort_values("date")
            .groupby("driver_number")
            .last()
            .reset_index()
            [["driver_number", "position"]]
        )
    else:
        final_pos = pos_df.groupby("driver_number")["position"].last().reset_index()

    # Get driver info
    drivers = api_get("drivers", {"session_key": session_key})
    if not drivers:
        return []

    driver_df = pd.DataFrame(drivers)[
        ["driver_number","name_acronym","full_name","team_name","country_code"]
    ].drop_duplicates("driver_number")

    # Split full_name into first + last
    driver_df["first_name"] = driver_df["full_name"].apply(
        lambda x: str(x).split()[0] if pd.notna(x) and len(str(x).split()) > 0 else ""
    )
    driver_df["last_name"] = driver_df["full_name"].apply(
        lambda x: " ".join(str(x).split()[1:]) if pd.notna(x) and len(str(x).split()) > 1 else str(x)
    )

    # Merge positions with driver info
    merged = final_pos.merge(driver_df, on="driver_number", how="left")

    # Get pit stop data for stint info
    pits = api_get("pit", {"session_key": session_key})
    pit_df = pd.DataFrame(pits) if pits else pd.DataFrame()

    # Build result rows in our standard format
    rows = []
    race_name   = RACE_NAMES_2026.get(round_num, session_info.get("meeting_name", f"Round {round_num}"))
    circuit_id  = str(session_info.get("circuit_short_name", "")).lower().replace(" ", "_")
    race_date   = str(session_info.get("date_start", ""))[:10]
    round_num   = session_info.get("meeting_key", 0)
    country     = session_info.get("country_name", "")
    location    = session_info.get("location", "")

    for _, row in merged.iterrows():
        driver_num = int(row.get("driver_number", 0))
        driver_id  = get_driver_id(
            driver_num,
            row.get("first_name", ""),
            row.get("last_name", "")
        )

        # Status — simplify
        finish_pos = row.get("position", None)
        status = "Finished" if pd.notna(finish_pos) and finish_pos <= 10 else "Classified"

        rows.append({
            "year":             CURRENT_YEAR,
            "round":            round_num,
            "race_name":        race_name,
            "circuit_name":     session_info.get("circuit_short_name", ""),
            "circuit_id":       circuit_id,
            "race_date":        race_date,
            "country":          country,
            "locality":         location,
            "lat":              None,
            "long":             None,
            "driver_id":        driver_id,
            "driver_code":      row.get("name_acronym", ""),
            "driver_name":      row.get("full_name", ""),
            "constructor_id":   str(row.get("team_name","")).lower().replace(" ","_"),
            "constructor_name": row.get("team_name", ""),
            "grid_position":    None,  # Will fill from quali
            "finish_position":  finish_pos,
            "points":           0.0,   # Will calculate from position
            "laps_completed":   None,
            "status":           status,
            "fastest_lap_rank": None,
            "fastest_lap_time": None,
        })

    return rows


# ─── STEP 3: GET RACE RESULT USING DEDICATED ENDPOINT ─────────────────────────

def get_race_results_2026():
    """
    Fetches all completed 2026 race results from OpenF1.

    Uses the /sessions endpoint to find race sessions,
    then /position for final results.

    Returns:
        pd.DataFrame: Race results in our standard format
    """
    print("\nFetching 2026 race results from OpenF1...")

    sessions_df = get_sessions_2026()
    if sessions_df.empty:
        return pd.DataFrame()

    # Separate main races from sprint races
    # Sprint data saved separately as features — NOT mixed into race_results
    if "session_name" in sessions_df.columns:
        # Main race only (not sprint)
        race_sessions = sessions_df[
            sessions_df["session_name"].str.lower().isin(["race"])
        ].copy()
        if race_sessions.empty:
            race_sessions = sessions_df[
                sessions_df["session_name"].str.lower().str.contains("race", na=False) &
                ~sessions_df["session_name"].str.lower().str.contains("sprint", na=False)
            ].copy()
    else:
        race_sessions = sessions_df[
            ~sessions_df.get("session_type", pd.Series()).str.lower().str.contains("sprint", na=False)
        ].copy()

    print(f"  Found {len(race_sessions)} race sessions")

    if race_sessions.empty:
        return pd.DataFrame()

    # Check which sessions have ended (date_end in the past)
    today = datetime.now()
    if "date_end" in race_sessions.columns:
        race_sessions["date_end_dt"] = pd.to_datetime(
            race_sessions["date_end"], errors="coerce", utc=True
        ).dt.tz_localize(None)
        completed = race_sessions[race_sessions["date_end_dt"] < today]
    else:
        completed = race_sessions

    print(f"  Completed race sessions: {len(completed)}")

    all_rows = []

    # Build meeting_key → round_number mapping
    # Each unique meeting = one race weekend = one round number
    # Sort meetings by date so Round 1 = first race, Round 2 = second etc.
    if "meeting_key" in completed.columns and "date_start" in completed.columns:
        meetings = (
            completed[["meeting_key","date_start"]]
            .drop_duplicates("meeting_key")
            .sort_values("date_start")
            .reset_index(drop=True)
        )
        meeting_to_round = {
            row["meeting_key"]: idx + 1
            for idx, (_, row) in enumerate(meetings.iterrows())
        }
    else:
        meeting_to_round = {}

    for _, session in completed.iterrows():
        session_key = session.get("session_key")
        if not session_key:
            continue

        rows = get_race_result_for_session(int(session_key), session)
        all_rows.extend(rows)

        # Use meeting_key for round number so sprints don't create extra rounds
        meeting_key = session.get("meeting_key", 0)
        round_num   = meeting_to_round.get(meeting_key, session.get("meeting_key", 0))

        for r in rows:
            r["round"] = round_num

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Calculate points from finishing position
    POINTS = {1:25,2:18,3:15,4:12,5:10,6:8,7:6,8:4,9:2,10:1}
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["points"] = df["finish_position"].apply(lambda p: POINTS.get(int(p), 0) if pd.notna(p) else 0)

    print(f"  Total 2026 race result rows: {len(df)}")
    return df


# ─── STEP 4: GET QUALIFYING RESULTS ───────────────────────────────────────────

def get_qualifying_results_2026():
    """
    Fetches 2026 qualifying results from OpenF1.
    Uses final sector times to determine grid order.

    Returns:
        pd.DataFrame: Qualifying results in our format
    """
    print("\nFetching 2026 qualifying results from OpenF1...")

    sessions_df = get_sessions_2026()
    if sessions_df.empty:
        return pd.DataFrame()

    # Filter to qualifying sessions
    if "session_name" in sessions_df.columns:
        quali_sessions = sessions_df[
            sessions_df["session_name"].str.lower().str.contains("qualifying|quali", na=False)
        ]
    else:
        return pd.DataFrame()

    print(f"  Found {len(quali_sessions)} qualifying sessions")

    all_rows = []
    today    = datetime.now()

    for idx, (_, session) in enumerate(quali_sessions.iterrows()):
        session_key = session.get("session_key")
        if not session_key:
            continue

        # Check if completed
        date_end = session.get("date_end", "")
        try:
            end_dt = pd.to_datetime(date_end, utc=True).tz_localize(None)
            if end_dt > today:
                continue
        except Exception:
            pass

        print(f"  Fetching quali {session_key} ({session.get('meeting_name','?')})...")

        # Get lap times to determine qualifying order
        laps = api_get("laps", {"session_key": session_key})

        if not laps:
            continue

        lap_df = pd.DataFrame(laps)

        if "lap_duration" not in lap_df.columns:
            continue

        lap_df["lap_duration"] = pd.to_numeric(lap_df["lap_duration"], errors="coerce")

        # Best lap per driver = qualifying position
        best_laps = (
            lap_df.dropna(subset=["lap_duration"])
            .groupby("driver_number")["lap_duration"]
            .min()
            .reset_index()
            .sort_values("lap_duration")
        )
        best_laps["quali_position"] = range(1, len(best_laps) + 1)

        # Get driver info
        drivers = api_get("drivers", {"session_key": session_key})
        driver_df = pd.DataFrame(drivers)[
            ["driver_number","name_acronym","full_name","team_name"]
        ].drop_duplicates("driver_number") if drivers else pd.DataFrame()

        merged = best_laps.merge(driver_df, on="driver_number", how="left")

        race_name   = session.get("meeting_name", "")
        circuit_id  = str(session.get("circuit_short_name","")).lower().replace(" ","_")
        race_date   = str(session.get("date_start",""))[:10]
        meeting_key = session.get("meeting_key", idx + 1)

        # Map meeting_key to round number using sorted meeting order
        if not quali_sessions.empty and "meeting_key" in quali_sessions.columns:
            sorted_meetings = sorted(quali_sessions["meeting_key"].unique())
            meeting_round_map = {mk: i+1 for i, mk in enumerate(sorted_meetings)}
            round_num = meeting_round_map.get(meeting_key, idx + 1)
        else:
            round_num = idx + 1

        for _, row in merged.iterrows():
            driver_num = int(row.get("driver_number", 0))
            driver_id  = get_driver_id(driver_num, "", row.get("full_name","").split()[-1] if row.get("full_name") else "")

            all_rows.append({
                "year":           CURRENT_YEAR,
                "round":          round_num,
                "race_name":      race_name,
                "circuit_id":     circuit_id,
                "driver_id":      driver_id,
                "driver_code":    row.get("name_acronym",""),
                "constructor_id": str(row.get("team_name","")).lower().replace(" ","_"),
                "quali_position": row.get("quali_position"),
                "best_lap_s":     row.get("lap_duration"),
                "q1_time":        None,
                "q2_time":        None,
                "q3_time":        None,
            })

    df = pd.DataFrame(all_rows)
    print(f"  Total 2026 qualifying rows: {len(df)}")
    return df


# ─── STEP 5: MERGE INTO EXISTING DATA ─────────────────────────────────────────

def merge_into_existing(new_df, csv_filename, key_cols=["year","round","driver_id"]):
    """
    Merges new OpenF1 data into existing CSV.
    Replaces any existing 2026 rows with fresh OpenF1 data.
    Keeps all 2016-2025 Jolpica data untouched.

    Args:
        new_df       (pd.DataFrame): New OpenF1 data
        csv_filename (str):          Filename in data/ folder
        key_cols     (list):         Columns that uniquely identify a row
    """
    path = os.path.join(DATA_DIR, csv_filename)

    if os.path.exists(path):
        existing = pd.read_csv(path)
        # Remove existing 2026 rows (will be replaced by OpenF1 data)
        historical = existing[existing["year"] != CURRENT_YEAR].copy()
        combined   = pd.concat([historical, new_df], ignore_index=True)
    else:
        combined = new_df

    # Sort by year, round
    if "year" in combined.columns and "round" in combined.columns:
        combined = combined.sort_values(["year","round"]).reset_index(drop=True)

    combined.to_csv(path, index=False)
    print(f"  Saved {csv_filename}: {len(combined):,} rows "
          f"({len(new_df)} from OpenF1 2026, "
          f"{len(combined) - len(new_df)} historical)")


# ─── STEP 6: GET DRIVER MAPPING FROM LIVE SESSION ─────────────────────────────

def update_driver_number_map():
    """
    Updates the driver number → driver_id mapping from the latest 2026 session.
    Prints any unknown driver numbers so you can update DRIVER_NUMBER_MAP.
    """
    print("\nChecking driver number mapping...")

    sessions_df = get_sessions_2026()
    if sessions_df.empty:
        return

    # Get most recent session
    latest_session = sessions_df.iloc[-1]
    session_key    = latest_session.get("session_key")

    drivers = api_get("drivers", {"session_key": session_key})
    if not drivers:
        return

    print("\nCurrent 2026 driver numbers:")
    print(f"  {'#':<5} {'Name':<30} {'Team':<25} {'Mapped to'}")
    print("  " + "-" * 70)

    for d in sorted(drivers, key=lambda x: x.get("driver_number", 0)):
        num   = d.get("driver_number", "?")
        name  = d.get("full_name", "?")
        team  = d.get("team_name", "?")
        mapped = DRIVER_NUMBER_MAP.get(num, "⚠️ NOT MAPPED")
        print(f"  {num:<5} {name:<30} {team:<25} {mapped}")


# ─── SPRINT FEATURES ─────────────────────────────────────────────────────────

def get_sprint_features_2026():
    """
    Fetches Sprint race results as FEATURES for the main race prediction.

    Sprint results are NOT added to race_results.csv — instead saved as
    sprint_features.csv which gets merged into master_dataset.csv.

    Sprint features tell us:
      - sprint_pos: sprint finishing position (car pace at this circuit NOW)
      - sprint_pts: sprint points (confidence/momentum indicator)
      - sprint_dnf: did car break down? (reliability flag)
      - sprint_positions_gained: overtaking ability shown in sprint
      - sprint_gap_to_leader_s: time gap to sprint winner (relative pace)

    Returns:
        pd.DataFrame: Sprint features per driver per round
    """
    print("\nFetching 2026 sprint features from OpenF1...")

    sessions_df = get_sessions_2026()
    if sessions_df.empty:
        return pd.DataFrame()

    # Get sprint sessions only
    if "session_name" in sessions_df.columns:
        sprint_sessions = sessions_df[
            sessions_df["session_name"].str.lower().str.contains("sprint", na=False) &
            ~sessions_df["session_name"].str.lower().str.contains("qualifying|shootout", na=False)
        ].copy()
    else:
        return pd.DataFrame()

    print(f"  Found {len(sprint_sessions)} sprint sessions")

    today = datetime.now()
    all_rows = []

    # Build meeting → round mapping
    sorted_meetings = sorted(sessions_df["meeting_key"].unique()) if "meeting_key" in sessions_df.columns else []
    meeting_round_map = {mk: i+1 for i, mk in enumerate(sorted_meetings)}

    for _, session in sprint_sessions.iterrows():
        session_key = session.get("session_key")
        if not session_key:
            continue

        # Check if completed
        date_end = session.get("date_end","")
        try:
            end_dt = pd.to_datetime(date_end, utc=True).tz_localize(None)
            if end_dt > today:
                continue
        except Exception:
            pass

        meeting_key = session.get("meeting_key", 0)
        round_num   = meeting_round_map.get(meeting_key, 0)
        circuit_id  = str(session.get("circuit_short_name","")).lower().replace(" ","_")

        print(f"  Fetching sprint session {session_key} (Round {round_num})...")

        # Get sprint positions
        positions = api_get("position", {"session_key": int(session_key)})
        if not positions:
            continue

        pos_df = pd.DataFrame(positions)
        if "date" in pos_df.columns:
            pos_df["date"] = pd.to_datetime(pos_df["date"], errors="coerce")
            final_pos = (
                pos_df.sort_values("date")
                .groupby("driver_number")
                .last()
                .reset_index()
                [["driver_number","position"]]
            )
        else:
            final_pos = pos_df.groupby("driver_number")["position"].last().reset_index()

        # Get sprint intervals (gap to leader)
        intervals = api_get("intervals", {"session_key": int(session_key)})
        int_df = pd.DataFrame(intervals) if intervals else pd.DataFrame()

        if not int_df.empty and "gap_to_leader" in int_df.columns:
            int_df["gap_to_leader"] = pd.to_numeric(int_df["gap_to_leader"], errors="coerce")
            last_interval = (
                int_df.sort_values("date") if "date" in int_df.columns else int_df
            ).groupby("driver_number")["gap_to_leader"].last().reset_index()
        else:
            last_interval = pd.DataFrame()

        # Get driver info
        drivers = api_get("drivers", {"session_key": int(session_key)})
        driver_df = pd.DataFrame(drivers)[
            ["driver_number","name_acronym","full_name","team_name"]
        ].drop_duplicates("driver_number") if drivers else pd.DataFrame()

        merged = final_pos.merge(driver_df, on="driver_number", how="left")
        if not last_interval.empty:
            merged = merged.merge(last_interval, on="driver_number", how="left")
        else:
            merged["gap_to_leader"] = None

        # Get sprint qualifying grid position
        sprint_quali = sessions_df[
            (sessions_df["meeting_key"] == meeting_key) &
            (sessions_df["session_name"].str.lower().str.contains("sprint qualifying|sprint shootout", na=False))
        ] if "session_name" in sessions_df.columns else pd.DataFrame()

        for _, row in merged.iterrows():
            driver_num = int(row.get("driver_number", 0))
            driver_id  = get_driver_id(driver_num, "", str(row.get("full_name","")).split()[-1])
            sprint_pos = row.get("position", None)

            # Sprint DNF = finished outside top 20 or position is null
            sprint_dnf = 1 if (sprint_pos is None or pd.isna(sprint_pos) or sprint_pos > 20) else 0

            # Points from sprint
            SPRINT_POINTS = {1:8,2:7,3:6,4:5,5:4,6:3,7:2,8:1}
            sprint_pts = SPRINT_POINTS.get(int(sprint_pos), 0) if sprint_pos and pd.notna(sprint_pos) else 0

            all_rows.append({
                "year":                   CURRENT_YEAR,
                "round":                  round_num,
                "circuit_id":             circuit_id,
                "driver_id":              driver_id,
                "driver_code":            row.get("name_acronym",""),
                "constructor_name":       row.get("team_name",""),
                "sprint_finish_pos":      sprint_pos,
                "sprint_pts":             sprint_pts,
                "sprint_dnf":             sprint_dnf,
                "sprint_gap_to_leader_s": row.get("gap_to_leader", None),
            })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["sprint_finish_pos"] = pd.to_numeric(df["sprint_finish_pos"], errors="coerce")

    # Sprint positions gained (needs grid — approximation from pos rank)
    df["sprint_positions_gained"] = 11 - df["sprint_finish_pos"].fillna(11)  # rough proxy

    # Save to sprint_features.csv
    path = os.path.join(DATA_DIR, "sprint_features_2026.csv")
    df.to_csv(path, index=False)
    print(f"  Saved sprint_features_2026.csv: {len(df)} rows")

    return df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_openf1_pipeline():
    """
    Full OpenF1 pipeline:
      1. Fetch all 2026 race results
      2. Fetch all 2026 qualifying results
      3. Merge into existing CSV files (keeping 2016-2025 Jolpica data)
      4. Print driver number mapping for verification
    """
    print("\n" + "=" * 60)
    print("OPENF1 2026 DATA PIPELINE")
    print("=" * 60)
    print(f"API    : {BASE_URL}")
    print(f"Season : {CURRENT_YEAR}")
    print(f"Free   : Yes (no API key needed)")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Update driver mapping first
    update_driver_number_map()

    # Fetch race results
    race_df = get_race_results_2026()

    if not race_df.empty:
        merge_into_existing(race_df, "race_results.csv")
        print(f"\n✓ Race results updated: {len(race_df)} 2026 rows")
    else:
        print("\n⚠ No race results found — API may not have 2026 data yet")

    # Fetch qualifying results
    quali_df = get_qualifying_results_2026()

    if not quali_df.empty:
        merge_into_existing(quali_df, "qualifying_results.csv")
        print(f"✓ Qualifying results updated: {len(quali_df)} 2026 rows")
    else:
        print("⚠ No qualifying results found yet")

    # Fetch sprint features
    sprint_df = get_sprint_features_2026()
    if not sprint_df.empty:
        print(f"✓ Sprint features saved: {len(sprint_df)} rows")
    else:
        print("⚠ No sprint features found yet")

    # Summary
    print("\n" + "=" * 60)
    print("OPENF1 UPDATE COMPLETE")
    print("=" * 60)
    print(f"Run this after every race for instant updates:")
    print(f"  python src/data_collection/get_openf1_data.py")
    print(f"\nThen rebuild models:")
    print(f"  python src/feature_engineering/build_dataset.py")
    print(f"  python src/models/ensemble.py")
    print("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_openf1_pipeline()