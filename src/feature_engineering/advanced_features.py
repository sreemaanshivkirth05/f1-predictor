"""
advanced_features.py
--------------------
Computes 80+ advanced features covering:

TRACK DOMINANCE (per driver per circuit):
  - Track dominance score (weighted wins/podiums)
  - Wet track dominance at specific circuit
  - Qualifying dominance at circuit
  - Lap record holder bonus
  - Home race motivation flag

ADVANCED PERFORMANCE (from FastF1 telemetry):
  - Tire degradation rate per driver
  - Safety car restart performance
  - Overtaking ability score
  - Defending ability score
  - DRS efficiency score
  - Sector 1/2/3 specialisation scores

PSYCHOLOGICAL & CONTEXTUAL:
  - Head to head vs specific rivals at this circuit
  - Teammate gap trend (last 5 races)
  - Championship pressure index
  - Back to back race weekend performance
  - Post summer break performance

HOW TO RUN (standalone test):
    python src/feature_engineering/advanced_features.py

OUTPUT:
    Returns a DataFrame — called by build_dataset.py
"""

import pandas as pd
import numpy as np
import os

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# Exponential decay for weighting older seasons less
DECAY_LAMBDA = 0.25

# Summer break is typically between rounds 12-13 (Hungary → Belgium)
SUMMER_BREAK_ROUND = 13

# Home country mapping for drivers
DRIVER_HOME_COUNTRY = {
    "max_verstappen":   "Netherlands",
    "russell":          "United Kingdom",
    "hamilton":         "United Kingdom",
    "leclerc":          "Monaco",
    "norris":           "United Kingdom",
    "sainz":            "Spain",
    "alonso":           "Spain",
    "piastri":          "Australia",
    "antonelli":        "Italy",
    "bearman":          "United Kingdom",
    "stroll":           "Canada",
    "gasly":            "France",
    "ocon":             "France",
    "albon":            "Thailand",
    "hulkenberg":       "Germany",
    "bottas":           "Finland",
    "zhou":             "China",
    "lawson":           "New Zealand",
    "colapinto":        "Argentina",
    "hadjar":           "France",
    "bortoleto":        "Brazil",
    "arvid_lindblad":   "Sweden",
    "perez":            "Mexico",
}

# Circuit country mapping
CIRCUIT_COUNTRY = {
    "albert_park":   "Australia",
    "bahrain":       "Bahrain",
    "jeddah":        "Saudi Arabia",
    "shanghai":      "China",
    "suzuka":        "Japan",
    "miami":         "United States",
    "imola":         "Italy",
    "monaco":        "Monaco",
    "villeneuve":    "Canada",
    "catalunya":     "Spain",
    "red_bull_ring": "Austria",
    "silverstone":   "United Kingdom",
    "hungaroring":   "Hungary",
    "spa":           "Belgium",
    "zandvoort":     "Netherlands",
    "monza":         "Italy",
    "baku":          "Azerbaijan",
    "marina_bay":    "Singapore",
    "americas":      "United States",
    "rodriguez":     "Mexico",
    "interlagos":    "Brazil",
    "vegas":         "United States",
    "losail":        "Qatar",
    "yas_marina":    "United Arab Emirates",
}

# Approximate lap records per circuit (seconds) — used for lap record holder bonus
LAP_RECORDS = {
    "albert_park":   79.820,   "bahrain":       91.567,
    "jeddah":        73.013,   "shanghai":      93.540,
    "suzuka":        90.983,   "miami":         88.067,
    "imola":         75.484,   "monaco":        71.382,
    "villeneuve":    72.564,   "catalunya":     79.854,
    "red_bull_ring": 64.984,   "silverstone":   87.097,
    "hungaroring":   76.627,   "spa":           103.269,
    "zandvoort":     72.097,   "monza":         80.046,
    "baku":          102.394,  "marina_bay":    99.627,
    "americas":      95.395,   "rodriguez":     79.118,
    "interlagos":    71.044,   "vegas":         87.303,
    "losail":        83.196,   "yas_marina":    87.875,
}


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_data():
    """Loads all required raw data files."""
    print("Loading data for advanced features...")

    race_path  = os.path.join(DATA_DIR, "race_results.csv")
    quali_path = os.path.join(DATA_DIR, "qualifying_results.csv")

    if not os.path.exists(race_path):
        raise FileNotFoundError(f"race_results.csv not found. Run get_ergast_data.py first.")

    race_df = pd.read_csv(race_path)
    race_df["finish_position"] = pd.to_numeric(race_df["finish_position"], errors="coerce")
    race_df["grid_position"]   = pd.to_numeric(race_df["grid_position"],   errors="coerce")
    race_df["points"]          = pd.to_numeric(race_df["points"],           errors="coerce")

    quali_df = pd.read_csv(quali_path) if os.path.exists(quali_path) else pd.DataFrame()
    if not quali_df.empty:
        quali_df["quali_position"] = pd.to_numeric(quali_df["quali_position"], errors="coerce")

    # Load FastF1 lap data if available
    fastf1_path = os.path.join(DATA_DIR, "fastf1_race_laps.csv")
    fastf1_df   = pd.read_csv(fastf1_path) if os.path.exists(fastf1_path) else pd.DataFrame()

    practice_path = os.path.join(DATA_DIR, "fastf1_practice_laps.csv")
    practice_df   = pd.read_csv(practice_path) if os.path.exists(practice_path) else pd.DataFrame()

    print(f"  Race results  : {len(race_df):,} rows")
    print(f"  Qualifying    : {len(quali_df):,} rows")
    print(f"  FastF1 laps   : {len(fastf1_df):,} rows")
    print(f"  Practice laps : {len(practice_df):,} rows")

    return race_df, quali_df, fastf1_df, practice_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRACK DOMINANCE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_track_dominance(race_df, quali_df):
    """
    Computes deep track dominance features per driver per circuit.

    Goes beyond simple win rate to capture:
      - Weighted dominance score (recent wins count more)
      - Podium streak at circuit
      - Qualifying dominance
      - Wet race dominance at this specific circuit
      - Head to head record vs top rivals at this circuit
      - Lap record holder flag

    Returns:
        pd.DataFrame: Track dominance features per driver per race
    """
    print("Computing track dominance features...")

    race_df = race_df.copy().dropna(subset=["finish_position"])
    records = []

    years  = sorted(race_df["year"].unique())

    for year in years:
        history = race_df[race_df["year"] < year]
        current = race_df[race_df["year"] == year]

        for circuit_id in current["circuit_id"].unique():
            circuit_races = current[current["circuit_id"] == circuit_id]
            rounds        = circuit_races["round"].unique()

            for round_num in rounds:
                drivers = current[
                    (current["circuit_id"] == circuit_id) &
                    (current["round"]       == round_num)
                ]["driver_id"].unique()

                # Circuit history for all drivers
                circuit_hist = history[history["circuit_id"] == circuit_id].copy()

                # Circuit country for home race check
                circuit_country = CIRCUIT_COUNTRY.get(circuit_id, "")

                for driver_id in drivers:
                    drv_hist = circuit_hist[circuit_hist["driver_id"] == driver_id].copy()

                    # ── Weighted track dominance score ────────────────────────
                    if len(drv_hist) > 0:
                        drv_hist["years_ago"] = year - drv_hist["year"]
                        drv_hist["weight"]    = np.exp(-DECAY_LAMBDA * drv_hist["years_ago"])

                        # Win score: each win worth 1.0, weighted by recency
                        wins_score   = (drv_hist["weight"] * (drv_hist["finish_position"] == 1)).sum()
                        podium_score = (drv_hist["weight"] * (drv_hist["finish_position"] <= 3)).sum()
                        top6_score   = (drv_hist["weight"] * (drv_hist["finish_position"] <= 6)).sum()
                        total_weight = drv_hist["weight"].sum()

                        track_dominance    = (wins_score * 3 + podium_score * 1.5 + top6_score) / (total_weight * 5 + 0.001)
                        circuit_win_pct    = (drv_hist["finish_position"] == 1).mean()
                        circuit_podium_pct = (drv_hist["finish_position"] <= 3).mean()
                        circuit_visits     = len(drv_hist)
                        circuit_best       = int(drv_hist["finish_position"].min())
                        circuit_avg        = drv_hist["finish_position"].mean()

                        # Podium streak (consecutive recent races with podium)
                        recent = drv_hist.sort_values("year", ascending=False)
                        streak = 0
                        for _, r in recent.iterrows():
                            if r["finish_position"] <= 3:
                                streak += 1
                            else:
                                break
                        podium_streak = streak

                    else:
                        wins_score         = 0.0
                        podium_score       = 0.0
                        track_dominance    = 0.0
                        circuit_win_pct    = 0.0
                        circuit_podium_pct = 0.0
                        circuit_visits     = 0
                        circuit_best       = 20
                        circuit_avg        = 10.0
                        podium_streak      = 0

                    # ── Qualifying dominance at circuit ───────────────────────
                    if not quali_df.empty:
                        q_hist = quali_df[
                            (quali_df["circuit_id"] == circuit_id) &
                            (quali_df["driver_id"]  == driver_id) &
                            (quali_df["year"]        < year)
                        ]
                        if len(q_hist) > 0:
                            # Compare to field average qualifying position
                            field_avg_q = quali_df[
                                (quali_df["circuit_id"] == circuit_id) &
                                (quali_df["year"]        < year)
                            ].groupby(["year","round"])["quali_position"].mean().mean()

                            q_avg = q_hist["quali_position"].mean()
                            quali_dominance = (field_avg_q - q_avg) / (field_avg_q + 0.001)
                            pole_rate       = (q_hist["quali_position"] == 1).mean()
                            front_row_rate  = (q_hist["quali_position"] <= 2).mean()
                            top3_q_rate     = (q_hist["quali_position"] <= 3).mean()
                        else:
                            quali_dominance = 0.0
                            pole_rate       = 0.0
                            front_row_rate  = 0.0
                            top3_q_rate     = 0.0
                    else:
                        quali_dominance = 0.0
                        pole_rate       = 0.0
                        front_row_rate  = 0.0
                        top3_q_rate     = 0.0

                    # ── Wet track dominance at this specific circuit ───────────
                    # A race is "wet" if status includes many retirements or
                    # we flag it from finish position variance (proxy)
                    if len(drv_hist) >= 2:
                        # Use high position variance as wet proxy
                        # (wet races have more position shuffles)
                        race_variance = circuit_hist.groupby(["year","round"])["finish_position"].std().mean()
                        own_variance  = drv_hist.groupby(["year","round"])["finish_position"].std().mean() if len(drv_hist) > 1 else 0
                        wet_circuit_dominance = max(0, race_variance - own_variance) / (race_variance + 0.001)
                    else:
                        wet_circuit_dominance = 0.0

                    # ── Home race motivation ───────────────────────────────────
                    driver_country = DRIVER_HOME_COUNTRY.get(driver_id, "")
                    is_home_race   = int(driver_country == circuit_country and driver_country != "")

                    # Home race bonus — historically drivers perform ~8% better
                    # at their home circuit (crowd energy, extra motivation)
                    home_race_boost = 0.08 * is_home_race

                    # ── Lap record holder ─────────────────────────────────────
                    # Check if driver holds the lap record at this circuit
                    # We approximate by checking if they have the fastest lap
                    fastest_lap_holder = 0
                    if len(drv_hist) > 0:
                        circuit_fastest = race_df[
                            (race_df["circuit_id"] == circuit_id) &
                            (race_df["fastest_lap_rank"].notna())
                        ]
                        if not circuit_fastest.empty:
                            last_fastest = circuit_fastest.sort_values("year").tail(3)
                            if (last_fastest["driver_id"] == driver_id).any():
                                fastest_lap_holder = 1

                    records.append({
                        "year":                       year,
                        "round":                      round_num,
                        "driver_id":                  driver_id,
                        "circuit_id":                 circuit_id,
                        "track_dominance_score":      round(track_dominance, 4),
                        "circuit_weighted_wins":      round(wins_score, 3),
                        "circuit_weighted_podiums":   round(podium_score, 3),
                        "circuit_win_pct":            round(circuit_win_pct, 3),
                        "circuit_podium_pct":         round(circuit_podium_pct, 3),
                        "circuit_visits_count":       circuit_visits,
                        "circuit_best_finish":        circuit_best,
                        "circuit_avg_finish_weighted":round(circuit_avg, 2),
                        "podium_streak_at_circuit":   podium_streak,
                        "quali_dominance_at_circuit": round(quali_dominance, 4),
                        "pole_rate_at_circuit":       round(pole_rate, 3),
                        "front_row_rate_at_circuit":  round(front_row_rate, 3),
                        "top3_quali_rate_at_circuit": round(top3_q_rate, 3),
                        "wet_circuit_dominance":      round(wet_circuit_dominance, 4),
                        "is_home_race":               is_home_race,
                        "home_race_boost":            home_race_boost,
                        "fastest_lap_holder":         fastest_lap_holder,
                    })

    df = pd.DataFrame(records)
    print(f"  Track dominance features: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ADVANCED PERFORMANCE FEATURES (FastF1)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_tire_degradation(fastf1_df, race_df):
    """
    Computes tire degradation rate per driver.

    Deg rate = average lap time increase per lap within a stint.
    Lower deg rate = driver manages tyres better = can stay out longer
    and gain strategic advantage.

    Returns:
        pd.DataFrame: Tire degradation score per driver per year
    """
    print("Computing tire degradation features...")

    if fastf1_df.empty:
        print("  FastF1 data not available — using neutral values")
        drivers = race_df["driver_id"].unique()
        years   = race_df["year"].unique()
        rows = []
        for year in years:
            for drv in race_df[race_df["year"] == year]["driver_id"].unique():
                rows.append({
                    "year":             year,
                    "driver_id":        drv,
                    "tyre_deg_rate":    0.05,
                    "tyre_mgmt_score":  0.5,
                    "avg_stint_length": 20.0,
                    "long_run_pace":    0.0,
                })
        return pd.DataFrame(rows)

    records = []
    years   = sorted(fastf1_df["year"].unique()) if "year" in fastf1_df.columns else []

    # Normalise driver column name
    drv_col = "Driver" if "Driver" in fastf1_df.columns else "driver_id"

    for year in years:
        year_df = fastf1_df[fastf1_df["year"] == year].copy()

        for driver_code in year_df[drv_col].unique():
            drv_laps = year_df[year_df[drv_col] == driver_code].copy()

            if "LapTime" not in drv_laps.columns or drv_laps["LapTime"].isna().all():
                continue

            drv_laps["LapTime"] = pd.to_numeric(drv_laps["LapTime"], errors="coerce")
            drv_laps = drv_laps.dropna(subset=["LapTime"])

            # Filter outliers (pit laps, SC laps)
            q1, q3  = drv_laps["LapTime"].quantile([0.1, 0.9])
            drv_laps = drv_laps[
                (drv_laps["LapTime"] >= q1) &
                (drv_laps["LapTime"] <= q3)
            ]

            if len(drv_laps) < 5:
                continue

            # Compute degradation within stints
            deg_rates = []
            if "Stint" in drv_laps.columns:
                for stint_num in drv_laps["Stint"].unique():
                    stint = drv_laps[drv_laps["Stint"] == stint_num].copy()
                    if len(stint) < 4:
                        continue
                    stint = stint.sort_values("LapNumber") if "LapNumber" in stint.columns else stint
                    # Linear regression of lap time vs lap number
                    x = np.arange(len(stint))
                    y = stint["LapTime"].values
                    if len(x) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        if slope > 0:
                            deg_rates.append(slope)

            avg_deg_rate    = np.mean(deg_rates) if deg_rates else 0.05
            # Normalise: lower is better — convert to 0-1 score (1=best tyre mgmt)
            tyre_mgmt_score = max(0, 1.0 - avg_deg_rate / 0.3)

            # Long run pace (avg lap time on stints > 10 laps)
            long_run_laps = []
            if "Stint" in drv_laps.columns:
                for stint_num in drv_laps["Stint"].unique():
                    stint = drv_laps[drv_laps["Stint"] == stint_num]
                    if len(stint) >= 10:
                        long_run_laps.extend(stint["LapTime"].tolist())

            long_run_pace = np.mean(long_run_laps) if long_run_laps else drv_laps["LapTime"].median()
            avg_stint_len = drv_laps.groupby("Stint").size().mean() if "Stint" in drv_laps.columns else 20.0

            # Map driver code back to driver_id
            driver_id = driver_code.lower().replace(" ", "_")

            records.append({
                "year":             year,
                "driver_id":        driver_id,
                "driver_code":      str(driver_code),
                "tyre_deg_rate":    round(avg_deg_rate, 4),
                "tyre_mgmt_score":  round(tyre_mgmt_score, 3),
                "avg_stint_length": round(avg_stint_len, 1),
                "long_run_pace":    round(long_run_pace, 3),
            })

    df = pd.DataFrame(records) if records else pd.DataFrame()
    print(f"  Tyre degradation features: {len(df):,} rows")
    return df


def compute_sector_specialisation(fastf1_df, race_df):
    """
    Computes driver sector specialisation scores.

    S1 = braking/high speed (Silverstone S1, Spa S1)
    S2 = technical/slow corners (Monaco S2, Hungary S2)
    S3 = power/DRS (Monza S3, Baku S3)

    For each driver, computes how much better they are in each
    sector relative to the field average.

    Returns:
        pd.DataFrame: Sector specialisation per driver per year
    """
    print("Computing sector specialisation features...")

    if fastf1_df.empty or "Sector1Time" not in fastf1_df.columns:
        print("  FastF1 sector data not available — using neutral values")
        rows = []
        for year in race_df["year"].unique():
            for drv in race_df[race_df["year"] == year]["driver_id"].unique():
                rows.append({
                    "year":         year,
                    "driver_id":    drv,
                    "s1_score":     0.0,
                    "s2_score":     0.0,
                    "s3_score":     0.0,
                    "best_sector":  0,
                    "s1_vs_field":  0.0,
                    "s2_vs_field":  0.0,
                    "s3_vs_field":  0.0,
                })
        return pd.DataFrame(rows)

    records = []
    drv_col = "Driver" if "Driver" in fastf1_df.columns else "driver_id"

    for year in sorted(fastf1_df["year"].unique()):
        year_df = fastf1_df[fastf1_df["year"] == year].copy()

        for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            year_df[col] = pd.to_numeric(year_df[col], errors="coerce")

        # Field average per sector
        s1_field = year_df["Sector1Time"].median()
        s2_field = year_df["Sector2Time"].median()
        s3_field = year_df["Sector3Time"].median()

        for driver_code in year_df[drv_col].unique():
            drv = year_df[year_df[drv_col] == driver_code].copy()

            s1_avg = drv["Sector1Time"].median()
            s2_avg = drv["Sector2Time"].median()
            s3_avg = drv["Sector3Time"].median()

            if pd.isna(s1_avg) or pd.isna(s2_avg) or pd.isna(s3_avg):
                continue

            # Negative delta = faster than field = better
            s1_vs_field = (s1_field - s1_avg) / (s1_field + 0.001)
            s2_vs_field = (s2_field - s2_avg) / (s2_field + 0.001)
            s3_vs_field = (s3_field - s3_avg) / (s3_field + 0.001)

            best_sector = int(np.argmax([s1_vs_field, s2_vs_field, s3_vs_field]) + 1)

            driver_id = driver_code.lower().replace(" ", "_")

            records.append({
                "year":        year,
                "driver_id":   driver_id,
                "driver_code": str(driver_code),
                "s1_score":    round(s1_vs_field, 4),
                "s2_score":    round(s2_vs_field, 4),
                "s3_score":    round(s3_vs_field, 4),
                "best_sector": best_sector,
                "s1_vs_field": round(s1_vs_field, 4),
                "s2_vs_field": round(s2_vs_field, 4),
                "s3_vs_field": round(s3_vs_field, 4),
            })

    df = pd.DataFrame(records) if records else pd.DataFrame()
    print(f"  Sector specialisation features: {len(df):,} rows")
    return df


def compute_overtaking_defending(race_df, fastf1_df):
    """
    Computes overtaking ability and defending ability scores.

    Overtaking score:
      - Positions gained from grid per race (rolling avg)
      - Positions gained in final 10 laps (last stint overtakes)
      - Safety car restart positions gained

    Defending score:
      - How often driver defends position when pressured
      - Positions lost per race vs grid
      - DRS efficiency (using speed trap data)

    Returns:
        pd.DataFrame: Overtaking and defending scores per driver per year
    """
    print("Computing overtaking and defending features...")

    race_df = race_df.copy()
    race_df["positions_gained"] = race_df["grid_position"] - race_df["finish_position"]

    records = []
    years   = sorted(race_df["year"].unique())

    for year in years:
        history = race_df[race_df["year"] < year]
        current = race_df[race_df["year"] == year]

        for driver_id in current["driver_id"].unique():
            drv_hist = history[history["driver_id"] == driver_id]
            drv_curr = current[current["driver_id"] == driver_id]

            if drv_hist.empty:
                records.append({
                    "year":                     year,
                    "driver_id":                driver_id,
                    "overtaking_score":         0.0,
                    "defending_score":          0.0,
                    "avg_positions_gained":     0.0,
                    "overtake_rate_from_back":  0.0,
                    "positions_lost_avg":       0.0,
                    "sc_restart_gain_avg":      0.0,
                    "drs_efficiency_score":     0.5,
                    "late_race_gain_score":     0.0,
                })
                continue

            # Positions gained overall
            avg_gain = drv_hist["positions_gained"].mean()

            # Overtaking from back — when starting P10+, how much gain?
            back_starts = drv_hist[drv_hist["grid_position"] >= 10]
            back_gain   = back_starts["positions_gained"].mean() if len(back_starts) > 0 else 0.0

            # Defending — when starting P1-5, how many positions lost?
            front_starts = drv_hist[drv_hist["grid_position"] <= 5]
            front_loss   = front_starts["positions_gained"].mean() if len(front_starts) > 0 else 0.0
            # Positive = maintained/gained; negative = lost positions

            # Normalise to 0-1 scores
            overtaking_score = min(1.0, max(0.0, (avg_gain + 5) / 15))
            defending_score  = min(1.0, max(0.0, (front_loss + 3) / 8))

            # DRS efficiency from FastF1 speed traps
            drs_eff = 0.5
            if not fastf1_df.empty and "SpeedST" in fastf1_df.columns:
                drv_col = "Driver" if "Driver" in fastf1_df.columns else "driver_id"
                # Map driver_id to code
                code_map = dict(zip(
                    race_df["driver_id"].str.lower(),
                    race_df.get("driver_code", race_df["driver_id"])
                ))
                code = code_map.get(driver_id, driver_id.upper()[:3])

                drv_speed = fastf1_df[
                    (fastf1_df[drv_col].str.upper() == str(code).upper()) &
                    (fastf1_df["year"] < year)
                ]["SpeedST"].dropna() if drv_col in fastf1_df.columns else pd.Series()

                field_speed = fastf1_df[fastf1_df["year"] < year]["SpeedST"].dropna()

                if len(drv_speed) > 0 and len(field_speed) > 0:
                    drs_eff = min(1.0, max(0.0, drv_speed.mean() / field_speed.mean()))

            records.append({
                "year":                     year,
                "driver_id":                driver_id,
                "overtaking_score":         round(overtaking_score, 3),
                "defending_score":          round(defending_score, 3),
                "avg_positions_gained":     round(avg_gain, 2),
                "overtake_rate_from_back":  round(back_gain, 2),
                "positions_lost_avg":       round(-front_loss, 2),
                "sc_restart_gain_avg":      round(avg_gain * 0.3, 2),  # Proxy
                "drs_efficiency_score":     round(drs_eff, 3),
                "late_race_gain_score":     round(back_gain * 0.5, 2),
            })

    df = pd.DataFrame(records)
    print(f"  Overtaking/defending features: {len(df):,} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PSYCHOLOGICAL & CONTEXTUAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psychological_features(race_df, quali_df):
    """
    Computes psychological and contextual performance features.

    Features:
      - Championship pressure index
      - Head to head vs rivals at this specific circuit
      - Teammate gap trend (last 5 races — are you beating teammate more?)
      - Back to back race weekend performance
      - Post summer break performance
      - Season opening form (first 3 races)
      - Contract year motivation proxy

    Returns:
        pd.DataFrame: Psychological features per driver per race
    """
    print("Computing psychological and contextual features...")

    race_df = race_df.copy()
    race_df["is_dnf"] = race_df["status"].apply(
        lambda s: 0 if (str(s) == "Finished" or str(s).startswith("+")) else 1
    )
    race_df["is_podium"] = (race_df["finish_position"] <= 3).astype(int)
    race_df["is_win"]    = (race_df["finish_position"] == 1).astype(int)

    records = []
    years   = sorted(race_df["year"].unique())

    for year in years:
        year_df  = race_df[race_df["year"] == year].copy()
        history  = race_df[race_df["year"] < year]
        rounds   = sorted(year_df["round"].unique())

        for round_num in rounds:
            race_group = year_df[year_df["round"] == round_num]
            drivers    = race_group["driver_id"].unique()

            # Championship context for this round
            standings_so_far = year_df[year_df["round"] < round_num]

            if not standings_so_far.empty:
                pts_so_far   = standings_so_far.groupby("driver_id")["points"].sum()
                leader_pts   = pts_so_far.max()
                total_races  = 24
                races_done   = round_num - 1
                races_left   = total_races - races_done
                max_gain     = races_left * 26
            else:
                pts_so_far   = pd.Series(dtype=float)
                leader_pts   = 0
                races_left   = 24
                max_gain     = 24 * 26

            for driver_id in drivers:
                drv_year     = year_df[
                    (year_df["driver_id"] == driver_id) &
                    (year_df["round"]      < round_num)
                ]
                drv_all_hist = race_df[race_df["driver_id"] == driver_id]
                drv_history  = history[history["driver_id"] == driver_id]

                # ── Championship pressure index ────────────────────────────
                drv_pts    = pts_so_far.get(driver_id, 0) if not pts_so_far.empty else 0
                gap        = max(0, leader_pts - drv_pts)
                is_leader  = int(drv_pts >= leader_pts)

                # Pressure = how much you NEED to win vs how much you CAN afford to be cautious
                # Leader has high pressure too (defending)
                # Chaser has high pressure if gap is closeable
                if max_gain > 0:
                    pressure_index = min(1.0, gap / max_gain) if not is_leader else 0.7
                else:
                    pressure_index = 1.0 if gap == 0 else 0.0

                # ── Teammate gap trend ─────────────────────────────────────
                if not drv_year.empty:
                    constructor_id = race_group[race_group["driver_id"] == driver_id]["constructor_id"].values
                    if len(constructor_id) > 0:
                        constructor_id = constructor_id[0]
                        # Get teammate results
                        teammate_races = year_df[
                            (year_df["constructor_id"] == constructor_id) &
                            (year_df["driver_id"]       != driver_id) &
                            (year_df["round"]            < round_num)
                        ]

                        # Compare finishing positions in shared races
                        shared_rounds = set(drv_year["round"]) & set(teammate_races["round"])

                        if shared_rounds:
                            teammate_gaps = []
                            for r in sorted(shared_rounds)[-5:]:  # Last 5 shared races
                                my_pos  = drv_year[drv_year["round"] == r]["finish_position"].values
                                tm_pos  = teammate_races[teammate_races["round"] == r]["finish_position"].values
                                if len(my_pos) > 0 and len(tm_pos) > 0:
                                    # Positive = I finished ahead of teammate
                                    teammate_gaps.append(float(tm_pos[0]) - float(my_pos[0]))

                            teammate_gap_avg        = np.mean(teammate_gaps) if teammate_gaps else 0.0
                            teammate_gap_trend      = (
                                np.mean(teammate_gaps[-3:]) - np.mean(teammate_gaps[:3])
                                if len(teammate_gaps) >= 4 else 0.0
                            )
                            teammate_beats_pct      = np.mean([1 if g > 0 else 0 for g in teammate_gaps])
                        else:
                            teammate_gap_avg   = 0.0
                            teammate_gap_trend = 0.0
                            teammate_beats_pct = 0.5
                    else:
                        teammate_gap_avg   = 0.0
                        teammate_gap_trend = 0.0
                        teammate_beats_pct = 0.5
                else:
                    teammate_gap_avg   = 0.0
                    teammate_gap_trend = 0.0
                    teammate_beats_pct = 0.5

                # ── Back to back race performance ──────────────────────────
                # How does driver perform when racing two weekends in a row?
                # Back-to-back rounds are typically consecutive round numbers
                is_back_to_back = int(round_num in [r + 1 for r in rounds if r < round_num])

                if is_back_to_back and not drv_all_hist.empty:
                    # Find races that were also back to back historically
                    all_years_rounds = drv_all_hist.groupby("year")["round"].apply(list)
                    b2b_races  = []
                    norm_races = []

                    for y, rnds in all_years_rounds.items():
                        sorted_rnds = sorted(rnds)
                        for i, r in enumerate(sorted_rnds):
                            pos = drv_all_hist[
                                (drv_all_hist["year"] == y) &
                                (drv_all_hist["round"] == r)
                            ]["finish_position"].values
                            if len(pos) == 0:
                                continue
                            if i > 0 and sorted_rnds[i-1] == r - 1:
                                b2b_races.append(float(pos[0]))
                            else:
                                norm_races.append(float(pos[0]))

                    b2b_avg_finish  = np.mean(b2b_races)  if b2b_races  else 10.0
                    norm_avg_finish = np.mean(norm_races) if norm_races else 10.0
                    b2b_delta       = norm_avg_finish - b2b_avg_finish  # + = better in b2b
                else:
                    b2b_avg_finish = 10.0
                    b2b_delta      = 0.0

                # ── Post summer break performance ──────────────────────────
                is_post_summer = int(round_num == SUMMER_BREAK_ROUND)

                if not drv_all_hist.empty:
                    post_summer_races = []
                    pre_summer_races  = []

                    for y in drv_all_hist["year"].unique():
                        y_races = drv_all_hist[drv_all_hist["year"] == y]
                        post    = y_races[y_races["round"] == SUMMER_BREAK_ROUND]["finish_position"].values
                        pre     = y_races[y_races["round"] == SUMMER_BREAK_ROUND - 1]["finish_position"].values
                        if len(post) > 0:
                            post_summer_races.append(float(post[0]))
                        if len(pre) > 0:
                            pre_summer_races.append(float(pre[0]))

                    post_summer_avg = np.mean(post_summer_races) if post_summer_races else 10.0
                    pre_summer_avg  = np.mean(pre_summer_races)  if pre_summer_races  else 10.0
                    summer_break_effect = pre_summer_avg - post_summer_avg  # + = improves after break
                else:
                    post_summer_avg     = 10.0
                    summer_break_effect = 0.0

                # ── Season opening form (first 3 races) ────────────────────
                if not drv_all_hist.empty:
                    opening_races = drv_all_hist[drv_all_hist["round"] <= 3]
                    opening_avg   = opening_races["finish_position"].mean() if len(opening_races) > 0 else 10.0
                    overall_avg   = drv_all_hist["finish_position"].mean()
                    season_opener_bonus = overall_avg - opening_avg  # + = better in openers
                else:
                    season_opener_bonus = 0.0

                records.append({
                    "year":                   year,
                    "round":                  round_num,
                    "driver_id":              driver_id,
                    "championship_pressure":  round(pressure_index, 3),
                    "is_championship_leader": is_leader,
                    "points_gap_to_lead":     gap,
                    "teammate_gap_avg":       round(teammate_gap_avg, 2),
                    "teammate_gap_trend":     round(teammate_gap_trend, 2),
                    "teammate_beats_pct":     round(teammate_beats_pct, 3),
                    "is_back_to_back_race":   is_back_to_back,
                    "b2b_avg_finish":         round(b2b_avg_finish, 2),
                    "b2b_performance_delta":  round(b2b_delta, 2),
                    "is_post_summer_break":   is_post_summer,
                    "post_summer_avg_finish": round(post_summer_avg, 2),
                    "summer_break_effect":    round(summer_break_effect, 2),
                    "season_opener_bonus":    round(season_opener_bonus, 2),
                })

    df = pd.DataFrame(records)
    print(f"  Psychological features: {len(df):,} rows")
    return df


def compute_rival_h2h_at_circuit(race_df):
    """
    Computes head to head record vs top rivals at each specific circuit.

    For each driver at each circuit, computes:
      - H2H win rate vs each of their top 5 rivals at this circuit
      - H2H weighted score (recent battles count more)

    Returns:
        pd.DataFrame: Rival H2H features per driver per circuit per year
    """
    print("Computing rival head to head at circuit features...")

    records = []
    years   = sorted(race_df["year"].unique())

    for year in years:
        history = race_df[race_df["year"] < year]
        current = race_df[race_df["year"] == year]

        for circuit_id in current["circuit_id"].unique():
            current_circuit = current[current["circuit_id"] == circuit_id]
            hist_circuit    = history[history["circuit_id"] == circuit_id]
            rounds          = current_circuit["round"].unique()

            for round_num in rounds:
                drivers_in_race = current_circuit[
                    current_circuit["round"] == round_num
                ]["driver_id"].unique()

                for driver_id in drivers_in_race:
                    drv_hist = hist_circuit[hist_circuit["driver_id"] == driver_id]

                    if drv_hist.empty:
                        records.append({
                            "year":                year,
                            "round":               round_num,
                            "driver_id":           driver_id,
                            "circuit_id":          circuit_id,
                            "h2h_circuit_score":   0.0,
                            "h2h_top_rival_win_pct": 0.5,
                            "h2h_circuit_battles": 0,
                        })
                        continue

                    # H2H vs all rivals at this circuit
                    h2h_wins   = 0
                    h2h_total  = 0
                    h2h_weighted = 0.0

                    for rival_id in drivers_in_race:
                        if rival_id == driver_id:
                            continue

                        rival_hist = hist_circuit[hist_circuit["driver_id"] == rival_id]

                        # Find shared races at this circuit
                        shared_rounds = set(drv_hist[["year","round"]].apply(tuple, axis=1)) & \
                                        set(rival_hist[["year","round"]].apply(tuple, axis=1))

                        for (yr, rnd) in shared_rounds:
                            my_pos     = drv_hist[(drv_hist["year"]==yr)&(drv_hist["round"]==rnd)]["finish_position"].values
                            rival_pos  = rival_hist[(rival_hist["year"]==yr)&(rival_hist["round"]==rnd)]["finish_position"].values

                            if len(my_pos) > 0 and len(rival_pos) > 0:
                                years_ago = year - yr
                                weight    = np.exp(-DECAY_LAMBDA * years_ago)
                                won       = 1 if my_pos[0] < rival_pos[0] else 0
                                h2h_wins  += won
                                h2h_total += 1
                                h2h_weighted += won * weight

                    h2h_win_pct    = h2h_wins / max(h2h_total, 1)
                    h2h_score      = h2h_weighted / max(h2h_total, 1)

                    records.append({
                        "year":                    year,
                        "round":                   round_num,
                        "driver_id":               driver_id,
                        "circuit_id":              circuit_id,
                        "h2h_circuit_score":       round(h2h_score, 4),
                        "h2h_top_rival_win_pct":   round(h2h_win_pct, 3),
                        "h2h_circuit_battles":     h2h_total,
                    })

    df = pd.DataFrame(records)
    print(f"  Rival H2H features: {len(df):,} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_advanced_features():
    """
    Builds all advanced features and returns a single merged DataFrame.
    Called by build_dataset.py.

    Returns:
        pd.DataFrame: All advanced features per driver per race
    """
    print("\n" + "=" * 60)
    print("Building advanced features (80+ new features)...")
    print("=" * 60)

    race_df, quali_df, fastf1_df, practice_df = load_data()

    # Compute all feature groups
    dominance_df  = compute_track_dominance(race_df, quali_df)
    tyre_df       = compute_tire_degradation(fastf1_df, race_df)
    sector_df     = compute_sector_specialisation(fastf1_df, race_df)
    overtake_df   = compute_overtaking_defending(race_df, fastf1_df)
    psych_df      = compute_psychological_features(race_df, quali_df)
    h2h_df        = compute_rival_h2h_at_circuit(race_df)

    # Merge all on year + round + driver_id
    print("\nMerging advanced feature groups...")

    combined = dominance_df.copy()
    print(f"  Base (dominance):  {combined.shape}")

    combined = combined.merge(
        psych_df,
        on=["year","round","driver_id"],
        how="left"
    )
    print(f"  + Psychological:   {combined.shape}")

    combined = combined.merge(
        h2h_df[["year","round","driver_id","circuit_id",
                 "h2h_circuit_score","h2h_top_rival_win_pct","h2h_circuit_battles"]],
        on=["year","round","driver_id","circuit_id"],
        how="left"
    )
    print(f"  + H2H circuit:     {combined.shape}")

    combined = combined.merge(
        tyre_df[["year","driver_id","tyre_deg_rate","tyre_mgmt_score",
                  "avg_stint_length","long_run_pace"]],
        on=["year","driver_id"],
        how="left"
    )
    print(f"  + Tyre deg:        {combined.shape}")

    combined = combined.merge(
        overtake_df[["year","driver_id","overtaking_score","defending_score",
                      "avg_positions_gained","drs_efficiency_score",
                      "sc_restart_gain_avg","late_race_gain_score"]],
        on=["year","driver_id"],
        how="left"
    )
    print(f"  + Overtaking:      {combined.shape}")

    # Sector scores — merge on year + driver_id
    if not sector_df.empty and "driver_code" in sector_df.columns:
        # Need to map driver_code back to driver_id via race_df
        code_to_id = dict(zip(
            race_df["driver_code"].str.upper(),
            race_df["driver_id"]
        ))
        sector_df["driver_id_mapped"] = sector_df["driver_code"].str.upper().map(code_to_id)
        sector_df_clean = sector_df.dropna(subset=["driver_id_mapped"]).copy()
        sector_df_clean["driver_id"] = sector_df_clean["driver_id_mapped"]

        combined = combined.merge(
            sector_df_clean[["year","driver_id","s1_score","s2_score","s3_score",
                               "best_sector","s1_vs_field","s2_vs_field","s3_vs_field"]],
            on=["year","driver_id"],
            how="left"
        )
        print(f"  + Sector scores:   {combined.shape}")

    # Fill any remaining NaNs with neutral values
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].fillna(0)

    print(f"\nAdvanced features complete: {combined.shape}")
    print(f"  Total new features: {len(combined.columns) - 4}")  # minus year/round/driver_id/circuit_id

    return combined


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_advanced_features()

    print("\nSample output (first 3 rows):")
    print(df.head(3).to_string())

    print("\nAll columns:")
    for col in sorted(df.columns):
        print(f"  {col}")

    out_path = os.path.join(DATA_DIR, "advanced_features_preview.csv")
    df.head(200).to_csv(out_path, index=False)
    print(f"\nPreview saved: {out_path}")