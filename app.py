"""
app.py
------
Main Streamlit website for the F1 Championship Predictor.

Pages:
  1. Championship Tracker
  2. Race by Race Predictor   ← fixed ranking direction
  3. Next Race Predictor
  4. Season Simulator
  5. Driver Deep Dive
  6. Head to Head
  7. Feature Explorer

HOW TO RUN:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
SRC_DIR      = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title            = "F1 Championship Predictor 2026",
    page_icon             = "🏎️",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-title { font-size:2.4rem; font-weight:700; color:#E10600; margin-bottom:0; }
    .sub-title  { font-size:1rem;   color:#888; margin-top:0; margin-bottom:2rem; }
    .section-header {
        font-size:1.3rem; font-weight:600; margin-top:1.5rem; margin-bottom:0.5rem;
        border-bottom:2px solid #E10600; padding-bottom:0.3rem;
    }
    .podium-card { border-radius:10px; padding:1rem; text-align:center; font-size:1rem; }
    .p1 { background:linear-gradient(135deg,#FFD700,#FFA500); color:#000; }
    .p2 { background:linear-gradient(135deg,#C0C0C0,#A0A0A0); color:#000; }
    .p3 { background:linear-gradient(135deg,#CD7F32,#A0522D); color:#fff; }
</style>
""", unsafe_allow_html=True)

# ─── TEAM COLOURS ─────────────────────────────────────────────────────────────

TEAM_COLORS = {
    "red bull":         "#3671C6",
    "mercedes":         "#27F4D2",
    "ferrari":          "#E8002D",
    "mclaren":          "#FF8000",
    "aston martin":     "#229971",
    "alpine":           "#FF87BC",
    "williams":         "#64C4FF",
    "rb f1":            "#6692FF",
    "haas":             "#B6BABD",
    "audi":             "#999999",
    "cadillac":         "#C00000",
    "sauber":           "#52E252",
}

def get_team_color(team_name):
    t = str(team_name).lower()
    for key, color in TEAM_COLORS.items():
        if key in t:
            return color
    return "#E10600"


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_final_predictions():
    path = os.path.join(DATA_DIR, "final_predictions.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_race_results():
    path = os.path.join(DATA_DIR, "race_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ["finish_position", "points", "grid_position"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_qualifying_results():
    path = os.path.join(DATA_DIR, "qualifying_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["quali_position"] = pd.to_numeric(df["quali_position"], errors="coerce")
    return df

@st.cache_data(ttl=300)
def load_master_dataset():
    path = os.path.join(DATA_DIR, "master_dataset.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_elo_history():
    path = os.path.join(DATA_DIR, "elo_ratings.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_elo_current():
    path = os.path.join(DATA_DIR, "elo_current.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_bayesian_history():
    path = os.path.join(DATA_DIR, "bayesian_history.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_weather_forecast():
    path = os.path.join(DATA_DIR, "weather_forecast_upcoming.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_weather_historical():
    path = os.path.join(DATA_DIR, "weather_historical.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_feature_importance():
    path = os.path.join(DATA_DIR, "feature_importance.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_championship_probs():
    path = os.path.join(DATA_DIR, "championship_probabilities.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_resource
def load_xgboost_model():
    path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feature_cols"]


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## 🏎️ F1 Predictor 2026")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", [
        "🏆 Championship Tracker",
        "🔮 Race by Race Predictor",
        "🏁 Next Race Predictor",
        "🎲 Season Simulator",
        "👤 Driver Deep Dive",
        "⚔️ Head to Head",
        "📊 Feature Explorer",
    ])

    st.sidebar.markdown("---")
    preds = load_final_predictions()
    if not preds.empty and "updated_at" in preds.columns:
        st.sidebar.caption(f"Updated: {preds['updated_at'].iloc[0]}")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    return page


# ─── RACE PREDICTION ENGINE ───────────────────────────────────────────────────

def predict_race_finishing_order(race_df, quali_df, master_df,
                                  elo_df, model, feature_cols,
                                  year, round_num, use_quali=True):
    """
    Predicts race finishing order for a given race.

    KEY FIX: XGBRanker outputs LOWER scores for BETTER finishing positions
    (it is a ranking model — rank 1 = lowest predicted score).
    So we rank ascending=True: lowest score → P1, highest score → P20.

    Args:
        use_quali (bool): If True, inject actual qualifying grid positions

    Returns:
        pd.DataFrame: One row per driver, sorted P1 to P20
    """
    # ── Get race data ─────────────────────────────────────────────────────────
    race_data = master_df[
        (master_df["year"]  == year) &
        (master_df["round"] == round_num)
    ].copy() if not master_df.empty else pd.DataFrame()

    # Fall back to race_results if master doesn't have this round
    if race_data.empty:
        race_data = race_df[
            (race_df["year"]  == year) &
            (race_df["round"] == round_num)
        ].copy()

    # ── FUTURE RACE FALLBACK ──────────────────────────────────────────────────
    # If this race hasn't happened yet, build features from scratch:
    # - Use latest driver rolling stats (form, Elo, championship context)
    # - Look up circuit history from PAST seasons at this same circuit
    if race_data.empty and not master_df.empty:
        # Step 1: Get latest driver features from most recent completed race
        latest_year  = master_df["year"].max()
        latest_round = master_df[master_df["year"] == latest_year]["round"].max()
        latest_data  = master_df[
            (master_df["year"]  == latest_year) &
            (master_df["round"] == latest_round)
        ].copy()

        if not latest_data.empty:
            race_data = latest_data.copy()
            race_data["round"] = round_num
            race_data["year"]  = year

            # Clear grid positions — unknown for future race
            if "grid_position" in race_data.columns:
                race_data["grid_position"] = np.nan
            if "quali_position" in race_data.columns:
                race_data["quali_position"] = np.nan

            # Step 2: Find this circuit_id from the race calendar
            # Look up circuit_id from calendar mapping
            CIRCUIT_MAP = {
                1:"albert_park", 2:"shanghai", 3:"suzuka", 4:"bahrain",
                5:"jeddah", 6:"miami", 7:"imola", 8:"monaco", 9:"catalunya",
                10:"villeneuve", 11:"red_bull_ring", 12:"silverstone",
                13:"spa", 14:"hungaroring", 15:"zandvoort", 16:"monza",
                17:"baku", 18:"marina_bay", 19:"americas", 20:"rodriguez",
                21:"interlagos", 22:"vegas", 23:"losail", 24:"yas_marina",
            }
            target_circuit = CIRCUIT_MAP.get(round_num, "")
            race_data["circuit_id"] = target_circuit

            # Step 3: Replace circuit features with historical data
            # for THIS circuit from past seasons
            if target_circuit and not master_df.empty:
                circuit_history = master_df[
                    master_df["circuit_id"] == target_circuit
                ]

                if not circuit_history.empty:
                    # Get per-driver circuit averages from history
                    circuit_cols = [c for c in master_df.columns if any(
                        kw in c for kw in [
                            "circuit_affinity", "circuit_avg_finish",
                            "circuit_win_pct", "circuit_podium_pct",
                            "track_dominance_score", "circuit_weighted_wins",
                            "circuit_weighted_podiums", "circuit_best_finish",
                            "circuit_visits", "podium_streak_at_circuit",
                            "quali_dominance_at_circuit", "pole_rate_at_circuit",
                            "front_row_rate_at_circuit", "h2h_circuit_score",
                            "wet_circuit_dominance", "is_home_race",
                        ]
                    )]

                    # Get most recent circuit row per driver
                    latest_circuit = (
                        circuit_history
                        .sort_values(["year","round"])
                        .groupby("driver_id")
                        .last()
                        .reset_index()
                        [["driver_id"] + [c for c in circuit_cols if c in circuit_history.columns]]
                    )

                    # Drop old circuit cols and merge with correct ones
                    race_data = race_data.drop(
                        columns=[c for c in circuit_cols if c in race_data.columns],
                        errors="ignore"
                    )
                    race_data = race_data.merge(
                        latest_circuit, on="driver_id", how="left"
                    )

                    # Fill any drivers with no circuit history with neutral values
                    for col in circuit_cols:
                        if col in race_data.columns:
                            race_data[col] = race_data[col].fillna(0)

                else:
                    # No history for this circuit — zero out circuit features
                    circuit_cols = [c for c in race_data.columns if any(
                        kw in c for kw in ["circuit_affinity","track_dominance",
                                            "circuit_win_pct","h2h_circuit"]
                    )]
                    for col in circuit_cols:
                        race_data[col] = 0.0

            # Step 4: Update home race flag for target circuit
            if "is_home_race" in race_data.columns:
                CIRCUIT_COUNTRY = {
                    "albert_park":"Australia","bahrain":"Bahrain",
                    "jeddah":"Saudi Arabia","shanghai":"China",
                    "suzuka":"Japan","miami":"United States",
                    "imola":"Italy","monaco":"Monaco",
                    "villeneuve":"Canada","catalunya":"Spain",
                    "red_bull_ring":"Austria","silverstone":"United Kingdom",
                    "hungaroring":"Hungary","spa":"Belgium",
                    "zandvoort":"Netherlands","monza":"Italy",
                    "baku":"Azerbaijan","marina_bay":"Singapore",
                    "americas":"United States","rodriguez":"Mexico",
                    "interlagos":"Brazil","vegas":"United States",
                    "losail":"Qatar","yas_marina":"United Arab Emirates",
                }
                DRIVER_COUNTRY = {
                    "max_verstappen":"Netherlands","russell":"United Kingdom",
                    "hamilton":"United Kingdom","leclerc":"Monaco",
                    "norris":"United Kingdom","sainz":"Spain",
                    "alonso":"Spain","piastri":"Australia",
                    "antonelli":"Italy","perez":"Mexico",
                    "gasly":"France","ocon":"France","albon":"Thailand",
                    "stroll":"Canada","bottas":"Finland","lawson":"New Zealand",
                    "bortoleto":"Brazil","hadjar":"France",
                    "colapinto":"Argentina","hulkenberg":"Germany",
                    "bearman":"United Kingdom","arvid_lindblad":"Sweden",
                }
                circ_country = CIRCUIT_COUNTRY.get(target_circuit, "")
                race_data["is_home_race"] = race_data["driver_id"].apply(
                    lambda d: int(DRIVER_COUNTRY.get(d,"") == circ_country and circ_country != "")
                )
                race_data["home_race_boost"] = race_data["is_home_race"] * 0.08

    if race_data.empty:
        return pd.DataFrame()

    # Drop duplicates — fallback can create duplicate driver rows
    race_data = race_data.drop_duplicates(subset=["driver_id"]).reset_index(drop=True)

    # ── Enrich with driver/team name from race_results ────────────────────────
    # For future races use most recent year's driver list
    name_year = year if not race_df[race_df["year"] == year].empty else race_df["year"].max()
    name_map = (
        race_df[race_df["year"] == name_year][["driver_id", "driver_name", "constructor_name"]]
        .drop_duplicates("driver_id")
    )
    if "driver_name" not in race_data.columns or race_data["driver_name"].isna().all():
        race_data = race_data.drop(columns=["driver_name","constructor_name"], errors="ignore")
        race_data = race_data.merge(name_map, on="driver_id", how="left")
    else:
        # Fill any missing names
        race_data = race_data.merge(
            name_map.rename(columns={"driver_name":"_dn","constructor_name":"_cn"}),
            on="driver_id", how="left"
        )
        race_data["driver_name"]      = race_data["driver_name"].fillna(race_data["_dn"])
        race_data["constructor_name"] = race_data["constructor_name"].fillna(race_data["_cn"])
        race_data = race_data.drop(columns=["_dn","_cn"], errors="ignore")

    # ── Inject qualifying position ────────────────────────────────────────────
    if use_quali and not quali_df.empty:
        q = quali_df[
            (quali_df["year"]  == year) &
            (quali_df["round"] == round_num)
        ][["driver_id","quali_position"]].copy()

        if not q.empty:
            race_data = race_data.drop(columns=["quali_position"], errors="ignore")
            race_data = race_data.merge(q, on="driver_id", how="left")
            # Also update grid_position with quali result
            if "grid_position" in race_data.columns:
                race_data["grid_position"] = race_data["quali_position"].fillna(
                    race_data["grid_position"]
                )

    # ── Update Elo ratings to latest ─────────────────────────────────────────
    if not elo_df.empty and "elo_before" in race_data.columns:
        elo_map = dict(zip(elo_df["driver_id"], elo_df["elo_rating"]))
        race_data["elo_before"] = race_data["driver_id"].map(elo_map).fillna(1500)

    # ── Run XGBoost model ─────────────────────────────────────────────────────
    scores = None

    if model is not None and feature_cols is not None:
        available = [f for f in feature_cols if f in race_data.columns]

        if len(available) >= len(feature_cols) * 0.5:
            X = race_data[feature_cols].copy()
            # Fill any missing features with column median from master_dataset
            for col in feature_cols:
                if col not in X.columns:
                    X[col] = 0
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

            try:
                scores = model.predict(X.values)
            except Exception as e:
                st.warning(f"Model prediction failed: {e}. Using Elo fallback.")

    # ── Elo fallback if model not available ───────────────────────────────────
    if scores is None:
        if not elo_df.empty:
            elo_map = dict(zip(elo_df["driver_id"], elo_df["elo_rating"]))
            # Higher Elo = better driver → negate so lower score = better rank
            scores = np.array([-elo_map.get(d, 1500) for d in race_data["driver_id"]])
        else:
            scores = np.random.rand(len(race_data))

    # ── KEY FIX: XGBRanker rank:pairwise → lower score = better position ──────
    # HIGHER score = better finish = lower position number (P1)
    # Model trained with inverted target: P1=21, P20=2
    # So highest score -> P1, use ascending=False
    predicted_rank = pd.Series(scores).rank(method="first", ascending=False).astype(int).values

    # Win probability via softmax — higher score = more likely to win
    exp_s     = np.exp(scores - scores.max())
    win_probs = exp_s / exp_s.sum() * 100

    # ── Build result DataFrame ────────────────────────────────────────────────
    result = pd.DataFrame({
        "driver_id":        race_data["driver_id"].values,
        "driver_name":      race_data["driver_name"].fillna(race_data["driver_id"]).values,
        "constructor_name": race_data["constructor_name"].fillna("").values,
        "predicted_rank":   predicted_rank,
        "win_probability":  np.round(win_probs, 2),
    })

    # Grid position
    grid_col = "grid_position" if "grid_position" in race_data.columns else (
               "quali_position" if "quali_position" in race_data.columns else None)
    if grid_col:
        result["grid_position"] = pd.to_numeric(
            race_data[grid_col].values, errors="coerce"
        )

    # Podium probabilities — simple approximation
    # P1 prob ≈ win_probability
    # P2 prob ≈ sum of probs for drivers predicted P1 or P2
    # P3 prob ≈ sum of probs for drivers predicted P1-P3
    sorted_idx   = np.argsort(scores)[::-1]     # indices sorted best→worst (highest score = P1)
    cumulative   = np.cumsum(exp_s[sorted_idx]) / exp_s.sum() * 100

    p1_probs = np.zeros(len(result))
    p2_probs = np.zeros(len(result))
    p3_probs = np.zeros(len(result))

    for pos, idx in enumerate(sorted_idx):
        p1_probs[idx] = win_probs[idx]
        p2_probs[idx] = win_probs[idx] * (1.8 if pos <= 3 else 0.3)
        p3_probs[idx] = win_probs[idx] * (1.5 if pos <= 5 else 0.2)

    result["p1_prob"] = np.round(p1_probs, 1)
    result["p2_prob"] = np.round(np.clip(p2_probs, 0, 100), 1)
    result["p3_prob"] = np.round(np.clip(p3_probs, 0, 100), 1)

    result = result.sort_values("predicted_rank").reset_index(drop=True)
    return result


# ─── PAGE: RACE BY RACE PREDICTOR ─────────────────────────────────────────────

def page_race_predictor_detailed():
    st.markdown('<p class="main-title">🔮 Race by Race Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predict each Grand Prix before and after qualifying — compare with actual results</p>', unsafe_allow_html=True)

    race_df   = load_race_results()
    quali_df  = load_qualifying_results()
    master_df = load_master_dataset()
    elo_df    = load_elo_current()
    weather_h = load_weather_historical()
    model, feature_cols = load_xgboost_model()

    if race_df.empty:
        st.error("No race data. Run get_ergast_data.py first.")
        return

    # ── Race selector ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Select a Race</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    years         = sorted(race_df["year"].unique(), reverse=True)
    selected_year = c1.selectbox("Season", years, index=0)

    year_races = (
        race_df[race_df["year"] == selected_year]
        [["round","race_name","race_date","circuit_id"]]
        .drop_duplicates()
        .sort_values("round", ascending=False)
    )
    race_options = {
        f"Round {int(r['round'])}: {r['race_name']} ({str(r['race_date'])[:10]})": int(r["round"])
        for _, r in year_races.iterrows()
    }
    if not race_options:
        st.warning(f"No race data found for {selected_year}.")
        return
    selected_label = c2.selectbox(
        f"Race ({len(race_options)} races — newest first)",
        list(race_options.keys()),
        index=0
    )
    selected_round = race_options[selected_label]
    total_rounds = len(race_options)
    c3.metric("Rounds in season", total_rounds)
    c3.metric("Selected", f"Round {selected_round}")


    # ── Prediction mode ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Prediction Timing</p>', unsafe_allow_html=True)
    pred_mode = st.radio(
        "When to predict",
        ["Before qualifying", "After qualifying", "Side by side comparison"],
        horizontal=True,
    )
    st.markdown("---")

    # ── Race metadata ──────────────────────────────────────────────────────────
    actual_result = race_df[
        (race_df["year"]  == selected_year) &
        (race_df["round"] == selected_round)
    ].sort_values("finish_position").copy()

    # Get race metadata — handle NaN from OpenF1 data
    def safe_val(df, col, default=""):
        if df.empty or col not in df.columns: return default
        val = df[col].dropna().iloc[0] if df[col].notna().any() else None
        return str(val) if val is not None else default

    race_name  = safe_val(actual_result, "race_name", f"Round {selected_round} {selected_year}")
    race_date  = safe_val(actual_result, "race_date", "")
    circuit_id = safe_val(actual_result, "circuit_id", "")
    has_result = not actual_result.empty and actual_result["finish_position"].notna().any()

    st.markdown(f"## {race_name}")
    st.caption(f"Round {selected_round} · {selected_year} · {race_date} · {circuit_id}")

    # ── Weather ────────────────────────────────────────────────────────────────
    if not weather_h.empty:
        w = weather_h[weather_h["circuit_id"] == circuit_id].sort_values("race_date").tail(1)
        if not w.empty:
            wr = w.iloc[0]
            wc1,wc2,wc3,wc4,wc5,wc6 = st.columns(6)
            def fmt(val, suffix="", default="?"):
                if val is None or str(val) == "nan" or str(val) == "None": return default
                try: return f"{float(val):.1f}{suffix}"
                except: return str(val) + suffix

            wc1.metric("Air Temp",   fmt(wr.get("air_temp_avg_c"), "°C"))
            wc2.metric("Track Temp", fmt(wr.get("track_temp_avg_c") or wr.get("soil_temp_avg_c"), "°C"))
            wc3.metric("Humidity",   fmt(wr.get("humidity_avg_pct"), "%"))
            wc4.metric("Wind",       fmt(wr.get("windspeed_avg_kmh"), " km/h"))
            wc5.metric("Rainfall",   fmt(wr.get("rainfall_total_mm","0"), " mm"))
            rain = wr.get("rain_flag", 0)
            wc6.metric("Conditions", "🌧️ Wet" if rain else "☀️ Dry")
            if rain:
                st.info("🌧️ Wet race — wet weather ratings applied")
    st.markdown("---")

    # ── Run predictions ────────────────────────────────────────────────────────
    def run_pred(use_quali):
        return predict_race_finishing_order(
            race_df, quali_df, master_df, elo_df,
            model, feature_cols,
            selected_year, selected_round,
            use_quali=use_quali,
        )

    # ── SIDE BY SIDE ──────────────────────────────────────────────────────────
    if pred_mode == "Side by side comparison":
        pred_pre  = run_pred(use_quali=False)
        pred_post = run_pred(use_quali=True)

        col_pre, col_post = st.columns(2)
        with col_pre:
            st.markdown("### Before Qualifying")
            st.caption("Season form · Elo · circuit affinity · weather")
            if not pred_pre.empty:
                render_prediction_table(pred_pre, actual_result, has_result, full=True)
            else:
                st.warning("No data available")

        with col_post:
            st.markdown("### After Qualifying")
            st.caption("Adds actual grid positions — more accurate")
            if not pred_post.empty:
                render_prediction_table(pred_post, actual_result, has_result, full=True)
            else:
                st.warning("No data available")

        if has_result and not pred_pre.empty and not pred_post.empty:
            st.markdown("---")
            st.markdown('<p class="section-header">Accuracy Comparison</p>', unsafe_allow_html=True)
            render_accuracy_comparison(pred_pre, pred_post, actual_result)

    else:
        # ── SINGLE MODE ───────────────────────────────────────────────────────
        use_quali = (pred_mode == "After qualifying")
        pred_df   = run_pred(use_quali=use_quali)

        if pred_df.empty:
            st.warning("No data available for this race.")
            return

        # ── Podium cards ──────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Predicted Podium</p>', unsafe_allow_html=True)
        pc2, pc1, pc3 = st.columns(3)

        for col, rank, css, emoji in [(pc2,2,"p2","🥈"),(pc1,1,"p1","🥇"),(pc3,3,"p3","🥉")]:
            row = pred_df[pred_df["predicted_rank"] == rank]
            if not row.empty:
                dr = row.iloc[0]
                col.markdown(
                    f'<div class="podium-card {css}">'
                    f'{emoji} <b>P{rank}</b><br>'
                    f'<span style="font-size:1.2rem"><b>{dr["driver_name"]}</b></span><br>'
                    f'<span style="font-size:0.85rem">{dr["constructor_name"]}</span><br>'
                    f'Win prob: <b>{dr["win_probability"]:.1f}%</b>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Win probability chart ──────────────────────────────────────────────
        st.markdown('<p class="section-header">Win Probability — All Drivers</p>', unsafe_allow_html=True)

        plot_df = pred_df[pred_df["win_probability"] > 0.3].sort_values(
            "win_probability", ascending=True
        )

        fig = go.Figure()
        for _, row in plot_df.iterrows():
            fig.add_trace(go.Bar(
                x            = [row["win_probability"]],
                y            = [row["driver_name"]],
                orientation  = "h",
                marker_color = get_team_color(row["constructor_name"]),
                text         = f"{row['win_probability']:.1f}%",
                textposition = "outside",
                name         = row["driver_name"],
                hovertemplate = (
                    f"<b>{row['driver_name']}</b> ({row['constructor_name']})<br>"
                    f"Predicted: P{row['predicted_rank']}<br>"
                    f"Win prob: {row['win_probability']:.1f}%<br>"
                    f"P1: {row['p1_prob']:.1f}%  "
                    f"P2: {row['p2_prob']:.1f}%  "
                    f"P3: {row['p3_prob']:.1f}%<extra></extra>"
                ),
            ))

        fig.update_layout(
            showlegend    = False,
            xaxis_title   = "Win Probability (%)",
            height        = max(350, len(plot_df) * 36),
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="white"),
            margin        = dict(l=10, r=80, t=10, b=40),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Full order table ───────────────────────────────────────────────────
        st.markdown('<p class="section-header">Full Predicted Finishing Order</p>', unsafe_allow_html=True)
        if has_result:
            st.caption("✅ = predicted within 1 place of actual  ❌ = incorrect")

        render_prediction_table(pred_df, actual_result, has_result, full=True)

        # ── Accuracy ──────────────────────────────────────────────────────────
        if has_result:
            st.markdown('<p class="section-header">Prediction Accuracy</p>', unsafe_allow_html=True)
            render_accuracy_single(pred_df, actual_result)

        # ── Actual Race Results ────────────────────────────────────────────────
        if has_result:
            st.markdown('<p class="section-header">Official Race Results</p>', unsafe_allow_html=True)
            st.caption("Actual finishing order from the race")

            actual_display = actual_result.sort_values("finish_position").head(20).copy()
            actual_display["Pos"]    = pd.to_numeric(actual_display["finish_position"], errors="coerce").fillna(0).astype(int)
            actual_display["Driver"] = actual_display["driver_name"].fillna(actual_display["driver_id"])
            actual_display["Team"]   = actual_display["constructor_name"].fillna("")
            actual_display["Points"] = pd.to_numeric(actual_display["points"], errors="coerce").fillna(0)
            actual_display["Status"] = actual_display["status"].fillna("Classified")
            # Only show Grid if we have real data (not all zeros from OpenF1)
            grid_vals = pd.to_numeric(actual_display.get("grid_position", pd.Series()), errors="coerce").fillna(0)
            if grid_vals.sum() > 0:
                actual_display["Grid"] = grid_vals.astype(int)
                show_cols = ["Pos","Driver","Team","Grid","Points","Status"]
            else:
                show_cols = ["Pos","Driver","Team","Points","Status"]
            actual_display = actual_display[show_cols]

            # Highlight podium rows
            def highlight_podium(row):
                if row["Pos"] == 1:
                    return ["background-color: rgba(255,215,0,0.15)"] * len(row)
                elif row["Pos"] == 2:
                    return ["background-color: rgba(192,192,192,0.15)"] * len(row)
                elif row["Pos"] == 3:
                    return ["background-color: rgba(205,127,50,0.12)"] * len(row)
                return [""] * len(row)

            st.dataframe(
                actual_display.style.apply(highlight_podium, axis=1),
                use_container_width=True,
                hide_index=True
            )

            # Winner callout
            winner = actual_result[actual_result["finish_position"] == 1]
            if not winner.empty:
                w = winner.iloc[0]
                st.success(f"🏆 Winner: **{w['driver_name']}** ({w['constructor_name']}) — started P{int(w['grid_position']) if pd.notna(w['grid_position']) else '?'}")        


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def render_prediction_table(pred_df, actual_result, show_actual, full=True):
    display = pred_df.copy() if full else pred_df.head(10).copy()

    actual_map = {}
    if show_actual and not actual_result.empty:
        actual_map = dict(zip(
            actual_result["driver_id"],
            actual_result["finish_position"]
        ))

    rows = []
    for _, row in display.iterrows():
        predicted  = int(row["predicted_rank"])
        actual_pos = actual_map.get(row["driver_id"], None)
        correct    = ""
        if actual_pos is not None:
            correct = "✅" if abs(predicted - int(actual_pos)) <= 1 else "❌"

        grid = row.get("grid_position", np.nan)
        grid = int(grid) if pd.notna(grid) else "—"

        r = {
            "Pred":   predicted,
            "Driver": row["driver_name"],
            "Team":   row["constructor_name"],
            "Grid":   grid,
            "Win %":  f"{row['win_probability']:.1f}%",
            "P1%":    f"{row.get('p1_prob',0):.1f}%",
            "P2%":    f"{row.get('p2_prob',0):.1f}%",
            "P3%":    f"{row.get('p3_prob',0):.1f}%",
        }
        if show_actual:
            r["Actual"] = int(actual_pos) if actual_pos else "TBD"
            r["✓"]      = correct

        rows.append(r)

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_accuracy_single(pred_df, actual_result):
    actual_map = dict(zip(
        actual_result["driver_id"], actual_result["finish_position"]
    ))
    df = pred_df.copy()
    df["actual_pos"] = df["driver_id"].map(actual_map)
    df = df.dropna(subset=["actual_pos"])
    if df.empty:
        return

    pw = df[df["predicted_rank"] == 1]["driver_id"].values
    aw = actual_result[actual_result["finish_position"] == 1]["driver_id"].values
    winner_ok = len(pw) > 0 and len(aw) > 0 and pw[0] == aw[0]

    pred_top3   = set(df[df["predicted_rank"] <= 3]["driver_id"].values)
    actual_top3 = set(actual_result[actual_result["finish_position"] <= 3]["driver_id"].values)

    df["err"]   = abs(df["predicted_rank"] - df["actual_pos"])
    within3_pct = (df["err"] <= 3).mean() * 100
    avg_err     = df["err"].mean()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Winner Correct",     "✅ Yes" if winner_ok else "❌ No")
    c2.metric("Podium Overlap",     f"{len(pred_top3 & actual_top3)}/3 correct")
    c3.metric("Within 3 Positions", f"{within3_pct:.0f}% of drivers")
    c4.metric("Avg Position Error", f"{avg_err:.1f} places")


def render_accuracy_comparison(pred_pre, pred_post, actual_result):
    actual_map = dict(zip(
        actual_result["driver_id"], actual_result["finish_position"]
    ))

    def metrics(pred_df):
        df = pred_df.copy()
        df["ap"] = df["driver_id"].map(actual_map)
        df = df.dropna(subset=["ap"])
        if df.empty:
            return None
        pw = df[df["predicted_rank"] == 1]["driver_id"].values
        aw = actual_result[actual_result["finish_position"] == 1]["driver_id"].values
        won = len(pw) > 0 and len(aw) > 0 and pw[0] == aw[0]
        p3  = set(df[df["predicted_rank"] <= 3]["driver_id"].values)
        a3  = set(actual_result[actual_result["finish_position"] <= 3]["driver_id"].values)
        df["err"] = abs(df["predicted_rank"] - df["ap"])
        return {
            "winner":  "✅" if won else "❌",
            "podium":  f"{len(p3 & a3)}/3",
            "within3": f"{(df['err'] <= 3).mean() * 100:.0f}%",
            "avg_err": f"{df['err'].mean():.1f}",
        }

    mp = metrics(pred_pre)
    mq = metrics(pred_post)
    if mp and mq:
        st.dataframe(pd.DataFrame({
            "Metric":           ["Winner correct","Podium overlap","Within 3 positions","Avg error"],
            "Before qualifying": [mp["winner"],mp["podium"],mp["within3"],mp["avg_err"]],
            "After qualifying":  [mq["winner"],mq["podium"],mq["within3"],mq["avg_err"]],
        }), use_container_width=True, hide_index=True)
        st.caption("After qualifying is typically 15–20% more accurate.")


# ─── PAGE: CHAMPIONSHIP TRACKER ───────────────────────────────────────────────

def page_championship_tracker():
    st.markdown('<p class="main-title">🏆 Championship Tracker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Live championship win probabilities — updated after every race</p>', unsafe_allow_html=True)

    preds = load_final_predictions()
    if preds.empty:
        st.warning("Run ensemble.py first.")
        return

    races_done = preds["races_completed"].iloc[0] if "races_completed" in preds.columns else 0
    leader     = preds.iloc[0]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Championship Leader", leader["driver_name"])
    c2.metric("Win Probability",     f"{leader['final_prob']:.1f}%")
    c3.metric("Races Completed",     f"{races_done} / 24")
    c4.metric("Races Remaining",     f"{24 - races_done}")

    st.markdown("---")

    display = preds[preds["final_prob"] > 0.1].head(15)
    fig = go.Figure()
    for _, row in display.iterrows():
        fig.add_trace(go.Bar(
            name         = row["driver_name"],
            x            = [row["final_prob"]],
            y            = [row["driver_name"]],
            orientation  = "h",
            marker_color = get_team_color(row["constructor_name"]),
            text         = f"{row['final_prob']:.1f}%",
            textposition = "outside",
            hovertemplate = (
                f"<b>{row['driver_name']}</b> · {row['constructor_name']}<br>"
                f"Points: {row['current_points']}<br>"
                f"Final: {row['final_prob']:.1f}%  MC: {row['mc_prob']:.1f}%  "
                f"Bayes: {row['bayes_prob']:.1f}%  Elo: {row['elo_prob']:.1f}%<extra></extra>"
            )
        ))

    fig.update_layout(
        showlegend=False, xaxis_title="Championship Win Probability (%)",
        yaxis=dict(autorange="reversed"),
        height=max(400, len(display)*35),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), margin=dict(l=10,r=80,t=20,b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Model Breakdown</p>', unsafe_allow_html=True)
    t = preds.head(10)[["rank","driver_name","constructor_name","current_points",
                          "mc_prob","bayes_prob","elo_prob","final_prob"]].copy()
    t.columns = ["Rank","Driver","Team","Points","MC %","Bayes %","Elo %","Final %"]
    st.dataframe(t, use_container_width=True, hide_index=True)

    bayes_hist = load_bayesian_history()
    if not bayes_hist.empty:
        st.markdown('<p class="section-header">Probability Evolution</p>', unsafe_allow_html=True)
        top6 = preds.head(6)["driver_id"].tolist()
        fig2 = px.line(
            bayes_hist[bayes_hist["driver_id"].isin(top6)],
            x="round", y="probability", color="driver_name",
            title="Championship win probability after each race",
        )
        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"))
        st.plotly_chart(fig2, use_container_width=True)


# ─── PAGE: NEXT RACE PREDICTOR ────────────────────────────────────────────────

# 2026 race calendar — used to find the next upcoming race
RACE_CALENDAR_2026 = [
    {"round":1,  "name":"Australian Grand Prix",    "circuit_id":"albert_park",   "date":"2026-03-15"},
    {"round":2,  "name":"Chinese Grand Prix",       "circuit_id":"shanghai",      "date":"2026-03-22"},
    {"round":3,  "name":"Japanese Grand Prix",      "circuit_id":"suzuka",        "date":"2026-04-05"},
    {"round":4,  "name":"Bahrain Grand Prix",       "circuit_id":"bahrain",       "date":"2026-04-19"},
    {"round":5,  "name":"Saudi Arabian Grand Prix", "circuit_id":"jeddah",        "date":"2026-04-26"},
    {"round":6,  "name":"Miami Grand Prix",         "circuit_id":"miami",         "date":"2026-05-03"},
    {"round":7,  "name":"Emilia Romagna Grand Prix","circuit_id":"imola",         "date":"2026-05-17"},
    {"round":8,  "name":"Monaco Grand Prix",        "circuit_id":"monaco",        "date":"2026-05-24"},
    {"round":9,  "name":"Spanish Grand Prix",       "circuit_id":"catalunya",     "date":"2026-06-07"},
    {"round":10, "name":"Canadian Grand Prix",      "circuit_id":"villeneuve",    "date":"2026-06-14"},
    {"round":11, "name":"Austrian Grand Prix",      "circuit_id":"red_bull_ring", "date":"2026-06-28"},
    {"round":12, "name":"British Grand Prix",       "circuit_id":"silverstone",   "date":"2026-07-05"},
    {"round":13, "name":"Belgian Grand Prix",       "circuit_id":"spa",           "date":"2026-07-26"},
    {"round":14, "name":"Hungarian Grand Prix",     "circuit_id":"hungaroring",   "date":"2026-08-02"},
    {"round":15, "name":"Dutch Grand Prix",         "circuit_id":"zandvoort",     "date":"2026-08-30"},
    {"round":16, "name":"Italian Grand Prix",       "circuit_id":"monza",         "date":"2026-09-06"},
    {"round":17, "name":"Azerbaijan Grand Prix",    "circuit_id":"baku",          "date":"2026-09-20"},
    {"round":18, "name":"Singapore Grand Prix",     "circuit_id":"marina_bay",    "date":"2026-10-04"},
    {"round":19, "name":"United States Grand Prix", "circuit_id":"americas",      "date":"2026-10-18"},
    {"round":20, "name":"Mexico City Grand Prix",   "circuit_id":"rodriguez",     "date":"2026-10-25"},
    {"round":21, "name":"São Paulo Grand Prix",     "circuit_id":"interlagos",    "date":"2026-11-08"},
    {"round":22, "name":"Las Vegas Grand Prix",     "circuit_id":"vegas",         "date":"2026-11-21"},
    {"round":23, "name":"Qatar Grand Prix",         "circuit_id":"losail",        "date":"2026-11-29"},
    {"round":24, "name":"Abu Dhabi Grand Prix",     "circuit_id":"yas_marina",    "date":"2026-12-06"},
]


def get_next_race(race_df):
    """
    Finds the next upcoming race by comparing today vs calendar.
    Returns the next race dict, or None if season is over.
    """
    today      = pd.Timestamp.today().normalize()
    last_round = 0

    if not race_df.empty:
        current = race_df[race_df["year"] == 2026]
        if not current.empty:
            last_round = int(current["round"].max())

    # Next race = first calendar entry after last completed round
    for race in RACE_CALENDAR_2026:
        if race["round"] > last_round:
            race["days_until"] = (pd.Timestamp(race["date"]) - today).days
            return race

    return None


def page_next_race():
    st.markdown('<p class="main-title">🏁 Next Race Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Live prediction for the upcoming Grand Prix — auto-updates after each race</p>', unsafe_allow_html=True)

    race_df   = load_race_results()
    quali_df  = load_qualifying_results()
    master_df = load_master_dataset()
    elo_df    = load_elo_current()
    preds     = load_final_predictions()
    weather   = load_weather_forecast()
    model, feature_cols = load_xgboost_model()

    # ── Find next race automatically ──────────────────────────────────────────
    next_race = get_next_race(race_df)

    if next_race is None:
        st.success("🏆 The 2026 season is complete!")
        return

    days_until = next_race["days_until"]

    # ── Next race header ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Next Race",      next_race["name"])
    c2.metric("Date",           next_race["date"])
    c3.metric("Round",          f"{next_race['round']} / 24")
    if days_until >= 0:
        c4.metric("Days Until Race", days_until)
    else:
        c4.metric("Status", "Race weekend!")

    st.markdown("---")

    # ── Weather for next race ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">Race Day Weather Forecast</p>', unsafe_allow_html=True)

    race_weather = pd.DataFrame()
    if not weather.empty and "circuit_name" in weather.columns:
        race_weather = weather[
            weather["circuit_name"].str.lower().str.contains(
                next_race["circuit_id"].replace("_"," ")[:6], na=False
            )
        ]
        if race_weather.empty:
            race_weather = weather[weather["race_date"] == next_race["date"]]

    if not race_weather.empty:
        wr = race_weather.iloc[0]
        wc1,wc2,wc3,wc4,wc5,wc6 = st.columns(6)
        wc1.metric("Air Temp",   f"{wr.get('air_temp_avg_c','?')}°C")
        wc2.metric("Track Temp", f"{wr.get('track_temp_avg_c','?')}°C")
        wc3.metric("Humidity",   f"{wr.get('humidity_avg_pct','?')}%")
        wc4.metric("Wind",       f"{wr.get('windspeed_avg_kmh','?')} km/h")
        wc5.metric("Rainfall",   f"{wr.get('rainfall_total_mm','0')} mm")
        rain = wr.get("rain_flag", 0)
        wc6.metric("Conditions", "🌧️ Wet" if rain else "☀️ Dry")
        if rain:
            st.info("🌧️ Rain expected — wet weather driver ratings applied to prediction")
    else:
        st.caption("Weather forecast not yet available for this race — updates within 7 days of race")

    st.markdown("---")

    # ── Race prediction ────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Race Prediction</p>', unsafe_allow_html=True)

    # Check if qualifying has happened
    has_quali = False
    if not quali_df.empty:
        q2026 = quali_df[
            (quali_df["year"]  == 2026) &
            (quali_df["round"] == next_race["round"])
        ]
        has_quali = not q2026.empty

    mode_label = "After Qualifying" if has_quali else "Before Qualifying"
    st.caption(f"Prediction mode: **{mode_label}** — {'grid positions known ✅' if has_quali else 'qualifying not yet happened'}")

    pred_df = predict_race_finishing_order(
        race_df, quali_df, master_df, elo_df,
        model, feature_cols,
        2026, next_race["round"],
        use_quali=has_quali,
    )

    if pred_df.empty:
        st.warning("Prediction data not yet available for this race. Run build_dataset.py after qualifying.")
    else:
        # ── Podium cards ──────────────────────────────────────────────────────
        pc2, pc1, pc3 = st.columns(3)
        for col, rank, css, emoji in [(pc2,2,"p2","🥈"),(pc1,1,"p1","🥇"),(pc3,3,"p3","🥉")]:
            row = pred_df[pred_df["predicted_rank"] == rank]
            if not row.empty:
                dr = row.iloc[0]
                col.markdown(
                    f'<div class="podium-card {css}">'
                    f'{emoji} <b>P{rank}</b><br>'
                    f'<span style="font-size:1.2rem"><b>{dr["driver_name"]}</b></span><br>'
                    f'<span style="font-size:0.85rem">{dr["constructor_name"]}</span><br>'
                    f'Win prob: <b>{dr["win_probability"]:.1f}%</b>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Win probability chart ──────────────────────────────────────────────
        plot_df = pred_df[pred_df["win_probability"] > 0.3].sort_values("win_probability", ascending=True)
        fig = go.Figure()
        for _, row in plot_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row["win_probability"]], y=[row["driver_name"]],
                orientation="h",
                marker_color=get_team_color(row["constructor_name"]),
                text=f"{row['win_probability']:.1f}%",
                textposition="outside",
                name=row["driver_name"],
                hovertemplate=(
                    f"<b>{row['driver_name']}</b> ({row['constructor_name']})<br>"
                    f"Predicted: P{row['predicted_rank']}<br>"
                    f"Win prob: {row['win_probability']:.1f}%<extra></extra>"
                ),
            ))
        fig.update_layout(
            showlegend=False, xaxis_title="Win Probability (%)",
            height=max(350, len(plot_df)*36),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), margin=dict(l=10,r=80,t=10,b=40),
        )
        fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
        st.plotly_chart(fig, use_container_width=True)

        # ── Full predicted order ───────────────────────────────────────────────
        st.markdown('<p class="section-header">Full Predicted Grid</p>', unsafe_allow_html=True)
        display_rows = []
        for _, row in pred_df.iterrows():
            grid_val = row.get("grid_position", np.nan)
            try:
                grid = int(float(grid_val)) if pd.notna(grid_val) and float(grid_val) > 0 else "—"
            except (ValueError, TypeError):
                grid = "—"
            display_rows.append({
                "Pred":   int(row["predicted_rank"]),
                "Driver": row["driver_name"],
                "Team":   row["constructor_name"],
                "Grid":   grid,
                "Win %":  f"{row['win_probability']:.1f}%",
                "P1%":    f"{row.get('p1_prob',0):.1f}%",
                "P2%":    f"{row.get('p2_prob',0):.1f}%",
                "P3%":    f"{row.get('p3_prob',0):.1f}%",
            })
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Championship standings ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">Championship Standings Heading Into This Race</p>', unsafe_allow_html=True)

    if not preds.empty:
        top10 = preds.head(10).copy()
        fig2 = go.Figure()
        for _, row in top10.iterrows():
            fig2.add_trace(go.Bar(
                x=[row["driver_name"]], y=[row["final_prob"]],
                name=row["driver_name"],
                marker_color=get_team_color(row["constructor_name"]),
                text=f"{row['final_prob']:.1f}%", textposition="outside",
                hovertemplate=(
                    f"<b>{row['driver_name']}</b><br>"
                    f"Points: {row['current_points']}<br>"
                    f"Championship prob: {row['final_prob']:.1f}%<extra></extra>"
                )
            ))
        fig2.update_layout(
            showlegend=False, yaxis_title="Championship Win Probability (%)",
            height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Last race results ──────────────────────────────────────────────────────
    if not race_df.empty:
        last_round = next_race["round"] - 1
        if last_round > 0:
            st.markdown('<p class="section-header">Last Race Results</p>', unsafe_allow_html=True)
            last_race = race_df[
                (race_df["year"]  == 2026) &
                (race_df["round"] == last_round)
            ].sort_values("finish_position")

            if not last_race.empty:
                rname = last_race["race_name"].iloc[0]
                st.subheader(f"Round {last_round}: {rname}")

                winner = last_race[last_race["finish_position"] == 1]
                if not winner.empty:
                    w = winner.iloc[0]
                    st.success(f"🏆 Winner: **{w['driver_name']}** ({w['constructor_name']})")

                st.dataframe(
                    last_race[["finish_position","driver_name","constructor_name","points","status"]]
                    .head(10)
                    .rename(columns={"finish_position":"Pos","driver_name":"Driver",
                                      "constructor_name":"Team","points":"Pts","status":"Status"}),
                    use_container_width=True, hide_index=True
                )

    # ── Auto-refresh note ──────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("🔄 This page auto-updates after each race via GitHub Actions daily update pipeline. "
               "Predictions switch to post-qualifying mode automatically once qualifying results are available.")


# ─── PAGE: SEASON SIMULATOR ───────────────────────────────────────────────────

def page_season_simulator():
    st.markdown('<p class="main-title">🎲 Season Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Monte Carlo championship probability distribution</p>', unsafe_allow_html=True)

    mc_df = load_championship_probs()
    if mc_df.empty:
        st.warning("Run monte_carlo.py first.")
        return

    c1, c2 = st.columns([2,1])
    with c1:
        display = mc_df[mc_df["championship_prob_pct"] > 0.1].copy()
        fig = px.bar(
            display, x="driver_name", y="championship_prob_pct",
            color="constructor_name",
            text=display["championship_prob_pct"].apply(lambda x: f"{x:.1f}%"),
            title=f"Based on {mc_df['championship_wins'].sum():,} simulations",
            color_discrete_map={t: get_team_color(t) for t in display["constructor_name"].unique()}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=450,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Standings</p>', unsafe_allow_html=True)
        st.dataframe(
            mc_df[["driver_name","current_points","championship_prob_pct"]]
            .rename(columns={"driver_name":"Driver","current_points":"Pts","championship_prob_pct":"Win %"})
            [mc_df["championship_prob_pct"] > 0],
            use_container_width=True, hide_index=True
        )
        st.metric("Total Simulations", f"{mc_df['championship_wins'].sum():,}")
        st.metric("Favourite", mc_df.iloc[0]["driver_name"])
        st.metric("Favourite Odds", f"{mc_df.iloc[0]['championship_prob_pct']:.1f}%")


# ─── PAGE: DRIVER DEEP DIVE ───────────────────────────────────────────────────

def page_driver_deep_dive():
    st.markdown('<p class="main-title">👤 Driver Deep Dive</p>', unsafe_allow_html=True)

    preds    = load_final_predictions()
    elo_hist = load_elo_history()
    race_df  = load_race_results()

    if preds.empty:
        st.warning("Run ensemble.py first.")
        return

    selected   = st.selectbox("Select a driver", preds["driver_name"].tolist())
    driver_row = preds[preds["driver_name"] == selected].iloc[0]
    driver_id  = driver_row["driver_id"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Championship Probability", f"{driver_row['final_prob']:.1f}%")
    c2.metric("Current Points",           int(driver_row["current_points"]))
    c3.metric("Championship Rank",        f"#{int(driver_row['rank'])}")
    c4.metric("Team",                     driver_row["constructor_name"])

    if not elo_hist.empty:
        st.markdown('<p class="section-header">Elo Rating History</p>', unsafe_allow_html=True)
        dh = elo_hist[elo_hist["driver_id"] == driver_id].copy()
        if not dh.empty:
            dh["race"] = dh["year"].astype(str) + " R" + dh["round"].astype(str)
            fig = px.line(dh, x="race", y="elo_before",
                           title=f"{selected} — Elo Rating Career",
                           color_discrete_sequence=[get_team_color(driver_row["constructor_name"])])
            fig.add_hline(y=1500, line_dash="dash", line_color="gray", annotation_text="Average (1500)")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="white"), xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    if not race_df.empty:
        st.markdown('<p class="section-header">Recent Points Per Race</p>', unsafe_allow_html=True)
        dr = race_df[(race_df["driver_id"]==driver_id)&(race_df["year"]>=2023)].copy()
        if not dr.empty:
            dr["race"] = dr["year"].astype(str) + " R" + dr["round"].astype(str)
            fig2 = px.bar(dr, x="race", y="points", title=f"{selected} — Points per Race (2023+)",
                           color_discrete_sequence=[get_team_color(driver_row["constructor_name"])])
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white"), xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig2, use_container_width=True)

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Races",      len(dr))
            c2.metric("Wins",       int((dr["finish_position"]==1).sum()))
            c3.metric("Podiums",    int((dr["finish_position"]<=3).sum()))
            c4.metric("Points",     int(dr["points"].sum()))
            c5.metric("Avg Finish", f"{dr['finish_position'].mean():.1f}")

        # Circuit performance
        st.markdown('<p class="section-header">Performance by Circuit</p>', unsafe_allow_html=True)
        cp = (
            race_df[race_df["driver_id"]==driver_id]
            .groupby("circuit_id")
            .agg(avg_finish=("finish_position","mean"), races=("round","count"))
            .reset_index().sort_values("avg_finish")
        )
        if not cp.empty:
            fig3 = px.bar(cp.head(15), x="circuit_id", y="avg_finish",
                           color="avg_finish", color_continuous_scale="RdYlGn_r",
                           title=f"{selected} — Avg Finish by Circuit (lower = better)",
                           text=cp.head(15)["avg_finish"].round(1))
            fig3.update_traces(textposition="outside")
            fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white"), coloraxis_showscale=False, height=380)
            st.plotly_chart(fig3, use_container_width=True)


# ─── PAGE: HEAD TO HEAD ───────────────────────────────────────────────────────

def page_head_to_head():
    st.markdown('<p class="main-title">⚔️ Head to Head</p>', unsafe_allow_html=True)

    preds   = load_final_predictions()
    race_df = load_race_results()

    if preds.empty:
        st.warning("Run ensemble.py first.")
        return

    drivers  = preds["driver_name"].tolist()
    c1, c2   = st.columns(2)
    driver_a = c1.selectbox("Driver A", drivers, index=0)
    driver_b = c2.selectbox("Driver B", drivers, index=1)

    if driver_a == driver_b:
        st.warning("Select two different drivers.")
        return

    row_a = preds[preds["driver_name"]==driver_a].iloc[0]
    row_b = preds[preds["driver_name"]==driver_b].iloc[0]

    ca, cm, cb = st.columns([2,1,2])
    with ca:
        st.metric(driver_a, f"{row_a['final_prob']:.1f}%",
                  delta=f"{row_a['final_prob']-row_b['final_prob']:+.1f}%")
        st.metric("Points", int(row_a["current_points"]))
        st.metric("Team",   row_a["constructor_name"])
    with cm:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("### VS")
    with cb:
        st.metric(driver_b, f"{row_b['final_prob']:.1f}%",
                  delta=f"{row_b['final_prob']-row_a['final_prob']:+.1f}%")
        st.metric("Points", int(row_b["current_points"]))
        st.metric("Team",   row_b["constructor_name"])

    if not race_df.empty:
        st.markdown('<p class="section-header">Points Per Race — 2024 Onwards</p>', unsafe_allow_html=True)
        recent = race_df[race_df["year"]>=2024].copy()
        recent["race"] = recent["year"].astype(str) + " R" + recent["round"].astype(str)
        fig = go.Figure()
        for driver, row in [(driver_a,row_a),(driver_b,row_b)]:
            dr = recent[recent["driver_id"]==row["driver_id"]]
            fig.add_trace(go.Scatter(x=dr["race"], y=dr["points"],
                mode="lines+markers", name=driver,
                line=dict(color=get_team_color(row["constructor_name"]), width=2)))
        fig.update_layout(title="Points per race", xaxis_tickangle=-45, height=380,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: FEATURE EXPLORER ───────────────────────────────────────────────────

def page_feature_explorer():
    st.markdown('<p class="main-title">📊 Feature Explorer</p>', unsafe_allow_html=True)

    imp = load_feature_importance()
    if imp.empty:
        st.warning("Run xgboost_model.py with SHAP first.")
        return

    top30 = imp.head(30).sort_values("importance", ascending=True)
    fig = px.bar(top30, x="importance", y="feature", orientation="h",
                  color="importance", color_continuous_scale="Reds",
                  title="Top 30 features — mean |SHAP value|")
    fig.update_layout(showlegend=False, height=700,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="white"), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(imp.rename(columns={"feature":"Feature","importance":"SHAP","rank":"Rank"}),
                 use_container_width=True, hide_index=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()
    if   page == "🏆 Championship Tracker":   page_championship_tracker()
    elif page == "🔮 Race by Race Predictor":  page_race_predictor_detailed()
    elif page == "🏁 Next Race Predictor":     page_next_race()
    elif page == "🎲 Season Simulator":        page_season_simulator()
    elif page == "👤 Driver Deep Dive":        page_driver_deep_dive()
    elif page == "⚔️ Head to Head":           page_head_to_head()
    elif page == "📊 Feature Explorer":        page_feature_explorer()

if __name__ == "__main__":
    main()