"""
app.py
------
Main Streamlit website for the F1 Championship Predictor.

Pages:
  1. Championship Tracker  — live probability bars for all drivers
  2. Race by Race Predictor — predict each race before/after qualifying
  3. Race Predictor        — next race podium odds + weather
  4. Season Simulator      — run Monte Carlo live in browser
  5. Driver Deep Dive      — circuit affinity, wet weather, Elo trend
  6. Head to Head          — compare two drivers side by side
  7. Feature Explorer      — SHAP feature importance chart

HOW TO RUN:
    streamlit run app.py

REQUIREMENTS:
    pip install streamlit plotly pandas numpy
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
    .main-title {
        font-size: 2.4rem; font-weight: 700;
        color: #E10600; margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem; color: #888;
        margin-top: 0; margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.3rem; font-weight: 600;
        margin-top: 1.5rem; margin-bottom: 0.5rem;
        border-bottom: 2px solid #E10600; padding-bottom: 0.3rem;
    }
    .podium-card {
        border-radius: 10px; padding: 1rem;
        text-align: center; font-size: 1rem;
    }
    .p1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; }
    .p2 { background: linear-gradient(135deg, #C0C0C0, #A0A0A0); color: #000; }
    .p3 { background: linear-gradient(135deg, #CD7F32, #A0522D); color: #fff; }
    .weather-rain   { color: #4fc3f7; font-weight: bold; }
    .weather-dry    { color: #ffb300; font-weight: bold; }
    .correct-pred   { background-color: rgba(0,200,81,0.15);  border-radius: 4px; }
    .incorrect-pred { background-color: rgba(255,68,68,0.10); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── TEAM COLOURS ─────────────────────────────────────────────────────────────

TEAM_COLORS = {
    "Red Bull":         "#3671C6", "Mercedes":      "#27F4D2",
    "Ferrari":          "#E8002D", "McLaren":       "#FF8000",
    "Aston Martin":     "#229971", "Alpine F1 Team":"#FF87BC",
    "Williams":         "#64C4FF", "RB F1 Team":    "#6692FF",
    "Haas F1 Team":     "#B6BABD", "Audi":          "#999999",
    "Cadillac F1 Team": "#C00000", "Sauber":        "#52E252",
}

def get_team_color(team_name):
    for key, color in TEAM_COLORS.items():
        if key.lower() in str(team_name).lower():
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
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
        df["points"]          = pd.to_numeric(df["points"],           errors="coerce")
        df["grid_position"]   = pd.to_numeric(df["grid_position"],    errors="coerce")
        return df
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_qualifying_results():
    path = os.path.join(DATA_DIR, "qualifying_results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["quali_position"] = pd.to_numeric(df["quali_position"], errors="coerce")
        return df
    return pd.DataFrame()

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
    if os.path.exists(path):
        with open(path, "rb") as f:
            saved = pickle.load(f)
        return saved["model"], saved["feature_cols"]
    return None, None


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## 🏎️ F1 Predictor 2026")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate", [
        "🏆 Championship Tracker",
        "🔮 Race by Race Predictor",
        "🏁 Race Predictor",
        "🎲 Season Simulator",
        "👤 Driver Deep Dive",
        "⚔️  Head to Head",
        "📊 Feature Explorer",
    ])

    st.sidebar.markdown("---")
    preds = load_final_predictions()
    if not preds.empty and "updated_at" in preds.columns:
        st.sidebar.caption(f"Last updated: {preds['updated_at'].iloc[0]}")

    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    return page


# ─── RACE PREDICTION ENGINE ───────────────────────────────────────────────────

def predict_race_finishing_order(race_df, quali_df, master_df,
                                  elo_df, model, feature_cols,
                                  year, round_num, use_quali=True):
    """
    Generates predicted finishing order and probabilities for one race.

    Mode 1 — Before qualifying (use_quali=False):
      Uses rolling form, Elo ratings, circuit affinity, and weather.
      Grid position is NOT known yet.

    Mode 2 — After qualifying (use_quali=True):
      Also uses actual qualifying position as a feature.
      This is significantly more accurate.

    Returns:
        pd.DataFrame: One row per driver with predicted rank and probabilities
    """
    # Get drivers in this race from master dataset or race results
    race_data = master_df[
        (master_df["year"] == year) &
        (master_df["round"] == round_num)
    ].copy() if not master_df.empty else pd.DataFrame()

    # Fall back to race results if master doesn't have this round yet
    if race_data.empty:
        race_data = race_df[
            (race_df["year"] == year) &
            (race_df["round"] == round_num)
        ].copy()

    if race_data.empty:
        return pd.DataFrame()

    # If using qualifying, inject actual grid position into features
    if use_quali and not quali_df.empty:
        quali_round = quali_df[
            (quali_df["year"] == year) &
            (quali_df["round"] == round_num)
        ][["driver_id", "quali_position"]]

        if not quali_round.empty and "quali_position" in race_data.columns:
            race_data = race_data.drop(columns=["quali_position"], errors="ignore")
            race_data = race_data.merge(quali_round, on="driver_id", how="left")

    # Get Elo ratings
    if not elo_df.empty:
        elo_map = dict(zip(elo_df["driver_id"], elo_df["elo_rating"]))
        if "elo_before" in race_data.columns:
            race_data["elo_before"] = race_data["driver_id"].map(elo_map).fillna(1500)

    # Run XGBoost prediction if model available
    if model is not None and feature_cols is not None:
        available_features = [f for f in feature_cols if f in race_data.columns]

        if len(available_features) >= len(feature_cols) * 0.7:
            X = race_data[available_features].fillna(0).values

            # Pad missing features with zeros
            if len(available_features) < len(feature_cols):
                pad = np.zeros((len(X), len(feature_cols) - len(available_features)))
                X   = np.hstack([X, pad])

            try:
                scores = model.predict(X)
            except Exception:
                scores = np.random.rand(len(race_data))
        else:
            scores = np.random.rand(len(race_data))
    else:
        # Elo-based fallback
        elo_map = dict(zip(elo_df["driver_id"], elo_df["elo_rating"])) if not elo_df.empty else {}
        scores  = np.array([elo_map.get(d, 1500) for d in race_data["driver_id"]])

    # Convert scores to probabilities using softmax
    exp_s = np.exp(scores - scores.max())
    probs = exp_s / exp_s.sum()

    # Win probability
    win_probs = probs.copy()

    # Podium probability (sum of top-3 softmax)
    top3_idx    = np.argsort(scores)[::-1][:3]
    podium_mask = np.zeros(len(scores))
    podium_mask[top3_idx] = 1

    result = pd.DataFrame({
        "driver_id":       race_data["driver_id"].values,
        "driver_name":     race_data.get("driver_name", race_data["driver_id"]).values,
        "constructor_name": race_data.get("constructor_name",
                             pd.Series([""] * len(race_data))).values,
        "raw_score":       scores,
        "win_probability": np.round(win_probs * 100, 2),
        "podium_flag":     podium_mask,
    })

    # Add grid position if available
    if "grid_position" in race_data.columns:
        result["grid_position"] = race_data["grid_position"].values
    elif use_quali and "quali_position" in race_data.columns:
        result["grid_position"] = race_data["quali_position"].values

    # Predicted rank
    result["predicted_rank"] = result["raw_score"].rank(ascending=False).astype(int)

    # Podium probability per driver
    result["p1_prob"] = np.round(
        np.where(result["predicted_rank"] == 1, win_probs * 100, win_probs * 100 * 0.3), 1
    )
    result["p2_prob"] = np.round(win_probs * 100 * 0.7 *
                                  np.where(result["predicted_rank"] <= 3, 1, 0.2), 1)
    result["p3_prob"] = np.round(win_probs * 100 * 0.5 *
                                  np.where(result["predicted_rank"] <= 5, 1, 0.1), 1)

    result = result.sort_values("predicted_rank").reset_index(drop=True)
    return result


# ─── PAGE: RACE BY RACE PREDICTOR ─────────────────────────────────────────────

def page_race_predictor_detailed():
    st.markdown('<p class="main-title">🔮 Race by Race Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predict each Grand Prix — before qualifying and after qualifying. Compare with actual results.</p>', unsafe_allow_html=True)

    # Load all data
    race_df    = load_race_results()
    quali_df   = load_qualifying_results()
    master_df  = load_master_dataset()
    elo_df     = load_elo_current()
    weather_h  = load_weather_historical()
    weather_f  = load_weather_forecast()

    model, feature_cols = load_xgboost_model()

    if race_df.empty:
        st.error("No race data found. Run get_ergast_data.py first.")
        return

    # ── Race selector ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Select a Race</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        years = sorted(race_df["year"].unique(), reverse=True)
        selected_year = st.selectbox("Season", years, index=0)

    with col2:
        year_races = (
            race_df[race_df["year"] == selected_year]
            [["round", "race_name"]]
            .drop_duplicates()
            .sort_values("round")
        )
        race_options = {
            f"Round {row['round']}: {row['race_name']}": row["round"]
            for _, row in year_races.iterrows()
        }
        selected_label = st.selectbox("Race", list(race_options.keys()))
        selected_round = race_options[selected_label]

    # ── Prediction mode toggle ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">Prediction Timing</p>', unsafe_allow_html=True)

    pred_mode = st.radio(
        "When to predict",
        ["Before qualifying", "After qualifying", "Side by side comparison"],
        horizontal=True,
        help="Before qualifying uses only form/weather. After qualifying adds grid position."
    )

    st.markdown("---")

    # ── Get actual result for this race ───────────────────────────────────────
    actual_result = race_df[
        (race_df["year"]  == selected_year) &
        (race_df["round"] == selected_round)
    ].sort_values("finish_position").copy()

    race_name   = actual_result["race_name"].iloc[0] if not actual_result.empty else f"Round {selected_round}"
    race_date   = actual_result["race_date"].iloc[0] if not actual_result.empty else ""
    circuit_id  = actual_result["circuit_id"].iloc[0] if not actual_result.empty else ""
    has_result  = not actual_result.empty and actual_result["finish_position"].notna().any()

    # ── Race header ────────────────────────────────────────────────────────────
    st.markdown(f"## {race_name}")
    st.caption(f"Round {selected_round} • {selected_year} • {race_date} • {circuit_id}")

    # ── Weather for this race ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">Race Day Weather</p>', unsafe_allow_html=True)

    weather_row = pd.DataFrame()
    if not weather_h.empty and has_result:
        weather_row = weather_h[
            weather_h["circuit_id"] == circuit_id
        ].sort_values("race_date").tail(1)

    if not weather_row.empty:
        wr = weather_row.iloc[0]
        wc1, wc2, wc3, wc4, wc5, wc6 = st.columns(6)
        wc1.metric("Air Temp",   f"{wr.get('air_temp_avg_c',   '?')}°C")
        wc2.metric("Track Temp", f"{wr.get('track_temp_avg_c', '?')}°C")
        wc3.metric("Humidity",   f"{wr.get('humidity_avg_pct', '?')}%")
        wc4.metric("Wind",       f"{wr.get('windspeed_avg_kmh','?')} km/h")
        wc5.metric("Rainfall",   f"{wr.get('rainfall_total_mm','0')} mm")
        rain = wr.get("rain_flag", 0)
        wc6.metric("Conditions", "🌧️ Wet" if rain else "☀️ Dry")

        if rain:
            st.info("🌧️ **Wet race detected** — wet weather driver ratings have been applied to predictions")
    else:
        st.caption("No weather data available for this race")

    st.markdown("---")

    # ── Generate predictions ───────────────────────────────────────────────────
    def run_prediction(use_quali):
        return predict_race_finishing_order(
            race_df, quali_df, master_df, elo_df,
            model, feature_cols,
            selected_year, selected_round,
            use_quali=use_quali
        )

    # ── SIDE BY SIDE COMPARISON ────────────────────────────────────────────────
    if pred_mode == "Side by side comparison":
        pred_pre  = run_prediction(use_quali=False)
        pred_post = run_prediction(use_quali=True)

        col_pre, col_post = st.columns(2)

        with col_pre:
            st.markdown("### Before Qualifying")
            st.caption("Based on season form, Elo ratings, circuit affinity, weather")
            if not pred_pre.empty:
                render_prediction_table(pred_pre, actual_result, show_actual=has_result)
            else:
                st.warning("No pre-qualifying prediction available")

        with col_post:
            st.markdown("### After Qualifying")
            st.caption("Adds actual grid positions — significantly more accurate")
            if not pred_post.empty:
                render_prediction_table(pred_post, actual_result, show_actual=has_result)
            else:
                st.warning("No post-qualifying prediction available")

        # Accuracy comparison
        if has_result and not pred_pre.empty and not pred_post.empty:
            st.markdown("---")
            st.markdown('<p class="section-header">Prediction Accuracy Comparison</p>', unsafe_allow_html=True)
            render_accuracy_comparison(pred_pre, pred_post, actual_result)

    else:
        # ── SINGLE MODE ───────────────────────────────────────────────────────
        use_quali = (pred_mode == "After qualifying")
        pred_df   = run_prediction(use_quali=use_quali)

        if pred_df.empty:
            st.warning("Could not generate prediction for this race. Feature data may not be available.")
            return

        # ── Podium cards ──────────────────────────────────────────────────────
        st.markdown('<p class="section-header">Predicted Podium</p>', unsafe_allow_html=True)

        top3 = pred_df.head(3)
        pc1, pc2, pc3 = st.columns(3)

        podium_cols  = [pc2, pc1, pc3]   # P2, P1, P3 layout
        podium_ranks = [2, 1, 3]
        podium_css   = ["p2", "p1", "p3"]
        podium_emojis = ["🥈", "🥇", "🥉"]

        for col, rank, css, emoji in zip(podium_cols, podium_ranks, podium_css, podium_emojis):
            driver_row = pred_df[pred_df["predicted_rank"] == rank]
            if not driver_row.empty:
                dr = driver_row.iloc[0]
                col.markdown(
                    f'<div class="podium-card {css}">'
                    f'{emoji} <b>P{rank}</b><br>'
                    f'<span style="font-size:1.2rem"><b>{dr["driver_name"]}</b></span><br>'
                    f'<span style="font-size:0.85rem">{dr["constructor_name"]}</span><br>'
                    f'<span style="font-size:0.9rem">Win prob: {dr["win_probability"]:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Win probability chart ──────────────────────────────────────────────
        st.markdown('<p class="section-header">Win Probability — All Drivers</p>', unsafe_allow_html=True)

        plot_df = pred_df[pred_df["win_probability"] > 0.5].copy()
        plot_df = plot_df.sort_values("win_probability", ascending=True)

        fig_win = go.Figure()
        for _, row in plot_df.iterrows():
            color = get_team_color(row["constructor_name"])
            fig_win.add_trace(go.Bar(
                x            = [row["win_probability"]],
                y            = [row["driver_name"]],
                orientation  = "h",
                marker_color = color,
                text         = f"{row['win_probability']:.1f}%",
                textposition = "outside",
                name         = row["driver_name"],
                hovertemplate = (
                    f"<b>{row['driver_name']}</b><br>"
                    f"Team: {row['constructor_name']}<br>"
                    f"Win probability: {row['win_probability']:.1f}%<br>"
                    f"Predicted P1 prob: {row['p1_prob']:.1f}%<br>"
                    f"Predicted P2 prob: {row['p2_prob']:.1f}%<br>"
                    f"Predicted P3 prob: {row['p3_prob']:.1f}%<extra></extra>"
                ),
            ))

        fig_win.update_layout(
            showlegend    = False,
            xaxis_title   = "Win Probability (%)",
            height        = max(350, len(plot_df) * 35),
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="white"),
            margin        = dict(l=10, r=80, t=20, b=40),
        )
        fig_win.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
        st.plotly_chart(fig_win, use_container_width=True)

        # ── Full predicted order vs actual ─────────────────────────────────────
        st.markdown('<p class="section-header">Full Predicted Finishing Order</p>', unsafe_allow_html=True)

        if has_result:
            st.caption("✅ = correct position prediction  ❌ = incorrect")

        render_prediction_table(pred_df, actual_result, show_actual=has_result, full=True)

        # ── Accuracy metrics ──────────────────────────────────────────────────
        if has_result and not pred_df.empty:
            st.markdown('<p class="section-header">Prediction Accuracy</p>', unsafe_allow_html=True)
            render_accuracy_single(pred_df, actual_result)


# ─── HELPER: RENDER PREDICTION TABLE ──────────────────────────────────────────

def render_prediction_table(pred_df, actual_result, show_actual=False, full=False):
    """
    Renders a styled table of predicted vs actual finishing order.

    Args:
        pred_df       (pd.DataFrame): Predictions
        actual_result (pd.DataFrame): Actual race results
        show_actual   (bool):         Whether to show actual finish column
        full          (bool):         Whether to show all 20 drivers or top 10
    """
    display = pred_df.copy() if full else pred_df.head(10).copy()

    # Build actual position map
    actual_map = {}
    if show_actual and not actual_result.empty:
        actual_map = dict(zip(
            actual_result["driver_id"],
            actual_result["finish_position"]
        ))

    rows = []
    for _, row in display.iterrows():
        driver_id    = row["driver_id"]
        predicted    = int(row["predicted_rank"])
        actual_pos   = actual_map.get(driver_id, None)

        correct = "✅" if actual_pos is not None and abs(predicted - actual_pos) <= 1 else (
                  "❌" if actual_pos is not None else "")

        grid = row.get("grid_position", "?")
        grid = int(grid) if pd.notna(grid) and grid != "?" else "?"

        rows.append({
            "Pred": predicted,
            "Driver": row["driver_name"],
            "Team": row["constructor_name"],
            "Grid": grid,
            "Win %": f"{row['win_probability']:.1f}%",
            "P1%": f"{row.get('p1_prob', 0):.1f}%",
            "P2%": f"{row.get('p2_prob', 0):.1f}%",
            "P3%": f"{row.get('p3_prob', 0):.1f}%",
            **({"Actual": int(actual_pos) if actual_pos else "TBD",
                "✓": correct} if show_actual else {}),
        })

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)


# ─── HELPER: ACCURACY METRICS ─────────────────────────────────────────────────

def render_accuracy_single(pred_df, actual_result):
    """Shows accuracy metrics for a single prediction."""
    actual_map = dict(zip(
        actual_result["driver_id"],
        actual_result["finish_position"]
    ))

    pred_df = pred_df.copy()
    pred_df["actual_pos"] = pred_df["driver_id"].map(actual_map)
    pred_df = pred_df.dropna(subset=["actual_pos"])

    if pred_df.empty:
        return

    # Winner correct?
    predicted_winner = pred_df[pred_df["predicted_rank"] == 1]["driver_id"].values
    actual_winner    = actual_result[actual_result["finish_position"] == 1]["driver_id"].values

    winner_correct = (len(predicted_winner) > 0 and len(actual_winner) > 0 and
                      predicted_winner[0] == actual_winner[0])

    # Top 3 accuracy
    pred_top3   = set(pred_df[pred_df["predicted_rank"] <= 3]["driver_id"].values)
    actual_top3 = set(actual_result[actual_result["finish_position"] <= 3]["driver_id"].values)
    top3_overlap = len(pred_top3 & actual_top3)

    # Within 3 positions accuracy
    pred_df["pos_error"]    = abs(pred_df["predicted_rank"] - pred_df["actual_pos"])
    within3_acc = (pred_df["pos_error"] <= 3).mean() * 100
    avg_error   = pred_df["pos_error"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winner Correct",    "✅ Yes" if winner_correct else "❌ No")
    c2.metric("Podium Overlap",    f"{top3_overlap}/3 drivers correct")
    c3.metric("Within 3 Positions", f"{within3_acc:.0f}% of drivers")
    c4.metric("Avg Position Error", f"{avg_error:.1f} places")


def render_accuracy_comparison(pred_pre, pred_post, actual_result):
    """Compares accuracy between pre and post qualifying predictions."""
    actual_map = dict(zip(
        actual_result["driver_id"],
        actual_result["finish_position"]
    ))

    def get_metrics(pred_df):
        df = pred_df.copy()
        df["actual_pos"] = df["driver_id"].map(actual_map)
        df = df.dropna(subset=["actual_pos"])
        if df.empty:
            return None

        winner_correct = False
        pw = df[df["predicted_rank"] == 1]["driver_id"].values
        aw = actual_result[actual_result["finish_position"] == 1]["driver_id"].values
        if len(pw) > 0 and len(aw) > 0:
            winner_correct = pw[0] == aw[0]

        pred_top3   = set(df[df["predicted_rank"] <= 3]["driver_id"].values)
        actual_top3 = set(actual_result[actual_result["finish_position"] <= 3]["driver_id"].values)

        df["pos_error"] = abs(df["predicted_rank"] - df["actual_pos"])

        return {
            "winner":    "✅" if winner_correct else "❌",
            "podium":    f"{len(pred_top3 & actual_top3)}/3",
            "within3":   f"{(df['pos_error'] <= 3).mean() * 100:.0f}%",
            "avg_error": f"{df['pos_error'].mean():.1f}",
        }

    m_pre  = get_metrics(pred_pre)
    m_post = get_metrics(pred_post)

    if m_pre and m_post:
        metric_df = pd.DataFrame({
            "Metric":             ["Winner correct", "Podium overlap", "Within 3 positions", "Avg error"],
            "Before qualifying":  [m_pre["winner"],  m_pre["podium"],  m_pre["within3"],     m_pre["avg_error"]],
            "After qualifying":   [m_post["winner"], m_post["podium"], m_post["within3"],     m_post["avg_error"]],
        })
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
        st.caption("After qualifying is typically 15–20% more accurate because grid position is the single strongest predictor of race outcome.")


# ─── PAGE 1: CHAMPIONSHIP TRACKER ────────────────────────────────────────────

def page_championship_tracker():
    st.markdown('<p class="main-title">🏆 Championship Tracker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Live championship win probabilities — updated after every race</p>', unsafe_allow_html=True)

    preds   = load_final_predictions()
    race_df = load_race_results()

    if preds.empty:
        st.warning("No predictions found. Run ensemble.py first.")
        return

    races_done      = preds["races_completed"].iloc[0] if "races_completed" in preds.columns else 0
    races_remaining = preds["races_remaining"].iloc[0] if "races_remaining" in preds.columns else 24
    leader          = preds.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Championship Leader", leader["driver_name"])
    col2.metric("Win Probability",     f"{leader['final_prob']:.1f}%")
    col3.metric("Races Completed",     f"{races_done} / 24")
    col4.metric("Races Remaining",     f"{races_remaining}")

    st.markdown("---")

    st.markdown('<p class="section-header">Championship Win Probability</p>', unsafe_allow_html=True)
    display_df = preds[preds["final_prob"] > 0.1].head(15).copy()

    fig = go.Figure()
    for _, row in display_df.iterrows():
        color = get_team_color(row["constructor_name"])
        fig.add_trace(go.Bar(
            name         = row["driver_name"],
            x            = [row["final_prob"]],
            y            = [row["driver_name"]],
            orientation  = "h",
            marker_color = color,
            text         = f"{row['final_prob']:.1f}%",
            textposition = "outside",
            hovertemplate = (
                f"<b>{row['driver_name']}</b><br>"
                f"Team: {row['constructor_name']}<br>"
                f"Points: {row['current_points']}<br>"
                f"Final: {row['final_prob']:.1f}%<br>"
                f"MC: {row['mc_prob']:.1f}%  Bayes: {row['bayes_prob']:.1f}%  Elo: {row['elo_prob']:.1f}%"
                f"<extra></extra>"
            )
        ))

    fig.update_layout(
        showlegend    = False,
        xaxis_title   = "Championship Win Probability (%)",
        yaxis         = dict(autorange="reversed"),
        height        = max(400, len(display_df) * 35),
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(color="white"),
        margin        = dict(l=10, r=80, t=20, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Model Breakdown</p>', unsafe_allow_html=True)
    table_df = preds.head(10)[["rank", "driver_name", "constructor_name",
                                "current_points", "mc_prob", "bayes_prob",
                                "elo_prob", "final_prob"]].copy()
    table_df.columns = ["Rank", "Driver", "Team", "Points",
                         "Monte Carlo %", "Bayesian %", "Elo %", "Final %"]
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    bayes_hist = load_bayesian_history()
    if not bayes_hist.empty:
        st.markdown('<p class="section-header">Probability Evolution Through the Season</p>', unsafe_allow_html=True)
        top_drivers   = preds.head(6)["driver_id"].tolist()
        hist_filtered = bayes_hist[bayes_hist["driver_id"].isin(top_drivers)]

        fig2 = px.line(
            hist_filtered,
            x     = "round",
            y     = "probability",
            color = "driver_name",
            title = "Championship win probability after each race",
            labels = {"round": "Race Round", "probability": "Win Probability (%)"},
        )
        fig2.update_layout(
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="white"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─── PAGE: RACE PREDICTOR (NEXT RACE) ─────────────────────────────────────────

def page_race_predictor():
    st.markdown('<p class="main-title">🏁 Next Race Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upcoming race weather and championship standings</p>', unsafe_allow_html=True)

    preds   = load_final_predictions()
    weather = load_weather_forecast()
    race_df = load_race_results()

    if not weather.empty:
        st.markdown('<p class="section-header">Upcoming Race Weather Forecast</p>', unsafe_allow_html=True)
        for _, race in weather.iterrows():
            with st.expander(f"🌤️ {race['circuit_name']} — {race['race_date']}", expanded=True):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Air Temp",   f"{race.get('air_temp_avg_c',   '?')}°C")
                c2.metric("Track Temp", f"{race.get('track_temp_avg_c', '?')}°C")
                c3.metric("Humidity",   f"{race.get('humidity_avg_pct', '?')}%")
                c4.metric("Wind",       f"{race.get('windspeed_avg_kmh','?')} km/h")
                rain_prob = race.get("rain_probability_pct", 0)
                c5.metric("Rain Prob",  f"{rain_prob:.0f}%",
                           "🌧️ Likely" if rain_prob > 50 else "☀️ Dry expected")
    else:
        st.info("No upcoming race weather. Run update_weather.py.")

    if not preds.empty:
        st.markdown('<p class="section-header">Current Championship Standings</p>', unsafe_allow_html=True)
        top10 = preds.head(10).copy()
        fig = go.Figure()
        for _, row in top10.iterrows():
            fig.add_trace(go.Bar(
                x            = [row["driver_name"]],
                y            = [row["final_prob"]],
                name         = row["driver_name"],
                marker_color = get_team_color(row["constructor_name"]),
                text         = f"{row['final_prob']:.1f}%",
                textposition = "outside",
            ))
        fig.update_layout(
            showlegend    = False,
            yaxis_title   = "Championship Win Probability (%)",
            height        = 400,
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

    if not race_df.empty:
        st.markdown('<p class="section-header">Most Recent Race Results</p>', unsafe_allow_html=True)
        ly = race_df["year"].max()
        lr = race_df[race_df["year"] == ly]["round"].max()
        latest = race_df[(race_df["year"] == ly) & (race_df["round"] == lr)].sort_values("finish_position").head(10)
        if not latest.empty:
            st.subheader(latest["race_name"].iloc[0])
            st.dataframe(
                latest[["finish_position","driver_name","constructor_name","points","status"]]
                .rename(columns={"finish_position":"Pos","driver_name":"Driver",
                                  "constructor_name":"Team","points":"Pts","status":"Status"}),
                use_container_width=True, hide_index=True
            )


# ─── PAGE: SEASON SIMULATOR ───────────────────────────────────────────────────

def page_season_simulator():
    st.markdown('<p class="main-title">🎲 Season Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Monte Carlo championship probability distribution</p>', unsafe_allow_html=True)

    mc_df = load_championship_probs()
    if mc_df.empty:
        st.warning("Run monte_carlo.py first.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        display = mc_df[mc_df["championship_prob_pct"] > 0.1].copy()
        fig = px.bar(
            display, x="driver_name", y="championship_prob_pct",
            color="constructor_name",
            text=display["championship_prob_pct"].apply(lambda x: f"{x:.1f}%"),
            title=f"Based on {mc_df['championship_wins'].sum():,} simulations",
            labels={"championship_prob_pct": "Win Probability (%)", "driver_name": "Driver"},
            color_discrete_map={t: get_team_color(t) for t in display["constructor_name"].unique()}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=450,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Current Standings</p>', unsafe_allow_html=True)
        st.dataframe(
            mc_df[["driver_name","current_points","championship_prob_pct"]]
            .rename(columns={"driver_name":"Driver","current_points":"Pts","championship_prob_pct":"Win %"})
            [mc_df["championship_prob_pct"] > 0].sort_values("Pts", ascending=False),
            use_container_width=True, hide_index=True
        )


# ─── PAGE: DRIVER DEEP DIVE ───────────────────────────────────────────────────

def page_driver_deep_dive():
    st.markdown('<p class="main-title">👤 Driver Deep Dive</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Explore any driver\'s full performance profile</p>', unsafe_allow_html=True)

    preds    = load_final_predictions()
    elo_hist = load_elo_history()
    race_df  = load_race_results()

    if preds.empty:
        st.warning("Run ensemble.py first.")
        return

    selected  = st.selectbox("Select a driver", preds["driver_name"].tolist())
    driver_row = preds[preds["driver_name"] == selected].iloc[0]
    driver_id  = driver_row["driver_id"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Championship Probability", f"{driver_row['final_prob']:.1f}%")
    c2.metric("Current Points",           int(driver_row["current_points"]))
    c3.metric("Championship Rank",        f"#{int(driver_row['rank'])}")
    c4.metric("Team",                     driver_row["constructor_name"])

    if not elo_hist.empty:
        st.markdown('<p class="section-header">Elo Rating History</p>', unsafe_allow_html=True)
        driver_elo = elo_hist[elo_hist["driver_id"] == driver_id].copy()
        if not driver_elo.empty:
            driver_elo["race_label"] = driver_elo["year"].astype(str) + " R" + driver_elo["round"].astype(str)
            fig = px.line(driver_elo, x="race_label", y="elo_before",
                           title=f"{selected} — Elo Rating Career",
                           color_discrete_sequence=[get_team_color(driver_row["constructor_name"])])
            fig.add_hline(y=1500, line_dash="dash", line_color="gray", annotation_text="Average")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="white"), xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    if not race_df.empty:
        st.markdown('<p class="section-header">Recent Points Per Race</p>', unsafe_allow_html=True)
        dr = race_df[(race_df["driver_id"] == driver_id) & (race_df["year"] >= 2023)].copy()
        if not dr.empty:
            dr["race_label"] = dr["year"].astype(str) + " R" + dr["round"].astype(str)
            fig2 = px.bar(dr, x="race_label", y="points",
                           color_discrete_sequence=[get_team_color(driver_row["constructor_name"])],
                           title=f"{selected} — Points per Race (2023+)")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="white"), xaxis_tickangle=-45, height=350)
            st.plotly_chart(fig2, use_container_width=True)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Races",      len(dr))
            c2.metric("Wins",       int((dr["finish_position"] == 1).sum()))
            c3.metric("Podiums",    int((dr["finish_position"] <= 3).sum()))
            c4.metric("Points",     int(dr["points"].sum()))
            c5.metric("Avg Finish", f"{dr['finish_position'].mean():.1f}")


# ─── PAGE: HEAD TO HEAD ───────────────────────────────────────────────────────

def page_head_to_head():
    st.markdown('<p class="main-title">⚔️ Head to Head</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Compare two drivers across all dimensions</p>', unsafe_allow_html=True)

    preds   = load_final_predictions()
    race_df = load_race_results()

    if preds.empty:
        st.warning("Run ensemble.py first.")
        return

    drivers = preds["driver_name"].tolist()
    c1, c2  = st.columns(2)
    driver_a = c1.selectbox("Driver A", drivers, index=0)
    driver_b = c2.selectbox("Driver B", drivers, index=1)

    if driver_a == driver_b:
        st.warning("Select two different drivers.")
        return

    row_a = preds[preds["driver_name"] == driver_a].iloc[0]
    row_b = preds[preds["driver_name"] == driver_b].iloc[0]

    ca, cm, cb = st.columns([2, 1, 2])
    with ca:
        st.metric(driver_a, f"{row_a['final_prob']:.1f}%",
                  delta=f"{row_a['final_prob'] - row_b['final_prob']:+.1f}%")
        st.metric("Points", int(row_a["current_points"]))
        st.metric("Team",   row_a["constructor_name"])
    with cm:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### VS")
    with cb:
        st.metric(driver_b, f"{row_b['final_prob']:.1f}%",
                  delta=f"{row_b['final_prob'] - row_a['final_prob']:+.1f}%")
        st.metric("Points", int(row_b["current_points"]))
        st.metric("Team",   row_b["constructor_name"])

    if not race_df.empty:
        st.markdown('<p class="section-header">Points Per Race — 2024 Onwards</p>', unsafe_allow_html=True)
        recent = race_df[race_df["year"] >= 2024].copy()
        recent["race_label"] = recent["year"].astype(str) + " R" + recent["round"].astype(str)

        fig = go.Figure()
        for driver, row in [(driver_a, row_a), (driver_b, row_b)]:
            dr = recent[recent["driver_id"] == row["driver_id"]]
            fig.add_trace(go.Scatter(
                x=dr["race_label"], y=dr["points"],
                mode="lines+markers", name=driver,
                line=dict(color=get_team_color(row["constructor_name"]), width=2),
            ))
        fig.update_layout(
            title="Points per race", xaxis_tickangle=-45, height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── PAGE: FEATURE EXPLORER ───────────────────────────────────────────────────

def page_feature_explorer():
    st.markdown('<p class="main-title">📊 Feature Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Which features drive the championship predictions most?</p>', unsafe_allow_html=True)

    importance_df = load_feature_importance()
    if importance_df.empty:
        st.warning("Run xgboost_model.py with SHAP first.")
        return

    top30 = importance_df.head(30).sort_values("importance", ascending=True)
    fig = px.bar(
        top30, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Reds",
        title="Top 30 features — mean |SHAP value|",
        labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
    )
    fig.update_layout(
        showlegend=False, height=700,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"), coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(importance_df.rename(columns={"feature":"Feature","importance":"SHAP","rank":"Rank"}),
                 use_container_width=True, hide_index=True)


# ─── MAIN APP ─────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "🏆 Championship Tracker":
        page_championship_tracker()
    elif page == "🔮 Race by Race Predictor":
        page_race_predictor_detailed()
    elif page == "🏁 Race Predictor":
        page_race_predictor()
    elif page == "🎲 Season Simulator":
        page_season_simulator()
    elif page == "👤 Driver Deep Dive":
        page_driver_deep_dive()
    elif page == "⚔️  Head to Head":
        page_head_to_head()
    elif page == "📊 Feature Explorer":
        page_feature_explorer()


if __name__ == "__main__":
    main()