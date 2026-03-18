"""
xgboost_model.py
----------------
Trains an XGBoost ranker to predict F1 race finishing positions.

KEY FIX: XGBRanker with rank:pairwise learns that HIGHER label = BETTER rank.
But finish_position has P1=1 (best) and P20=20 (worst).
So we train on INVERTED target: rank_score = 21 - finish_position
  P1  → rank_score = 20  (highest = best)
  P20 → rank_score = 1   (lowest  = worst)

At prediction time, HIGHER score = better predicted finish = lower position number.
So predicted_rank uses ascending=False (highest score → P1).

HOW TO RUN:
    python src/models/xgboost_model.py

REQUIREMENTS:
    pip install xgboost scikit-learn shap
"""

import pandas as pd
import numpy as np
import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb

# ─── PATH SETUP ───────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)
SRC_DIR      = os.path.dirname(os.path.dirname(THIS_FILE))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

TARGET_COL = "finish_position"

# Columns to never use as model features
NON_FEATURE_COLS = [
    "finish_position", "points", "grid_position",
    "driver_id", "constructor_id", "circuit_id",
    "driver_enc", "constructor_enc", "circuit_type_enc",
    "year", "round",
]

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    "objective":        "rank:pairwise",
    "eval_metric":      "ndcg",
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "verbosity":        0,
    "n_jobs":           -1,
}

NEW_REG_YEARS = [2022, 2026]

# Maximum number of cars per race (used for target inversion)
MAX_CARS = 22


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_dataset():
    """Loads the master dataset built by build_dataset.py."""
    path = os.path.join(DATA_DIR, "master_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"master_dataset.csv not found.\nPlease run build_dataset.py first."
        )
    df = pd.read_csv(path)
    print(f"Loaded master dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Seasons : {sorted(df['year'].unique())}")
    print(f"Drivers : {df['driver_id'].nunique()}")
    return df


# ─── TARGET INVERSION ─────────────────────────────────────────────────────────

def invert_target(finish_positions, max_cars=MAX_CARS):
    """
    Inverts finish_position so XGBRanker learns correctly.

    XGBRanker ranks items so that HIGHER label = BETTER = ranked first.
    But finish_position has P1=1 (best) and P20=20 (worst).

    Fix: rank_score = (max_cars + 1) - finish_position
      P1  → 22 (highest score → ranked first)
      P20 → 2  (lowest score  → ranked last)

    Args:
        finish_positions: array of finish positions (1=best, 20=worst)
        max_cars: maximum cars in a race

    Returns:
        array of rank scores (higher = better finish)
    """
    return (max_cars + 1) - np.array(finish_positions, dtype=float)


# ─── FEATURE SELECTION ────────────────────────────────────────────────────────

def get_feature_columns(df):
    """Returns numeric feature columns excluding ID and target columns."""
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    print(f"Using {len(feature_cols)} features for training")
    return feature_cols


# ─── SAMPLE WEIGHTS ───────────────────────────────────────────────────────────

def compute_group_weights(df):
    """
    One weight per race group (not per row — XGBRanker requirement).
    More recent seasons weighted higher. Post-regulation seasons boosted.
    """
    groups = (
        df.groupby(["year", "round"], sort=False)
        .first()
        .reset_index()[["year", "round"]]
    )

    min_year   = groups["year"].min()
    max_year   = groups["year"].max()
    year_range = max(max_year - min_year, 1)

    weights = (groups["year"] - min_year + 1) / year_range

    for reg_year in NEW_REG_YEARS:
        weights[groups["year"] >= reg_year] *= 1.5

    w_min = weights.min()
    w_max = weights.max()
    if w_max > w_min:
        weights = 0.1 + 0.9 * (weights - w_min) / (w_max - w_min)
    else:
        weights = np.ones(len(weights))

    return weights.values


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def leave_one_season_out_validation(df, feature_cols):
    """
    Leave-one-season-out cross validation.
    Train on past seasons, predict on future seasons.
    Uses INVERTED target for training.
    """
    print("\nRunning leave-one-season-out validation...")

    years     = sorted(df["year"].unique())
    val_years = [y for y in years if y >= 2020]
    results   = {}

    for val_year in val_years:
        train_df = df[df["year"] < val_year].copy().sort_values(["year","round"]).reset_index(drop=True)
        test_df  = df[df["year"] == val_year].copy().sort_values(["year","round"]).reset_index(drop=True)

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train  = train_df[feature_cols].values
        # KEY: use inverted target for training
        y_train  = invert_target(train_df[TARGET_COL].values)
        X_test   = test_df[feature_cols].values

        train_groups  = train_df.groupby(["year","round"], sort=False).size().values
        group_weights = compute_group_weights(train_df)

        model = xgb.XGBRanker(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            group=train_groups,
            sample_weight=group_weights,
            verbose=False,
        )

        # Predict — higher score = better finish = lower position number
        scores   = model.predict(X_test)
        test_df  = test_df.copy()
        test_df["pred_score"] = scores

        # HIGHER score → P1: rank descending (highest score = rank 1)
        test_df["predicted_rank"] = (
            test_df.groupby(["year","round"])["pred_score"]
            .rank(ascending=False)
            .astype(int)
        )

        # Accuracy metrics
        test_df["actual_top3"]    = (test_df[TARGET_COL] <= 3).astype(int)
        test_df["predicted_top3"] = (test_df["predicted_rank"] <= 3).astype(int)
        top3_acc = (test_df["actual_top3"] == test_df["predicted_top3"]).mean()

        winner_correct = 0
        total_races    = 0
        for (yr, rnd), grp in test_df.groupby(["year","round"]):
            aw = grp[grp[TARGET_COL]        == 1]["driver_id"].values
            pw = grp[grp["predicted_rank"]   == 1]["driver_id"].values
            if len(aw) > 0 and len(pw) > 0:
                winner_correct += int(aw[0] == pw[0])
                total_races    += 1

        winner_acc = winner_correct / total_races if total_races > 0 else 0

        results[val_year] = {
            "top3_accuracy":   round(top3_acc, 3),
            "winner_accuracy": round(winner_acc, 3),
            "races":           total_races,
        }
        print(f"  {val_year}: Winner={winner_acc:.1%}  Top-3={top3_acc:.1%}  ({total_races} races)")

    return results


# ─── TRAIN FINAL MODEL ────────────────────────────────────────────────────────

def train_final_model(df, feature_cols):
    """Trains final XGBoost model on ALL data using inverted target."""
    print("\nTraining final model on all data...")

    df = df.sort_values(["year","round"]).reset_index(drop=True)

    X       = df[feature_cols].values
    y       = invert_target(df[TARGET_COL].values)   # ← inverted target
    groups  = df.groupby(["year","round"], sort=False).size().values
    weights = compute_group_weights(df)

    model = xgb.XGBRanker(**XGBOOST_PARAMS)
    model.fit(X, y, group=groups, sample_weight=weights, verbose=False)

    print(f"  Trained on {len(df):,} rows, {len(feature_cols)} features")
    return model


# ─── PREDICT RACE OUTCOME ─────────────────────────────────────────────────────

def predict_race(model, race_features_df, feature_cols):
    """
    Predicts finishing order for all drivers in one race.

    Higher model score = better predicted finish = lower position number.
    So we rank DESCENDING (highest score → P1).

    Args:
        model            (XGBRanker):  Trained model
        race_features_df (DataFrame):  One row per driver
        feature_cols     (list):       Feature columns

    Returns:
        DataFrame: Drivers with predicted_rank (1=winner) and win_probability
    """
    X      = race_features_df[feature_cols].values
    scores = model.predict(X)

    result = race_features_df[["driver_id"]].copy().reset_index(drop=True)
    result["raw_score"] = scores

    # Softmax win probability — higher score = more likely to win
    exp_s = np.exp(scores - scores.max())
    result["win_probability"] = exp_s / exp_s.sum() * 100

    # DESCENDING rank: highest score → P1
    result["predicted_rank"] = result["raw_score"].rank(ascending=False).astype(int)

    return result.sort_values("predicted_rank").reset_index(drop=True)


# ─── SHAP FEATURE IMPORTANCE ──────────────────────────────────────────────────

def compute_shap_importance(model, df, feature_cols):
    """Computes SHAP feature importance values."""
    try:
        import shap
        print("\nComputing SHAP feature importance...")
        sample   = df.sample(min(2000, len(df)), random_state=42)
        X_sample = sample[feature_cols].values

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        importance_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        importance_df["rank"] = importance_df.index + 1

        print("\nTop 20 most important features:")
        print("-" * 52)
        max_val = importance_df["importance"].max()
        for _, row in importance_df.head(20).iterrows():
            bar = "█" * int(row["importance"] / max_val * 25)
            print(f"  {int(row['rank']):>2}. {row['feature']:<38} {bar}")

        imp_path = os.path.join(DATA_DIR, "feature_importance.csv")
        importance_df.to_csv(imp_path, index=False)
        print(f"\nSaved: {imp_path}")
        return importance_df

    except ImportError:
        print("  SHAP not installed — run: pip install shap")
        scores = model.get_booster().get_fscore()
        return pd.DataFrame(
            list(scores.items()), columns=["feature","importance"]
        ).sort_values("importance", ascending=False).reset_index(drop=True)


# ─── GENERATE ALL PREDICTIONS ─────────────────────────────────────────────────

def generate_all_predictions(model, df, feature_cols):
    """Generates predictions for every race in the dataset."""
    print("\nGenerating predictions for all races...")
    all_predictions = []

    for (year, round_num), race_group in df.groupby(["year","round"]):
        pred      = predict_race(model, race_group, feature_cols)
        pred["year"]  = year
        pred["round"] = round_num

        actual = race_group[["driver_id", TARGET_COL]].copy()
        actual.columns = ["driver_id","actual_position"]
        pred   = pred.merge(actual, on="driver_id", how="left")
        all_predictions.append(pred)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    print(f"  Generated {len(predictions_df):,} driver-race predictions")
    return predictions_df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """Full training pipeline with corrected target inversion."""
    print("\n" + "=" * 60)
    print("F1 CHAMPIONSHIP PREDICTOR — XGBOOST MODEL")
    print("=" * 60)
    print("Target: inverted finish_position (P1=21, P20=2)")
    print("        so XGBRanker ranks P1 as highest score")
    print("=" * 60)

    df           = load_dataset()
    feature_cols = get_feature_columns(df)

    # Validation
    val_results    = leave_one_season_out_validation(df, feature_cols)
    avg_winner_acc = np.mean([v["winner_accuracy"] for v in val_results.values()]) if val_results else 0
    avg_top3_acc   = np.mean([v["top3_accuracy"]   for v in val_results.values()]) if val_results else 0

    print(f"\nAverage validation accuracy:")
    print(f"  Winner prediction : {avg_winner_acc:.1%}")
    print(f"  Top-3 prediction  : {avg_top3_acc:.1%}")

    # Train final model
    model = train_final_model(df, feature_cols)

    # SHAP importance
    importance_df = compute_shap_importance(model, df, feature_cols)

    # Generate all predictions
    predictions_df = generate_all_predictions(model, df, feature_cols)

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "val_results":  val_results,
        }, f)

    # Save feature list
    with open(os.path.join(MODELS_DIR, "feature_columns.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    # Save predictions
    pred_path = os.path.join(DATA_DIR, "race_predictions.csv")
    predictions_df.to_csv(pred_path, index=False)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Features         : {len(feature_cols)}")
    print(f"  Winner accuracy  : {avg_winner_acc:.1%}")
    print(f"  Top-3 accuracy   : {avg_top3_acc:.1%}")
    print(f"  Model saved      : {model_path}")
    print(f"  Predictions      : {pred_path}")

    return model, feature_cols, predictions_df


# ─── HELPER: LOAD SAVED MODEL ─────────────────────────────────────────────────

def load_model():
    """Loads saved XGBoost model. Called by monte_carlo.py and app.py."""
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found. Run xgboost_model.py first.")
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["feature_cols"]


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, feature_cols, predictions = run_full_pipeline()