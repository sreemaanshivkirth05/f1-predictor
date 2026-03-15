"""
xgboost_model.py
----------------
Trains an XGBoost model to predict F1 race finishing position
probabilities for each driver at each race.

What this file does:
  1. Loads master_dataset.csv
  2. Trains XGBoost using leave-one-season-out validation
  3. Outputs finishing position probability distributions
  4. Saves the trained model to models/xgboost_model.pkl
  5. Computes SHAP feature importance values
  6. Saves predictions to data/race_predictions.csv

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

# Seasons where regulations changed — get extra weight in training
NEW_REG_YEARS = [2022, 2026]


# ─── LOAD DATA ────────────────────────────────────────────────────────────────

def load_dataset():
    """Loads master_dataset.csv built by build_dataset.py."""
    path = os.path.join(DATA_DIR, "master_dataset.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"master_dataset.csv not found at {path}\n"
            f"Please run build_dataset.py first."
        )

    df = pd.read_csv(path)
    print(f"Loaded master dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"Seasons : {sorted(df['year'].unique())}")
    print(f"Drivers : {df['driver_id'].nunique()}")
    return df


# ─── FEATURE SELECTION ────────────────────────────────────────────────────────

def get_feature_columns(df):
    """
    Returns numeric columns to use as model features.
    Excludes ID columns, target, and non-numeric columns.
    """
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    print(f"Using {len(feature_cols)} features for training")
    return feature_cols


# ─── SAMPLE WEIGHTS (PER GROUP) ───────────────────────────────────────────────

def compute_group_weights(df):
    """
    Computes one sample weight per race group (not per row).

    XGBRanker with group= requires weights to be per-group, not per-row.
    More recent seasons get higher weight. Post-regulation-change seasons
    get an extra boost since old car data is less relevant.

    Args:
        df (pd.DataFrame): Training data — must have year and round columns

    Returns:
        np.array: One weight value per race group, in the same order
                  as df.groupby(['year','round'])
    """
    # Get one row per race group, in order
    groups = (
        df.groupby(["year", "round"], sort=False)
        .first()
        .reset_index()[["year", "round"]]
    )

    min_year   = groups["year"].min()
    max_year   = groups["year"].max()
    year_range = max(max_year - min_year, 1)

    # Base weight: linearly increases with year
    weights = (groups["year"] - min_year + 1) / year_range

    # Boost post-regulation-change seasons
    for reg_year in NEW_REG_YEARS:
        weights[groups["year"] >= reg_year] *= 1.5

    # Normalise to [0.1, 1.0]
    w_min = weights.min()
    w_max = weights.max()
    if w_max > w_min:
        weights = 0.1 + 0.9 * (weights - w_min) / (w_max - w_min)
    else:
        weights = np.ones(len(weights))

    return weights.values


# ─── LEAVE ONE SEASON OUT VALIDATION ─────────────────────────────────────────

def leave_one_season_out_validation(df, feature_cols):
    """
    Validates the model using leave-one-season-out cross validation.

    For each season from 2020 onwards:
      - Train on all PREVIOUS seasons
      - Predict on this season
      - Measure winner and top-3 accuracy

    This is the correct approach for time-series — we never train on
    future data to predict the past.

    Args:
        df           (pd.DataFrame): Master dataset
        feature_cols (list):         Feature columns

    Returns:
        dict: Validation accuracy per season
    """
    print("\nRunning leave-one-season-out validation...")
    print("(Training on past seasons, predicting future seasons)")

    years     = sorted(df["year"].unique())
    val_years = [y for y in years if y >= 2020]
    results   = {}

    for val_year in val_years:
        train_df = df[df["year"] < val_year].copy()
        test_df  = df[df["year"] == val_year].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Sort both sets by year then round — important for group ordering
        train_df = train_df.sort_values(["year", "round"]).reset_index(drop=True)
        test_df  = test_df.sort_values(["year", "round"]).reset_index(drop=True)

        X_train = train_df[feature_cols].values
        y_train = train_df[TARGET_COL].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[TARGET_COL].values

        # Group sizes — number of drivers per race
        train_groups = train_df.groupby(["year", "round"], sort=False).size().values
        test_groups  = test_df.groupby(["year", "round"],  sort=False).size().values

        # One weight per race group (not per row)
        group_weights = compute_group_weights(train_df)

        model = xgb.XGBRanker(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            group=train_groups,
            sample_weight=group_weights,
            verbose=False,
        )

        # Predict scores
        test_scores = model.predict(X_test)
        test_df     = test_df.copy()
        test_df["predicted_score"] = test_scores

        # Convert scores to predicted rank per race
        test_df["predicted_rank"] = (
            test_df.groupby(["year", "round"])["predicted_score"]
            .rank(ascending=False)
            .astype(int)
        )

        # Top-3 accuracy
        test_df["actual_top3"]    = (test_df[TARGET_COL] <= 3).astype(int)
        test_df["predicted_top3"] = (test_df["predicted_rank"] <= 3).astype(int)
        top3_accuracy = (test_df["actual_top3"] == test_df["predicted_top3"]).mean()

        # Winner accuracy per race
        winner_correct = 0
        total_races    = 0

        for (yr, rnd), race_group in test_df.groupby(["year", "round"]):
            actual_winner    = race_group.loc[race_group[TARGET_COL] == 1,    "driver_id"].values
            predicted_winner = race_group.loc[race_group["predicted_rank"] == 1, "driver_id"].values

            if len(actual_winner) > 0 and len(predicted_winner) > 0:
                if actual_winner[0] == predicted_winner[0]:
                    winner_correct += 1
                total_races += 1

        winner_accuracy = winner_correct / total_races if total_races > 0 else 0

        results[val_year] = {
            "top3_accuracy":   round(top3_accuracy, 3),
            "winner_accuracy": round(winner_accuracy, 3),
            "races":           total_races,
        }

        print(f"  {val_year}: Winner={winner_accuracy:.1%}  "
              f"Top-3={top3_accuracy:.1%}  "
              f"({total_races} races)")

    return results


# ─── TRAIN FINAL MODEL ────────────────────────────────────────────────────────

def train_final_model(df, feature_cols):
    """
    Trains the final XGBoost model on ALL available data.
    This model is used for future race predictions.

    Args:
        df           (pd.DataFrame): Full master dataset
        feature_cols (list):         Feature columns

    Returns:
        xgb.XGBRanker: Trained model
    """
    print("\nTraining final model on all data...")

    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    X      = df[feature_cols].values
    y      = df[TARGET_COL].values
    groups = df.groupby(["year", "round"], sort=False).size().values
    weights = compute_group_weights(df)

    model = xgb.XGBRanker(**XGBOOST_PARAMS)
    model.fit(
        X, y,
        group=groups,
        sample_weight=weights,
        verbose=False,
    )

    print(f"  Trained on {len(df):,} rows, {len(feature_cols)} features")
    return model


# ─── PREDICT RACE OUTCOME ─────────────────────────────────────────────────────

def predict_race(model, race_features_df, feature_cols):
    """
    Predicts finishing probabilities for all drivers in one race.

    The model outputs a raw score per driver. We convert these to
    win probability using softmax normalisation.

    Args:
        model            (xgb.XGBRanker): Trained model
        race_features_df (pd.DataFrame):  One row per driver
        feature_cols     (list):          Feature columns

    Returns:
        pd.DataFrame: Drivers with predicted rank and win probability
    """
    X      = race_features_df[feature_cols].values
    scores = model.predict(X)

    result = race_features_df[["driver_id"]].copy().reset_index(drop=True)
    result["raw_score"] = scores

    # Softmax win probability
    exp_scores = np.exp(scores - scores.max())
    result["win_probability"] = exp_scores / exp_scores.sum()

    # Predicted rank
    result["predicted_rank"] = result["raw_score"].rank(ascending=False).astype(int)

    result = result.sort_values("predicted_rank").reset_index(drop=True)
    return result


# ─── SHAP FEATURE IMPORTANCE ──────────────────────────────────────────────────

def compute_shap_importance(model, df, feature_cols):
    """
    Computes SHAP feature importance — explains which features
    drove the model's predictions most strongly.

    Args:
        model        (xgb.XGBRanker): Trained model
        df           (pd.DataFrame):  Dataset
        feature_cols (list):          Feature columns

    Returns:
        pd.DataFrame: Features ranked by importance
    """
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
        print("-" * 50)
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
        print("  Using XGBoost built-in importance instead...")

        scores = model.get_booster().get_fscore()
        importance_df = pd.DataFrame(
            list(scores.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return importance_df


# ─── GENERATE ALL PREDICTIONS ─────────────────────────────────────────────────

def generate_all_predictions(model, df, feature_cols):
    """
    Generates predictions for every race in the dataset.
    Used for backtesting and dashboard display.

    Args:
        model        (xgb.XGBRanker): Trained model
        df           (pd.DataFrame):  Full dataset
        feature_cols (list):          Feature columns

    Returns:
        pd.DataFrame: Predictions for all driver-race combinations
    """
    print("\nGenerating predictions for all races...")

    all_predictions = []

    for (year, round_num), race_group in df.groupby(["year", "round"]):
        pred      = predict_race(model, race_group, feature_cols)
        pred["year"]  = year
        pred["round"] = round_num

        actual    = race_group[["driver_id", TARGET_COL]].copy()
        actual.columns = ["driver_id", "actual_position"]
        pred = pred.merge(actual, on="driver_id", how="left")

        all_predictions.append(pred)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    print(f"  Generated {len(predictions_df):,} driver-race predictions")
    return predictions_df


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run_full_pipeline():
    """
    Full training pipeline:
      1. Load data
      2. Select features
      3. Validate with leave-one-season-out CV
      4. Train final model on all data
      5. Compute SHAP feature importance
      6. Generate all predictions
      7. Save model, features, and predictions
    """
    print("\n" + "=" * 60)
    print("F1 CHAMPIONSHIP PREDICTOR — XGBOOST MODEL")
    print("=" * 60)

    df           = load_dataset()
    feature_cols = get_feature_columns(df)

    # Validation
    val_results    = leave_one_season_out_validation(df, feature_cols)
    avg_winner_acc = np.mean([v["winner_accuracy"] for v in val_results.values()])
    avg_top3_acc   = np.mean([v["top3_accuracy"]   for v in val_results.values()])

    print(f"\nAverage validation accuracy:")
    print(f"  Winner prediction : {avg_winner_acc:.1%}")
    print(f"  Top-3 prediction  : {avg_top3_acc:.1%}")

    # Train final model
    model = train_final_model(df, feature_cols)

    # SHAP importance
    importance_df = compute_shap_importance(model, df, feature_cols)

    # Generate predictions
    predictions_df = generate_all_predictions(model, df, feature_cols)

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        model,
            "feature_cols": feature_cols,
            "val_results":  val_results,
        }, f)
    print(f"\nModel saved   : {model_path}")

    # Save feature list
    feat_path = os.path.join(MODELS_DIR, "feature_columns.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"Features saved: {feat_path}")

    # Save predictions
    pred_path = os.path.join(DATA_DIR, "race_predictions.csv")
    predictions_df.to_csv(pred_path, index=False)
    print(f"Predictions   : {pred_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Features         : {len(feature_cols)}")
    print(f"  Winner accuracy  : {avg_winner_acc:.1%}")
    print(f"  Top-3 accuracy   : {avg_top3_acc:.1%}")

    return model, feature_cols, predictions_df


# ─── HELPER: LOAD SAVED MODEL ─────────────────────────────────────────────────

def load_model():
    """
    Loads the saved XGBoost model from disk.
    Called by monte_carlo.py and the Streamlit app.

    Returns:
        tuple: (model, feature_cols)
    """
    model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Please run xgboost_model.py first."
        )

    with open(model_path, "rb") as f:
        saved = pickle.load(f)

    return saved["model"], saved["feature_cols"]


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, feature_cols, predictions = run_full_pipeline()