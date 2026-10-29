"""Train and evaluate an XGBoost lap-time model using race-grouped validation."""

import logging
import pickle
from typing import TypedDict

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from src.config import ENCODER_PATH, MODEL_PATH, RANDOM_SEED, TEST_FOLDS
from src.features import ALL_FEATURE_COLS, build_features

logger = logging.getLogger(__name__)


class FoldMetrics(TypedDict):
    """Metrics collected for one race-held-out fold."""

    fold: int
    mae: float
    rmse: float
    r2: float
    train_races: int
    validation_races: int


class EvaluationMetrics(TypedDict):
    """Aggregate out-of-fold metrics and the baseline comparison."""

    folds: list[FoldMetrics]
    mae: float
    rmse: float
    r2: float
    baseline_mae: float
    improvement_pct: float
    rows: int
    races: int


def _target_means(df: pd.DataFrame) -> dict[str, pd.Series | float]:
    """Fit category means on a training partition only."""
    return {
        "driver": df.groupby("Driver")["LapTimeSeconds"].mean(),
        "event": df.groupby("EventName")["LapTimeSeconds"].mean(),
        "global": float(df["LapTimeSeconds"].mean()),
    }


def _apply_means(
    df: pd.DataFrame, means: dict[str, pd.Series | float]
) -> pd.DataFrame:
    """Apply fitted category means with a global-mean fallback."""
    result = df.copy()
    global_mean = float(means["global"])
    result["Driver_te"] = result["Driver"].map(means["driver"]).fillna(global_mean)
    result["Event_te"] = result["EventName"].map(means["event"]).fillna(global_mean)
    return result


def _cross_fitted_training_features(
    df: pd.DataFrame, groups: pd.Series, n_splits: int
) -> pd.DataFrame:
    """Generate training encodings without allowing a row to encode its own target."""
    encoded = df.copy()
    encoded["Driver_te"] = np.nan
    encoded["Event_te"] = np.nan
    splitter = GroupKFold(n_splits=n_splits)
    for fit_idx, holdout_idx in splitter.split(df, groups=groups):
        means = _target_means(df.iloc[fit_idx])
        holdout = _apply_means(df.iloc[holdout_idx], means)
        encoded.loc[holdout_idx, ["Driver_te", "Event_te"]] = holdout[
            ["Driver_te", "Event_te"]
        ].to_numpy()
    if encoded[["Driver_te", "Event_te"]].isna().any().any():
        raise RuntimeError("Cross-fitted target encoding left missing values.")
    return encoded


def _race_groups(df: pd.DataFrame) -> pd.Series:
    """Build race IDs that remain unique if the data includes multiple seasons."""
    return df["Year"].astype(str) + "-" + df["Round"].astype(str)


def baseline_mae(df: pd.DataFrame, groups: pd.Series | None = None) -> float:
    """Evaluate the per-driver/event median baseline using held-out races only."""
    if groups is None:
        groups = _race_groups(df)
    n_groups = groups.nunique()
    if n_groups < 2:
        raise ValueError("At least two races are required to evaluate the baseline.")
    splitter = GroupKFold(n_splits=min(TEST_FOLDS, n_groups))
    predictions = np.empty(len(df), dtype=float)
    for fit_idx, holdout_idx in splitter.split(df, groups=groups):
        train = df.iloc[fit_idx]
        holdout = df.iloc[holdout_idx]
        global_median = float(train["LapTimeSeconds"].median())
        category_medians = train.groupby(["Driver", "EventName"])["LapTimeSeconds"].median()
        predictions[holdout_idx] = [
            float(category_medians.get((driver, event), global_median))
            for driver, event in zip(holdout["Driver"], holdout["EventName"])
        ]
    return float(mean_absolute_error(df["LapTimeSeconds"], predictions))


def _new_model() -> xgb.XGBRegressor:
    """Create the consistently configured regressor."""
    options = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_SEED,
        "objective": "reg:squarederror",
        "n_jobs": -1,
    }
    return xgb.XGBRegressor(**options)


def evaluate_model_cv(df: pd.DataFrame) -> EvaluationMetrics:
    """Evaluate the training workflow with out-of-fold predictions grouped by race."""
    if df.empty:
        raise ValueError("Cannot evaluate an empty DataFrame.")
    required = {
        "Year", "Round", "EventName", "Driver", "LapNumber",
        "LapTimeSeconds", "TyreLife", "Compound",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {sorted(missing)}")

    df = build_features(df).reset_index(drop=True)
    groups = _race_groups(df).reset_index(drop=True)
    n_groups = groups.nunique()
    if n_groups < 3:
        raise ValueError("At least three races are required for grouped validation.")
    outer = GroupKFold(n_splits=min(TEST_FOLDS, n_groups))

    fold_results: list[FoldMetrics] = []
    predictions = np.empty(len(df), dtype=float)
    for fold, (train_idx, valid_idx) in enumerate(outer.split(df, groups=groups), start=1):
        train = df.iloc[train_idx].reset_index(drop=True)
        valid = df.iloc[valid_idx].copy()
        train_groups = groups.iloc[train_idx].reset_index(drop=True)
        inner_splits = min(TEST_FOLDS, train_groups.nunique())
        train_encoded = _cross_fitted_training_features(train, train_groups, inner_splits)
        means = _target_means(train)
        valid_encoded = _apply_means(valid, means)

        # Keep the outer race fold untouched for evaluation; don't use it for early stopping.
        model = _new_model()
        model.fit(
            train_encoded[ALL_FEATURE_COLS],
            train_encoded["LapTimeSeconds"],
            verbose=False,
        )
        fold_predictions = model.predict(valid_encoded[ALL_FEATURE_COLS])
        predictions[valid_idx] = fold_predictions
        actual = valid_encoded["LapTimeSeconds"]
        mae = mean_absolute_error(actual, fold_predictions)
        rmse = np.sqrt(mean_squared_error(actual, fold_predictions))
        r2 = r2_score(actual, fold_predictions)
        fold_results.append({
            "fold": fold,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "train_races": int(train_groups.nunique()),
            "validation_races": int(groups.iloc[valid_idx].nunique()),
        })

    y_true = df["LapTimeSeconds"]
    cv_mae = float(mean_absolute_error(y_true, predictions))
    cv_rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    cv_r2 = float(r2_score(y_true, predictions))
    baseline = baseline_mae(df, groups)
    improvement = (1 - cv_mae / baseline) * 100 if baseline else float("nan")
    return {
        "folds": fold_results,
        "mae": cv_mae,
        "rmse": cv_rmse,
        "r2": cv_r2,
        "baseline_mae": baseline,
        "improvement_pct": float(improvement),
        "rows": int(len(df)),
        "races": int(n_groups),
    }


def train_model(df: pd.DataFrame) -> xgb.XGBRegressor:
    """Run race-grouped CV, report metrics, then save a full-data deployment model."""
    evaluation = evaluate_model_cv(df)
    for result in evaluation["folds"]:
        logger.info(
            "Fold %s: MAE=%.3fs RMSE=%.3fs R2=%.3f",
            result["fold"], result["mae"], result["rmse"], result["r2"],
        )
    logger.info(
        "Out-of-fold MAE=%.3fs RMSE=%.3fs R2=%.3f",
        evaluation["mae"], evaluation["rmse"], evaluation["r2"],
    )
    logger.info("Leakage-safe baseline MAE: %.3f s", evaluation["baseline_mae"])
    logger.info("Improvement over baseline: %.1f%%", evaluation["improvement_pct"])

    df = build_features(df).reset_index(drop=True)
    groups = _race_groups(df).reset_index(drop=True)
    n_groups = groups.nunique()
    final_means = _target_means(df)
    final_training = _cross_fitted_training_features(
        df, groups, min(TEST_FOLDS, n_groups)
    )
    final = _new_model()
    final.fit(final_training[ALL_FEATURE_COLS], final_training["LapTimeSeconds"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(MODEL_PATH))
    with ENCODER_PATH.open("wb") as file:
        pickle.dump(final_means, file)
    logger.info("Saved model to %s and encoders to %s", MODEL_PATH, ENCODER_PATH)
    return final


if __name__ == "__main__":
    import logging

    from src.config import CLEAN_PARQUET

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_model(pd.read_parquet(CLEAN_PARQUET))
