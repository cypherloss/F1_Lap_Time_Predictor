"""SHAP explanations for lap-time model predictions."""

import pandas as pd
import shap
import xgboost as xgb

from src.features import ALL_FEATURE_COLS, apply_target_encodings, build_features


def generate_shap(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    encoders: dict[str, pd.Series | float],
    sample_n: int = 500,
) -> tuple[shap.TreeExplainer, shap.Explanation, pd.DataFrame]:
    """Return a fitted tree explainer, SHAP values, and their named feature rows."""
    if df.empty:
        raise ValueError("Cannot explain an empty DataFrame.")
    engineered = build_features(df)
    prepared = apply_target_encodings(engineered, encoders)
    sample = prepared.sample(min(sample_n, len(prepared)), random_state=42)
    feature_frame = sample[ALL_FEATURE_COLS]
    explainer = shap.TreeExplainer(model)
    values = explainer(feature_frame)
    return explainer, values, feature_frame


def plot_waterfall(
    explainer: shap.TreeExplainer,
    row: pd.Series,
    feature_cols: list[str] | None = None,
) -> None:
    """Render a waterfall plot for one already-engineered feature row."""
    columns = feature_cols or ALL_FEATURE_COLS
    values = explainer(row[columns].to_frame().T)
    shap.plots.waterfall(values[0], show=False)


def plot_summary(values: shap.Explanation) -> None:
    """Render the global SHAP feature-importance summary."""
    shap.plots.beeswarm(values, show=False)
