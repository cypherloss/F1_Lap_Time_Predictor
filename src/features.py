"""Feature engineering shared by training, explanation, and inference."""

import numpy as np
import pandas as pd

from src.config import VALID_COMPOUNDS

FEATURE_COLS = [
    "TyreLife",
    "TyreLifeLog",
    "FuelProxy",
    "LapNumber",
    "Cmp_SOFT",
    "Cmp_MEDIUM",
    "Cmp_HARD",
]
ENCODED_COLS = ["Driver_te", "Event_te"]
ALL_FEATURE_COLS = FEATURE_COLS + ENCODED_COLS


def add_fuel_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate fuel load by normalized race lap progress."""
    df = df.copy()
    max_lap = df.groupby(["Year", "Round"])["LapNumber"].transform("max").clip(lower=1)
    df["FuelProxy"] = (1.0 - (df["LapNumber"] - 1) / max_lap).clip(0.0, 1.0)
    return df


def add_tyre_degradation(df: pd.DataFrame) -> pd.DataFrame:
    """Use a log-transformed tyre age to represent accelerating wear."""
    df = df.copy()
    df["TyreLifeLog"] = np.log1p(df["TyreLife"].clip(lower=0))
    return df


def encode_compound(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable one-hot compound columns even when a compound is absent."""
    df = df.copy()
    for compound in VALID_COMPOUNDS:
        df[f"Cmp_{compound}"] = (df["Compound"] == compound).astype("int8")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compose the reusable, deterministic feature transformations."""
    return encode_compound(add_tyre_degradation(add_fuel_proxy(df)))


def apply_target_encodings(
    df: pd.DataFrame, encoders: dict[str, pd.Series | float]
) -> pd.DataFrame:
    """Apply training-fold category means, using the global mean for unseen values."""
    result = df.copy()
    global_mean = float(encoders["global"])
    result["Driver_te"] = result["Driver"].map(encoders["driver"]).fillna(global_mean)
    result["Event_te"] = result["EventName"].map(encoders["event"]).fillna(global_mean)
    return result
