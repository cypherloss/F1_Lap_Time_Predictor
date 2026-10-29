"""Unit tests for feature engineering."""

import numpy as np
import pandas as pd

from src.features import ALL_FEATURE_COLS, build_features


def make_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Year": [2024, 2024],
        "Round": [1, 1],
        "LapNumber": [1, 20],
        "TyreLife": [1.0, 10.0],
        "Compound": ["SOFT", "HARD"],
    })


def test_build_features_has_stable_columns_and_values() -> None:
    features = build_features(make_df())
    assert np.allclose(features["FuelProxy"], [1.0, 0.05])
    assert np.allclose(features["TyreLifeLog"], np.log1p([1.0, 10.0]))
    assert features["Cmp_SOFT"].tolist() == [1, 0]
    assert features["Cmp_MEDIUM"].tolist() == [0, 0]
    assert features["Cmp_HARD"].tolist() == [0, 1]
    assert all(column in features for column in ALL_FEATURE_COLS[:7])


def test_build_features_keeps_compound_columns_when_compound_is_absent() -> None:
    features = build_features(make_df().iloc[:1])
    assert {"Cmp_SOFT", "Cmp_MEDIUM", "Cmp_HARD"}.issubset(features.columns)
