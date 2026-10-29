"""Tests for leakage-safe training utilities."""

import pandas as pd

from src.train import _cross_fitted_training_features, _target_means


def test_target_means_fall_back_to_global_for_unseen_categories() -> None:
    train = pd.DataFrame({
        "Driver": ["A", "A"],
        "EventName": ["X", "X"],
        "LapTimeSeconds": [90.0, 100.0],
    })
    means = _target_means(train)
    assert means["global"] == 95.0
    assert means["driver"]["A"] == 95.0


def test_cross_fitted_encodings_do_not_use_own_race_targets() -> None:
    frame = pd.DataFrame({
        "Driver": ["A", "A", "A", "A"],
        "EventName": ["X", "X", "X", "X"],
        "LapTimeSeconds": [10.0, 20.0, 30.0, 40.0],
    })
    groups = pd.Series(["r1", "r2", "r3", "r4"])
    encoded = _cross_fitted_training_features(frame, groups, n_splits=2)
    assert encoded["Driver_te"].tolist() == [30.0, 20.0, 30.0, 20.0]
    for index in range(len(frame)):
        changed = frame.copy()
        changed.loc[index, "LapTimeSeconds"] += 100
        changed_encoding = _cross_fitted_training_features(changed, groups, n_splits=2)
        assert changed_encoding.loc[index, "Driver_te"] == encoded.loc[index, "Driver_te"]
        assert changed_encoding.loc[index, "Event_te"] == encoded.loc[index, "Event_te"]
