"""Unit tests for lap cleaning rules."""

import pandas as pd

from src.clean import (
    finalize,
    remove_first_lap,
    remove_non_green_laps,
    remove_outliers,
    remove_pit_laps,
)


def make_df() -> pd.DataFrame:
    """Build small race data containing each filter condition."""
    return pd.DataFrame({
        "Year": [2024] * 6,
        "Round": [1] * 6,
        "EventName": ["Test GP"] * 6,
        "Driver": ["VER"] * 6,
        "LapNumber": [1, 2, 3, 4, 5, 6],
        "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.Timedelta(seconds=60), pd.NaT],
        "PitOutTime": [pd.NaT] * 6,
        "TrackStatus": ["1", "1", "1", "4", "1", "1"],
        "LapTime": pd.to_timedelta([95, 90, 91, 130, 92, 120], unit="s"),
        "TyreLife": [1, 2, 3, 4, 5, 6],
        "Compound": ["SOFT"] * 6,
    })


def test_remove_pit_laps() -> None:
    assert 5 not in remove_pit_laps(make_df())["LapNumber"].values


def test_remove_non_green_laps() -> None:
    assert 4 not in remove_non_green_laps(make_df())["LapNumber"].values


def test_remove_first_lap() -> None:
    assert 1 not in remove_first_lap(make_df())["LapNumber"].values


def test_remove_outliers_uses_driver_race_best() -> None:
    assert 6 not in remove_outliers(make_df())["LapNumber"].values


def test_finalize_creates_model_schema() -> None:
    finalized = finalize(make_df())
    assert "LapTimeSeconds" in finalized.columns
    assert finalized.loc[0, "LapTimeSeconds"] == 95.0
    assert "LapTime" not in finalized.columns
