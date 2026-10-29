"""Tests for race-grouped model evaluation."""

import numpy as np
import pandas as pd
import pytest

from src import train


def test_evaluate_model_cv_returns_out_of_fold_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class MeanRegressor:
        def fit(self, features: pd.DataFrame, target: pd.Series, **kwargs: object) -> None:
            self.mean = float(target.mean())

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.full(len(features), self.mean)

    monkeypatch.setattr(train, "_new_model", lambda: MeanRegressor())
    frame = pd.DataFrame([
        {
            "Year": 2024,
            "Round": round_number,
            "EventName": f"Event {round_number}",
            "Driver": driver,
            "LapNumber": 2,
            "LapTimeSeconds": 90.0 + round_number + driver_number,
            "TyreLife": 2.0,
            "Compound": "SOFT",
        }
        for round_number in range(1, 4)
        for driver_number, driver in enumerate(("A", "B"))
    ])

    results = train.evaluate_model_cv(frame)

    assert results["races"] == 3
    assert len(results["folds"]) == 3
    assert results["rows"] == len(frame)
    assert results["mae"] > 0
    assert results["rmse"] > 0
    assert np.isfinite(results["r2"])
    assert results["baseline_mae"] > 0
    assert np.isfinite(results["improvement_pct"])


def test_evaluate_model_cv_requires_three_races() -> None:
    frame = pd.DataFrame({
        "Year": [2024, 2024],
        "Round": [1, 2],
        "EventName": ["A", "B"],
        "Driver": ["A", "A"],
        "LapNumber": [2, 2],
        "LapTimeSeconds": [90.0, 91.0],
        "TyreLife": [2.0, 2.0],
        "Compound": ["SOFT", "SOFT"],
    })

    with pytest.raises(ValueError, match="At least three races"):
        train.evaluate_model_cv(frame)
