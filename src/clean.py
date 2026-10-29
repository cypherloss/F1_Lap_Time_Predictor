"""Apply the core cleaning rules to raw FastF1 lap data."""

import pandas as pd

from src.config import GREEN_FLAG_STATUS, OUTLIER_PCT, VALID_COMPOUNDS


def remove_pit_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove laps with pit entry or exit timestamps, which are not racing pace."""
    return df[df["PitInTime"].isna() & df["PitOutTime"].isna()].copy()


def remove_non_green_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep green-flag laps only so caution periods do not define normal pace."""
    return df[df["TrackStatus"].astype("string") == GREEN_FLAG_STATUS].copy()


def remove_first_lap(df: pd.DataFrame) -> pd.DataFrame:
    """Drop lap one because a standing start is not comparable to a flying lap."""
    return df[df["LapNumber"] > 1].copy()


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Keep laps within the configured percentage of each driver's race best."""
    best = df.groupby(["Year", "Round", "Driver"])["LapTime"].transform("min")
    return df[df["LapTime"].notna() & (df["LapTime"] <= best * OUTLIER_PCT)].copy()


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Apply pit, flag, first-lap, then per-driver race outlier filters."""
    df = remove_pit_laps(df)
    df = remove_non_green_laps(df)
    df = remove_first_lap(df)
    return remove_outliers(df)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert lap durations to seconds and enforce the model's clean-data schema."""
    df = df.copy()
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
    df = df[df["Compound"].isin(VALID_COMPOUNDS)]
    columns = [
        "Year",
        "Round",
        "EventName",
        "Driver",
        "LapNumber",
        "LapTimeSeconds",
        "TyreLife",
        "Compound",
        "TrackStatus",
    ]
    return df[columns].dropna(subset=["LapTimeSeconds", "TyreLife"]).reset_index(drop=True)


if __name__ == "__main__":
    import logging

    from src.config import CLEAN_PARQUET, RAW_PARQUET

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw = pd.read_parquet(RAW_PARQUET)
    clean = finalize(clean_laps(raw))
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(CLEAN_PARQUET, index=False)
    logging.info("Raw: %s -> clean: %s", len(raw), len(clean))
