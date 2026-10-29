"""Load FastF1 race sessions and merge their lap data."""

import logging
from collections.abc import Sequence

import fastf1
import pandas as pd

from src.config import CACHE_DIR, RAW_PARQUET, ROUNDS, YEAR

logger = logging.getLogger(__name__)


def enable_cache() -> None:
    """Enable FastF1's on-disk cache before loading sessions."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_race_laps(year: int, rnd: int) -> pd.DataFrame | None:
    """Load one race; return None for a failed session so other rounds continue."""
    try:
        session = fastf1.get_session(year, rnd, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()
        laps["Year"] = year
        laps["Round"] = rnd
        laps["EventName"] = session.event["EventName"]
        return laps
    except Exception:
        logger.exception("Could not load race %s round %s", year, rnd)
        return None


def load_season(year: int = YEAR, rounds: Sequence[int] = ROUNDS) -> pd.DataFrame:
    """Load and concatenate race laps, preserving a unique row index."""
    enable_cache()
    frames: list[pd.DataFrame] = []
    for rnd in rounds:
        race_laps = load_race_laps(year, rnd)
        if race_laps is not None:
            frames.append(race_laps)
            logger.info("Loaded %s laps from %s round %s", len(race_laps), year, rnd)

    if not frames:
        raise RuntimeError("No sessions loaded; check the year, rounds, and FastF1 access.")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw = load_season()
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(RAW_PARQUET, index=False)
    logger.info("Saved %s raw laps to %s", len(raw), RAW_PARQUET)
