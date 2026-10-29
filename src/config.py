"""Central configuration for paths, data scope, and model settings."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "f1_cache"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_PARQUET = RAW_DIR / "f1_raw_laps.parquet"
CLEAN_PARQUET = PROCESSED_DIR / "f1_lap_times_clean.parquet"
MODEL_PATH = PROCESSED_DIR / "xgb_lap_model.json"
ENCODER_PATH = PROCESSED_DIR / "encoders.pkl"

YEAR = 2024
ROUNDS = tuple(range(1, 11))

OUTLIER_PCT = 1.07
GREEN_FLAG_STATUS = "1"
VALID_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

RANDOM_SEED = 42
TEST_FOLDS = 5
