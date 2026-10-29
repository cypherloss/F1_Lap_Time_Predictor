# F1 Lap Time Predictor

A Python project for collecting Formula 1 race-lap data, preparing representative
lap records, and evaluating an XGBoost model for lap-time prediction. The
project includes a FastF1 data pipeline, race-grouped model evaluation, SHAP
explanations, and a Streamlit prediction app.

## What the project does

- Downloads race-session laps with FastF1 and caches them locally.
- Removes pit, non-green-flag, first-lap, and unusually slow laps.
- Engineers tyre-age, race-progress (fuel proxy), compound, driver, and event
  features.
- Evaluates an XGBoost regressor using race-grouped cross-validation.
- Compares the model with a per-driver/per-event median baseline.
- Provides a Streamlit interface with per-prediction SHAP explanations.

## Requirements and setup

Use Python 3.10 or newer. From the project root, create an environment and
install the pinned dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

FastF1 needs network access the first time data is downloaded. Later runs use
the local `f1_cache` directory. The configured season and rounds are in
[`src/config.py`](src/config.py).

## Run the pipeline

Run commands from the project root:

```powershell
python -m src.ingest
python -m src.clean
python -m src.evaluate
python -m src.train
```

The commands create:

| Output | Purpose |
|---|---|
| `data/raw/f1_raw_laps.parquet` | Raw laps downloaded from FastF1 |
| `data/processed/f1_lap_times_clean.parquet` | Filtered laps used for analysis |
| `data/processed/xgb_lap_model.json` | Trained XGBoost model |
| `data/processed/encoders.pkl` | Driver and event target-encoding values |

Evaluation can be run independently after cleaning:

```powershell
python -m src.evaluate
```

Save a machine-readable report or evaluate another clean parquet file with:

```powershell
python -m src.evaluate --output data/processed/evaluation.json
python -m src.evaluate --data PATH_TO_CLEAN_PARQUET
```

The evaluator fits fold models and scores only held-out races; it does not
report in-sample performance from the saved model. Training also runs grouped
cross-validation before fitting and saving the final model, so running both
commands calculates evaluation metrics twice.

## Run the app

Train the model first, then start Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

The app prompts for tyre age, fuel proxy, compound, driver, and event, then
shows a predicted lap time and SHAP waterfall.

## Methodology and limitations

- Validation groups use a `Year-Round` race identifier. Fold count is reduced
  when fewer than the configured five races are available; at least three
  races are required by the nested training-encoding workflow.
- Driver and event target encodings are fitted using training races only.
  Training-row encodings are cross-fitted so a row's target does not encode
  itself.
- The median baseline is evaluated on held-out races, with unseen categories
  falling back to the training global median.
- `FuelProxy` represents normalized race progress; it is not measured fuel
  load or telemetry.
- Cleaning retains the configured compounds and removes pit, non-green,
  first-lap, and per-driver/per-race outlier laps. Review those assumptions
  when extending the dataset to other seasons or conditions.

## Tests

Run the unit tests from the project root:

```powershell
python -m pytest
```

The tests cover cleaning rules, final data schema, feature construction,
cross-fitted target encoding, and grouped evaluation.

## Project structure

```text
src/                 Ingestion, cleaning, features, training, evaluation, SHAP
app/                 Streamlit prediction interface
tests/               Unit tests
notebooks/            Walkthrough notebook
data/raw/             Downloaded raw-lap parquet
data/processed/       Clean data and model artifacts
f1_cache/             FastF1 on-disk cache
```