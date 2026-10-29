# F1 Lap Time Predictor

A Python project for collecting Formula 1 race-lap data, preparing representative
lap records, and evaluating an XGBoost model for lap-time prediction. The
project includes a FastF1 data pipeline, race-grouped model evaluation, SHAP
explanations, and a Streamlit prediction app.

> **Current model status:** The latest recorded evaluation is a useful baseline,
> not yet a reliable lap-time predictor. Its errors remain large and its
> out-of-fold R² is negative. See [Observed evaluation](#observed-evaluation)
> before interpreting predictions.

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

## Observed evaluation

The following results are transcribed from the project's captured run with
10 races and 8,557 cleaned laps:

| Metric | Observed | PRD target | Assessment |
|---|---:|---:|---|
| Raw laps collected | 11,135 | — | Input to cleaning |
| Clean laps | 8,557 | At least 5,000 across 10+ races | Volume met; 10 races in this run |
| Out-of-fold MAE | 9.919 s | At most 0.35 s | Not met |
| Out-of-fold RMSE | 10.991 s | Not specified | High error |
| Out-of-fold R² | -0.783 | Not specified | Model underperforms a constant-mean prediction on these folds |
| Baseline MAE | 12.176 s | Model improves by at least 50% | Reference score |
| Improvement over baseline | 18.5% | At least 50% | Not met |

Fold performance varied substantially: MAE ranged from 5.007 s to 14.438 s,
and four of five fold R² scores were negative. The overall 18.5% improvement
does not offset the high absolute error. Do not treat these results as
race-strategy-grade predictions.

The evaluation holds out complete races, which is a more demanding and
realistic test of generalization to unseen events than a random row split.
Before claiming improved performance, investigate the large fold errors,
validate lap and feature alignment, and establish an appropriate comparison
strategy. Potential next steps include error analysis by race/driver/compound
and considering track-specific pace normalization. Re-run
`python -m src.evaluate` after any change and compare against this recorded
snapshot. Metrics can change when the dataset, configuration, or code changes.

## Run the app

Train the model first, then start Streamlit:

```powershell
streamlit run app/streamlit_app.py
```

The app prompts for tyre age, fuel proxy, compound, driver, and event, then
shows a predicted lap time and SHAP waterfall. Predictions are estimates from
the trained model; their accuracy is limited by the evaluation results above.

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
- PRD success metrics are goals, not guaranteed performance. The observed run
  has not met the model MAE or baseline-improvement targets.

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

## Suggested project schedule

For an intermediate data analyst working part-time, allow **2-3 weeks
(approximately 25-35 focused hours)** for a portfolio-ready version, including
analysis and iteration rather than only getting the code to run:

| Phase | Estimate | Work |
|---|---:|---|
| Scope and data acquisition | 3-4 hours | Configure FastF1, collect races, inspect coverage |
| Cleaning and data quality | 4-5 hours | Validate filters, lap counts, and missing values |
| Feature engineering and baseline | 4-5 hours | Build features and document assumptions |
| Modeling and error analysis | 6-9 hours | Grouped validation, compare folds, investigate errors |
| Explainability and Streamlit | 4-6 hours | SHAP plots and an end-to-end prediction flow |
| Tests, documentation, and polish | 4-6 hours | Run tests, capture results, prepare the portfolio walkthrough |

Downloading and inspecting data can take longer on a first run or when FastF1
access is slow. Given the current evaluation, reserve time in the modeling
phase for investigating and improving generalization; do not present the PRD
targets as achieved until a new held-out evaluation supports that conclusion.
