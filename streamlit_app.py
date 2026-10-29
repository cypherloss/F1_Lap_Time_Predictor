"""Interactive lap-time predictor with a per-prediction SHAP explanation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

from src.config import ENCODER_PATH, MODEL_PATH, VALID_COMPOUNDS
from src.features import ALL_FEATURE_COLS


@st.cache_resource
def load_artifacts() -> tuple[xgb.XGBRegressor, dict]:
    """Load the model and category encoders once per Streamlit process."""
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run the ingestion, cleaning, and training "
            "steps described in README.md before starting the app."
        )
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    with ENCODER_PATH.open("rb") as file:
        encoders = pickle.load(file)
    return model, encoders


def predict(
    tyre_life: int,
    fuel_proxy: float,
    compound: str,
    driver: str,
    event: str,
    model: xgb.XGBRegressor,
    encoders: dict,
) -> tuple[pd.DataFrame, float]:
    """Build the same feature contract used in training and return one prediction."""
    if compound not in VALID_COMPOUNDS:
        raise ValueError(f"Unsupported compound: {compound}")
    feats = {
        "TyreLife": tyre_life,
        "TyreLifeLog": np.log1p(tyre_life),
        "FuelProxy": fuel_proxy,
        "LapNumber": int(round((1 - fuel_proxy) * 60 + 1)),
        "Cmp_SOFT": int(compound == "SOFT"),
        "Cmp_MEDIUM": int(compound == "MEDIUM"),
        "Cmp_HARD": int(compound == "HARD"),
        "Driver_te": encoders["driver"].get(driver, encoders["global"]),
        "Event_te": encoders["event"].get(event, encoders["global"]),
    }
    features = pd.DataFrame([feats], columns=ALL_FEATURE_COLS)
    return features, float(model.predict(features)[0])


st.set_page_config(page_title="F1 Lap Time Predictor", page_icon="🏎️")
st.title("F1 Lap Time Predictor")
st.caption("Estimate flying-lap pace from tyre age, fuel proxy, compound, driver, and event.")

model, encoders = load_artifacts()
col1, col2 = st.columns(2)
with col1:
    driver = st.selectbox("Driver", sorted(encoders["driver"].index))
    event = st.selectbox("Event", sorted(encoders["event"].index))
    compound = st.selectbox("Compound", list(VALID_COMPOUNDS))
with col2:
    tyre_life = st.slider("Tyre age (laps)", 1, 40, 10)
    fuel_proxy = st.slider("Fuel load proxy (1 = full)", 0.0, 1.0, 0.6, 0.05)

if st.button("Predict lap time"):
    features, prediction = predict(
        tyre_life, fuel_proxy, compound, driver, event, model, encoders
    )
    st.metric("Predicted lap time", f"{prediction:.3f} s")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(features)
    shap.plots.waterfall(shap_values[0], show=False)
    figure = plt.gcf()
    st.pyplot(figure)
    plt.close(figure)
