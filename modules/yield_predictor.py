
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import YIELD_MODEL_PATH


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_yield_model():
    """
    Load the saved Weather XGBoost pipeline.
    No model training is performed here.
    """

    model_path = Path(YIELD_MODEL_PATH)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Yield model not found: {model_path}"
        )

    artifact = joblib.load(model_path)

    required_keys = ["model", "features", "target"]

    for key in required_keys:
        if key not in artifact:
            raise ValueError(
                f"Invalid model artifact: missing '{key}'"
            )

    return artifact


# ==========================================
# 2. VALIDATE INPUT DATA
# ==========================================

def validate_yield_inputs(input_data, features):
    """
    Validate all features required by the trained model.
    """

    if not isinstance(input_data, dict):
        raise TypeError("Input must be a dictionary.")

    missing_features = [
        feature
        for feature in features
        if feature not in input_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing model inputs: {missing_features}"
        )

    if not str(input_data["district"]).strip():
        raise ValueError("District cannot be empty.")

    numeric_features = [
        feature
        for feature in features
        if feature != "district"
    ]

    cleaned_data = {
        "district": str(input_data["district"]).strip()
    }

    for feature in numeric_features:

        try:
            value = float(input_data[feature])
        except (TypeError, ValueError):
            raise ValueError(
                f"{feature} must be numeric."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{feature} must be a finite number."
            )

        cleaned_data[feature] = value

    if cleaned_data["year_start"] < 1900:
        raise ValueError("Invalid season start year.")

    if not cleaned_data["year_start"].is_integer():
        raise ValueError("Season start year must be a whole number.")

    if cleaned_data["year_start"] > 2100:
        raise ValueError("Season start year is outside the supported range.")

    non_negative_features = [
        "yield_lag_1",
        "yield_lag_2",
        "historical_avg_yield",
        "rainfall_mm",
        "humidity_pct"
    ]

    for feature in non_negative_features:
        if feature in cleaned_data and cleaned_data[feature] < 0:
            raise ValueError(
                f"{feature} cannot be negative."
            )

    if "humidity_pct" in cleaned_data:
        if cleaned_data["humidity_pct"] > 100:
            raise ValueError(
                "Humidity must be between 0 and 100."
            )

    return cleaned_data


# ==========================================
# 3. PREDICT WHEAT YIELD
# ==========================================

def predict_wheat_yield(input_data):
    """
    Predict wheat yield using the saved Weather XGBoost model.

    Returns yield in tonnes per hectare.
    """

    artifact = load_yield_model()

    model = artifact["model"]
    features = artifact["features"]

    cleaned_data = validate_yield_inputs(
        input_data,
        features
    )

    input_df = pd.DataFrame(
        [cleaned_data],
        columns=features
    )

    prediction = float(
        model.predict(input_df)[0]
    )

    if not np.isfinite(prediction):
        raise ValueError(
            "Model returned an invalid prediction."
        )

    if prediction < 0:
        raise ValueError(
            "Model returned a negative yield estimate. "
            "Please verify the inputs."
        )

    return {
        "predicted_yield_tonnes_per_hectare": round(
            prediction, 3
        ),
        "district": cleaned_data["district"],
        "season_start_year": int(
            cleaned_data["year_start"]
        ),
        "model_name": artifact.get(
            "model_name", "Weather XGBoost"
        ),
        "prediction_unit": "tonnes/hectare"
    }


# ==========================================
# 4. CALCULATE FARM PRODUCTION
# ==========================================

def calculate_total_production(
    yield_per_hectare,
    area_acres
):
    """
    Convert farm acreage into hectares and estimate production.
    """

    yield_per_hectare = float(yield_per_hectare)
    area_acres = float(area_acres)

    if not np.isfinite(yield_per_hectare):
        raise ValueError("Yield must be a valid number.")

    if not np.isfinite(area_acres):
        raise ValueError("Farm area must be a valid number.")

    if yield_per_hectare < 0:
        raise ValueError("Yield cannot be negative.")

    if area_acres <= 0:
        raise ValueError("Farm area must be greater than zero.")

    # 1 acre = 0.404686 hectares
    area_hectares = area_acres * 0.404686

    production_tonnes = (
        yield_per_hectare * area_hectares
    )

    return {
        "farm_area_acres": round(area_acres, 2),
        "farm_area_hectares": round(
            area_hectares, 3
        ),
        "estimated_production_tonnes": round(
            production_tonnes, 3
        ),
        "estimated_production_kg": round(
            production_tonnes * 1000, 2
        )
    }


# ==========================================
# 5. MODEL INFORMATION
# ==========================================

def get_yield_model_info():

    artifact = load_yield_model()

    return {
        "model_name": artifact.get(
            "model_name", "Weather XGBoost"
        ),
        "features": artifact["features"],
        "target": artifact["target"],
        "training_rows": artifact.get(
            "training_rows", "Not available"
        ),
        "training_year_min": artifact.get(
            "training_year_min", "Not available"
        ),
        "training_year_max": artifact.get(
            "training_year_max", "Not available"
        )
    }
