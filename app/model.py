import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# ✅ Suppress XGBoost version mismatch warning during pickle load
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*If you are loading a serialized model.*"
    )

    # 🔁 Load model and preprocessors
    ARTIFACT_DIR = Path("app/artifacts")
    model = joblib.load(ARTIFACT_DIR / "price_model.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "scaler.pkl")
    le_loc = joblib.load(ARTIFACT_DIR / "location_encoder.pkl")

# 🔁 Load location-to-avg-pps map
with open(ARTIFACT_DIR / "loc_avg_pps.json", "r") as f:
    loc_pps_mean = json.load(f)

overall_pps_mean = np.mean(list(loc_pps_mean.values()))

def preprocess_input(data):
    """
    Preprocess user input for prediction:
    - Encode location
    - Map loc_avg_pps
    - Scale features
    """
    if data.location in le_loc.classes_:
        loc_encoded = le_loc.transform([data.location])[0]
    else:
        loc_encoded = le_loc.transform(['other'])[0]

    loc_avg_pps = loc_pps_mean.get(data.location, overall_pps_mean)

    input_row = pd.DataFrame([{
        'total_sqft': data.total_sqft,
        'bath': data.bath,
        'balcony': data.balcony,
        'bhk': data.bhk,
        'location_encoded': loc_encoded,
        'loc_avg_pps': loc_avg_pps
    }])

    input_scaled = scaler.transform(input_row)
    return input_scaled

def predict_price(data):
    """
    Make a prediction and return price in Lakhs + price per sqft.
    """
    processed = preprocess_input(data)
    log_price = model.predict(processed)[0]
    pred_price_lakhs = np.expm1(log_price)
    pred_ppsqft = (pred_price_lakhs * 1e5) / data.total_sqft
    return pred_price_lakhs, pred_ppsqft
