import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Load model, scaler, and columns
model = joblib.load(Path("app/artifacts/xgb_model.pkl"))
scaler = joblib.load(Path("app/artifacts/scaler.pkl"))

with open(Path("app/artifacts/columns.json"), "r") as f:
    model_columns = json.load(f)

def preprocess_input(data):
    # Create input dict
    base = {
        'total_sqft': data.total_sqft,
        'bath': data.bath,
        'balcony': data.balcony,
        'price_per_sqft': data.price_per_sqft
    }

    for col in model_columns:
        if col.startswith("location_"):
            base[col] = 1 if col == f"location_{data.location}" else 0

    # Ensure all model columns are in input
    for col in model_columns:
        if col not in base:
            base[col] = 0

    df = pd.DataFrame([base])
    df[['total_sqft', 'bath', 'balcony']] = scaler.transform(df[['total_sqft', 'bath', 'balcony']])
    return df[model_columns]

def predict_price(data):
    processed = preprocess_input(data)
    log_price = model.predict(processed)[0]
    predicted_price = np.expm1(log_price) / 1e5  # convert to Lakhs
    return predicted_price