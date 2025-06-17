from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PropertyInput, PredictionResponse
from app.model import predict_price

app = FastAPI(title="Real Estate Price Predictor API")

# ✅ CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://real-estate-price-predictor-fronten.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the Real Estate Price Predictor API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PropertyInput):
    pred_price_lakhs, pred_ppsqft = predict_price(data)
    return {
        "predicted_price_lakhs": round(pred_price_lakhs, 2),
        "predicted_price_per_sqft_inr": round(pred_ppsqft, 2)
    }
