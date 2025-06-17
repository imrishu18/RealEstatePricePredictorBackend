from pydantic import BaseModel

class PropertyInput(BaseModel):
    total_sqft: float
    bath: int
    balcony: int
    bhk: int
    location: str  # We keep location as input
    # Removed property_type — no longer used in model

class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
    predicted_price_per_sqft_inr: float
