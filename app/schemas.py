from pydantic import BaseModel

class PropertyInput(BaseModel):
    total_sqft: float
    bath: int
    balcony: int
    price_per_sqft: float
    location: str

class PredictionResponse(BaseModel):
    predicted_price_lakhs: float
