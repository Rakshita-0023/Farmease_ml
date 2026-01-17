from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create app
app = FastAPI()

# Load trained model
model = joblib.load("crop_model.pkl")

# Define input structure
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Root route (test)
@app.get("/")
def home():
    return {"message": "Farmease Crop Recommendation API is running"}

# Prediction route
@app.post("/predict-crop")
def predict_crop(data: CropInput):
    input_data = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]]
    
    prediction = model.predict(input_data)
    return {"recommended_crop": prediction[0]}
