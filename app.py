# ----------------------
# Import libraries
# ----------------------
import pandas as pd
import joblib
from fastapi import FastAPI
from datetime import datetime
import numpy as np
import pytz
from train_uber import load_and_train, MODEL_PATH   # Reuse training code

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="🚖 Uber Fare Prediction API")

# ----------------------
# Utility: Haversine distance
# ----------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

# ----------------------
# Load model at startup
# ----------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded from disk")
    except:
        print("⚠️ Model not found, training a new one...")
        load_and_train()
        model = joblib.load(MODEL_PATH)

# ----------------------
# Root endpoint
# ----------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to Uber Fare Prediction API 🚖",
        "usage_example": "/predict?pickup_longitude=-73.985&pickup_latitude=40.758&"
                         "dropoff_longitude=-73.985&dropoff_latitude=40.768&"
                         "passenger_count=2&pickup_datetime=2025-08-20%2005:20:00",
        "datetime_format": "YYYY-MM-DD HH:MM:SS (US/Eastern Time)",
        "example_pickup_datetime": "2025-08-20 05:20:00"
    }

# ----------------------
# Prediction endpoint
# ----------------------
@app.get("/predict")
def predict_fare(
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int,
    pickup_datetime: str   # Required param → "YYYY-MM-DD HH:MM:SS" in US/Eastern
):
    """
    Predict Uber fare price.
    Required query parameters:
    - pickup_longitude (float)
    - pickup_latitude (float)
    - dropoff_longitude (float)
    - dropoff_latitude (float)
    - passenger_count (int)
    - pickup_datetime (str, format: "YYYY-MM-DD HH:MM:SS", US/Eastern Time)
    
    Example:
    /predict?pickup_longitude=-73.985&pickup_latitude=40.758&
             dropoff_longitude=-73.985&dropoff_latitude=40.768&
             passenger_count=2&pickup_datetime=2025-08-20%2005:20:00
    """

    # Step 1: Distance
    distance_km = haversine_distance(
        pickup_latitude, pickup_longitude,
        dropoff_latitude, dropoff_longitude
    )

    # Step 2: Time features (ensure US/Eastern timezone)
    try:
        dt_naive = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
        eastern = pytz.timezone("US/Eastern")
        dt = eastern.localize(dt_naive)  # assign timezone
    except ValueError:
        return {
            "error": "Invalid pickup_datetime format. "
                     "Use 'YYYY-MM-DD HH:MM:SS' (e.g. 2025-08-20 05:20:00)",
            "timezone": "US/Eastern"
        }

    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month

    # Step 3: Prepare input for model
    data = pd.DataFrame([{
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count,
        "distance_km": distance_km,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month
    }])

    # Step 4: Predict fare
    fare = model.predict(data)[0]

    return {
        "predicted_fare_usd": round(float(fare), 2),
        "datetime_format": "YYYY-MM-DD HH:MM:SS (US/Eastern Time)",
        "input": {
            "pickup_longitude": pickup_longitude,
            "pickup_latitude": pickup_latitude,
            "dropoff_longitude": dropoff_longitude,
            "dropoff_latitude": dropoff_latitude,
            "passenger_count": passenger_count,
            "pickup_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "distance_km": round(distance_km, 2),
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month
        }
    }
