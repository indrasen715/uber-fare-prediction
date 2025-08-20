
# Import required libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import pytz


# Configuration

DATASET_PATH = "./training-data/uber.csv"   # Path to NEW generated dataset
MODEL_PATH = "./model/uber_fare_model.pkl"          # Save trained model here



# Utility: Haversine distance

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance (in km) between two points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))



# Main function

def load_and_train():
    """Load dataset, engineer features, train Random Forest, evaluate, and save."""

    # Step 1: Load dataset
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    # Step 2: Clean dataset
    df = df.dropna()
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 200)]

    # Step 3: Feature Engineering
    
    # If distance_km column is not in data → calculate
    if "distance_km" not in df.columns:
        df["distance_km"] = haversine_distance(
            df["pickup_latitude"], df["pickup_longitude"],
            df["dropoff_latitude"], df["dropoff_longitude"]
        )

    df = df[(df["distance_km"] > 0.1) & (df["distance_km"] < 100)]

    # Ensure datetime is parsed
    
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime"])

    eastern = pytz.timezone("US/Eastern")

    if df["pickup_datetime"].dt.tz is None:
        df["pickup_datetime"] = df["pickup_datetime"].dt.tz_localize("UTC").dt.tz_convert(eastern)
    else:
        df["pickup_datetime"] = df["pickup_datetime"].dt.tz_convert(eastern)

    # Extract datetime features (if not already in dataset)
    if "hour" not in df.columns:
        df["hour"] = df["pickup_datetime"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    if "month" not in df.columns:
        df["month"] = df["pickup_datetime"].dt.month

    # Step 4: Select Features & Target
    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance_km",
        "hour",
        "day_of_week",
        "month"
    ]

    X = df[feature_cols]
    y = df["fare_amount"]

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 6: Train model
    print("Training model...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Step 7: Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained! MAE: {mae:.2f} USD")

    # Step 8: Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")



# Run script

if __name__ == "__main__":
    load_and_train()
