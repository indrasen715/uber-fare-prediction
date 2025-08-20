# ----------------------
# generate_uber_data.py
# ----------------------
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import pytz
import math

# This sets approximate latitude/longitude limits around New York City.

NYC_BOUNDS = {
    "lat_min": 40.55,
    "lat_max": 40.95,
    "lon_min": -74.05,
    "lon_max": -73.70
}

# Uber Fare constants
BASE_FARE = 7.19
PER_MINUTE = 0.88
PER_MILE = 1.65
TOLL_MEAN = 6.94
SURCHARGE_MEAN = 7.67
RESERVATION_FEE = 1.00
WAIT_FEE_PER_MIN = 0.81

# US/Eastern timezone
eastern = pytz.timezone("US/Eastern")


def haversine(lon1, lat1, lon2, lat2):
    """Calculate shortest distance between two points on a sphere (like Earth) using their latitude and longitude. (in km)."""
    R = 6371  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def generate_trip():
    """Generate one synthetic Uber trip row."""
    # Random pickup and dropoff points
    pickup_lat = random.uniform(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
    pickup_lon = random.uniform(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
    dropoff_lat = random.uniform(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
    dropoff_lon = random.uniform(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])

    # Distance
    distance_km = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    distance_miles = distance_km * 0.621371

    # Duration (assume avg 30 km/h ± randomness)
    avg_speed = random.uniform(20, 40)  # km/h
    duration_min = (distance_km / avg_speed) * 60
    duration_min = max(duration_min, 1)  # at least 1 min

    # Pickup datetime (within past year)
    days_offset = random.randint(0, 365)
    seconds_offset = random.randint(0, 24*3600)
    pickup_dt = datetime.now(eastern) - timedelta(days=days_offset, seconds=seconds_offset)

    # Features from datetime
    hour = pickup_dt.hour
    day_of_week = pickup_dt.weekday()
    month = pickup_dt.month

    # Passenger count
    passenger_count = random.randint(1, 4)

    # Tolls, surcharges, wait time
    tolls = max(0, np.random.normal(TOLL_MEAN, 2))
    surcharges = max(0, np.random.normal(SURCHARGE_MEAN, 2))
    wait_time = max(0, random.randint(0, 10))  # minutes
    wait_fee = WAIT_FEE_PER_MIN * max(0, wait_time - 5)

    # Fare calculation
    fare = (
        BASE_FARE
        + (PER_MINUTE * duration_min)
        + (PER_MILE * distance_miles)
        + tolls
        + surcharges
        + RESERVATION_FEE
        + wait_fee
    )
    fare = round(fare, 2)

    return {
        "fare_amount": fare,
        "pickup_datetime": pickup_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "pickup_longitude": pickup_lon,
        "pickup_latitude": pickup_lat,
        "dropoff_longitude": dropoff_lon,
        "dropoff_latitude": dropoff_lat,
        "passenger_count": passenger_count,
        "distance_km": round(distance_km, 2),
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month
    }


def generate_dataset(n=20000, filename="uber_synthetic_20000.csv"):
    """Generate dataset and save as CSV."""
    data = [generate_trip() for _ in range(n)]
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"✅ Generated {n} rows and saved to {filename}")


if __name__ == "__main__":
    generate_dataset()
