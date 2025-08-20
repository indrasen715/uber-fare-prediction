# Uber Fare Prediction (Synthetic Data, 2025)

This project demonstrates how to **predict Uber ride fares** using a machine learning pipeline trained on a **synthetic dataset** that mimics real NYC Uber rides.  

The repository contains:  
- **Synthetic dataset** (20,000+ rides generated programmatically)  
- **Machine learning model** (Random Forest)  
- **Training pipeline** (data preprocessing, feature engineering, model training)  
- **FastAPI service** for fare prediction in real-time  

---

## Dataset

The dataset was generated using a pricing formula (base fare, per mile, per minute, surcharges, and tolls).  
It contains **synthetic NYC Uber trips** for safe, public machine learning use.

### Features
- `pickup_datetime` → Date and time of pickup (US/Eastern timezone)  
- `pickup_longitude`, `pickup_latitude` → Pickup location  
- `dropoff_longitude`, `dropoff_latitude` → Dropoff location  
- `passenger_count` → Number of passengers  
- `distance_km` → Haversine distance (in kilometers)  
- `hour` → Hour of the day (0–23)  
- `day_of_week` → Day of week (0 = Monday, 6 = Sunday)  
- `month` → Month of year (1–12)  
- `fare_amount` → Estimated fare (USD)  

**Note**: This is **synthetic data** generated for ML purposes, not real Uber data.

---

## Project Structure

```php 
├── training-data/
│ └── uber.csv # Synthetic dataset (20,000+ rows)
├── train_uber.py # Training script (feature engineering + model training)
├── uber_api.py # FastAPI service for real-time predictions
├── uber_fare_model.pkl # Saved trained RandomForest model
├── requirements.txt # Python dependencies
├── README.md # Project documentation

```
---

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt

```
### Train the model
```bash
python train_uber.py
```

This will:

- Load and clean the dataset

- Engineer features (distance, time, etc.)

- Train a Random Forest Regressor

Save the model as uber_fare_model.pkl

### Run the API

```bash
uvicorn uber_api:app --reload
```

### Example API call

```

curl "http://127.0.0.1:8000/predict?pickup_longitude=-73.9849&pickup_latitude=40.7511&dropoff_longitude=-73.8035&dropoff_latitude=40.6591&passenger_count=1&pickup_datetime=2025-08-20 05:20:00"

```


### Response:

```json

{
  "predicted_fare_usd": 51.96,
  "datetime_format": "YYYY-MM-DD HH:MM:SS (US/Eastern Time)",
  "input": {
    "pickup_longitude": -73.9849562,
    "pickup_latitude": 40.7511257,
    "dropoff_longitude": -73.8035325,
    "dropoff_latitude": 40.659113,
    "passenger_count": 1,
    "pickup_datetime": "2025-08-20 05:20:00",
    "distance_km": 18.4,
    "hour": 5,
    "day_of_week": 2,
    "month": 8
  }
}
```

### Model

1. Algorithm: Random Forest Regressor

2. Metrics: Mean Absolute Error (MAE) on test set

### Features used:

1. Coordinates (pickup/dropoff)

2. Passenger count

3. Distance (Haversine formula)

4. Datetime features (hour, weekday, month)

The model was tuned for stability and generalization.