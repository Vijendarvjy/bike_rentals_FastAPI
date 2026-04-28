from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import pandas as pd
import joblib
import os

# -------------------------------
# APP SETUP
# -------------------------------
app = FastAPI(
    title="🚲 Bike Rental Analytics API",
    description="FastAPI backend for Bike Rental Analytics & Demand Prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LOAD DATA
# -------------------------------
_data: Optional[pd.DataFrame] = None

def get_data() -> pd.DataFrame:
    global _data
    if _data is None:
        df = pd.read_csv("hour.csv")
        df["dteday"] = pd.to_datetime(df["dteday"])
        df["day_name"] = df["dteday"].dt.day_name()
        df["is_weekend"] = df["dteday"].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        def categorize_hour(hour):
            if 7 <= hour <= 9:
                return "Morning Rush"
            elif 17 <= hour <= 19:
                return "Evening Rush"
            elif 0 <= hour <= 5:
                return "Low Demand"
            else:
                return "Normal Hours"

        df["time_category"] = df["hr"].apply(categorize_hour)
        _data = df
    return _data

# -------------------------------
# LOAD MODEL
# -------------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists("tuned_xgboost_model.pkl"):
            _model = joblib.load("tuned_xgboost_model.pkl")
    return _model

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
MODEL_FEATURES = [
    "yr","holiday","workingday","temp","atemp","hum","windspeed",
    "day_of_week","is_weekend",
    "season_2","season_3","season_4",
    "weathersit_2","weathersit_3","weathersit_4",
    "mnth_2","mnth_3","mnth_4","mnth_5","mnth_6","mnth_7","mnth_8","mnth_9","mnth_10","mnth_11","mnth_12",
    "hr_1","hr_2","hr_3","hr_4","hr_5","hr_6","hr_7","hr_8","hr_9","hr_10","hr_11","hr_12",
    "hr_13","hr_14","hr_15","hr_16","hr_17","hr_18","hr_19","hr_20","hr_21","hr_22","hr_23",
    "weekday_1","weekday_2","weekday_3","weekday_4","weekday_5","weekday_6",
    "time_category_Low Demand","time_category_Morning Rush","time_category_Normal Hours"
]

def build_input(hour: int, temp: float, hum: float, windspeed: float, is_weekend: int) -> pd.DataFrame:
    row = dict.fromkeys(MODEL_FEATURES, 0)
    row["temp"] = temp
    row["atemp"] = temp
    row["hum"] = hum
    row["windspeed"] = windspeed
    row["is_weekend"] = is_weekend
    row["yr"] = 1
    row["holiday"] = 0
    row["workingday"] = 0 if is_weekend else 1
    row["day_of_week"] = 6 if is_weekend else 2
    if hour != 0:
        row[f"hr_{hour}"] = 1
    weekday = 6 if is_weekend else 2
    if weekday != 0:
        row[f"weekday_{weekday}"] = 1
    row["mnth_5"] = 1
    row["season_2"] = 1
    if 7 <= hour <= 9:
        row["time_category_Morning Rush"] = 1
    elif 0 <= hour <= 5:
        row["time_category_Low Demand"] = 1
    else:
        row["time_category_Normal Hours"] = 1
    return pd.DataFrame([row])

# -------------------------------
# SCHEMAS
# -------------------------------
class PredictionRequest(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature (0.0-1.0)")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity (0.0-1.0)")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized windspeed (0.0-1.0)")
    day_type: str = Field(..., pattern="^(Weekday|Weekend)$", description="'Weekday' or 'Weekend'")

class PredictionResponse(BaseModel):
    predicted_rentals: int
    hour: int
    day_type: str
    time_category: str

class KPIResponse(BaseModel):
    total_rentals: int
    avg_rentals: float
    peak_hour: int

class HourlyTrendItem(BaseModel):
    hr: int
    avg_cnt: float

class DemandDistributionItem(BaseModel):
    time_category: str
    total_cnt: int

class DayTypeFilter(BaseModel):
    day_types: List[str] = Field(default=["Weekday", "Weekend"])

# -------------------------------
# HEALTH
# -------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Bike Rental API is running 🚲"}

@app.get("/health", tags=["Health"])
def health():
    model_loaded = get_model() is not None
    try:
        df = get_data()
        data_loaded = True
        rows = len(df)
    except Exception:
        data_loaded = False
        rows = 0
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "data_loaded": data_loaded,
        "total_rows": rows
    }

# -------------------------------
# KPI METRICS
# -------------------------------
@app.get("/kpi", response_model=KPIResponse, tags=["Analytics"])
def get_kpis(
    day_types: str = Query(default="Weekday,Weekend", description="Comma-separated: Weekday,Weekend")
):
    """
    Returns total rentals, average rentals per hour, and peak hour.
    Filter by day type: Weekday, Weekend, or both.
    """
    df = get_data()
    day_filter = [d.strip() for d in day_types.split(",")]
    filtered = df[df["is_weekend"].map({0: "Weekday", 1: "Weekend"}).isin(day_filter)]
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found for given filters.")
    return KPIResponse(
        total_rentals=int(filtered["cnt"].sum()),
        avg_rentals=round(float(filtered["cnt"].mean()), 2),
        peak_hour=int(filtered.groupby("hr")["cnt"].mean().idxmax())
    )

# -------------------------------
# DEMAND DISTRIBUTION
# -------------------------------
@app.get("/demand-distribution", response_model=List[DemandDistributionItem], tags=["Analytics"])
def get_demand_distribution(
    day_types: str = Query(default="Weekday,Weekend", description="Comma-separated: Weekday,Weekend")
):
    """
    Returns total rentals grouped by time category (for donut chart).
    """
    df = get_data()
    day_filter = [d.strip() for d in day_types.split(",")]
    filtered = df[df["is_weekend"].map({0: "Weekday", 1: "Weekend"}).isin(day_filter)]
    grouped = filtered.groupby("time_category")["cnt"].sum().reset_index()
    return [
        DemandDistributionItem(time_category=row["time_category"], total_cnt=int(row["cnt"]))
        for _, row in grouped.iterrows()
    ]

# -------------------------------
# HOURLY TREND
# -------------------------------
@app.get("/hourly-trend", response_model=List[HourlyTrendItem], tags=["Analytics"])
def get_hourly_trend(
    day_types: str = Query(default="Weekday,Weekend", description="Comma-separated: Weekday,Weekend")
):
    """
    Returns average rentals per hour (for line chart).
    """
    df = get_data()
    day_filter = [d.strip() for d in day_types.split(",")]
    filtered = df[df["is_weekend"].map({0: "Weekday", 1: "Weekend"}).isin(day_filter)]
    hourly = filtered.groupby("hr")["cnt"].mean().reset_index()
    return [
        HourlyTrendItem(hr=int(row["hr"]), avg_cnt=round(float(row["cnt"]), 2))
        for _, row in hourly.iterrows()
    ]

# -------------------------------
# PREDICT
# -------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_demand(req: PredictionRequest):
    """
    Predict bike demand given hour, temperature, humidity, windspeed, and day type.
    """
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available. Ensure tuned_xgboost_model.pkl is present.")

    is_weekend = 1 if req.day_type == "Weekend" else 0

    try:
        input_df = build_input(req.hour, req.temp, req.hum, req.windspeed, is_weekend)
        prediction = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    if 7 <= req.hour <= 9:
        time_cat = "Morning Rush"
    elif 17 <= req.hour <= 19:
        time_cat = "Evening Rush"
    elif 0 <= req.hour <= 5:
        time_cat = "Low Demand"
    else:
        time_cat = "Normal Hours"

    return PredictionResponse(
        predicted_rentals=int(prediction),
        hour=req.hour,
        day_type=req.day_type,
        time_category=time_cat
    )
