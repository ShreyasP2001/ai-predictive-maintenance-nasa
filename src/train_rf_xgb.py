import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from utils import load_data, normalize_features

# Paths
DATA_PATH = "../data/sample_engine_data.csv"
RF_MODEL_PATH = "../models/rf_model.pkl"
XGB_MODEL_PATH = "../models/xgb_model.pkl"

# Features and target
FEATURES = ['operational_setting_1', 'operational_setting_2',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
TARGET = 'RUL'

# Load and preprocess data
df = load_data(DATA_PATH)
df, scaler = normalize_features(df, FEATURES)

X = df[FEATURES].values
y = df[TARGET].values

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
joblib.dump(rf, RF_MODEL_PATH)

# Train XGBoost
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X, y)
joblib.dump(xgb, XGB_MODEL_PATH)

# Evaluate
for name, model in zip(["Random Forest", "XGBoost"], [rf, xgb]):
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    print(f"{name} RMSE: {rmse:.2f}, R²: {r2:.2f}")
