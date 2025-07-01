import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from utils import load_data, normalize_features, create_sequences

# Paths
DATA_PATH = "../data/sample_engine_data.csv"
LSTM_MODEL_PATH = "../models/lstm_model.h5"
RF_MODEL_PATH = "../models/rf_model.pkl"
XGB_MODEL_PATH = "../models/xgb_model.pkl"

RESULT_PLOT = "../results/predicted_vs_true_rul.png"
SHAP_PLOT = "../results/shap_summary_plot.png"

FEATURES = ['operational_setting_1', 'operational_setting_2',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
TARGET = 'RUL'

# Load and preprocess data
df = load_data(DATA_PATH)
df, scaler = normalize_features(df, FEATURES)
X_seq, y_seq = create_sequences(df, FEATURES, target=TARGET)
X_flat = df[FEATURES].values
y_flat = df[TARGET].values

# Evaluate LSTM
lstm = load_model(LSTM_MODEL_PATH)
y_pred_lstm = lstm.predict(X_seq).flatten()
y_true_lstm = y_seq

# Evaluate RF/XGB
rf = joblib.load(RF_MODEL_PATH)
xgb = joblib.load(XGB_MODEL_PATH)
y_pred_rf = rf.predict(X_flat)
y_pred_xgb = xgb.predict(X_flat)

# Metrics
def print_metrics(y_true, y_pred, model_name):
    print(f"--- {model_name} ---")
    print(f"RMSE: {mean_squared_error(y_true, y_pred, squared=False):.2f}")
    print(f"MAE : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"R²  : {r2_score(y_true, y_pred):.2f}\n")

print_metrics(y_true_lstm, y_pred_lstm, "LSTM")
print_metrics(y_flat, y_pred_rf, "Random Forest")
print_metrics(y_flat, y_pred_xgb, "XGBoost")

# Plot predicted vs true for LSTM
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y_true_lstm)), y_true_lstm, label="True RUL", alpha=0.6)
plt.scatter(range(len(y_pred_lstm)), y_pred_lstm, label="Predicted RUL", alpha=0.6)
plt.title("LSTM: True vs Predicted RUL")
plt.xlabel("Sample")
plt.ylabel("RUL")
plt.legend()
plt.tight_layout()
plt.savefig(RESULT_PLOT)
plt.close()

# SHAP for Random Forest
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_flat)
shap.summary_plot(shap_values, X_flat, feature_names=FEATURES, show=False)
plt.tight_layout()
plt.savefig(SHAP_PLOT)
plt.close()
