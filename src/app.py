import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils import normalize_features, create_sequences

st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")
st.title("🔧 Predictive Maintenance Dashboard")

# Load models and scalers
@st.cache_resource
def load_models():
    rf = joblib.load("../models/rf_model.pkl")
    xgb = joblib.load("../models/xgb_model.pkl")
    lstm = load_model("../models/lstm_model.h5")
    return rf, xgb, lstm

rf, xgb, lstm = load_models()

# Upload data
uploaded_file = st.file_uploader("Upload Engine Sensor Data CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")

    # Select features
    FEATURES = ['operational_setting_1', 'operational_setting_2',
                'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
    LABEL = 'RUL' if 'RUL' in df.columns else None

    df, _ = normalize_features(df, FEATURES)

    st.subheader("📊 Model Predictions")
    X_seq, y_seq = create_sequences(df, FEATURES)
    X_flat = df[FEATURES].values

    lstm_preds = lstm.predict(X_seq).flatten()
    rf_preds = rf.predict(X_flat)
    xgb_preds = xgb.predict(X_flat)

    st.write("**LSTM Prediction (Sample):**", lstm_preds[:5])
    st.write("**Random Forest Prediction (Sample):**", rf_preds[:5])
    st.write("**XGBoost Prediction (Sample):**", xgb_preds[:5])

    st.line_chart({"LSTM": lstm_preds[:50], "RF": rf_preds[:50], "XGB": xgb_preds[:50]})

    if LABEL:
        st.subheader("✅ True vs Predicted (if RUL exists)")
        st.line_chart({
            "True": y_seq[:50],
            "LSTM": lstm_preds[:50]
        })

    st.success("✅ Dashboard ready.")
else:
    st.info("Upload a CSV to begin. Sample format: 7 feature columns with 'engine_id', 'cycle', and optional 'RUL'.")
