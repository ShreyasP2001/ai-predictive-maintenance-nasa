import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils import load_data, normalize_features, create_sequences

# Paths
DATA_PATH = "../data/sample_engine_data.csv"
MODEL_SAVE_PATH = "../models/lstm_model.h5"
PLOT_SAVE_PATH = "../results/training_loss_plot.png"

# Feature and label columns
FEATURES = ['operational_setting_1', 'operational_setting_2',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
LABEL = 'RUL'

# Load and preprocess data
df = load_data(DATA_PATH)
df, scaler = normalize_features(df, FEATURES)
X, y = create_sequences(df, FEATURES, target=LABEL)

# Build model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2,
                    callbacks=[EarlyStopping(patience=2)], verbose=1)

# Save model
model.save(MODEL_SAVE_PATH)

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('LSTM Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH)
plt.close()
