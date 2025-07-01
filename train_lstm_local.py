import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os

# Load data
df = pd.read_csv("data/sample_engine_data.csv")
FEATURES = ['operational_setting_1', 'operational_setting_2',
            'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5']
TARGET = 'RUL'

# Normalize
scaler = MinMaxScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

# Create sequences
window_size = 10
sequences, labels = [], []

for engine_id in df['engine_id'].unique():
    engine_data = df[df['engine_id'] == engine_id]
    for i in range(len(engine_data) - window_size):
        seq = engine_data[FEATURES].iloc[i:i+window_size].values
        label = engine_data[TARGET].iloc[i+window_size]
        sequences.append(seq)
        labels.append(label)

X = np.array(sequences)
y = np.array(labels)

# Build model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2,
          callbacks=[EarlyStopping(patience=2)])

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.h5")
print("✅ Model saved to models/lstm_model.h5")
