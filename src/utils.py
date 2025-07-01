import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    return pd.read_csv(path)

def normalize_features(df, feature_cols):
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def create_sequences(df, features, target='RUL', window_size=10):
    sequences, labels = [], []
    for engine_id in df['engine_id'].unique():
        engine_data = df[df['engine_id'] == engine_id]
        for i in range(len(engine_data) - window_size):
            seq = engine_data[features].iloc[i:i+window_size].values
            label = engine_data[target].iloc[i+window_size]
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)
