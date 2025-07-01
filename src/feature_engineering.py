import pandas as pd

def add_cycle_features(df):
    df['max_cycle'] = df.groupby('engine_id')['cycle'].transform('max')
    df['norm_cycle'] = df['cycle'] / df['max_cycle']
    return df

def add_statistical_features(df, sensors):
    agg_funcs = ['mean', 'std', 'min', 'max']
    stat_features = df.groupby('engine_id')[sensors].agg(agg_funcs)
    stat_features.columns = ['_'.join(col) for col in stat_features.columns]
    stat_features.reset_index(inplace=True)
    return stat_features

def merge_features(df, stat_features):
    return df.merge(stat_features, on='engine_id', how='left')
