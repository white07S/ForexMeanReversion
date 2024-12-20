import numpy as np
import pandas as pd
from ..config import FEATURE_WINDOW_SHORT, FEATURE_WINDOW_LONG

def compute_features(df):
    # df has columns: Datetime, LogClose
    df = df.copy()
    df['Return'] = df['LogClose'].diff()
    # Short-term MA
    df['MA_short'] = df['LogClose'].rolling(FEATURE_WINDOW_SHORT).mean()
    # Long-term MA
    df['MA_long'] = df['LogClose'].rolling(FEATURE_WINDOW_LONG).mean()
    # Short-term Vol
    df['Vol_short'] = df['Return'].rolling(FEATURE_WINDOW_SHORT).std()
    # Long-term Vol
    df['Vol_long'] = df['Return'].rolling(FEATURE_WINDOW_LONG).std()

    df.dropna(inplace=True)
    feature_cols = ['MA_short', 'MA_long', 'Vol_short', 'Vol_long']
    return df, feature_cols