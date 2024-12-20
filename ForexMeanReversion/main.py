import pandas as pd
import numpy as np
import os

from .config import PROCESSED_DATA_PATH, TIME_STEP, ROLL_WINDOW, DEVIATION_THRESHOLD, TRANSACTION_COST
from .data.data_preparation import load_and_clean_data
from .ou_model.ou_estimation import estimate_ou_params_mle, rolling_ou_estimation
from .ou_model.features import compute_features
from .ou_model.ml_model import train_nn, predict_params
from .backtest.signal_generation import generate_signals
from .backtest.backtester import run_backtest
from .backtest.performance import performance_metrics

def main():
    # 1. Data Preparation
    if not os.path.exists(PROCESSED_DATA_PATH):
        df = load_and_clean_data()
    else:
        df = pd.read_parquet(PROCESSED_DATA_PATH)

    # 2. Baseline OU parameter estimation
    # Use first year as training for baseline
    # Assume data is large; say we take first 50000 points for baseline
    baseline_data = df['LogClose'].values[:50000]
    theta_b, mu_b, sigma_b = estimate_ou_params_mle(baseline_data, TIME_STEP)

    # 3. Rolling window estimation for ML training set
    x = df['LogClose'].values
    times, params_series = rolling_ou_estimation(x, TIME_STEP, ROLL_WINDOW)
    # times, params_series align so that params_series[i] is for x[times[i]]

    # 4. Compute features for entire dataset
    df_feat, fcols = compute_features(df)
    # Align params with df_feat index
    # df_feat starts later due to rolling windows for features
    start_idx = df_feat.index[0]
    # We need to align times and feature indices. times gives indices in x;
    # we must ensure times correspond to df index.
    # df is continuous and zero-based, times is zero-based, so times are also indices in df
    # We'll only use overlapping indices
    valid_idx = (times >= start_idx)
    times = times[valid_idx]
    params_series = params_series[valid_idx]

    # Align features and params
    df_feat = df_feat.loc[times]
    feat_matrix = df_feat[fcols].values

    # 5. Split into train/val/test for ML
    n = len(times)
    train_size = int(n*0.7)
    val_size = int(n*0.15)
    test_size = n - train_size - val_size

    train_X = feat_matrix[:train_size]
    train_Y = params_series[:train_size]
    val_X = feat_matrix[train_size:train_size+val_size]
    val_Y = params_series[train_size:train_size+val_size]
    test_X = feat_matrix[train_size+val_size:]
    test_Y = params_series[train_size+val_size:]

    # 6. Train NN model for parameter prediction
    model = train_nn(train_X, train_Y)

    # 7. Predict parameters on test set
    pred_params = predict_params(model, test_X)
    # pred_params: Nx3 (theta, mu, sigma)
    # Align with df_feat index on test portion
    test_times = times[train_size+val_size:]
    # Insert predicted params into a dataframe
    pred_df = pd.DataFrame(pred_params, index=test_times, columns=['theta','mu','sigma'])

    # 8. Generate signals
    test_df = df.loc[test_times]
    signals = generate_signals(test_df, pred_df['theta'].values, pred_df['mu'].values, pred_df['sigma'].values, DEVIATION_THRESHOLD)

    # 9. Backtest
    pnl = run_backtest(test_df, signals, transaction_cost=TRANSACTION_COST)

    # 10. Performance
    metrics = performance_metrics(pnl)
    print("Performance Metrics:")
    for k,v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()