import numpy as np
from ..ou_model.ou_params import ou_stationary_var

def generate_signals(df, theta_series, mu_series, sigma_series, threshold):
    # df with columns Datetime, LogClose
    # theta_series, mu_series, sigma_series aligned with df index
    # We generate signals based on deviation from mu:
    # If (X - mu) > threshold * sqrt(var) -> short
    # if (X - mu) < -threshold * sqrt(var) -> long
    # else flat

    signals = np.zeros(len(df))
    for i in range(len(df)):
        X = df['LogClose'].iloc[i]
        theta = theta_series[i]
        mu = mu_series[i]
        sigma = sigma_series[i]
        var = ou_stationary_var(theta, sigma)
        std = np.sqrt(var)
        dev = X - mu
        if dev > threshold * std:
            signals[i] = -1  # short
        elif dev < -threshold * std:
            signals[i] = 1   # long
        else:
            signals[i] = 0   # flat
    return signals