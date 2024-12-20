import numpy as np
import pandas as pd

def sharpe_ratio(pnl, freq=252*24*60):
    # freq: number of bars per year approx
    mean_return = np.mean(pnl)
    vol = np.std(pnl)
    if vol == 0:
        return 0
    sr = (mean_return * np.sqrt(freq)) / vol
    return sr

def max_drawdown(cum_pnl):
    peak = np.maximum.accumulate(cum_pnl)
    dd = (peak - cum_pnl)/peak
    return np.max(dd)

def performance_metrics(pnl):
    cum_pnl = pnl.cumsum()
    sr = sharpe_ratio(pnl)
    mdd = max_drawdown(cum_pnl)
    return {
        'Total PnL': cum_pnl.iloc[-1],
        'Sharpe Ratio': sr,
        'Max Drawdown': mdd,
        'Hit Ratio': (pnl > 0).mean()
    }