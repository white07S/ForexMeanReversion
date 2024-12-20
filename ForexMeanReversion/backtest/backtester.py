import numpy as np

def run_backtest(df, signals, transaction_cost=0.00002):
    # df: has Datetime, LogClose
    # signals: array of {1,0,-1}
    # Compute P&L from going long or short one unit
    # Returns = (X_(t+1)-X_t) * position
    # position changes at each step according to signal
    # For simplicity: position at time t is signals[t]
    # Apply transaction costs when position changes.

    returns = df['LogClose'].diff().shift(-1) # next minute return
    # position at time t trades at time t, realized at t+1
    position = signals
    # transaction costs when position changes
    # cost = abs(position_t - position_{t-1}) * cost_per_unit
    pos_change = np.abs(np.diff(position, prepend=0))
    cost_array = pos_change * transaction_cost

    pnl = (position * returns) - cost_array
    pnl = pnl[:-1]  # last return might be NaN due to shift
    pnl = pnl.fillna(0)
    return pnl