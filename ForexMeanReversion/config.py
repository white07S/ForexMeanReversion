import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw', 'eurusd')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'eurusd_clean.parquet')

# OU model configs
TIME_STEP = 1.0 / (252 * 24 * 60)  # Assuming 1-minute bars, ~252 trading days, ~24*60 minutes per day for scaling
# Above is approximate; adjust as needed.

# Rolling window length (in minutes)
ROLL_WINDOW = 30 * 24 * 60  # 30 days of 1-min data

# ML model configs
FEATURE_WINDOW_SHORT = 30    # 30-minute window for short MA/Vol
FEATURE_WINDOW_LONG = 1440   # 1-day window for long MA/Vol
HIDDEN_LAYERS = [64, 32, 16]
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EPOCHS = 10

# Trading signal thresholds
DEVIATION_THRESHOLD = 1.0

# Backtest configs
TRANSACTION_COST = 0.00002  # Example cost per trade