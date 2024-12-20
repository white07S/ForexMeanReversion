import pandas as pd
import os
from ..config import RAW_DATA_DIR, PROCESSED_DATA_PATH

def load_and_clean_data():
    # Load all CSV files
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    dfs = []
    for f in files:
        # Format: YYYYMMDD HHMMSS;OPEN_BID;HIGH_BID;LOW_BID;CLOSE_BID;VOLUME
        path = os.path.join(RAW_DATA_DIR, f)
        df = pd.read_csv(path, sep=';', header=None, names=['Datetime','Open','High','Low','Close','Volume'], 
                         parse_dates=[0], date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d %H%M%S'))
        dfs.append(df)
    
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.drop_duplicates(subset=['Datetime'], inplace=True)
    full_df.sort_values('Datetime', inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    # Basic cleaning
    full_df = full_df.dropna()
    # Ensure regularly spaced 1-minute intervals (if missing, forward fill or drop)
    # For simplicity, assume data is continuous. Otherwise, reindex: #TODO: need to be fixed

    # Use log-price
    full_df['LogClose'] = (full_df['Close']).apply(lambda x: x if x>0 else None)
    full_df.dropna(subset=['LogClose'], inplace=True)
    full_df['LogClose'] = full_df['LogClose'].apply(lambda x: pd.np.log(x))

    # Save processed data
    if not os.path.exists(os.path.dirname(PROCESSED_DATA_PATH)):
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH))
    full_df.to_parquet(PROCESSED_DATA_PATH)
    return full_df

if __name__ == "__main__":
    load_and_clean_data()