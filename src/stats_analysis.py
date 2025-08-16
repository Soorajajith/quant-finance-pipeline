import numpy as np
import pandas as pd
import logging
class StatsAnalysis:
    "Class for statistical analysis of financial data"

    @staticmethod
    def calculate_returns(data: pd.DataFrame, rolling_window:int) -> pd.DataFrame:
        if data.empty:
            logging.warning("Market data is empty. Cannot calculate returns and volatility.")
            return pd.DataFrame()

        # Make a copy to keep the original data intact
        stats_data = data.copy()

        # Calculate returns and volatility
        stats_data["returns"] = stats_data["Close"].pct_change()
        stats_data["log_returns"] = np.log(stats_data["Close"] / stats_data["Close"].shift(1))
        stats_data["volatility"] = stats_data["returns"].rolling(window=rolling_window).std()
        stats_data["volatility_log"] = stats_data["log_returns"].rolling(window=rolling_window).std()

        # Convert only numeric columns to float
        numeric_cols = ["Close", "Open", "High", "Low", "Volume", 
                        "returns", "log_returns", "volatility", "volatility_log"]
        stats_data[numeric_cols] = stats_data[numeric_cols].astype(float)

        return stats_data.dropna()
    