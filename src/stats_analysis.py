import numpy as np
import pandas as pd
import logging
class StatsAnalysis:
    "Class for statistical analysis of financial data"

    @staticmethod
    def calculate_returns(data: pd.DataFrame, rolling_window:int) -> pd.DataFrame:
        "Calculate returns and volatility for the given market data"
        if data.empty or "Close" not in data.columns:
            logging.warning("Market data is empty or without Close. Cannot calculate returns and volatility.")
            return pd.DataFrame(index=data.index)
        stats_data = pd.DataFrame(index=data.index)
        stats_data["returns"] = data["Close"].pct_change()
        stats_data["log_returns"] = np.log(data["Close"] / data["Close"].shift(1))
        stats_data["volatility"] = stats_data["returns"].rolling(window=rolling_window).std()
        stats_data["volatility_log"] = stats_data["log_returns"].rolling(window=rolling_window).std()
        # Ensure numeric types for consistency
        stats_data = stats_data.astype(float)

        # Metadata: track calculation info (optional, can help in larger pipelines)
        stats_data.attrs["rolling_window"] = rolling_window
        stats_data.attrs["calculated"] = True

        return stats_data
    