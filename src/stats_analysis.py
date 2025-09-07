import numpy as np
import pandas as pd
import logging
from scipy.stats import gaussian_kde
from typing import Tuple

class StatsAnalysis:
    "Class for statistical analysis of financial data"

    @staticmethod
    def calculate_returns(data: pd.DataFrame, rolling_window:int, windows: int=[5,10,20]) -> pd.DataFrame:
        if data.empty:
            logging.warning("Market data is empty. Cannot calculate returns and volatility.")
            return pd.DataFrame()
        # Make a copy to keep the original data intact
        stats_data = data.copy()
        # Calculate returns and volatility
        for w in windows:
            stats_data[f"returns_{w}"] = data["Close"].pct_change(periods=w)
        stats_data["returns_1d"] = stats_data["Close"].pct_change(periods=1)
        stats_data["log_returns"] = np.log(stats_data["Close"] / stats_data["Close"].shift(1))
        stats_data["volatility"] = stats_data["returns_1d"].rolling(window=rolling_window).std()
        stats_data["volatility_log"] = stats_data["log_returns"].rolling(window=rolling_window).std()
        # Convert only numeric columns to float
        numeric_cols = stats_data.select_dtypes(include=[np.number]).columns
        stats_data[numeric_cols] = stats_data[numeric_cols].astype(float)
        return stats_data.dropna()
    
    @staticmethod 
    def compute_atr(data: pd.DataFrame, window:int = 14) -> pd.DataFrame:
        """Compute Average True Range (ATR)"""
        if data.empty:
            logging.warning("Market data is empty.")
            return pd.DataFrame()
        for col in ['High', 'Low', 'Close']:
            if col not in data.columns:
                logging.error("Missing required column: %s", col)
                return data
            
        atr_data = pd.DataFrame(index=data.index)
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low']- data['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_data[f'atr_{window}'] = tr.rolling(window=window).mean()
        return atr_data
    @staticmethod
    def compute_roc(data: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Compute Rate of Change (ROC)"""
        if data.empty:
            logging.warning("Market data is empty. Cannot compute Rate of Change (ROC).")
            return pd.DataFrame()
        roc_data = pd.DataFrame(index=data.index)
        roc_data["roc"] = data["Close"].pct_change(periods=window)
        return roc_data

    @staticmethod
    def calculate_descriptive_stats(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            logging.warning("Market data is empty. Cannot calculate descriptive statistics.")
            return pd.DataFrame(index=data.index)
        # Calculate descriptive statistics
        descriptive_data = pd.DataFrame({
        "returns_mean": [data["returns"].mean()],
        "returns_std": [data["returns"].std()],
        "returns_median": [data["returns"].median()],
        "skewness": [data["returns"].skew()],
        "kurtosis": [data["returns"].kurtosis()],
        "returns_min": [data["returns"].min()],
        "returns_max": [data["returns"].max()],
        ## Quantiles 5% and 95% for better understanding of distribution
        "returns_5pct": [data["returns"].quantile(0.05)],
        "returns_95pct": [data["returns"].quantile(0.95)]
        })
        return descriptive_data
    
    @staticmethod
    def estimate_return_distribution(data: pd.DataFrame, num_points: int = 500) -> pd.DataFrame:
        if data.empty:
            logging.warning("Market data is empty. Cannot estimate return distribution.")
            return pd.DataFrame("x","kde")

        returns = data["returns"].dropna()
        kde = gaussian_kde(returns)
        ## x_range is using min() and max() of the observed returns. Sometimes, for smoother visual tails, 
        ## extending the range slightly (e.g., min()-3*std to max()+3*std) is better.
        x_min = returns.min() - 3 * returns.std()
        x_max = returns.max() + 3 * returns.std()
        x_range = np.linspace(x_min, x_max, num_points)
        kde_values = kde(x_range)
        return pd.DataFrame({
            "x" : x_range,
            "kde": kde_values
        })
    
    @staticmethod
    def calculate_rolling_volume_average(data: pd.DataFrame, size1: int = 20, size2: int = 50) -> pd.DataFrame:
        if data.empty:
            logging.Warning("Market data is empty. Cannot calculate rolling volume average")
            return pd.DataFrame("rolling_volume_20","rolling_volume_50")
        volume_data = pd.DataFrame(index=data.index)
        volume_data["rolling_volume_20"] = data["Volume"].rolling(size1).mean()
        volume_data["rolling_volume_50"] = data["Volume"].rolling(size2).mean()
        return volume_data
    
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.DataFrame:
        """"Calculate Volume Weighted Average Price (VWAP)"""
        if data.empty:
            logging.Warning("Market data is empty. Cannot calculate volume weighted average price")
            return pd.DataFrame("typical_price","cum_vol","cum_vol_price","vwap")
        vwap_data = pd.DataFrame(index=data.index)
        vwap_data["typical_price"] = (data["High"] + data["Low"] + data["Close"])/3
        vwap_data["cum_vol"] = data["Volume"].cumsum()
        vwap_data["cum_vol_price"] = (vwap_data["typical_price"] * data["Volume"]).cumsum()
        vwap_data["vwap"] = vwap_data["cum_vol_price"] / vwap_data["cum_vol"]
        return vwap_data
    
    @staticmethod
    def calculate_rolling_close_averages(data: pd.DataFrame, windows=[20,50,200]) -> pd.DataFrame:
        "Calculate rolling averages for closing prices"
        if data.empty:
            logging.warning("Market data is empty. Cannot compute rolling averages for closing prices")
            return pd.DataFrame()
        rolling_average = pd.DataFrame(index=data.index)
        for w in windows:
            rolling_average[f"SMA_{w}"] = data["Close"].rolling(window=w).mean()
        return rolling_average
    
    @staticmethod
    def calculate_correlations(data: pd.DataFrame, return_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute correlation matrices:
        - OHLCV features
        - Return features
        """
        if data.empty or return_data.empty:
            logging.warning("Either market data or return data is empty")
            return pd.DataFrame(), pd.DataFrame()
        features = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in features if c in data.columns]
        correlation_ohlcv = data[available].corr()
        correlation_returns = return_data.corr()
        return correlation_ohlcv, correlation_returns