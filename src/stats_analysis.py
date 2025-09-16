import numpy as np
import pandas as pd
import logging
from scipy.stats import gaussian_kde
from typing import Tuple
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller, kpss

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
    def compute_obv(data: pd.DataFrame) -> pd.DataFrame:
        """Compute On-Balance Volume (OBV)"""
        if data.empty:
            logging.warning("Market data is empty.")
            return pd.DataFrame()
        for col in ['Close','Volume']:
            if col not in data.columns:
                logging.error("Missing required column: %s", col)
                return data
        obv_data = pd.DataFrame(index=data.index)
        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        obv_data['obv'] = obv
        return obv_data

            

    @staticmethod
    def calculate_descriptive_stats(data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            logging.warning("Market data is empty. Cannot calculate descriptive statistics.")
            return pd.DataFrame(index=data.index)
        # Calculate descriptive statistics
        descriptive_data = pd.DataFrame({
        "returns_mean": [data["returns_1d"].mean()],
        "returns_std": [data["returns_1d"].std()],
        "returns_median": [data["returns_1d"].median()],
        "skewness": [data["returns_1d"].skew()],
        "kurtosis": [data["returns_1d"].kurtosis()],
        "returns_min": [data["returns_1d"].min()],
        "returns_max": [data["returns_1d"].max()],
        ## Quantiles 5% and 95% for better understanding of distribution
        "returns_5pct": [data["returns_1d"].quantile(0.05)],
        "returns_95pct": [data["returns_1d"].quantile(0.95)]
        })
        return descriptive_data
    
    @staticmethod
    def estimate_return_distribution(data: pd.DataFrame, num_points: int = 500) -> pd.DataFrame:
        if data.empty:
            logging.warning("Market data is empty. Cannot estimate return distribution.")
            return pd.DataFrame(columns=["x","kde"])

        returns = data["returns_1d"].dropna()
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
            logging.warning("Market data is empty. Cannot calculate rolling volume average")
            return pd.DataFrame()
        volume_data = pd.DataFrame(index=data.index)
        volume_data["rolling_volume_20"] = data["Volume"].rolling(size1).mean()
        volume_data["rolling_volume_50"] = data["Volume"].rolling(size2).mean()
        return volume_data
    
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.DataFrame:
        """"Calculate Volume Weighted Average Price (VWAP)"""
        if data.empty:
            logging.warning("Market data is empty. Cannot calculate volume weighted average price")
            return pd.DataFrame()
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
    
    @staticmethod
    def compute_rolling_sharpe(data: pd. DataFrame, window: int = 60) -> pd.Series:
        """Compute rolling sharpe ratio"""
        if data.empty:
            logging.warning("Market data is empty.")
            return pd.Series()
        rolling_sharpe = (data["returns_1d"].rolling(window).mean())
        rolling_sharpe /= data["returns_1d"].rolling(window).std()
        rolling_sharpe *= np.sqrt(252)  # Annualize assuming 252 trading days
        return rolling_sharpe
    
    @staticmethod
    def compute_drawdown(data: pd.DataFrame) -> pd.Series:
        """Compute drawdown from peak"""
        if data.empty:
            logging.warning("Market data is empty.")
            return pd.Series()
        cumulative = (1 + data['returns_1d']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.rename('drawdown')
    @staticmethod
    def pca_factors(returns: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Compute PCA factors on returns cross section."""
        if returns.empty:
            logging.warning("Returns data is empty. Cannot compute PCA factors.")
            return pd.DataFrame(),[]
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(returns.dropna())
        factors_df = pd.DataFrame(factors, index=returns.dropna().index,
                                  columns=[f"factor_{i+1}" for i in range(n_components)])
        return factors_df, pca.explained_variance_ratio_
    
    @staticmethod
    def adf_test(series: pd.Series) -> Tuple[float, float]:
        """ Return ADF statistic and p-value"""
        if series.empty:
            logging.warning("Input series is empty")
            return {"stats": None, "pval": None}
        result = adfuller(series.dropna(), autolag='AIC')
        return {"stat": result[0], "pval": result[1]}
    
    @staticmethod
    def kpss_test(series: pd.Series, regression: str = 'c') -> Tuple[float, float]:
        """Return KPSS test statistic and p-value"""
        if series.empty:
            logging.warning("Input series is empty")
            return {"stat": None, "pval": None}
        result = kpss(series.dropna(), regression=regression, nlags="auto")
        return {"stat": result[0], "pval": result[1]}
    