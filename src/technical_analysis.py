import talib as ta
import numpy as np
import pandas as pd 
import logging 
class TechnicalIndicators:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def compute_sma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """"Compute Simple Moving Average (SMA)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.Series()
        talib_sma = ta.SMA(data["Close"], timeperiod=window)
        return talib_sma

    def compute_ema(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute Exponential Moving Average (EMA)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.Series()
        talib_ema = ta.EMA(data["Close"], timeperiod=window)
        return talib_ema
    
    def compute_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """"Compute Relative Strength Index (RSI)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.Series()
        talib_rsi = ta.RSI(data["Close"], timeperiod=window)
        return talib_rsi
    
    def compute_macd(self, data: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """"Compute Moving Average Convergence Divergence (MACD)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.DataFrame()
        macd, signal, hist = ta.MACD(data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        macd_df = pd.DataFrame({"macd" : macd, "macd_signal": signal, "macd_hist": hist})
        return macd_df
    
    def compute_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """""Compute Bollinger Bands."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.DataFrame()
        upperband, middleband, lowerband = ta.BBANDS(data["Close"], timeperiod=window, nbdevup=num_std, nbdevdn=num_std, matype=0)
        bb_df = pd.DataFrame({"bb_upper": upperband, "bb_middle": middleband, "bb_lower": lowerband})
        return bb_df