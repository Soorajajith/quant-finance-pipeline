import talib as ta
import numpy as np
import pandas as pd 
import logging 
class TechnicalIndicators:

    @staticmethod
    def compute_sma(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """"Compute Simple Moving Average (SMA)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.Series()
        talib_sma = ta.SMA(data["Close"], timeperiod=window)
        return talib_sma

    @staticmethod
    def compute_ema(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Compute Exponential Moving Average (EMA)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.Series()
        talib_ema = ta.EMA(data["Close"], timeperiod=window)
        return talib_ema
    
    @staticmethod
    def compute_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
        """"Compute Relative Strength Index (RSI)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.Series()
        talib_rsi = ta.RSI(data["Close"], timeperiod=window)
        return talib_rsi
    
    @staticmethod
    def compute_macd(data: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
        """"Compute Moving Average Convergence Divergence (MACD)."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.DataFrame()
        macd, signal, hist = ta.MACD(data["Close"], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        macd_df = pd.DataFrame({"macd" : macd, "macd_signal": signal, "macd_hist": hist})
        return macd_df
    
    @staticmethod
    def compute_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """""Compute Bollinger Bands."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.DataFrame()
        upperband, middleband, lowerband = ta.BBANDS(data["Close"], timeperiod=window, nbdevup=num_std, nbdevdn=num_std, matype=0)
        bb_df = pd.DataFrame({"bb_upper": upperband, "bb_middle": middleband, "bb_lower": lowerband})
        return bb_df