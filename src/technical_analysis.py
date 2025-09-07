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
    
    @staticmethod
    def compute_macd_signal_ratio(data: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD / Signal ration. Requires MACD + Signal columns."""
        if data.empty:
            logging.warning("Input data is empty.")
            return pd.DataFrame()
        if "macd" not in data.columns or "macd_signal" not in data.columns:
            logging.warning("MACD or MACD Signal columns missing.")
            return pd.DataFrame()
        macd_signal_ratio = pd.DataFrame(index=data.index)
        macd_signal_ratio = data["macd"] / data["macd_signal"]
        return pd.DataFrame({"macd_signal_ratio": macd_signal_ratio})
    
    @staticmethod
    def compute_bollinger_percent_b(market_data: pd.DataFrame, bollinger_data: pd.DataFrame) -> pd.DataFrame:
        """Compute Bollinger %B: position of price within Bollinger Bands"""
        for col in ['bb_upper', 'bb_lower']:
            if col not in bollinger_data.columns:
                logging.error(f"{col} not in Bollinger data")
                return bollinger_data
        if ['Close'] not in market_data.columns:
            logging.error("Close price not in market data")
            return market_data
        bb_percent_b = pd.DataFrame(index=market_data.index)
        bb_percent_b['bb_percent_b'] = (market_data['Close'] - bollinger_data['bb_lower']) / (bollinger_data['bb_upper'] - bollinger_data['bb_lower'])
        return bb_percent_b
