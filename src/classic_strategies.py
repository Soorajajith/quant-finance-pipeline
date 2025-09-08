import pandas as pd 
import numpy as np

class ClassicStrategies:

    @staticmethod
    def moving_average_crossover(data: pd.DataFrame, short: int = 50, long:int = 200) -> pd.Series:
        """"
        Buy when short MA crosses above long MA, sell when below.
        """
        short_ma = data['Close'].rolling(window=short).mean()
        long_ma = data['Close'].rolling(window=long).mean()
        signal = np.where(short_ma > long_ma, 1, -1)
        return pd.Series(signal, index=data.index, name=f"ma_crossover_{short}_{long}")
    
    @staticmethod
    def mean_reversion_bollinger(data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.Series:
        """
        Long if price < lower band, short if price > upper band.
        """
        min = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper = min + num_std * std
        lower = min - num_std * std
        signal = np.where(data['Close'] < lower, 1,
                          np.where(data['Close'] > upper, -1, 0))
        return pd.Series(signal, index=data.index, name=f"bollinger_reversion_{window}")
    
    @staticmethod
    def breakout_strategy(data: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Long if price breaks above rolling max, short if below rolling min."""
        rolling_max = data['Close'].rolling(window=window).max()
        rolling_min = data['Close'].rolling(window=window).min()
        signal = np.where(data['Close'] > rolling_max.shift(1), 1,
                          np.where(data['Close'] < rolling_min.shift(1), -1, 0))
        return pd.Series(signal, index=data.index, name=f"breakout_{window}")
    
    @staticmethod
    def macd_crossover(data: pd.DataFrame, short: int = 12, long: int = 26, signal_window: int = 9) -> pd.Series:
        """
        Long when MACD crosses above singal line, short when below.
        """
        ema_short = data['Close'].ewm(span=short, adjust=False).mean()
        ema_long = data['Close'].ewm(span=long, adjust=False).mean()
        macd = ema_short - ema_long
        signal_line = macd.ewm(span=signal_window, adjust=False).mean()

        signal = np.where(macd > signal_line, 1, -1)
        return pd.Series(signal, index=data.index, name=f"macd_crossover_{short}_{long}")
    @staticmethod
    def compute_momentum(data: pd.DataFrame, window: list = [20, 60, 250]) -> pd.DataFrame:
        """
        Compute momentum features as past return over different horizons
        windows: list of lookback period
        """
        momentum = pd.DataFrame(index=data.index)
        for w in window:
            momentum[f"momentum_{w}d"] = data['Close'].pct_change(periods=w)
            momentum[f"momentum_rank_{w}d"] = momentum[f"momentum_{w}d"].rank(method='average', na_option='keep')
        return momentum
