import numpy as np
import pandas as pd
import logging
from datetime import datetime
from scipy.stats import norm
from scipy.optimize import brentq

class OptionsAnalyzer:
    """
    Tools to build options snapshot, compute implied vol surface / term-structure,
    aggregate OI/volumr and produce Plotly visualization
    """
    def __init__(self, risk_free_rate: float = 0.01):
        self.r = risk_free_rate
    @staticmethod
    def _bs_price(S, K, T, r, sigma, option_type='call'):
        """
        Black-Scholes price from European option (call/put)
        """
        if T <= 0:
            return max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)
        if sigma <= 0:
            return max(0.0, S - K * np.exp(-r * T)) if option_type == 'call' else max(0.0, K * np.exp(-r * T) - S)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d2))
    @staticmethod
    def implied_volatility_from_price(market_price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
        """
        Invert Black_Scholes to obtain implied volatility.
        Return np.nan if inversion fails or market price <= intrinsic.
        """
        if pd.isna(market_price) or market_price <= 0:
            return np.nan
        
        intrinsic = max(0.0, S - K) if option_type == 'call' else max(0.0, K - S)

        if market_price <= intrinsic + 1e-12:
            return np.nan
        
        def objective(sig):
            return OptionsAnalyzer._bs_price(S, K, T, r, sig, option_type) - market_price
        low,high = 1e-6, 5.0

        try:
            if objective(low) * objective(high) > 0:
                return np.nan
            vol = brentq(objective, low, high, xtol=tol, maxiter=max_iter)
            return float(vol)
        except Exception:
            return np.nan
        
    def compute_implied_vols(self, options_df: pd.DataFrame, spot_price: float) -> pd.DataFrame:
        """
        Compute implied volatility for each option row
        """
        if options_df.empty:
            logging.warning("Options data is empty")
            return pd.DataFrame()
        df = options_df.copy()
        df['mid'] = np.where(df['mid'].notna() & df['ask'].notna(),
                             (df['bid'] + df['ask']) / 2.0,
                             df['lastPrice'])
        
        df['time_to_expiry'] = (pd.to_datetime(df['expiry']) - pd.Timestamp.today()).dt.days / 365.25
        df.loc[df['time_to_expiry'] < 0, 'time_to_expiry'] = 0.0

        df['impliedVolatility'] = df.apply(
            lambda row: self.implied_volatility_from_price(
                market_price=row['mid'],
                S=spot_price,
                K=row['strike'],
                T=row['time_to_expiry'],
                r=self.r,
                option_type='call' if row["Type"] == 'calls' else 'put'
            ), axis=1
        )
        return df
    @staticmethod
    def calculate_open_interest_by_strike(options_df: pd.DataFrame) -> pd.DataFrame:
        if options_df.empty:
            logging.warning("Options data is empty")
            return pd.DataFrame()
        return options_df.groupby(["strike", "Type"])["openInterest"].sum().reset_index()
    
    @staticmethod
    def calculate_iv_term_structure(options_df: pd.DataFrame, agg='median') -> pd.DataFrame:
        if options_df.empty:
            logging.warning("Option's data is empty")
            return pd.DataFrame()
        df = options_df.copy()
        grp=df.groupby('expiry')['impliedVolatility']
        iv_series = grp.mean() if agg == 'median' else grp.mean()
        out = iv_series.reset_index().rename(columns={'impliedVolatility': f'IV_{agg}'})
        out['time_to_expiry'] = (pd.to_datetime(out['expiry']) - pd.Timestamp.today()).dt.days / 365.25
        return out
    
    @staticmethod
    def build_vol_surfaces(options_df: pd.DataFrame) -> pd.DataFrame:
        if options_df.empty:
            logging.warning("Options data is empty")
            return pd.DataFrame()
        records = []
        for _,row in options_df.iterrows():
            records.append({
                'expiry': row['expiry'],
                'strike': row['strike'],
                'IV': row['impliedVolatility'],
                'Type': row['Type']
            })
        return pd.DataFrame(records)