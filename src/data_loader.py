import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from dataclasses import dataclass
import numpy as np

@dataclass
class TickerData:
    market_data: pd.DataFrame
    options_data: pd.DataFrame
    financials: pd.DataFrame

class DataLoader:
    def __init__(self):
        self.source = 'yfinance'
        self.tickers = self._get_top_tickers()  # Get top tickers from a predefined list
    
    def _get_top_tickers(self) -> list:
        # Hardcode top 10 tickers (US large-cap companies)
        ticker_data = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "BRK-B", "META", "V", "JPM"]
        return ticker_data
    
    def _validate_input(self, ticker: str, interval: str = "1d", start_date: str = "1995-05-02", end_date: str = "1995-09-13") ->bool:
        # Validate input parameters
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string.")
        valid_intervals = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
        if interval not in valid_intervals:
            raise ValueError(f"Interval must be one of {valid_intervals}")
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Start and end dates must be in YYYY-MM-DD format.")
        if start >= end:
            raise ValueError("Start date must be before end date.")
        return True
    
    def download_history_data(self, ticker: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Downloads data for all tickers, returns dict of DataFrames
        try:
            self._validate_input(ticker, interval, start_date, end_date)
            history_data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
            if history_data.empty:
                logging.warning(f"No historical data found for {ticker} from {start_date} to {end_date}.")
                return pd.DataFrame()
            return history_data
        except Exception as e:
            logging.error(f"Error downloading historical data for {ticker}: {e}")
            return {}
    
    def market_data(self, ticker: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Download market data
        try:
            self._validate_input(ticker, interval, start_date, end_date)
            raw_data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
            if raw_data.empty :
                logging.warning(f"No data found for {ticker} from {start_date} to {end_date}.")
                return pd.DataFrame()
            data = raw_data[["Open", "High", "Low", "Close", "Volume"]]
            data = data.astype(float)
            data = data.dropna()
            data["Symbol"] = ticker
            data["Date"] = raw_data.index  # keep the datetime index as a column
            data = data.reset_index(drop=True)
            if data.isnull().values.any():
               data = data.dropna()
            return data
        except Exception as e:
            logging.error(f"Error downloading market data for {ticker}: {e}")
            return pd.DataFrame()
    
    def download_options_data(self, ticker: str) -> pd.DataFrame:
        # Downloads options data for a single ticker
        try:
            self._validate_input(ticker)
            options_data = yf.Ticker(ticker).option_chain()
            if options_data is None:
                logging.warning(f"No options data found for {ticker}.")
                return pd.DataFrame()
            standardized_options_data = []
            for opt_type, opt_data in [("calls", options_data.calls), ("puts", options_data.puts)]:
                if not opt_data.empty:
                    opt_data = opt_data[["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume", "openInterest"]].dropna()
                    opt_data = opt_data.astype({
                        "strike": float,
                        "lastPrice": float,
                        "bid": float,
                        "ask": float,
                        "volume": float,
                        "openInterest": float,
                    })
                    opt_data["Type"] = opt_type
                    opt_data["Ticker"] = ticker
                    standardized_options_data.append(opt_data)
            if not standardized_options_data:
                logging.warning(f"No options data found for {ticker}.")
                return pd.DataFrame()
            options_data = pd.concat(standardized_options_data).reset_index(drop=True)
            if options_data.isnull().values.any():
                options_data = options_data.dropna()
            return options_data
        except Exception as e:
            logging.error(f"Error downloading options data for {ticker}: {e}")
            return pd.DataFrame()
    
    def download_financials(self, ticker: str) -> pd.DataFrame:
        # Downloads financial data for a single ticker
        try:
            self._validate_input(ticker)
            raw_data= yf.Ticker(ticker).financials
            if raw_data.empty:
                logging.warning(f"No financial data found for the {ticker}.")
                return pd.DataFrame()
            data = raw_data.transpose()
            data["Symbol"] = ticker
            data = data.dropna(axis=1, how='all')  # Drop columns with all NaN values
            data = data.reset_index()
            return data
        except Exception as e:
            logging.error(f"Error downloading financial data for {ticker}: {e}")
            return pd.DataFrame()
    
    def download_all_data(self, ticker: str, interval: str, start_date: str, end_date: str) -> TickerData:
        # Downloads all data for a single ticker
        try:
            self._validate_input(ticker, interval, start_date, end_date)
            market_data = self.market_data(ticker, interval, start_date, end_date)
            options_data = self.download_options_data(ticker)
            financials = self.download_financials(ticker)
            return TickerData(
                market_data=market_data,
                options_data=options_data,
                financials=financials
            )
        except Exception as e:
            logging.error(f"Error downloading all data for {ticker}: {e}")
            return TickerData(
                market_data=pd.DataFrame(),
                options_data=pd.DataFrame(),
                financials=pd.DataFrame()
            )
