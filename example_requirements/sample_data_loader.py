import pandas as pd
import yfinance as yf
from datetime import datetime
import logging 

class DataLoader:
    def __init__(self, source: str, tickers: list):
        self.source = source
        self.tickers = tickers

    def _validate_input(self, ticker: str, interval: str, start_date: str, end_date: str) ->bool:
        # Validate input parameters
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string.")
        valid_intervals = ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max']
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
            history_data = yf.download(ticker, interval=interval, start=start_date, end=end_date)
            if history_data.empty:
                logging.warning(f"No historical data found for {ticker} from {start_date} to {end_date}.")
                return {}
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error downloading historical data for {ticker}: {e}")
            return {}
    
    def market_data(self, ticker: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Download market data
        try:
            self._validate_input(ticker, interval, start_date, end_date)
            raw_data = yf.Ticker(ticker).history(start=start_date, end=start_date, interval=interval)
            if raw_data.empty :
                logging.warning(f"No data found for {ticker} from {start_date} to {end_date}.")
                return {}
            data = raw_data[["Open", "High", "Low", "Close", "Volume"]]
            data = data.dropna()
            data = data.astype(float)
            data["Symbol"] = ticker
            return data.reset_index()
        except Exception as e:
            logging.error(f"Error downloading market data for {ticker}: {e}")
            return pd.DataFrame()
    
    def download_options_data(self, ticker: str, interval: str, start_date: str, end_date: str) -> dict:
        # Downloads options data for a single ticker
        try:
            self._validate_input(ticker, interval= interval, start_date=start_date, end_date=end_date)
            options_data = yf.Ticker(ticker).option_chain()
            if options_data.empty:
                logging.warning(f"No options data found for {ticker} from {start_date} to {end_date}.")
                return {}
            return {'options_data': options_data}
        except Exception as e:
            logging.error(f"Error downloading options data for {ticker}: {e}")
            return {}
    
    def download_financials(self, ticker: str, interval: str, start_date: str, end_date: str) -> dict:
        # Downloads financial data for a single ticker
        try:
            financials= yf.Ticker(ticker).financials
            if financials.empty:
                logging.warning(f"No financial data found for {ticker} from {start_date} to {end_date}.")
                return {}
            return {'financials': financials}
        except Exception as e:
            logging.error(f"Error downloading financial data for {ticker}: {e}")
            return {}
    
    def download_all_data(self, ticker: str, interval: str, start_date: str, end_date: str) -> dict:
        # Downloads all data for a single ticker
        try:
            self._validate_input(ticker, interval, start_date, end_date)
            history_data = self.download_history_data(ticker, interval, start_date, end_date)
            market_data = self.market_data(ticker, interval, start_date, end_date)
            options_data = self.download_options_data(ticker, interval, start_date, end_date)
            financials = self.download_financials(ticker, interval, start_date, end_date)
            return {
                'history_data': history_data.get('history_data'),
                'market_data': market_data.get('market_data'),
                'options_data': options_data.get('options_data'),
                'financials': financials.get('financials')
            }
        except Exception as e:
            logging.error(f"Error downloading all data for {ticker}: {e}")
            return {}
