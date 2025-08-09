class DataLoader:
    def __init__(self, source: str, tickers: list):
        self.source = source
        self.tickers = tickers

    def download(self, start_date: str, end_date: str) -> dict:
        # Downloads data for all tickers, returns dict of DataFrames
        pass

    def _download_single_ticker(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Helper for individual ticker download
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Cleans data (missing values, duplicates)
        pass

    def save_data(self, data: dict, path: str) -> None:
        # Saves data to disk
        pass
