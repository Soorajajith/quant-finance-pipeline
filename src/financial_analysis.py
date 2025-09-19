import pandas as pd
import numpy as np
import logging

class FinancialAnalysis:
    """
    Comprehensive financial analysis for quantitative strategies.
    Calculates trends, growth rates, ratios and turnover metrics
    """
    @staticmethod
    def prepare_financials(financial: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Filter by ticker, relevant columns, sort by Date.
        Columns: Date, Total Revenue, EBITDA, Net Income, Diluted EPS,
                 Operating Expends, R&D, SG&A, Cost of Revenue 
        """
        if financial.empty:
            logging.warning("Financial data is empty")
            return pd.DataFrame()
        financial = financial.reset_index().rename(columns={"index": "Date"})
        financial["Date"] = pd.to_datetime(financial["Date"], utc=True)
        financial = financial.set_index("Date")
        cols = ['Total Revenue', 'EBITDA', 'Net Income', 'Diluted EPS', 
                'Operating Expense', 'Research And Development', 'Selling General And Administration',
                'Cost Of Revenue', 'Symbol']
        df_fin = financial[financial['Symbol']==ticker].copy()
        df_fin = df_fin[cols].sort_index()
        numeric_cols = [c for c in df_fin.columns if c != 'Symbol']
        df_fin[numeric_cols] = df_fin[numeric_cols].astype(float)
        return df_fin
    
    @staticmethod
    def compute_growth(financial: pd.DataFrame, period='YoY') -> pd.DataFrame:
        """
        Compute growth rates for revenue, EBITDA, net income, EPS.
        period: 'YoY' (4 quarters), 'QoQ' (1 quarter)
        """
        if financial.empty:
            logging.warning("Financial data is empty. Cannot compute growth rate")
            return pd.DataFrame()
        df_growth = financial.copy()
        # Detect frequency: quarterly vs annual
        freq = pd.infer_freq(df_growth.index)
        if freq is None:
            logging.warning("Could not infer frequency, assuming annual")
            shift_period = 1 if period == 'YoY' else 1
        else:
            if period == 'YoY':
                shift_period = 4 if freq.startswith('Q') else 1
            else:  # QoQ
                shift_period = 1
        for col in ['Total Revenue', 'EBITDA', 'Net Income', 'Diluted EPS']:
            if col not in df_growth.columns:
                logging.warning(f"{col} not in financials, skipping")
                continue
            df_growth[f"{col}_growth_{period}"] = (
                (df_growth[col] / df_growth[col].shift(shift_period) - 1) * 100
            )
        return df_growth.dropna()   # don't dropna, let later steps handle alignment
    
    @staticmethod
    def compute_ratios(financial: pd.DataFrame, stock_price: pd.DataFrame) -> pd.DataFrame:
        """
        Compute P/E, P/B (if book value available), ROE, EBITDA margin
        """
        if financial.empty or stock_price.empty:
            logging.warning("Either financial daya or stock price data is available")
            return pd.DataFrame()
        df = financial.copy()
        stock_price['Date'] = stock_price['Date'].dt.tz_convert('UTC')
        # print((df.dtypes))
        df = pd.merge_asof(financial,
                   stock_price[['Date', 'Close']],
                   left_index=True,  # financial index
                   right_on='Date',
                   direction='backward')  # last available price
        df['p/e'] = df['Close'] / df['Diluted EPS']
        df['ebitda_margin'] = df['EBITDA'] / df['Total Revenue']
        # if 'Total Shareholder Equity' in df.columns:
        #     df['roe'] = df['Net Income'] / df['Total Shareholder Equity']
        # if 'Total Assets' in df.columns and 'Total Liabilities' in df.columns:
        #     df['book_value'] = df['Total Assets'] - df['Total Liabilities']
        #     df['p/b'] = df['Close'] / df['book_value']
        
        return df
    
    @staticmethod
    def compute_revenue_growth_vs_stock(financials: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Revenue YoY growth vs Stock Returns for scatter plot
        """
        if financials.empty or market_data.empty:
            logging.warning("Either financial or market data is empty")
            return pd.DataFrame()
        
        df = financials[['Total Revenue']].copy()
        freq = pd.infer_freq(df.index)
        if freq is None:
            logging.warning("Could not infer frequency, assuming annual")
            shift_period = 1
        else:
            shift_period = 4 if freq.startswith('Q') else 1
        df = pd.merge_asof(financials[['Total Revenue']], 
                   market_data[['Date','Close']], 
                   left_index=True, 
                   right_on='Date', 
                   direction='backward')  # takes last available stock price
        df['revenue_growth_yoy'] = df['Total Revenue'].pct_change(periods=shift_period) * 100
        df['stock_returns'] = df['Close'].pct_change(periods=shift_period) * 100
        print(df[['revenue_growth_yoy','stock_returns']])
        df = df.dropna(subset=['revenue_growth_yoy','stock_returns'])
        return df
    
    @staticmethod
    def compute_piotroski(financial: pd.DataFrame) -> pd.Series:
        """Compute piotroski F score"""
        fscore = pd.Series(0, index=financial.index, dtype=int)
        fscore += (financial['Net Income'] > 0).astype(int)
        fscore += (financial['Operating Cash Flow'] > 0).astype(int)
        fscore += (financial['Operating Case Flow'] > financial['Net Income']).astype(int)
        return fscore
    @staticmethod
    def compute_altman_z_score(financial: pd.DataFrame) -> pd.Series:
        """Compute Altman Zscore"""
        z = (1.2 * financial['Working Capital'] / financial['Total Assets'] +
        1.4 * financial['Retained Earnings'] / financial['Total Assets'] +
        3.3 * financial['EBIT'] / financial['Total Assets'] +
        0.6 * financial['Market Value Equity'] / financial['Total Assets'] +
        1.0 * financial['Total Revenue'] / financial['Total Assets'])
        return z

