import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import logging

class PlotterClass:
    "Class for plotting financial data"
    template = 'plotly_dark'
    
    class MarketData:
        @staticmethod
        def plot_closing_data(market_data: pd.DataFrame):
            "Plot closing data from the market data "
            closing_fig = go.Figure()
            closing_fig.add_trace(go.Scatter(
                x=market_data["Date"],
                y=market_data["Close"],
                mode='lines',
                name='Close Price'
            ))
            closing_fig.update_layout(
                title='Closing Price Over Time',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                template=PlotterClass.template
            )
            return closing_fig
        
        @staticmethod
        def plot_candlestick_data(market_data: pd.DataFrame, sma1: int = 0, sma2: int = 0):
            "Plot candlestick data from the market data"
            if market_data.empty:
                logging.warning("Market data is empty. Cannot plot candlestick chart.")
                return  
            candlestick_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])
            candlestick_fig.add_trace(go.Candlestick(
                x=market_data["Date"],
                open=market_data["Open"],
                high=market_data["High"],
                low=market_data["Low"],
                close=market_data["Close"],
                name='Candlestick'
            ), row=1, col=1)
            candlestick_fig.update_traces(increasing_line_width=1.5, decreasing_line_width=1.5)
            if sma1 > 0:
                candlestick_fig.add_trace(go.Scatter(
                    x=market_data["Date"],
                    y=market_data["Close"].rolling(window=int(sma1)).mean(),
                    mode='lines',
                    name=f'SMA {sma1} - moving day average',
                    line=dict(color='lightblue', width=1.5)
                ), row=1, col=1)
            if sma2 > 0:
                candlestick_fig.add_trace(go.Scatter(
                    x=market_data["Date"],
                    y=market_data["Close"].rolling(window=int(sma2)).mean(),
                    mode='lines',
                    name=f'SMA {sma2} - moving day average',
                    line=dict(color='red', width=1.5)
                ), row=1, col=1)
            candlestick_fig.add_trace(go.Bar(
                x=market_data["Date"],
                y=market_data["Volume"],
                name='Volume',
            ), row=2, col=1)
            candlestick_fig.update_layout(
                title = {
                    'text': f' {market_data["Symbol"][0]} Candlestick Chart with volume',
                    'x' : 0.5,
                    'y': 0.9,
                    'font': {'size' : 15},
                    'xanchor': 'center',
                    'yanchor': 'top',},
                    xaxis_rangeslider_visible=False,
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    hovermode='x',
                    bargap = 0,
                    bargroupgap = 0.0,
                    template=PlotterClass.template)
            candlestick_fig.update_xaxes(title_text = '',row=1, col=1, showgrid=True)
            candlestick_fig.update_xaxes(title_text = 'Date',row=2, col=1)
            candlestick_fig.update_yaxes(title_text = 'Volume',row=2, col=1)
            return candlestick_fig
        
        @staticmethod
        def plot_rolling_volatility(market_data: pd.DataFrame):
            "Plot rolling volatility from the market data"
            if market_data.empty:
                logging.warning("Market data is empty. Cannot plot rolling volatility.")
                return
            volatility_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])
            volatility_fig.add_trace(go.Scatter(
                x=market_data["Date"],
                y=market_data["volatility"],
                mode='lines',
                name='Rolling Volatility'
            ), row=1, col=1)
            volatility_fig.add_trace(go.Scatter(
                x=market_data["Date"],
                y=market_data["volatility_log"],
                mode='lines',
                name='Log Returns Volatility'
            ), row=1, col=1)
            volatility_fig.add_trace(go.Scatter(
                x=market_data["Date"],
                y=market_data["returns"],
                mode='lines',
                name='Returns'
            ), row=2, col=1)
            volatility_fig.add_trace(go.Scatter(
                x=market_data["Date"],
                y=market_data["log_returns"],
                mode='lines',
                name='Log Returns'
            ), row=2, col=1)
            volatility_fig.update_layout(
                title="Returns and Rolling volatility Over Time",
                xaxis2_title='Date',
                yaxis_title='Volatility',
                yaxis2_title='Returns',
                template=PlotterClass.template
            )
            return volatility_fig
            
        @staticmethod
        def plot_high_low_range(market_data: pd.DataFrame):
            if market_data.empty:
                logging.warning("Market data is empty. Cannot plot high-low range.")
                return
            high_low_fig = go.Figure()
            high_low_fig.add_trace(go.Scatter(
                x=market_data["Date"],
                y=market_data["High"] - market_data["Low"],
                mode='lines',
                name='High Price',
                line=dict(color='green')
            ))
            high_low_fig.update_layout(
                title='High-Low Range Over Time',
                xaxis_title='Date',
                yaxis_title='Price Range (USD)',
                template=PlotterClass.template
            )
            return high_low_fig
            


                
