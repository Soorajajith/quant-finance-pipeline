import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import logging
import plotly.figure_factory as ff
from typing import List 

class PlotterClass:
    "Class for plotting financial data"
    template = 'plotly_dark'
    
    class MarketDataPlotter:
        @staticmethod
        def plot_closing_data(market_data: pd.DataFrame) -> go.Figure:
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
        def plot_candlestick_data(market_data: pd.DataFrame, sma1: int = 0, sma2: int = 0) -> go.Figure:
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
        def plot_rolling_volatility(market_data: pd.DataFrame) -> go.Figure:
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
        def plot_high_low_range(market_data: pd.DataFrame) -> go.Figure:
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
        
        @staticmethod
        def plot_return_distribution(market_data: pd.DataFrame, descriptive_data: pd.DataFrame, gaussian_kde: pd.DataFrame) -> go.Figure:
            """Plot return distribution with histogram and Gaussian KDE"""
            if market_data.empty or descriptive_data.empty:
                logging.warning("Market data is empty. Cannot plot return distribution.")
                return
            histogram_plot = go.Figure()
            # Histogram
            histogram_plot.add_trace(go.Histogram(
                x=market_data["returns"],
                histnorm="probability density",
                name="Return Histogram",
                opacity=0.6
            ))
            # KDE line
            histogram_plot.add_trace(go.Scatter(
                x=gaussian_kde["x"],
                y=gaussian_kde["kde"],
                mode="lines",
                name="KDE",
                line=dict(color="red", width=2)
            ))
            # Mean and Median lines
            histogram_plot.add_vline(
                x=descriptive_data["returns_mean"].iloc[0],
                line_dash="dash", line_color="green",
                annotation_text="Mean", annotation_position="top right"
            )
            histogram_plot.add_vline(
                x=descriptive_data["returns_median"].iloc[0],
                line_dash="dot", line_color="blue",
                annotation_text="Median", annotation_position="top left"
            )
            # Skewness & Kurtosis annotation
            skew = descriptive_data["skewness"].iloc[0]
            kurt = descriptive_data["kurtosis"].iloc[0]
            histogram_plot.add_annotation(
                x=0.98, y=0.95, xref="paper", yref="paper", showarrow=False,
                text=f"Skewness: {skew:.2f}<br>Kurtosis: {kurt:.2f}",
                align="right", bgcolor="black", bordercolor="black", opacity=0.8
            )

            histogram_plot.update_layout(
                title="Returns Distribution with KDE",
                xaxis_title="Returns",
                yaxis_title="Density",
                template=PlotterClass.template
            )
            return histogram_plot
        @staticmethod
        def plot_rolling_volume_average(data: pd.DataFrame, volume_data: pd.DataFrame) -> go.Figure:
            if data.empty or volume_data.empty:
                logging.warning("Market data or Volume data empty. Cannot plot volume distribution")
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Scatter(
                x=data.index,
                y=data["Close"],
                mode='lines',
                name='Close Price',
                line=dict(color="blue")
            ))
            volume_fig.add_trace(go.Scatter(
                x=volume_data.index,
                y=volume_data["vwap"],
                mode='lines',
                name='VWAP',
                line=dict(color="orange", dash="dot")
            ))
            volume_fig.update_layout(
                title="Price vs VWAP",
                xaxis_title="Date",
                yaxis_title="Price",
                template=PlotterClass.template
            )
            return volume_fig
        @staticmethod
        def plot_volume_rolling(data: pd.DataFrame, rolling_volume_data: pd.DataFrame) -> go.Figure:
            if data.empty or rolling_volume_data.empty:
                logging.warning("Market data or Volume data empty. Cannot plot volume distribution")
            rolling_volume_fig = go.Figure()
            rolling_volume_fig.add_trace(go.Bar(
                x=data.index,
                y=data["Volumes"],
                name="Daily Volume",
                marker=dict(color="lightblue")
            ))
            rolling_volume_fig.add_trace(go.Scatter(
                x=rolling_volume_data.index,
                y=rolling_volume_data["rolling_volume_20"],
                mode="lines",
                name="20-day Rolling Volume",
                line=dict(color="red")
            ))
            rolling_volume_fig.add_trace(go.Scatter(
                x=rolling_volume_data.index,
                y=rolling_volume_data["rolling_volume_50"],
                mode="lines",
                name="50-day Rolling Volume",
                line=dict(color="green")
            ))
            rolling_volume_fig.update_layout(
                title="Volume with Rolling Averages",
                xaxis_title="Date",
                yaxis_title="Volume",
                template=PlotterClass.template,
                barmode="overlay"
            )
            return rolling_volume_fig

        @staticmethod
        def plot_price_with_sma(market_data: pd.DataFrame) -> go.Figure:
            """Plot price with rolling averages (20d, 50d, 200d)."""
            if market_data.empty:
                logging.warning("Market data is empty. Cannot plot SMA.")
                return

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=market_data["Date"], y=market_data["Close"],
                                    mode="lines", name="Close"))
            for w in [20, 50, 200]:
                if f"SMA_{w}" in market_data.columns:
                    fig.add_trace(go.Scatter(
                        x=market_data["Date"], y=market_data[f"SMA_{w}"],
                        mode="lines", name=f"SMA {w}"
                    ))
            fig.update_layout(title="Price with Rolling Averages",
                            xaxis_title="Date", yaxis_title="Price",
                            template=PlotterClass.template)
            return fig

        @staticmethod
        def plot_price_vs_volume_dual(market_data: pd.DataFrame) -> go.Figure:
            """Dual-axis chart: Price vs Volume (overlayed)."""
            if market_data.empty:
                logging.warning("Market data is empty. Cannot plot price vs volume.")
                return

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=market_data["Date"], y=market_data["Close"],
                name="Close Price", line=dict(color="blue")
            ), secondary_y=False)
            fig.add_trace(go.Bar(
                x=market_data["Date"], y=market_data["Volume"],
                name="Volume", opacity=0.3
            ), secondary_y=True)

            fig.update_layout(title="Price vs Volume (Dual Axis)",
                            template=PlotterClass.template)
            fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
            fig.update_yaxes(title_text="Volume", secondary_y=True)
            return fig
        @staticmethod
        def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title="Correlation Heatmap") -> go.Figure:
            "Plot heatmap of correlation"
            if corr_matrix.empty:
                logging.warning("Correlation matrix is empty. Cannot plot the correlation")
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                annotation_text=corr_matrix.round(2).values,
                colorscale="Viridis",
                showscale=True,
                hoverinfo="z"
            )
            fig.update_layout(
                title=title,
                xaxis=dict(tickangle=45),
                yaxis=dict(autorange="reversed")
            )
            return fig
    class FinancialPlotter:
        @staticmethod
        def plot_trend(financials: pd.DataFrame, cols: List[str] = None, title: str = "Financial Trends") -> go.Figure:
            """
            Plot line trends for selected financial columns (Revenue, EBITDA, Net Income, EPS, etc.)
            """
            if financials.empty:
                logging.warning("Financials data is empty. Cannot plot trend.")
                return go.Figure()
            
            if cols is None:
                cols = [c for c in financials.columns if c not in ['Date', 'Symbol']]
            
            fig = go.Figure()
            for col in cols:
                if col in financials.columns:
                    fig.add_trace(go.Scatter(
                        x=financials['Date'], y=financials[col],
                        mode='lines+markers', name=col
                    ))
            fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')
            return fig

        @staticmethod
        def plot_growth(financials: pd.DataFrame, col: str, title: str = None) -> go.Figure:
            """
            Plot growth rates for a single metric column (e.g., Total Revenue_growth_YoY)
            """
            if financials.empty or col not in financials.columns:
                logging.warning(f"{col} not found in financials data. Cannot plot growth.")
                return go.Figure()
            
            if title is None:
                title = f"{col} Growth"
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=financials['Date'], y=financials[col], name=col))
            fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Growth (%)')
            return fig

        @staticmethod
        def plot_eps_vs_stock(financials: pd.DataFrame, market_data: pd.DataFrame, eps_col: str = "Diluted EPS", stock_col: str = "Close", title: str = "EPS vs Stock Price") -> go.Figure:
            """
            Dual-axis plot of EPS and stock price over time
            """
            if financials.empty or market_data.empty:
                logging.warning("Financial or market data is empty. Cannot plot EPS vs Stock.")
                return go.Figure()
            
            df = financials.merge(market_data[['Date', stock_col]], on='Date', how='left')
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=df['Date'], y=df[eps_col], name='EPS', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=df['Date'], y=df[stock_col], name='Stock Price', mode='lines'), secondary_y=True)
            fig.update_layout(title=title, xaxis_title='Date')
            fig.update_yaxes(title_text=eps_col, secondary_y=False)
            fig.update_yaxes(title_text=stock_col, secondary_y=True)
            return fig

        @staticmethod
        def plot_ratios(financials: pd.DataFrame, ratio_cols: List[str], title: str = "Financial Ratios") -> go.Figure:
            """
            Plot any ratios already computed (e.g., P/E, P/B, ROE, EBITDA margin)
            """
            if financials.empty or not ratio_cols:
                logging.warning("Financial data or ratio columns are empty. Cannot plot ratios.")
                return go.Figure()
            
            fig = go.Figure()
            for r in ratio_cols:
                if r in financials.columns:
                    fig.add_trace(go.Scatter(x=financials['Date'], y=financials[r], name=r, mode='lines+markers'))
            
            fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Value')
            return fig

        @staticmethod
        def plot_revenue_vs_stock_growth(df: pd.DataFrame, rev_col: str = "revenue_growth_yoy", stock_col: str = "stock_returns", title: str = "Revenue Growth vs Stock Returns") -> go.Figure:
            """
            Scatter plot of revenue growth vs stock returns (already computed)
            """
            if df.empty or rev_col not in df.columns or stock_col not in df.columns:
                logging.warning("Data or columns missing. Cannot plot revenue vs stock.")
                return go.Figure()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[rev_col], y=df[stock_col],
                                    mode='markers', name='Revenue vs Stock'))
            fig.update_layout(title=title, xaxis_title='Revenue Growth (%)', yaxis_title='Stock Returns (%)')
            return fig

    class OptionsPlotter:     
        @staticmethod
        def plot_open_interest_by_strike(options_df, top_n=20, ticker='TICKER'):
            """Bar chart: Open Interest by strike (calls vs puts)"""
            if options_df.empty:
                return go.Figure()
            
            calls = options_df[options_df['Type']=='calls'].sort_values('openInterest', ascending=False).head(top_n)
            puts = options_df[options_df['Type']=='put'].sort_values('openInterest', ascending=False).head(top_n)
            
            fig = go.Figure()
            if not calls.empty:
                dfc = calls.iloc[::-1]
                fig.add_trace(go.Bar(x=dfc['openInterest'], y=dfc['strike'].astype(str),
                                    name='Calls OI', orientation='h'))
            if not puts.empty:
                dfp = puts.iloc[::-1]
                fig.add_trace(go.Bar(x=dfp['openInterest'], y=dfp['strike'].astype(str),
                                    name='Puts OI', orientation='h'))
            fig.update_layout(title=f"{ticker} Open Interest by Strike (Top {top_n})",
                            xaxis_title="Open Interest", yaxis_title="Strike",
                            barmode='group')
            return fig

        @staticmethod
        def plot_iv_term_structure(iv_term_df, ticker='TICKER'):
            """Line chart: IV term structure"""
            if iv_term_df.empty:
                return go.Figure()
            
            iv_col = [c for c in iv_term_df.columns if c.startswith('IV')][0]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=iv_term_df['time_to_expiry'], y=iv_term_df[iv_col],
                                    mode='lines+markers', name='IV'))
            fig.update_layout(title=f"{ticker} Implied Volatility Term Structure",
                            xaxis_title="Time to Expiry (years)",
                            yaxis_title="Implied Volatility")
            return fig

        @staticmethod
        def plot_vol_surface(vol_surface_df, ticker='TICKER'):
            """3D Surface plot of volatility surface"""
            if vol_surface_df.empty:
                return go.Figure()
            
            piv = vol_surface_df.pivot_table(index='expiry', columns='strike', values='IV', aggfunc='median')
            piv = piv.sort_index()
            strikes = np.array(piv.columns.tolist(), dtype=float)
            expiries = np.array(piv.index.tolist())
            Z = piv.values
            
            y = np.array([(pd.to_datetime(e) - pd.to_datetime(expiries[0])).days for e in expiries], dtype=float)
            x = strikes.astype(float)
            
            fig = go.Figure()
            fig.add_trace(go.Surface(z=Z, x=x, y=y))
            fig.update_layout(title=f"{ticker} Implied Volatility Surface",
                            scene=dict(xaxis_title="Strike",
                                        yaxis_title="Days from first expiry",
                                        zaxis_title="Implied Volatility"))
            return fig

        @staticmethod
        def plot_iv_vs_underlying(iv_history_df, underlying_df, ticker='TICKER'):
            """
            Plot IV (median across strikes) vs underlying price time series
            iv_history_df: ['as_of', 'impliedVolatility']
            underlying_df: ['Date', 'Close']
            """
            if iv_history_df.empty or underlying_df.empty:
                return go.Figure()
            
            iv_ts = iv_history_df.groupby('as_of')['impliedVolatility'].median().reset_index()
            merged = iv_ts.merge(underlying_df.rename(columns={'Date':'as_of'}), on='as_of', how='left')
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=merged['as_of'], y=merged['impliedVolatility'],
                                    name='IV', mode='lines+markers'), secondary_y=False)
            fig.add_trace(go.Scatter(x=merged['as_of'], y=merged['Close'],
                                    name='Underlying', mode='lines'), secondary_y=True)
            fig.update_layout(title=f"{ticker} IV vs Underlying Price",
                            xaxis_title='Date')
            fig.update_yaxes(title_text="Implied Volatility", secondary_y=False)
            fig.update_yaxes(title_text="Underlying Price", secondary_y=True)
            return fig