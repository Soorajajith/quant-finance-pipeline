# Quantitative Finance Data Pipeline

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

This repository provides a **Python-based pipeline** to download, process, and analyze financial and options data for quantitative finance research and feature engineering. It is designed for tasks such as computing growth features, analyzing market signals, and visualizing financial data — ideal for factor modeling or quant strategy prototyping.

---

## 📂 Repository Structure

- **`financial_analysis.py`**  
  Functions to download, clean, and process financial statement data (income statement, balance sheet, cash flow).  
  Key functions:
  - `prepare_financials(financial, ticker)`  
    Filters by ticker, selects relevant columns, sets `Date` as index, and converts numeric fields to float.
  - `compute_growth(financial, period='YoY')`  
    Computes quarterly (QoQ) or yearly (YoY) growth for revenue, EBITDA, net income, and EPS. Handles both annual and quarterly data.

- **`options_analysis.py`**  
  Functions for downloading, parsing, and analyzing options data.  
  Includes pricing, implied volatility, and Greeks computations.

- **`data_plotter.py`**  
  Functions to visualize financial and market data, such as revenue growth, EPS growth, stock price trends, and options surfaces.

---

## ⚡ Features

1. **Financial Feature Engineering**
   - YoY and QoQ growth for key financial metrics
   - Handles quarterly and annual reporting periods
   - Produces numeric data ready for factor analysis or ML

2. **Ticker-based Filtering**
   - Extracts data for a single company using its ticker
   - Aligns financials with market time series

3. **Options Analysis**
   - Extract options chain data
   - Compute standard options metrics for trading strategies

4. **Visualization**
   - Plot historical financial performance and growth metrics
   - Supports exploratory data analysis and presentation-ready charts

---

## 🔧 Usage Example

```python
import data_loader as dl
import data_plotter as dp
import stats_analysis as sa
import options_analysis as oa
import financial_analysis as fa
stock_data = dl.DataLoader()
market_data_plotter_func = dp.PlotterClass().MarketDataPlotter()
statistics_analysis_func = sa.StatsAnalysis()
options_analysis_func = oa.OptionsAnalysis()
financial_analysis_func = fa.FinancialAnalysis()

# Download financials
ticker="AAPL"
interval="1d"
start_date="2019-01-01"
end_date="2023-01-01"
market_data = stock_data.market_data(ticker,  interval="1d", start_date="2019-01-01", end_date="2023-01-01")
financial_data = stock_data.download_financials(ticker)
options_data = stock_data.download_options_data(ticker)

# Prepare financials
###
plot_data = market_data_plotter_func.plot_closing_data(market_data)
candlestick_data = market_data_plotter_func.plot_candlestick_data(market_data)
###
stats_data = statistics_analysis_func.calculate_returns(market_data, 20)
return_descriptive_stats = statistics_analysis_func.calculate_descriptive_stats(stats_data)
kde_estimation = statistics_analysis_func.estimate_return_distribution(stats_data)
###
# print(financial_data.columns)
process_financial_data = financial_analysis_func.prepare_financials(financial_data, ticker)
# print(process_financial_data['Diluted EPS'])
financial_growth = financial_analysis_func.compute_growth(process_financial_data, period='YoY')
ratios = financial_analysis_func.compute_ratios(process_financial_data,market_data)
revenue_growth = financial_analysis_func.compute_revenue_growth_vs_stock(process_financial_data,market_data)
