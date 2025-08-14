import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import data_loader as dl
import pandas as pd
from unittest.mock import patch, MagicMock

def test_load_data():
    data = {
        "2023-05-10":[100,100,None],
        "2023-05-11":[200,200,None]
    }
    index = ["Total Revenue", "Net Income", "EmptyMetric"]
    return pd.DataFrame(data, index=index)

@patch("yfinance.Ticker")
def test_load_data_with_mock(mock_ticker_class):
    mock_instance = MagicMock()
    mock_instance.financials = test_load_data()
    mock_ticker_class.return_value = mock_instance

    df = dl.DataLoader().download_financials("AAPL")
    ## Assert that the DataFrame is not empty
    assert not df.empty
    assert["Symbol"] in df.columns
    # assert "Total Revenue" in df.columns
    assert "EmptyMetric" not in df.columns  # Should be dropped because all NaNs
    assert df["Symbol"].iloc[0] == "AAPL"

