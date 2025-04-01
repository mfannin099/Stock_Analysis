import yfinance as yf
import ta
import pandas as pd
import numpy as np

def pull_data(ticker):
    data = yf.Ticker(ticker)

    df = pd.DataFrame(data.history(period='max'))

    ##Filter to 3 years of history (if available)... else use all history
    last_date = df.index[-1]
    three_years_ago = last_date - pd.DateOffset(years=3)

    if df.index[0] < three_years_ago:
        df_filtered = df[df.index >= three_years_ago]  # Use last 3 years
        print("Filtered to last 3 years: ", df_filtered.index.min())
    else:
        df_filtered = df  # Use all data if less than 3 years
        print("Less than 3 years of data: ", df_filtered.index.min())

    return df_filtered