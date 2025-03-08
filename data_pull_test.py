import yfinance as yf
import pandas as pd
import numpy as np

data = yf.Ticker("NKE")
print(data) ## Prints a ticker object
print(data.info)
print("--------------------")
print("--------------------")
print(data.history(period='max')) ## must be of the format 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max, etc.

nke_df = pd.DataFrame(data.history(period='max'))
nke_df.head()