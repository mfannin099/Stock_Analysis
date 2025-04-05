import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("..") 
from utils import pull_data
from utils import plot_data_line_chart
from utils import create_pred_df
from utils import naive_random_walk
from utils import get_lag_values
from utils import xgboost
from utils import plot_oos
import streamlit as st

# Get and pull data from Yahoo Finance
st.title("Stock Predictions and Analysis")
ticker = st.text_input("Enter valid stock ticker: ")
if not ticker:
    st.write("Please Input a Valid Ticker to Continue")
else:
    df = pull_data(ticker)
    df = df.reset_index()
    df['Date'] = df['Date'].dt.tz_localize(None).dt.date

    st.divider()
    st.subheader("Stock Price Chart")


    # Plot stock data
    start_date = st.date_input('Start Date', df['Date'].min())
    end_date = st.date_input('End Date', df['Date'].max())
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Add Moving Averages
    filtered_df["MA_20"] = filtered_df["Close"].rolling(window=20).mean()
    filtered_df["MA_50"] = filtered_df["Close"].rolling(window=50).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['Date'], filtered_df['Close'], label="Close Price", color='blue')
    plt.plot(filtered_df['Date'], filtered_df["MA_20"], label="20-Day MA", color='orange', linestyle="dashed")
    plt.plot(filtered_df['Date'], filtered_df["MA_50"], label="50-Day MA", color='red', linestyle="dashed")

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt)


    st.divider()
    st.subheader("RSI (Relative Stength Index)")

    # Plot Relative Strength Index
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    macd_indicator = ta.trend.MACD(df['Close'])
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()

    plt.figure(figsize=(12, 4))
    sns.lineplot(data=df, x=df.index, y='RSI', color='purple', linewidth=2)
    plt.axhline(70, color='red', linestyle='dashed', label="Overbought (70)")
    plt.axhline(30, color='green', linestyle='dashed', label="Oversold (30)")
    plt.title(f'{ticker} RSI Indicator')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    st.pyplot(plt)

    st.divider()
    st.subheader("Predict Future Prices")

    days_of_trading_to_predict = st.text_input("Enter Trading Days to Predict i.e (100): ")
    xgboost_weight = st.text_input("Enter weight of XGBoost.. closer to 1 creates more stable model i.e (0-1)): ")
    naive_weight = st.text_input("Enter weight of Random Walk.. closer to 1 creates more random model i.e (0-1)): ")




    if not (days_of_trading_to_predict and xgboost_weight and naive_weight):
        st.write("Fill out all Model Parameters to Create Predictions")
    else:
        days_of_trading_to_predict = int(days_of_trading_to_predict)
        xgboost_weight = float(xgboost_weight)
        naive_weight = float(naive_weight) 

        if (xgboost_weight + naive_weight) != 1:
            st.write("Sum of Weights MUST equal 1")
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # 'coerce' turns invalid parsing into NaT

            pred_df = create_pred_df(df,days_of_trading_to_predict)

            ##Creating predictions
            preds_naive = naive_random_walk(df, pred_df)
            preds_xgboost = xgboost(df,pred_df,days_of_trading_to_predict)

            ## plotting predictions
            pred_df['preds'] = (xgboost_weight * preds_xgboost['preds']) + (naive_weight * preds_naive['preds'])

            # Plot predicts
            # Get the most recent date in the df_actual
            most_recent_date = df['Date'].max()
            
            # Filter the last year of data from df_actual
            one_year_ago = most_recent_date - pd.DateOffset(years=1)
            df_actual_filtered = df[df['Date'] >= one_year_ago]

            # Plot the actual and predicted data
            plt.figure(figsize=(12, 6))

            sns.lineplot(x=df_actual_filtered['Date'], y=df_actual_filtered['Close'], label="Actual Close Price", color="blue")
            sns.lineplot(x=pred_df['dt'], y=pred_df['preds'], label="Prediction", color="red", linestyle="dashed")

            plt.xlabel("Date")
            plt.ylabel("Stock Price")
            plt.title("Actual vs. Predicted Stock Prices")
            plt.legend()
            st.pyplot(plt)

    st.divider()

