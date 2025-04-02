import yfinance as yf
import ta
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import holidays 
import xgboost as xgb

def pull_data(ticker):
    data = yf.Ticker(ticker)

    df = pd.DataFrame(data.history(period='max'))

    ##Filter to 3 years of history (if available)... else use all history
    last_date = df.index[-1]
    three_years_ago = last_date - pd.DateOffset(years=3)

    if df.index[0] < three_years_ago:
        df_filtered = df[df.index >= three_years_ago]  # Use last 3 years
    else:
        df_filtered = df  # Use all data if less than 3 years

    return df_filtered


def plot_data_line_chart(df, start_date, end_date):
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the DataFrame based on the selected date range
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    filtered_df["MA_20"] = filtered_df["Close"].rolling(window=20).mean() 
    filtered_df["MA_50"] = filtered_df["Close"].rolling(window=50).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['Date'], filtered_df['Close'], label="Close Price", color='blue')
    plt.plot(filtered_df['Date'], filtered_df["MA_20"], label="20-Day MA", color='orange', linestyle="dashed")
    plt.plot(filtered_df['Date'], filtered_df["MA_50"], label="50-Day MA", color='red', linestyle="dashed")

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def create_pred_df(df, days):

    df_to_model = df[['Close', 'Volume']].copy()
    predictions_df = pd.DataFrame()

    ## Creating future dates to predict
    us_holidays = holidays.UnitedStates(years = [2025,2026])
    current_datetime = datetime.datetime.today()
    current_datetime = current_datetime + datetime.timedelta(days=1)

    formatted_date = current_datetime.strftime("%m/%d/%Y")
    future_dates = pd.date_range(start=formatted_date, periods=days, freq='B')

    # Remove U.S. holidays from the generated business days
    future_dates = [date for date in future_dates if date not in us_holidays]

    # Keep only the first `days_of_trading_to_predict` valid trading days
    future_dates = future_dates[:days]

    predictions_df = pd.DataFrame({'dt': future_dates})


    return predictions_df


def naive_random_walk(df, pred_df):
    predictions_df_naive = pred_df.copy()

    # parameters for Random Walk
    mu = 0  # Mean return (0 for pure random walk)
    sigma = df['Close'].std()  # Standard deviation of returns
    last_price = df['Close'].iloc[-1]

    closes = np.random.normal(mu, sigma, len(predictions_df_naive))
    closes = closes + last_price

    predictions_df_naive['preds'] = closes

    return predictions_df_naive


def get_lag_values(df,days_of_trading_to_predict):

    ##Shorter Lag
    if (days_of_trading_to_predict + 1) < len(df):
        lag1_feature = days_of_trading_to_predict + 1
    else:
        lag1_feature = 0
        print(f'No lag feature value for: {days_of_trading_to_predict + 1} days out')

    # Longer Lag
    if ((days_of_trading_to_predict*2) + 1) < len(df):
        lag2_feature = (days_of_trading_to_predict *2) + 1
    else:
        lag2_feature = 0
        print(f'No lag feature value for: {(days_of_trading_to_predict*2) + 1} days out')

    return lag1_feature, lag2_feature

def get_closest_business_day(lag, df, pred_df, column):

    last_index_pred_df = df.index[-1]
    target_index = last_index_pred_df - lag

    # Now, get the next `len(pred_df)` rows (which is the length of predictions) from df
    end_index = target_index + len(pred_df) # Make sure we don't exceed df length

    # Extract the 'Close' values for the range of indices from target_index to end_index
    rows = df.loc[target_index:end_index-1, column].values  # Adjust end_index to be exclusive
    return rows


def xgboost(df,pred_df, days_of_trading_to_predict):

    xgboost_df = df[['Close', 'Volume', 'Date']].copy()
    lag1, lag2 = get_lag_values(df,days_of_trading_to_predict)
    predictions_df_xgb = pred_df.copy()
    reference_date = xgboost_df['Date'].min()

    ##Begin Feature Creation for train data
    time_diff = reference_date - xgboost_df['Date']
    xgboost_df['days_past'] = time_diff.dt.days
    xgboost_df['months_past'] = time_diff.dt.days // 30  # Approximation of months
    xgboost_df['years_past'] = time_diff.dt.days // 365  # Approximation of years
    xgboost_df[['days_past', 'months_past', 'years_past']] = xgboost_df[['days_past', 'months_past', 'years_past']].apply(lambda x: x.abs())

    xgboost_df['month'] = xgboost_df['Date'].dt.month
    xgboost_df['day'] = xgboost_df['Date'].dt.day
    xgboost_df["day_of_week"] = xgboost_df['Date'].dt.weekday

    # Adding Lag Features
    xgboost_df[f'lag_{lag1}'] = xgboost_df['Close'].shift(lag1).fillna(0)
    xgboost_df[f'lag_{lag2}'] = xgboost_df['Close'].shift(lag2).fillna(0)

    xgboost_df[f'lag_{lag1}_volume'] = xgboost_df['Volume'].shift(lag1).fillna(0)
    xgboost_df[f'lag_{lag2}_volume'] = xgboost_df['Volume'].shift(lag2).fillna(0)


    ## Feature creation for the prediction dataset
    time_diff = reference_date - predictions_df_xgb['dt']

    predictions_df_xgb['days_past'] = time_diff.dt.days
    predictions_df_xgb['months_past'] = time_diff.dt.days // 30  # Approximation of months
    predictions_df_xgb['years_past'] = time_diff.dt.days // 365  # Approximation of years
    predictions_df_xgb[['days_past', 'months_past', 'years_past']] = predictions_df_xgb[['days_past', 'months_past', 'years_past']].apply(lambda x: x.abs())

    predictions_df_xgb['dt'] = pd.to_datetime(predictions_df_xgb['dt'])
    predictions_df_xgb['month'] = predictions_df_xgb['dt'].dt.month
    predictions_df_xgb['day'] = predictions_df_xgb['dt'].dt.day
    predictions_df_xgb['day_of_week'] = predictions_df_xgb['dt'].dt.weekday


    # Adding Lag Features
    predictions_df_xgb[f'lag_{lag1}'] = get_closest_business_day(lag1, xgboost_df, predictions_df_xgb, "Close")
    predictions_df_xgb[f'lag_{lag2}'] = get_closest_business_day(lag2, xgboost_df, predictions_df_xgb, "Close")

    predictions_df_xgb[f'lag_{lag1}_volume'] = get_closest_business_day(lag1, xgboost_df, predictions_df_xgb, "Volume")
    predictions_df_xgb[f'lag_{lag2}_volume'] = get_closest_business_day(lag2, xgboost_df, predictions_df_xgb, "Volume")



    X_train = xgboost_df[['days_past', 'months_past', 'years_past', f'lag_{lag1}', f'lag_{lag2}', 'month',f'lag_{lag1}_volume', f'lag_{lag2}_volume', 'day', 'day_of_week']]
    X_train = X_train.fillna(0)
    y_train = xgboost_df['Close']  

    X_forecast = predictions_df_xgb[['days_past', 'months_past', 'years_past', f'lag_{lag1}', f'lag_{lag2}', 'month',f'lag_{lag1}_volume', f'lag_{lag2}_volume', 'day', 'day_of_week']]


    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_forecast)
    predictions_df_xgb['preds'] = y_pred
    predictions_df_xgb = predictions_df_xgb[['dt', 'preds']].copy()

    return predictions_df_xgb

def plot_oos(df_actual, df_preds):
    
    # Get the most recent date in the df_actual
    most_recent_date = df_actual['Date'].max()
    
    # Filter the last year of data from df_actual
    one_year_ago = most_recent_date - pd.DateOffset(years=1)
    df_actual_filtered = df_actual[df_actual['Date'] >= one_year_ago]

    # Plot the actual and predicted data
    plt.figure(figsize=(12, 6))

    sns.lineplot(x=df_actual_filtered['Date'], y=df_actual_filtered['Close'], label="Actual Close Price", color="blue")
    sns.lineplot(x=df_preds['dt'], y=df_preds['preds'], label="Prediction", color="red", linestyle="dashed")

    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Actual vs. Predicted Stock Prices")
    plt.legend()

    # Show the plot
    plt.show()
