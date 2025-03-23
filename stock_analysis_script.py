import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

def monte_carlo_simulation(start_price, n_steps, n_simulations):
    simulations = np.zeros((n_simulations, n_steps))

    for i in range(n_simulations):
        price = start_price
        for j in range(n_steps):
            random_shock = np.random.normal(0.08, 1.5) #.08 comes from the average return of the S&P 500
            price = price + random_shock
            simulations[i, j] = price
    
    return simulations

ticker = input("Please enter vaild ticker to analyze: ")
print('User input was:', ticker)
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

df_filtered['Intraday'] = df_filtered['Close'] - df_filtered['Open']
# Calculate Log Returns
df_filtered['Log Return'] = np.log(df_filtered['Close'] / df_filtered['Close'].shift(1))
df_filtered['Pct Change'] = df_filtered['Close'].pct_change(periods=1)

##Basic EDA
print(df_filtered[['Close', 'Intraday','Volume']].describe())
print("-----------------------------")

df_filtered['Close_bin'] = pd.qcut(df_filtered['Close'], q=10, labels=False)
print(df_filtered.groupby('Close_bin')['Close'].describe())
print("---------------------")

## Fit Linear Regression Line
# Feature Creation
df_filtered['year'] = df_filtered.index.year
df_filtered['month'] = df_filtered.index.month
df_filtered['day'] = df_filtered.index.day

X = df_filtered[['year', 'month']]
X = sm.add_constant(X)  # Adds intercept term
y = df_filtered['Close']

# Fit the model
model = sm.OLS(y, X).fit()

# Get coefficients
intercept = model.params['const']
slope_year = model.params['year'] # this should really be # of months that have past
slope_month = model.params['month'] #this should really be # of days that have past

# Print the regression equation
print(f"Equation of the line: y = {intercept:.2f} + {slope_year:.2f} * year + {slope_month:.2f} * month")
print(model.summary())
print("---------------------")
print("--------------------")

## Monte Carlo Simulation
n = 1000
n_steps = 252
start_price = df_filtered['Close'].iloc[0]

simulated_prices = monte_carlo_simulation(start_price, n_steps, n)
final_prices = simulated_prices[:, -1]  # Last column contains final prices
lower_bound = np.percentile(final_prices, 2.5)  # 2.5th percentile
upper_bound = np.percentile(final_prices, 97.5)  # 97.5th percentile

print(f"95% Confidence Interval of Price in 1 Year From Monte Carlo Simulation: {lower_bound} to {upper_bound}")
print("--------------------")

##Calculating % return  chances
thresholds = [.05, 0.10, 0.20, 0.30, -.05, -0.10, -0.20, -0.30]
threshold_prices = {t: start_price * (1 + t) for t in thresholds}

# Calculate probabilities for each threshold
probabilities = {}
for t, price in threshold_prices.items():
    if t > 0:
        probabilities[f"{int(t * 100)}% return or higher"] = np.mean(final_prices >= price)
    else:
        probabilities[f"{int(t * 100)}% return or lower"] = np.mean(final_prices <= price)

# Print probabilities
for desc, prob in probabilities.items():
    print(f"Probability of {desc}: {prob * 100:.2f}% from Monte Carlo Simulation")
print("--------------------")
print("--------------------")

##Calculate worst Case and Worst Case returns 
best_case_index = np.argmax(simulated_prices[:, -1])  # Index of highest final price
worst_case_index = np.argmin(simulated_prices[:, -1])  # Index of lowest final price

best_case_val = simulated_prices[best_case_index][-1]
worst_case_val = simulated_prices[worst_case_index][-1]

best_case_ret = best_case_val - start_price
best_case_ret_pct = (best_case_val - start_price) / start_price * 100
print(f"Best Case Return: {best_case_ret} , %: {best_case_ret_pct}")

worst_case_ret = worst_case_val - start_price
worst_case_ret_pct = (worst_case_val - start_price) / start_price * 100
print(f"Worst Case Return: {worst_case_ret} , %: {worst_case_ret_pct}")
