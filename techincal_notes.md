## Virtual Env:

Code to create virtual env (Run this line): python -m venv name_of_env

Code for starting a virtual enviroment (Run this line): name_of_env\Scripts\activate (name_of_env) stocks\Scripts\activate stocks

To stop: deactivate

## Github: 

Link is for making feature branch: https://gist.github.com/vlandham/3b2b79c40bc7353ae95a

Merge new branch into main: git checkout main, git branch, git merge feature_branch_name, git push origin main

## Python 

- The multiprocessing package in Python is used to run multiple processes in parallel, utilizing multiple CPU cores to speed up execution. Use ChatGPT for examples
- Try simple moving average, random walk, Linear Regression
- try using this package: Pyalgotrade



The ML-based strategy is based on lagged return data that is binarized. That is, the ML algorithm learns from historical patterns of upward and downward <---- Could try to do something like this. Features for that would be made similar to below... But how would this work for values further out.... need to have lag values for x lags out
    cols = []
         for lag in range(1, lags + 1):
             col = 'lag_{}'.format(lag)
             data[col] = data['returns'].shift(lag)  1
             cols.append(col)

In [50]: data.dropna(inplace=True)

In [51]: data[cols] = np.where(data[cols] > 0, 1, 0)  2

In [52]: data['direction'] = np.where(data['returns'] > 0, 1, -1)  3

In [53]: data[cols + ['direction']].head()  