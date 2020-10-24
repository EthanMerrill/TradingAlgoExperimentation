#%%

from fastquant import get_stock_data
from fastquant import backtest

import yfinance as yf
import requests 
import keys
import json
import pandas as pd
from time import time
from datetime import date
import numpy as np
# import polygon
# from polygon import RESTClient"

# Set Finnhub api keys
# finnhubKey = keys.keys.get("finnhub")
alpacaKey = keys.keys.get("alpaca")
# IEXKey = keys.keys.get("iex")



# %%
def get_All_Tickers(date = (date.today())):
    # function to find & filter all symbols for any date
    # #Get all tickers:
    polygonTickersData = requests.get(f"https://api.polygon.io/v2/aggs/grouped/locale/US/market/STOCKS/{str(date)}?unadjusted=false&apiKey={alpacaKey}").json().get("results")

    PolygonTickersDataFrame = pd.DataFrame(polygonTickersData)
    # Filter 
    # Volume over 1mil, close price over 15&under 200
    PolygonTickersDataFrame = PolygonTickersDataFrame.loc[(PolygonTickersDataFrame["v"]>=1000000 ) & (PolygonTickersDataFrame["c"]>=15 ) & (PolygonTickersDataFrame["c"]<=200)]

    return PolygonTickersDataFrame

# someObject = backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30, init_cash=1000)
# backtest('smac', df, fast_period=10, slow_period=40,init_cash=1000)
# %%

def rsi_optimizer(periods_list, rsi_lower_list, rsi_upper_list, symbol, start_date, end_date=str(date.today()), init_cash=1000):
    start_time = time()
    # get historical stock data (may need to update this to use polygon, not Yfinance)
    stock_data_df = get_stock_data(symbol, str(start_date), str(end_date))
    # make a list of all the values to test
    periods_list = np.arange(periods_list[0],periods_list[1],periods_list[2], dtype=int)
    rsi_lower_list = np.arange(rsi_lower_list[0],rsi_lower_list[1],rsi_lower_list[2], dtype=int)
    rsi_upper_list = np.arange(rsi_upper_list[0],rsi_upper_list[1],rsi_upper_list[2], dtype=int)

    # Make a 3grid of 0 placeholders
    period_grid = np.zeros(shape=(len(periods_list),len(rsi_lower_list),len(rsi_upper_list)))
    period_grid.shape

    ## RSI Optimization; run the grid search

    for i, rsi_period in enumerate(periods_list):
        for j, rsi_lower in enumerate(rsi_lower_list):
            for k, rsi_upper in enumerate(rsi_upper_list):
                results = backtest('rsi',
                                    stock_data_df,
                                    rsi_period=rsi_period,
                                    rsi_lower=rsi_lower,
                                    rsi_upper=rsi_upper,
                                    init_cash=init_cash,
                                    plot=False
                )
                net_profit = results.final_value.values[0]-init_cash
                period_grid[i,j,k] = net_profit

                    # Find highest profit
    
    best_params_position = np.unravel_index(period_grid.argmax(), period_grid.shape)
    # search position with highest net profit
    optimal_rsi_period = periods_list[best_params_position[0]]
    optimal_rsi_lower = rsi_lower_list[best_params_position[1]]
    optimal_rsi_upper = rsi_upper_list[best_params_position[2]]
    profit = np.amax(period_grid)
    # print(f''''
    # profit: {profit}
    # Best rsi_period:{optimal_rsi_period}
    # Best rsi_lower:{optimal_rsi_lower}
    # best rsi_upper:{optimal_rsi_upper}
    # ''')

    end_time = time()

    time_basic = end_time-start_time
    print("Basic grid search took {:.1f} sec".format(time_basic))

    return time_basic, optimal_rsi_period, optimal_rsi_lower, optimal_rsi_upper, profit


#%%
PolygonTickersDataFrame = get_All_Tickers("2020-10-23")

#%%
print(rsi_optimizer([3,20,3], [25,35,4],[65,75,4], PolygonTickersDataFrame.iloc[0,0], "2020-01-01", "2020-10-23"))
# %%
best_params_position[1]
# %%
periods_list[0]
# %%
rsi_lower
rsi_upper_list
# %%

backtest('rsi', single_stock_test_df, rsi_period=7, rsi_upper=70, rsi_lower=30, init_cash=1000)
# %%

# %%
