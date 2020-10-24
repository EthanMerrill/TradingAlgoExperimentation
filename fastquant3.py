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

    polygon_tickers_dataframe = pd.DataFrame(polygonTickersData)
    # Filter 
    # Volume over 1mil, close price over 15&under 200
    polygon_tickers_dataframe = polygon_tickers_dataframe.loc[(polygon_tickers_dataframe["v"]>=1000000 ) & (polygon_tickers_dataframe["c"]>=15 ) & (polygon_tickers_dataframe["c"]<=200)]

    return polygon_tickers_dataframe

# someObject = backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30, init_cash=1000)
# backtest('smac', df, fast_period=10, slow_period=40,init_cash=1000)
# %%

def rsi_optimizer(periods_list, rsi_lower_list, rsi_upper_list, symbol, start_date, end_date=str(date.today()), init_cash=1000):
    # Start the timer!
    start_time = time()
    #Save all the inputs for recreation/debuggin/testing
    # input_conditions = locals()
    # get historical stock data (may need to update this to use polygon, not Yfinance)

    stock_data_df = get_stock_data(symbol, str(start_date), str(end_date))
    if stock_data_df.empty == True:
        print(f"unable to find yahoo data for {symbol}")
        return [symbol, "null", "null", "null"]   
    
    # make a list of all the values to test
    periods_list = np.arange(periods_list[0],periods_list[1],periods_list[2], dtype=int)
    rsi_lower_list = np.arange(rsi_lower_list[0],rsi_lower_list[1],rsi_lower_list[2], dtype=int)
    rsi_upper_list = np.arange(rsi_upper_list[0],rsi_upper_list[1],rsi_upper_list[2], dtype=int)

    # Make a 3grid of 0 placeholders
    period_grid = np.zeros(shape=(len(periods_list),len(rsi_lower_list),len(rsi_upper_list)))
 
    ## RSI Optimization; run the grid search
    try:
        for i, rsi_period in enumerate(periods_list):
            for j, rsi_lower in enumerate(rsi_lower_list):
                for k, rsi_upper in enumerate(rsi_upper_list):
                    results = backtest('rsi',
                                        stock_data_df,
                                        rsi_period=rsi_period,
                                        rsi_lower=rsi_lower,
                                        rsi_upper=rsi_upper,
                                        init_cash=init_cash,
                                        verbose=False,
                                        plot=False
                                        
                    )
                    net_profit = results.final_value.values[0]-init_cash
                    period_grid[i,j,k] = net_profit
    except Exception as e:
        print(f"error occured in rsi_optimizer: {e}")     
        return [symbol, "null", "null", "null"]   
    # Find highest profit
    best_params_position = np.unravel_index(period_grid.argmax(), period_grid.shape)
    # find position of highest profit
    optimal_rsi_period = periods_list[best_params_position[0]]
    optimal_rsi_lower = rsi_lower_list[best_params_position[1]]
    optimal_rsi_upper = rsi_upper_list[best_params_position[2]]
    profit = np.amax(period_grid)
    # make a dict with the values to return
    return_dict = {
        "symbol":symbol,
        "optimal_rsi_period" : optimal_rsi_period,
        "optimal_rsi_lower" : optimal_rsi_lower,
        "optimal_rsi_upper" : optimal_rsi_upper,
        "profit":profit
    }

    # Calculate total time taken by the function
    end_time = time()
    time_basic = end_time-start_time
        # print(f''''
    # profit: {profit}
    # Best rsi_period:{optimal_rsi_period}
    # Best rsi_lower:{optimal_rsi_lower}
    # best rsi_upper:{optimal_rsi_upper}
    # Basic grid search took {:.1f} sec
    # '''.format(time_basic))
    

    return return_dict

#%%
# create a test object to store backtest data
# class Backtest_Results:
#     def __init__(self, time_basic, input_conditions, optimal_rsi_period, optimal_rsi_lower, optimal_rsi_upper, profit):
#         self.time_basic = time_basic
#         self.input_conditions = input_conditions
#         self.optimal_rsi_period = optimal_rsi_period
#         self.optimal_rsi_lower = optimal_rsi_lower
#         self.optimal_rsi_upper = optimal_rsi_upper
#         self.profit = profit

#%%
def multi_stock_rsi_optimize(df_of_stocks):
    start_time = time()
    # Add empty columns to dataframe
    results_df = pd.DataFrame(columns = ["symbol", "optimal_rsi_period","optimal_rsi_lower","optimal_rsi_upper","profit"])
    symbol_count=0
    for symbol in df_of_stocks["T"]:

        symbol_dict = rsi_optimizer([3,18,5], [25,35,5],[65,75,5], symbol, "2020-01-01", "2020-10-23")
        results_df = results_df.append(symbol_dict, ignore_index=True)
        symbol_count=+1

    end_time = time()
    time_basic = end_time-start_time
    return results_df, time_basic

#%%
all_ticks = get_All_Tickers("2020-10-23").loc[0:10]
if all_ticks.empty==True :
    print("could not find ticks")
newDf, time_basic = multi_stock_rsi_optimize(all_ticks)
print(newDf,time_basic)
# %%
