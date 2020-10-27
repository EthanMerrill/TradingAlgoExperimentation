#%%
import requests 
import keys
import json
import pandas as pd
from time import time
from datetime import datetime, timedelta, date
import numpy as np
from simple_rsi import callable_rsi_backtest
# import polygon
# from polygon import RESTClient"

# Set Finnhub api keys
# finnhubKey = keys.keys.get("finnhub")
alpacaKey = keys.keys.get("alpaca")
# IEXKey = keys.keys.get("iex")

# MIsc global options
pd.set_option('display.max_rows', None)


# %%
def get_All_Tickers(date = (date.today())):

    #chech to see if a pickled version for requested date exists, if so, just return that:
    try:
        # SAVE THE DF Locally because this operation uses a lot of api pts wherever you do it. 
        polygon_tickers_dataframe = pd.read_pickle(f"Stock_universe_{date}")
        print("existing Universe for selected date found, loaded cache")
        return polygon_tickers_dataframe
    except:
        print(f"existing universe for {date} not found, querying polygon API")
        pass

    # function to find & filter all symbols for any date
    # #Get all tickers:
    polygonTickersData = requests.get(f"https://api.polygon.io/v2/aggs/grouped/locale/US/market/STOCKS/{str(date)}?unadjusted=false&apiKey={alpacaKey}").json().get("results")

    polygon_tickers_dataframe = pd.DataFrame(polygonTickersData)
    # Filter 
    # Volume over 1mil, close price over 15&under 200
    polygon_tickers_dataframe = polygon_tickers_dataframe.loc[(polygon_tickers_dataframe["v"]>=1000000 ) & (polygon_tickers_dataframe["c"]>=15 ) & (polygon_tickers_dataframe["c"]<=200)]
    polygon_tickers_dataframe = polygon_tickers_dataframe.sort_values(by=['T'])
    polygon_tickers_dataframe.reset_index(inplace = True)
    polygon_tickers_dataframe.to_pickle(f"Stock_universe_{date}")
    return polygon_tickers_dataframe

# %%

def rsi_optimizer(periods_list, rsi_lower_list, rsi_upper_list, symbol, start_date, end_date=date.today(), init_cash=1000):
    # Start the timer!
    start_time = time()
    # 
    null_return_dict = {
    "symbol":symbol,
    "optimal_rsi_period" : None,
    "optimal_rsi_lower" : None,
    "optimal_rsi_upper" : None,
    "profit": None,
    "roi":None,
    "buy_and_hold":None}

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
                    results = callable_rsi_backtest(symbol1 = symbol, 
                                                    start_date = start_date, 
                                                    end_date =  end_date, 
                                                    period = rsi_period, 
                                                    lower = rsi_lower, 
                                                    upper = rsi_upper, 
                                                    cash = init_cash
                                                    )
                    net_profit1 = results.pnl_val
                    period_grid[i,j,k] = net_profit1

    except Exception as e:
        print(f"error occured in rsi_optimizer: {e}")     

        return null_return_dict
  
    # Find highest profit
    best_params_position = np.unravel_index(period_grid.argmax(), period_grid.shape)
    # find position of highest profit
    optimal_rsi_period = periods_list[best_params_position[0]]
    optimal_rsi_lower = rsi_lower_list[best_params_position[1]]
    optimal_rsi_upper = rsi_upper_list[best_params_position[2]]
    profit = np.amax(period_grid)

    #calculate the percent return of the strategy (ROI)
    roi = profit/init_cash

    # calculate change in the equity over the time period 
    # buy_and_hold = (stock_data_df["close"].iloc[-1]-stock_data_df["close"].iloc[0])/stock_data_df["close"].iloc[0]
    buy_and_hold = results.analyzers.getbyname("basereturn").get_analysis()
    #TO get the data out, you have to do this: it basically takes the data out of its ordered list, or collection, then gets it within 2 indexes...
    buy_and_hold = list(buy_and_hold.items())[0][1]

    # make a dict with the values to return
    return_dict = {
        "symbol":symbol,
        "optimal_rsi_period" : optimal_rsi_period,
        "optimal_rsi_lower" : optimal_rsi_lower,
        "optimal_rsi_upper" : optimal_rsi_upper,
        "profit":profit,
        "roi": roi,
        "buy_and_hold": buy_and_hold ### !!! TEMP NOT WORKING
    }

    # Calculate total time taken by the function
    end_time = time()
    time_basic = end_time-start_time

    return return_dict


#%%
def multi_stock_rsi_optimize(df_of_stocks):
    start_time = time()
    # Add empty columns to dataframe
    results_df = pd.DataFrame(columns = ["symbol", "optimal_rsi_period","optimal_rsi_lower","optimal_rsi_upper","profit", "roi", "buy_and_hold"])
    symbol_count=0
    for symbol in df_of_stocks["T"]:

        symbol_dict = rsi_optimizer([3,18,4], [24,35,5],[64,75,5], symbol, datetime(2020, 1, 1), datetime(2020, 10, 1))
        results_df = results_df.append(symbol_dict, ignore_index=True)
        # Temp save function to salvage some data from a very long test
        if (symbol_count % 50 == 0):
            results_df.to_pickle(f"Partial_Backtest_Save")
        print(f"finished symbol: {symbol}. {symbol_count+1} analyized so far out of {len(df_of_stocks)}.")
        symbol_count = symbol_count+1


    end_time = time()
    time_basic = end_time-start_time
    return results_df, time_basic

#%%
all_ticks = get_All_Tickers("2020-10-23").loc[0:5]
# if all_ticks.empty==True :
#     print("could not find ticks")
newDf, time_basic = multi_stock_rsi_optimize(all_ticks)
todayStr=datetime.strftime(date.today(), "%Y-%m-%d")
newDf.to_pickle(f"Full_Backtest_{todayStr}")
print(newDf,time_basic)

#%%
Backtest = pd.read_pickle(f"Full_Backtest_{date.today()}")
# %%
Backtest["improvement"] = Backtest["roi"]-Backtest["buy_and_hold"]

print(Backtest.head())
# # 4 hours 
# #%%
# def add_business_days(d, business_days_to_add):
#     num_whole_weeks  = business_days_to_add / 5
#     extra_days       = num_whole_weeks * 2

#     first_weekday    = d.weekday()
#     remainder_days   = business_days_to_add % 5

#     natural_day      = first_weekday + remainder_days
#     if natural_day > 4:
#         if first_weekday == 5:
#             extra_days += 1
#         elif first_weekday != 6:
#             extra_days += 2

#     return d + timedelta(business_days_to_add + extra_days)
# #%%
# def current_rsi(symbol, lookback_period):

#     start_date = add_business_days(date.today-lookback_period)
    
#     historic_symbol_data = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{date.today()}?unadjusted=false&sort=asc&apiKey={alpacaKey}").json().get("results")
#     return historic_symbol_data
# temp = current_rsi("AAPL", 4)
# %%
