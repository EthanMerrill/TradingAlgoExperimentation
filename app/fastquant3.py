#%%
import requests 
import json
import pandas as pd
from time import time
import time as tm
from datetime import datetime, timedelta, date
import numpy as np
from simple_rsi import callable_rsi_backtest, get_symbol_data
import os
import networking
from helper_functions import ensure_dir
import params
# Set Finnhub api keys
# finnhubKey = keys.keys.get("finnhub")
polygon_KEY = os.environ['polygon']
# IEXKey = keys.keys.get("iex")
# MIsc global options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
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
    try:
        queryurl = (f"https://api.polygon.io/v2/aggs/grouped/locale/US/market/stocks/{str(date)}?unadjusted=false&apiKey={polygon_KEY}")
        polygonTickersData = requests.get(queryurl).json().get("results")

        polygon_tickers_dataframe = pd.DataFrame(polygonTickersData)
        print(polygon_tickers_dataframe.head())
        if not polygonTickersData:
            print("ERROR No Polygon tickers data found")
            raise Exception ("empty Polygon Tickers data, check Polygon api status")
    except Exception as e:
        print(f"failed to query polygon api {e}")
    # Filter 
    # Volume over 1mil, close price over 15&under 200
    polygon_tickers_dataframe = polygon_tickers_dataframe.loc[(polygon_tickers_dataframe["v"]>=1000000 ) & (polygon_tickers_dataframe["c"]>=15 ) & (polygon_tickers_dataframe["c"]<=200)]
    polygon_tickers_dataframe = polygon_tickers_dataframe.sort_values(by=["T"])#[:50]# !!!!!!!!!!!!!!UNCOMMENT THIS FOR PRODUCTION
    polygon_tickers_dataframe.reset_index(inplace = True)
    # polygon_tickers_dataframe.to_pickle(f"Stock_universe_{date}")
    return polygon_tickers_dataframe
#%%

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
    print(f"total possiblities:{len(periods_list)*len(rsi_lower_list)*len(rsi_upper_list)}")
    
    data = get_symbol_data(symbol, start_date, end_date)

    # make an empty list to push strats to
    stratsList = []
    ## RSI Optimization; run the grid search
    try:
        for i, rsi_period in enumerate(periods_list):
            for j, rsi_lower in enumerate(rsi_lower_list):
                for k, rsi_upper in enumerate(rsi_upper_list):
                    results = callable_rsi_backtest(symbol = symbol, 
                                                    data0 = data,
                                                    period = rsi_period, 
                                                    lower = rsi_lower, 
                                                    upper = rsi_upper, 
                                                    cash = init_cash
                                                    )
                    stratsList.append(results[0][0])

    except Exception as e:
        print(f"error occured in rsi_optimizer: {e}") 
        raise     
        # return null_return_dict, 0
  
    # Find highest profit
    # strats = [x[0] for x in stratsList]


    # Best Possible Strategy:
    bestStrat = max(stratsList, key = lambda item:list(item.analyzers.returns.get_analysis().items())[0][1] )
    # get params and all other information from the best possible strat and add them to a dict
    return_dict = {
        "symbol":symbol,
        "optimal_rsi_period" : bestStrat.p.rsi_period,
        "optimal_rsi_lower" : bestStrat.p.rsi_lower,
        "optimal_rsi_upper" : bestStrat.p.rsi_upper,
        "profit":bestStrat.p.pnl_val,
        "roi": list(bestStrat.analyzers.returns.get_analysis().items())[0][1],
        "buy_and_hold": list(bestStrat.analyzers.basereturn.get_analysis().items())[0][1]
    }


    # Calculate total time taken by the function
    end_time = time()
    time_basic = end_time-start_time

    return return_dict, time_basic


# https://bbs.archlinux.org/viewtopic.php?id=77634
def humanize_time(secs):
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d' % (hours, mins, secs)

#%%
def multi_stock_rsi_optimize(df_of_stocks, end_date):
    ensure_dir("/tmp/Backtests")
    TEMP_SAVE_DIR = "/tmp/Backtests"
    start_time = time()

        # set certain columns to smaller data types
    dtypes_dict = {
        'symbol':'string',
        'optimal_rsi_period':'int16',
        'optimal_rsi_lower':'int16',
        'optimal_rsi_upper':'int16',
        'roi':'float16',
        'buy_and_hold':'float16',
        'profit':'float32'
    }
    # Add empty columns to dataframe
    results_df = pd.DataFrame(columns = ["symbol", "optimal_rsi_period","optimal_rsi_lower","optimal_rsi_upper","profit", "roi", "buy_and_hold"])
    results_df = results_df.astype(dtypes_dict)


    symbol_count=0
    error_count=0
    for symbol in df_of_stocks:


        try:
            symbol_dict, time_elapsed = rsi_optimizer([params.backtest_period_start,34,10],[30,41,5],[64,75,5], symbol, datetime(2020, 6, 1), end_date=end_date, init_cash =1000)
            results_df = results_df.append(symbol_dict, ignore_index=True)
            symbol_count = symbol_count+1
        except Exception as e:
            error_count = error_count+1
            time_elapsed = 0
            print(f"error occured in rsi_optimizer during symbol: {symbol}: {e} so far {error_count} errors have occured. Successes: {symbol_count} .")

        
            # Time calculation function
        time_left = ((len(df_of_stocks)-(symbol_count+1))*time_elapsed)
        print(f"projected time left: {humanize_time(time_left)}")

        try:
            # Temp save function to salvage some data from a very long test [depreciated]
            if (symbol_count % 50 == 0 and symbol_count != 0):
                # if this is not the first cache save, get the previous one, and merge it to the existing one
                if (symbol_count != 50):
                    results_df = results_df.append(pd.read_pickle(TEMP_SAVE_DIR))
                # Save the newly merged, larger dataframe locally
                results_df.to_pickle(TEMP_SAVE_DIR)
                results_df = pd.DataFrame(columns = ["symbol", "optimal_rsi_period","optimal_rsi_lower","optimal_rsi_upper","profit", "roi", "buy_and_hold"])
                results_df = results_df.astype(dtypes_dict)
                print('partial save and wipe complete')
        except Exception as e:
            print("failed to do a temp save of Data ")    

        if (len(results_df)>0):
            print(f"finished symbol: {symbol}.STATS: {results_df.loc[len(results_df)-1]} {symbol_count+error_count} analyized so far out of {len(df_of_stocks)} (Successes: {symbol_count}).")
        
    try:
        results_df = pd.read_pickle(TEMP_SAVE_DIR)

    except Exception as e:
        print(e)
        print(f"returning the dataframe of length {len(results_df)}")
        time_basic = time()-start_time
        return results_df, time_basic

    end_time = time()
    time_basic = end_time-start_time


    return results_df, time_basic


# %%
def run_strategy_generator(date):
    # convert the passed date to string:
    date_str=datetime.strftime(date, "%Y-%m-%d")
    all_ticks = get_All_Tickers(date_str)#.loc[0:15]
    # just get what you need from all ticks-- the ticks! Should save some ram
    all_ticks = all_ticks['T']
    if all_ticks.empty==True :
        print("could not find tickers")
        raise Exception 
    # RUn the mult stock rsi Optimizer
    backtest, time_basic = multi_stock_rsi_optimize(all_ticks, date)
    # Pickle the results of the multistock Optimizer
    #print the total time to complete
    print(f"time to complete backtester: {time_basic}")

    # drop empty rows
    backtest.dropna(
        axis=0,
        how='any',
        thresh=None,
        subset=None,
        inplace=True
    )
    # good information for logging in the future:
    # newDf["alpha"] = newDf["roi"]-newDf["buy_and_hold"]
    # total_results = len(backtest)
    # positive_alpha = len(backtest.loc[Backtest["alpha"]>1])
    # pct_positive_alpha = positive_alpha/total_results

    return backtest




###################
# portfolio correlation tester: 
def portfolio_correlation_test(portfolio_symbols, start_date, end_date):
    '''determines the level of correlation between the prospective purchase and the rest of the portfolio'''
    portfolio_history = pd.DataFrame()
    # for each symbol, get the historical tick data from polygon API arrange the data in a array of series'
    i=0
    while(i<len(portfolio_symbols)):
        print(f'Creating historical price array for portfolio is the portfolio_correlation_test module. Symbol {i} of {len(portfolio_symbols)}')
        closing_vals = pd.DataFrame(networking.polygon_data().get_single_stock_daily_bars(portfolio_symbols[i],start_date,end_date))["c"]
        closing_vals = closing_vals.rename(portfolio_symbols[i])
        portfolio_history = pd.concat([portfolio_history, closing_vals], axis=1)
        i=i+1
    # corr = portfolio_history.corr()
    # print(pd.np.triu(corr.values))

    portfolio_correlation_lst = portfolio_history.corr().unstack().sort_values(kind="quicksort")
    # remove all 1s from the flattened correlation list, they mess with the average.
    portfolio_correlation_lst = portfolio_correlation_lst[portfolio_correlation_lst!=1]
    # drop duplicates because there are two of every value from the corr operation

    #get the average correlation of all stocks in portfolio
    mean_portfolio_corr = portfolio_correlation_lst.mean()

    return mean_portfolio_corr, portfolio_correlation_lst
