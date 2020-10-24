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
finnhubKey = keys.keys.get("finnhub")
alpacaKey = keys.keys.get("alpaca")
IEXKey = keys.keys.get("iex")

r = requests.get(f'https://finnhub.io/api/v1/stock/symbol?exchange=US&token={finnhubKey}')

#%%
# print(type(r.json()))
# allTickersList = json.loads(r.json())
allTickers = pd.DataFrame(r.json())

# Filter for just Equities:
allTickers = allTickers.loc[allTickers['type'] == "EQS"]
print(allTickers.head())
print(f"Found the symbols for {len(allTickers.index)} Equities")


# def get_MKT_Cap(symbol):
#     attempts=0
#     while attempts<5:
#         try:
#             time.sleep(1.01)
#             r = requests.get(f'https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={finnhubKey}')
#             mktCap = r.json().get("marketCapitalization")
#             print(f"success for symbol:{symbol} at market cap: {mktCap}")
#             break
#         except Exception as e:
#             attempts +=1
#             print(f"error in get_MKT_Cap function: {e} \n Symbol {symbol} \n MKTCAP {mktCap}")
    
#     return (mktCap)


#First, Filter by market cap using the finnhub api which has a higher rate limit (30 requests/second), but only market cap information. 
#%%
allTickers["mktCap"] = allTickers.symbol.apply(get_MKT_Cap)

# SAVE THE DF Locally because this operation uses a lot of api pts wherever you do it. 
allTickers.to_pickle(f"Stock_universe_{date.today()}")



#%%
# Depickle and filter by market cap larger than x
allTickers = pd.read_pickle(f"Stock_universe_{date.today()}")

allTickers = allTickers.loc[allTickers['mktCap']>=600 ]


#%%
def get_Volume_Price(symbol):
    attempts = 0
    while attempts<5:
        try:
            symbol = yf.Ticker(symbol)
        try:
            # time.sleep(1)
            
            volume = (symbol.info.get("averageVolume10days"))
            closePrice = (symbol.info.get("previousClose"))
            print(f"Symbol: {symbol} volume: {volume} closePrice:{closePrice}")
            break

        except Exception as e, err:
            attempts += 1
            print(f"error in Volume_Price {e} {symbol} ERR: {err} ")

    return volume, closePrice 

#%%
allTickers["averageVolume10days"],allTickers["closePrice"] = allTickers.symbol.apply(get_Volume_Price)

#%%


# %%
polygonData = requests.get(f"https://api.polygon.io/v2/reference/tickers?apiKey={alpacaKey}").json()

# %%
# polygonTickersDataFrame = pd.DataFrame()

# errors=0
# if errors<2:
#     pageNumber = 0
#     # ugly and bad:
#     polygonTickersData = requests.get(f"https://api.polygon.io/v2/reference/tickers?sort=ticker&type=cs&market=STOCKS&locale=us&perpage=50&page={pageNumber}&active=true&apiKey={alpacaKey}").json().get("tickers")

#     while (polygonTickersData != []):
#         try:
#             polygonTickersData = requests.get(f"https://api.polygon.io/v2/reference/tickers?sort=ticker&type=cs&market=STOCKS&locale=us&perpage=50&page={pageNumber}&active=true&apiKey={alpacaKey}").json().get("tickers")
#             pageNumber += 1
#             polygonTickersDataFrame = polygonTickersDataFrame.append(polygonTickersData)
#             print(f"working on page Number: {pageNumber}")
#         except Exception as e:
#             errors += 1
#             print(f"error in Volume_Price {e}  ")


#%%
# Filter out OTC things
polygonTickersDataFrame = polygonTickersDataFrame.loc[polygonTickersDataFrame["primaryExch"]!="OTC"]



# %%
polygonTickersData = requests.get(f"https://api.polygon.io/v2/aggs/grouped/locale/US/market/STOCKS/{str(date.today())}?unadjusted=false&apiKey={alpacaKey}").json().get("results")# %%

# %%
PolygonTickersDataFrame = pd.DataFrame(polygonTickersData)
# Filter 
# %%
# Volume over 1mil, close price over 15&under 200
PolygonTickersDataFrame= PolygonTickersDataFrame.loc[(PolygonTickersDataFrame["v"]>=1000000 ) & (PolygonTickersDataFrame["c"]>=15 ) & (PolygonTickersDataFrame["c"]<=200)]

# %%

single_stock_test_df = get_stock_data("nvda", "2020-01-01", str(date.today()))

# someObject = backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30, init_cash=1000)
# backtest('smac', df, fast_period=10, slow_period=40,init_cash=1000)
# %%
# make a list of all the values to test
periods_list = np.arange(3,20,4, dtype=int)
rsi_lower_list = np.arange(25,35,5, dtype=int)
rsi_upper_list = np.arange(65,75,5,dtype=int)

# Make a 3grid of 0 placeholders
period_grid = np.zeros(shape=(len(periods_list),len(rsi_lower_list),len(rsi_upper_list)))
period_grid.shape
## RSI Optimization; run the grid search
#%%

start_time = time()
init_cash=1000


for i, rsi_period in enumerate(periods_list):
    for j, rsi_lower in enumerate(rsi_lower_list):
        for k, rsi_upper in enumerate(rsi_upper_list):
            results = backtest('rsi',
                                single_stock_test_df,
                                rsi_period=rsi_period,
                                rsi_lower=rsi_lower,
                                rsi_upper=rsi_upper,
                                init_cash=1000,
                                plot=False
            )
            net_profit = results.final_value.values[0]-init_cash
            period_grid[i,j,k] = net_profit

end_time = time()

time_basic = end_time-start_time
print("Basic grid search took {:.1f} sec".format(time_basic))
# %%
# Find highest profit

# search position with highest net profit


print(f"max profit={np.amax(period_grid)} ")
best_params_position = np.unravel_index(period_grid.argmax(), period_grid.shape)

print(f''''
Best rsi_period:{periods_list[best_params_position[0]]}
Best rsi_lower:{rsi_lower_list[best_params_position[1]]}
best rsi_upper:{rsi_upper_list[best_params_position[2]]}


''')
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
