#%%

from fastquant import get_stock_data
from fastquant import backtest

import yfinance as yf
import requests 
import keys
import json
import pandas as pd
import time
from datetime import date
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


def get_MKT_Cap(symbol):
    attempts=0
    while attempts<5:
        try:
            time.sleep(1.01)
            r = requests.get(f'https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={finnhubKey}')
            mktCap = r.json().get("marketCapitalization")
            print(f"success for symbol:{symbol} at market cap: {mktCap}")
            break
        except Exception as e:
            attempts +=1
            print(f"error in get_MKT_Cap function: {e} \n Symbol {symbol} \n MKTCAP {mktCap}")
    
    return (mktCap)


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

df = get_stock_data("aapl", "2018-01-01", "2020-10-01")
print(df.head())

# backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30)
backtest('smac', df, fast_period=10, slow_period=40,init_cash=1000)

# %%
polygonData = requests.get(f"https://api.polygon.io/v2/reference/tickers?apiKey={alpacaKey}").json()

# %%
polygonTickersDataFrame = pd.DataFrame()

errors=0
if errors<2:
    pageNumber = 0
    # ugly and bad:
    polygonTickersData = requests.get(f"https://api.polygon.io/v2/reference/tickers?sort=ticker&type=cs&market=STOCKS&locale=us&perpage=50&page={pageNumber}&active=true&apiKey={alpacaKey}").json().get("tickers")

    while (polygonTickersData != []):
        try:
            polygonTickersData = requests.get(f"https://api.polygon.io/v2/reference/tickers?sort=ticker&type=cs&market=STOCKS&locale=us&perpage=50&page={pageNumber}&active=true&apiKey={alpacaKey}").json().get("tickers")
            pageNumber += 1
            polygonTickersDataFrame = polygonTickersDataFrame.append(polygonTickersData)
            print(f"working on page Number: {pageNumber}")
        except Exception as e:
            errors += 1
            print(f"error in Volume_Price {e}  ")


#%%
# Filter out OTC things
polygonTickersDataFrame = polygonTickersDataFrame.loc[polygonTickersDataFrame["primaryExch"]!="OTC"]



# %%
polygonTickersData = requests.get(f"https://api.polygon.io/v2/aggs/grouped/locale/US/market/STOCKS/{date.today()}?unadjusted=false&apiKey={alpacaKey}").json().get("results")# %%

# %%
PolygonTickersDataFrame = pd.DataFrame(polygonTickersData)
# Filter 
# %%
# Volume over 1mil, close price over 15&under 200
PolygonTickersDataFrame= PolygonTickersDataFrame.loc[(PolygonTickersDataFrame["v"]>=1000000 ) & (PolygonTickersDataFrame["c"]>=15 ) & (PolygonTickersDataFrame["c"]<=200)]

# %%
PolygonTickersDataFrame["mktCap"] = PolygonTickersDataFrame.T.apply(get_MKT_Cap)