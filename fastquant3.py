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
# Set Finnhub api keys
finnhubKey = keys.keys.get("finnhub")

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

allTickers = allTickers.loc[allTickers['mktCap']>=300 ]


#%%


msft = yf.Ticker("MSFT")
msft.info
#%%

df = get_stock_data("aapl", "2018-01-01", "2020-10-01")
print(df.head())

# backtest('rsi', df, rsi_period=14, rsi_upper=70, rsi_lower=30)
backtest('smac', df, fast_period=10, slow_period=40,init_cash=1000)

# %%



# %%
