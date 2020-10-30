#%%
import requests
import keys
from datetime import date, timedelta, datetime
import datetime as dt
import pandas as pd
import alpaca_trade_api as tradeapi
import numpy as np
#%%
symbol = "AAPL"
start_date = "2020-08-01"
alpacaKey = alpacaKey = keys.keys.get("alpaca")
#%%
historic_symbol_data = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{date.today()}?unadjusted=false&sort=asc&apiKey={alpacaKey}").json().get("results")
# %%
symboldf = pd.DataFrame(historic_symbol_data)


#Quick convert to normal time from epoch:
def epoch_to_msec(time):
    s = time / 1000.0
    # new_time=datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
    new_time=datetime.fromtimestamp(s).strftime('%Y-%m-%d')

    return new_time
######################################
#%%
def RSI_parser(symbol, end_date, period):
    # Set start date: 
    # day to account for weekends *** BUT NOT HOLIDAYS, hence roughly XX% day Buffer ** also will not work properly when run on weekends (buffer covers for this flaw also)
    # *2 because there needs to be additional data beyond the lookback period as rsi is recursive
    offset_days = -(period+((period//7)*2)+(period%7)+2)*2
    start_date = end_date + timedelta(days = offset_days)
    # Unneccessary formatting
    start_date = datetime.strftime(start_date, "%Y-%m-%d")
    end_date = datetime.strftime(date.today(), "%Y-%m-%d")
    # Get Symbol Data
    historic_symbol_data = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?unadjusted=false&sort=asc&apiKey={alpacaKey}").json().get("results")
    #shove it in a dataframe real quick:
    historic_symbol_data = pd.DataFrame.from_dict(historic_symbol_data)
    #fix dates 
    historic_symbol_data["t"] = historic_symbol_data.t.apply(epoch_to_msec)
    # Calcuate the RSI:
    RSI_df = RSI(historic_symbol_data, period)
    # Extract and return the RSI
    max_date_row = RSI_df.loc[RSI_df["t"]==(RSI_df["t"].max())]
    max_date_row.reset_index(inplace = True)
    final_rsi = max_date_row["RSI_current"][0]
    return final_rsi



# %%
def RSI(df, period, base="c"):
    """
    Function to compute Relative Strength Index (RSI)
    https://github.com/arkochhar/Technical-Indicators/blob/master/indicator/indicators.py
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the MACD needs to be computed from (Default Close)
        period : Integer indicates the period of computation in terms of number of candles
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            Relative Strength Index (RSI_$period)
    """
 
    delta = df[base].diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0
    
    rUp = up.ewm(com=period - 1,  adjust=False).mean()
    rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
    # Nice way to add the period to the column, we don't need that 
    # df['RSI_' + str(period)] = 100 - 100 / (1 + rUp / rDown)
    # df['RSI_' + str(period)].fillna(0, inplace=True)
    df['RSI_current'] = 100 - 100 / (1 + rUp / rDown)
    df['RSI_current'].fillna(0, inplace=True)

    return df



    def get_entries(backtest):

    backtest["alpha"] = backtest["roi"]-backtest["buy_and_hold"]
    # find the current RSI of every positive alpha item:
    backtest = backtest.loc[backtest["alpha"]>0]
    # There is a a better way to do this, other thna a loop...
    backtest["RSI_current"] = np.vectorize(RSI_parser)(backtest['symbol'],date.today(),backtest["optimal_rsi_period"])
    buying_opp = backtest.loc[backtest["optimal_rsi_lower"]>backtest["RSI_current"]]
    buying_opp = buying_opp.sort_values(by=['alpha'])
    buying_opp.reset_index(inplace=True)

    return buying_opp


#%%
if __name__ == "__main__":

    #get backtest data
    
    Backtest = pd.read_pickle(f"Partial_Backtest_Save")
    get_entries(Backtest)
