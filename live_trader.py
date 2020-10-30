#%%
import requests
import keys
from datetime import date, timedelta, datetime
import datetime as dt
import pandas as pd
import alpaca_trade_api as tradeapi
import numpy as np

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


# Entry Contiditions Evaluation
def get_entries(backtest):

    backtest["alpha"] = backtest["roi"]-backtest["buy_and_hold"]
    # find the current RSI of every positive alpha item:
    backtest = backtest.loc[backtest["alpha"]>0]
    # include only profitable strategies:
    backtest = backtest.loc[backtest["proft"]>0]
    # Get the Current RSI for each symbol given the RSI period. 
    backtest["RSI_current"] = np.vectorize(RSI_parser)(backtest['symbol'],date.today(),backtest["optimal_rsi_period"])
    # Of all items tested, get only those where the Current RSI is lower than the Optimal low RSI entry pt
    buying_opp = backtest.loc[backtest["optimal_rsi_lower"]>backtest["RSI_current"]]

    #sort and re-index the new dataframe before returning
    buying_opp = buying_opp.sort_values(by=['alpha'])
    buying_opp.reset_index(inplace=True)

    return buying_opp


#%%
temp = get_exits(Backtest)
#%%

# Exit Conditions Evaluation
def get_exits(current_positions):
    """
    for each symbol in the dataframe of current assets:
        if over the max RSI place sell order immediately and remove from the positions df
        get the stop value, if its tighter than the current one update it & place a new limit order in alpaca

    repickle the positions df 
    Positions DF:
    symbol|rsi_period|rsi_lower|rsi_upper|current_rsi|modeled_returns|alpha|entry_date|entry_price|exit_date|exit|price
    """
    current_positions["temp_stop_price"] = np.vectorize(get_stop)(current_positions["symbol"],
                                    date.today(),
                                    current_positions["optimal_rsi_period"],
                                    (current_positions["optimal_rsi_period"]*2),
                                    stop_factor=3)

    if current_positions["stop_price"]<current_positions["temp_stop_price"] or current_positions["stop_price"] == NoneType:
        current_positions["stop_price"]=current_positions["temp_stop_price"]

    return current_positions

# Calculate the volatility:

def get_stop(symbol, end_date, ema_period, atr_period, stop_factor=3):
    '''
    should really have a description of the function here
    calculates the Exponential Moving Average of the Average True Range. Designed so that it can be iterated through via numpy vectorize
    '''
    period = max(ema_period, atr_period)*2
    offset_days = -(period+((period//7)*2)+(period%7)+2)*2
    start_date = end_date + timedelta(days = offset_days)
    # Unneccessary formatting
    start_date = datetime.strftime(start_date, "%Y-%m-%d")
    end_date = datetime.strftime(date.today(), "%Y-%m-%d")
    # Get Symbol Data
    df = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?unadjusted=false&sort=asc&apiKey={alpacaKey}").json().get("results")

     #shove it in a dataframe real quick:
    df = pd.DataFrame.from_dict(df)

    df = ATR(df, atr_period, ohlc=["o","h","l","c"])

    df = EMA(df, "ATR", "EMA",ema_period)
    max_date_row = df.loc[df["t"]==(df["t"].max())]
    max_date_row.reset_index(inplace = True)
    smoothed_final_ema = max_date_row["EMA"][0]
    stop_price = max_date_row["c"][0]-(smoothed_final_ema*stop_factor)


    return stop_price

def EMA(df, base, target, period, alpha=False):
    """
    Function to compute Exponential Moving Average (EMA)
    https://github.com/arkochhar/Technical-Indicators/blob/master/indicator/indicators.py
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        base : String indicating the column name from which the EMA needs to be computed from
        target : String indicates the column name to which the computed data needs to be stored
        period : Integer indicates the period of computation in terms of number of candles
        alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)
        
    Returns :
        df : Pandas DataFrame with new column added with name 'target'
    """

    con = pd.concat([df[:period][base].rolling(window=period).mean(), df[period:][base]])
    
    if (alpha == True):
        # (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / period
        df[target] = con.ewm(alpha=1 / period, adjust=False).mean()
    else:
        # ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (period + 1)
        df[target] = con.ewm(span=period, adjust=False).mean()
    
    df[target].fillna(0, inplace=True)
    return df


def ATR(df, period, ohlc=['Open', 'High', 'Low', 'Close']):
    """
    Function to compute Average True Range (ATR)
    https://github.com/arkochhar/Technical-Indicators/blob/master/indicator/indicators.py
    Args :
        df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
        period : Integer indicates the period of computation in terms of number of candles
        ohlc: List defining OHLC Column names (default ['Open', 'High', 'Low', 'Close'])
        
    Returns :
        df : Pandas DataFrame with new columns added for 
            True Range (TR)
            ATR (ATR_$period)
    """
    atr = 'ATR'

    # Compute true range only if it is not computed and stored earlier in the df
    if not 'TR' in df.columns:
        df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
        df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
        df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())
         
        df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
         
        df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)

    # Compute EMA of true range using ATR formula after ignoring first row
    EMA(df, 'TR', atr, period, alpha=True)
    
    return df

#  Volatility of the market is determined by a 10-day Exponential Moving Average of the Average True Range
# Trailing stop at distance from the close 3-times the volatility
# The stop could only move in the direction of the trade
def update_prices(df):
    """simply queries polgon for the latest daily closing price of each sumbol and returns a df with a col
    called price
    """

class Place_Order():
    '''
    a class used for interacting with the alpaca api to place buy and sell orders
    '''

    def stop_order:
    
    def buy_order:
    
    def get_positions:

#%%
if __name__ == "__main__":

    #get backtest data
    
    Backtest = pd.read_pickle(f"Partial_Backtest_Save")
    get_entries(Backtest)

"""
get exits runs every day
Backtester runs when entries are needed
Get entries runs when entries are needed
"""
# %%
