#%%
import networking
from networking import cloud_object, alpaca_api
from datetime import date, timedelta
import datetime as dt
import pandas as pd
import numpy as np
import fastquant3
import os
import requests
from helper_functions import ensure_dir,list_files

# Create alpaca api Object

api = alpaca_api.create_api(alpaca_api(PAPER_TRADE=True))

#################################################

#Quick convert to normal time from epoch:
def epoch_to_msec(time):
    s = time / 1000.0
    # new_time=dt.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f')
    new_time=dt.datetime.fromtimestamp(s).strftime('%Y-%m-%d')

    return new_time
######################################
#%%
def RSI_parser(symbol, end_date, period):
    print(f'RSI Parser for {symbol} period of: {period} ending:{end_date}')
    # Set start date: 
    # day to account for weekends *** BUT NOT HOLIDAYS, hence roughly XX% day Buffer ** also will not work properly when run on weekends (buffer covers for this flaw also)
    # *2 because there needs to be additional data beyond the lookback period as rsi is recursive
    offset_days = -(period+((period//7)*2)+(period%7)+2)*2
    offset_days = int(offset_days)
    start_date = end_date + (timedelta(days = offset_days))
    # Unneccessary formatting
    start_date = dt.datetime.strftime(start_date, "%Y-%m-%d")
    end_date = dt.datetime.strftime(date.today(), "%Y-%m-%d")
    # Get Symbol Data
    historic_symbol_data = requests.get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?unadjusted=false&sort=asc&apiKey="+os.environ["alpaca_live"]).json().get("results")
    #shove it in a dataframe real quick:
    historic_symbol_data = pd.DataFrame.from_dict(historic_symbol_data)
    #fix dates 
    # historic_symbol_data["t"] = historic_symbol_data.t.apply(epoch_to_msec)
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
    #create the alpha column
    backtest["alpha"] = backtest["roi"]-backtest["buy_and_hold"]
    #  every positive alpha item:
    backtest = backtest.loc[backtest["alpha"]>0]
    # include only profitable strategies:
    backtest = backtest.loc[backtest["profit"]>0]
    # Get the Current RSI for each symbol given the RSI period. 
    backtest["RSI_current"] = np.vectorize(RSI_parser)(backtest['symbol'],date.today(),backtest["optimal_rsi_period"])
    # Of all items tested, get only those where the Current RSI is lower than the Optimal low RSI entry pt
    buying_opp = backtest.loc[backtest["optimal_rsi_lower"]>backtest["RSI_current"]]

    #sort and re-index the new dataframe before returning
    
    buying_opp = buying_opp.sort_values(by=['alpha'], ascending=False)
    buying_opp.reset_index(inplace=True)
    buying_opp.set_index('symbol')

    return buying_opp

#%%

def exits_helper(stop_price,temp_stop_price):
    try:
        if ((stop_price<temp_stop_price) | (math.isnan(stop_price)==True) ):
            return temp_stop_price
        else:
            print(type(stop_price), stop_price)
            return stop_price
    # if the stop field is empy, an error will be thrown, but we'll just put in the temp
    except :
        return temp_stop_price

        
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
    if current_positions.empty == True:
        return current_positions
    current_positions["temp_stop_price"] = current_positions.apply(lambda x:get_stop(x["symbol"],most_recent_weekday(), x["optimal_rsi_period"], (x["optimal_rsi_period"]*2), stop_factor=2.5), axis=1)
    # check if stop columns exist:
    if ('stop_price' in current_positions):
        # !!!!!!!!!!!!!check if there is data in columns if they exist
        #only move stop higher
        # make it faster using this:https://stackoverflow.com/questions/27474921/compare-two-columns-using-pandas
        current_positions['stop_price'] = current_positions.apply(lambda x:exits_helper(x['stop_price'],x['temp_stop_price']), axis = 1)
    else:
        current_positions["stop_price"]=current_positions["temp_stop_price"]
    del current_positions["temp_stop_price"]

    return current_positions

# Calculate the volatility:

def get_stop(symbol, end_date, ema_period, atr_period, stop_factor=3):
    '''
    should really have a description of the function here
    calculates the Exponential Moving Average of the Average True Range. Designed so that it can be iterated through via numpy vectorize

    #  Volatility of the market is determined by a 10-day Exponential Moving Average of the Average True Range
    # Trailing stop at distance from the close 3-times the volatility
    # The stop could only move in the direction of the trade
    '''


    period = max(ema_period, atr_period)*2
    offset_days = -(period+((period//7)*2)+(period%7)+2)*2
    # #Convert numpy float to normal days
    # offset_days = offset_days.item()
    start_date = end_date + timedelta(days = offset_days)
    # Unneccessary formatting
    start_date = dt.datetime.strftime(start_date, "%Y-%m-%d")
    end_date = dt.datetime.strftime(date.today(), "%Y-%m-%d")
    # Get Symbol Data
    polygon_symbol_request =  (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?unadjusted=false&sort=asc&apiKey="+os.environ["alpaca_live"])
    df = requests.get(polygon_symbol_request).json().get("results")

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
    period = int(period)
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


#%%
##################################
def get_positions(df = None):
    print('getting positions')
    '''
    This function updates the positions Dataframe, the old positions dataframe can be passed as an argument or it will be searched for using the REST API
    '''

    if df is not None:
        old_positions = df
    else:
        old_positions=None    
    positions = (api.list_positions())
    new_positions = pd.DataFrame({
                                # 'asset_class': [x.asset_class for x in positions],
                                # 'assset_id': [x.assset_id for x in positions],
                                'symbol': [x.symbol for x in positions],
                                'qty': [x.qty for x in positions],
                                'avg_entry_price': [x.avg_entry_price for x in positions],
                                'change_today': [x.change_today for x in positions],
                                'cost_basis': [x.cost_basis for x in positions],
                                'current_price': [x.current_price for x in positions],
                                'exchange': [x.exchange for x in positions],
                                'lastday_price': [x.lastday_price for x in positions],
                                'market_value': [x.market_value for x in positions],
                                'side': [x.side for x in positions],
                                'unrealized_intraday_pl': [x.unrealized_intraday_pl for x in positions],
                                'unrealized_intraday_plpc': [x.unrealized_intraday_plpc for x in positions],
                                'unrealized_pl': [x.unrealized_pl for x in positions],
                                })
    new_positions.set_index('symbol')
    # print("NEW POSITIONS DF \n", new_positions, "\n OLD POSITIONS DF \n", old_positions)
    #  if old positions has data, update the new positions and return it. 
    if(old_positions is not None):
        old_positions = old_positions.combine_first(new_positions)
        print("\nUPDATED POSITIONS\n",old_positions)
        return old_positions
    else:
        empty_portfolio = pd.DataFrame(columns = {
                                                'symbol',
                                                'qty',
                                                'avg_entry_price',
                                                'change_today',
                                                'cost_basis',
                                                'current_price',
                                                'exchange',
                                                'lastday_price',
                                                'market_value',
                                                'side',
                                                'unrealized_intraday_pl',
                                                'unrealized_intraday_plpc',
                                                'unrealized_pl',
                                                })
        return empty_portfolio
    # empty_portfolio = empty_portfolio.transpose()
    
    
 
#%%
class order():
# class keeps all the order types tidy

    def limit_sell(symbol, qty, price):
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='stop',
                stop_price = price,
                time_in_force='day'
            )
        except Exception as e:
                print(f"Limit Sell Failed with exception: {e}")
                return

    def sell(symbol, qty):
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='day'
            )
        except Exception as e:
                print(f"sell failed with exception: {e}")
                return

    def buy(symbol, qty):
        try:
            api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
        except Exception as e:
                print(f"buy failed with exception: {e}")
                return
    
    def oco(symbol, qty, stop_price, limit_price):
        try:
            api.submit_order(
                side= "buy",
                symbol= symbol,
                type= "market",
                qty= qty,
                time_in_force= "day",
                order_class= "bracket",
                take_profit= dict(
                    limit_price = limit_price
                ),
                stop_loss= dict(
                    stop_price= stop_price,
                    limit_price= stop_price*.95
                )
            )
        except Exception as e:
                print(f"OCO order failed with exception: {e}")
                return

#%%

def orderer(df, long_market_value, cash):
    '''
    scans the positions df and places orders to fill in the gaps.
    '''
    #first, see if there are any positions:
    if (df['qty'].notnull().values.any()==True):
        # if there are positions, make a seperate df with only positions:
        active_positions = df[df['qty'].notnull()]
        # if RSI level is above top limit, sell
        rsi_upper_exceeded = active_positions.loc[active_positions["optimal_rsi_upper"]<=active_positions["RSI_current"]]
        if rsi_upper_exceeded.empty == False:
            rsi_upper_exceeded.apply(lambda x:order.sell(x['symbol'], x['qty']), axis=1)
        # update all trailing limit orders with new prices
        active_positions.apply(lambda x:order.limit_sell(x["symbol"], x['qty'], x["stop_price"]), axis=1)

    # if there is no quantity of a position, open an order to acquire it
    # This is designed so that in the future multiple new positions can be added
    new_positions = df[df['qty'].isnull()]
    # reset index so it can be iterated
    new_positions.reset_index(inplace=True)
    i= 0 
    while i< len(new_positions):

        # each asset aims to be approx 10% of portfolio at purchase. 
        # get appropriate purchase value:
        equity = long_market_value+cash
        purchase_cash = (equity*.1)
        # find number of shares which can be purchased with that amount
        ask_price = api.get_last_quote(new_positions['symbol'][i]).askprice
        shares = purchase_cash//ask_price
        # Find the correct lower stop price for the oco order:
        get_stop(new_positions["symbol"][i], date.today(), new_positions["optimal_rsi_period"][i], (new_positions["optimal_rsi_period"][i]*2),stop_factor=3)
        # order.buy(new_positions['symbol'][0],shares)
        limit_price = ask_price+((ask_price - new_positions['stop_price'][i])*2)
        order.oco(new_positions['symbol'][i],shares, new_positions['stop_price'][i],limit_price)
        i = i+1
    return df
#%%

def most_recent_weekday(offset=0):
    '''
    does what it says on th tin
    '''

    today = date.today()+timedelta(offset)
    day_of_week = today.weekday()
    if day_of_week <5:
        return today
    else:
        time_delta = timedelta(day_of_week-4)
        most_recent = today - time_delta
  
    return most_recent

#%%
if __name__ == "__main__":
    new_positions = []
    # print(f"started live trader working directory:{os.getcwd()} /n MachineTime:{dt.datetime.now()}")
    # print(f"environ Variables: {os.environ}")
    cloud_connection = cloud_object('backtests-and-positions')
    recent_weekday = most_recent_weekday()
    # May want to put the logic below into the most recent weekday function with the use of a time_cutoff argument:
    today8pm = dt.datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
    # if today is a weekday and before 8, run with previous current day:
    if (date.today() == recent_weekday)  & (dt.datetime.now()<=today8pm):
        recent_weekday = most_recent_weekday(-1)
    # If none of the above are true, it is a weekday after 8, and the simple most recent weekday will work. 

    # # get current positions
    # if its monday, get positions from last friday, if that fails, keep going 1 day back till it works. 
    # if (date.today.weekday() == 0):
    #     previous_positions_date = recent_weekday-timedelta(2)
    # else: 
    #     previous_positions_date = recent_weekday
    # count back 10 days and check for portfolio records on each day. THis is why the iterator goes negative in a weird way
    i = 0
    while i>-10:
        try:
            recent_weekday_attempt = (str(most_recent_weekday(offset = i)))
            yesterdays_portfolio = cloud_connection.download_from_positions(recent_weekday_attempt)
            print(f'positions found for {recent_weekday_attempt}')

            updated_portfolio = get_positions(yesterdays_portfolio)
            existing_portfolio_corr_mean, existing_portfolio_corr_all_vals = fastquant3.portfolio_correlation_test(updated_portfolio["symbol"].tolist(), dt.datetime(2020,1,1),recent_weekday )
            break
        except Exception as e:
            print(f"positions_for {recent_weekday_attempt} not found, going one more day back (most likely due to holiday) {e}")
            i=i-1 
        if i == -10:
            print('Current positions not found, creating new file')
            updated_portfolio = get_positions()
            existing_portfolio_corr_mean = 1
            break
    # cancel all existing orders for the Day
    api.cancel_all_orders() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #then, determine if new acquisitions can occur
    cash = float(api.get_account().cash)
    long_mkt_val = float(api.get_account().long_market_value)
    
    equity = cash+long_mkt_val
    if cash > (equity*.1):
        #get opportunities:
        # first, check if a backtest for the current day exists:
        try: 
            # backtest = pd.read_pickle(f"app/tmp/2020-11-09")
            # print(f"getting backtest for {recent_weekday}") 
            backtest = cloud_connection.download_from_backtests(recent_weekday) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        except :
            print(f'backtest for current day not found, running for {recent_weekday} ')
            backtest = fastquant3.run_strategy_generator(recent_weekday)
        
        backtest.set_index('symbol')
        buying_opp = get_entries(backtest)

        # Save and wipe mem ASAP
        cloud_connection.save_to_backtests(backtest,recent_weekday) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        backtest = None
#%%
        
        MAX_NEW_POSITIONS = 2
        num_new_positions = min(MAX_NEW_POSITIONS, cash//(equity*.1))
        j = 0 #stocks purchased iterator
        while j<num_new_positions:
            purchase = None
            try:
                i = 0 #row in purchase options DF iterator
                # Create a new_positions df from the best buying opportunity, if new portfolio df doesn't exist:
                if(len(updated_portfolio)==0):
                     # # create empty current positions list:
                    current_positions_template = pd.DataFrame(columns=['symbol','qty','avg_entry_price','change_today','cost_basis','current_price','exchange','lastday_price','market_value','side','unrealized_intraday_pl','unrealized_intraday_plpc','unrealized_pl'])
                    purchase = buying_opp.loc[i]
                    # new_positions = purchase
                    purchase = purchase.to_frame()
                    purchase = purchase.transpose()
                    updated_portfolio = current_positions_template.append(purchase, verify_integrity=True, ignore_index=True)
                    print(purchase)
                    j = j+1
                # Check to see if position already exists, add it if not:
                while (i <= len(buying_opp)) & (j<num_new_positions):
                    # make sure that the asset isn't already owned, then move the the second or third best option if it is, to encourage diversity
                    is_not_owned = buying_opp["symbol"][i] not in updated_portfolio["symbol"].values
                    #check the correlation of the purchase with the rest of the portfolio. 
                    new_portfolio_list = (updated_portfolio["symbol"].tolist())
                    new_portfolio_list.append(buying_opp["symbol"][i])
                    print(new_portfolio_list)
                    new_portfolio_corr_mean, new_portfolio_corr_all_vals = fastquant3.portfolio_correlation_test(new_portfolio_list, dt.datetime(2020,1,1),recent_weekday)

                #if not owned and makes portfolio more diverse, add the stock to the portfolio
                    if (is_not_owned and (new_portfolio_corr_mean<existing_portfolio_corr_mean)):
                        purchase = buying_opp.loc[i]
                        updated_portfolio = updated_portfolio.append(purchase, verify_integrity=True, ignore_index=True)
                        break
                    i=i+1
            
            except Exception as e:
                if (e == KeyError(0)):
                    print(f"no buing opportunities found: {buying_opp.head()}")
                print(f"all entry opportunities already owned, buying opportunities:{e}")
            j = j+1
    # current_positions.set_index('symbol')
#%%

    if (len(updated_portfolio)>0):
        #Update Stops
        updated_portfolio = get_exits(updated_portfolio)
        #update RSI
        updated_portfolio["RSI"] = updated_portfolio.apply(lambda x:RSI_parser(x["symbol"],recent_weekday, x["optimal_rsi_period"]),axis=1)

        # new_positions.loc[len(new_positions)] = purchase
        # then update stops and rsi, and place any necessary puchase orders:
        updated_portfolio = orderer(updated_portfolio, long_mkt_val, cash)
        # save the updated positions to the CLOUD
        cloud_connection.save_to_positions(updated_portfolio, recent_weekday)
        

    print('success!')
    # try:
    #     os.system('sudo shutdown -h now')
    # except Exception as e:
    #     print(f'shutdown failed: {e}')
# https://www.googleapis.com/compute/v1/projects/myproject/zones/us-central1-f/instances/example-instance/start
# %%
