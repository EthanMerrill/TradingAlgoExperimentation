# Basic Algo

Some Experimenting with a trading algorithm for US Common Stock. 
## Processes Overview:
### Positions
1. Get all previous positions in the portfolio by querying the google cloud datastore. This will return a picked dataframe.
1. Get all current positions in the <a href='alpaca.markets'>alpaca.markets</a> portfolio. Updated the picked dataframe with the new positions data. This updates the close price in the dataframe to the most recent close price.
1. Get cash and long market value amounts. If the cash is greater than 10% of the equity (cash+long market value), run the backtester:

### Backtest
1. Filter universe of all stocks to those with large market caps and a reasonable amount of volume results in 8-1.5k securities
2. For each security in this universe, backtest RSI Strategies over the past *6 months*. These are backtested with different combinations of `rsi upper bound`, `rsi lower bound`, and `rsi period`. These three parameters are optimized for each security using grid optimization. An improved optimization strategy is on the roadmap. This operation creates a dataframe with all the securities, their optimized parameters, the strategy return, and the return of a buy and hold strategy for the given security.
1. Every 50 strategy generations, or securities run through the backests/parameter optimization, the dataframe is appended to a locally saved version. This is saved in a format called a pickle. This is a way of reducing the amount of ram required by the program.
1. The backtest dataframe and is saved to google cloud storage once all securities have been processed.

### Buying Opportunities
1. Once the Backtester is finished (2-5 hours) the results are filtered using the get entries function. This function creates the alpha column, then filters for positive alpha items (where the ROI is greater than the buy and hold).
2. Next, only profitable strategies are selected. A strategy can be better than buy and hold, but still have lost money. 
1. After these filters are finished, the current rsi is calculated for each strategy based the strategies' specified rsi period. If the current RSI is lower than the rsi_lower_bound specified in the strategy, add to the buying opportunities DF
4. Determine how many different shares can be purchased by calculating the amount of cash as a percentage of equity. The portfolio is Equally weighted with 10 positions, so if 20% cash is available, buy 2 new securities. This can also be overridden with a **MAX_NEW_POSITIONS** value, if this is less than the number of securities capable of being purchased with cash, just purchase the [max new positions] number of securities. This was added to limit the amount of shares purchased on a single day. The portfolio allocation methodology could be significantly improved. 
1. 


### Get Exits

    
*Positions DF:*
symbol|rsi_period|rsi_lower|rsi_upper|current_rsi|modeled_returns|alpha|entry_date|entry_price|exit_date|exit|price
*Backtests DF*
symbol|rsi_period|rsi_lower|rsi_upper|current_rsi|profit|ROI|Buy_and_hold

todo:
- [x] Add a better time estimator, this will be useful as we move to backtesting multiple equities at once. 
- [x] Remove all dependency on yfinance
- [x] Make algo built on backtrader not fastquant on top of backtrader. (cut out the middle package!)

- [x] make a function which evaluates and updates trailing stops
- [x] compare current rsi to rsi entry limits in entry calculator function
- [x] function to place buy and sell orders (switch for paper vs non paper trading)
- [x] simple comparator for rsi exit conditions
- [x] Make the Paths Portable: https://docs.python.org/3/library/os.path.html
- [x] optimize ram usage during backtesting

- [x] Clean up the key/path/variable management to use only environ variables. 
- [x] Manually update positions df on buy and sell orders. 
    - [ ] may want to build more error handling in in the future to handle manual buy and sell orders amongst other things. Just a better way to reconcile strategies and positions 
- [x] More complex buy ordering. Limit orders, not market orders
    - [x] Implement oco order on initial position
    - [x] create a class for order types
- [x] Set up containerization
    - [x] Create docker file
    - [x] Make a working build
- [x] Add error handling to polygon request, if it returns a blank set throw error

- [ ] Check that NYSE volumes are not 2x!!!

- [ ] put all variables in one place. and make a log of this data as a sort of metadata file. Add arguments on run, at least argument for PAPER_TRADING
    - stop price multiplier
    - volatility stop multiple
    - rsi_optimizer inputs
    - number of positions
- [x] handle the situation when 10% of portfolio isn't enough to purchase even one share.

- [ ] Check the RMA EMA functions, something is a little off there. 

- [ ] Pass 10% of equity to backtester for a more accurate test

- [x] GCP Cloud bucket!
- [ ] add a max time period for trades: https://community.backtrader.com/topic/2150/sell-a-position-after-2-days
    - [x] add to backtests
    - [ ] add to live_trader
- [x] add correlation checking for new positions https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

- [ ] fix to make backtester less error prone (problem is in the buy and hold analyzer I think)
- [ ] have the script turn off the system when done.
- [ ] place oco order types at day start

- ## Long term Features:
- [x] Use a format better than pickle for long term storage
- [x] CI/CD!! :o https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build
- [ ] improve optimization strategies
- [ ] integrate facebook prophet

pip freeze > requirements.txt
pip install -r requirements.txt