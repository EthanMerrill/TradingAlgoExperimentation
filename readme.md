# Basic Algo

Some Experimenting with a trading algorithm for US Common Stock. 

Positions DF:
symbol|rsi_period|rsi_lower|rsi_upper|current_rsi|modeled_returns|alpha|entry_date|entry_price|exit_date|exit|price

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

- [ ] Clean up the key/path/variable management to use only environ variables. 
- [x] Manually update positions df on buy and sell orders. 
    - [ ] may want to build more error handling in in the future to handle manual buy and sell orders amongst other things. Just a better way to reconcile strategies and positions 
- [ ] More complex buy ordering. Limit orders, not market orders
    - [ ] Implement oco order on initial position
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
- [ ] handle the situation when 10% of portfolio isn't enough to purchase even one share.

- [ ] Check the RMA EMA functions, something is a little off there. 
- [ ] use numpy.where instead of vectorize where possible
- [ ] GCP Cloud bucket!

- ## Long term Features:
- [ ] Use a format better than pickle for long term storage
- [ ] CI/CD!! :o https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build
- 
- pip freeze > requirements.txt
