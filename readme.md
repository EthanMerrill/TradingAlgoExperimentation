# Basic Algo

Some Experimenting with a trading algorithm for US Common Stock. 

Positions DF:
symbol|rsi_period|rsi_lower|rsi_upper|current_rsi|modeled_returns|alpha|entry_date|entry_price|exit_date|exit|price

todo:
- [x] Add a better time estimator, this will be useful as we move to backtesting multiple equities at once. 
- [x] Remove all dependency on yfinance
- [x] Make algo built on backtrader not fastquant on top of backtrader. (cut out the middle package!)

- [ ] make a function which evaluates and updates trailing stops
- [ ] compare current rsi to rsi entry limits in entry calculator function
- [ ] function to place buy and sell orders (switch for paper vs non paper trading)
- [ ] simple comparator for rsi exit conditions
- [ ] Check the RMA EMA functions, something is a little off there.