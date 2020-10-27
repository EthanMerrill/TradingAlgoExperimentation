# Basic Algo

Some Experimenting with a trading algorithm for US Common Stock. 

todo:
- [ ] Run RSI starat then buy and hold strat. and return both results
- [ ] Add a higher set of variables such as dates in which the backtest is run (currently hardcoded in multi_rsi_opti funct), save directories, 
- [ ] Add a better time estimator, this will be useful as we move to backtesting multiple equities at once. 
- [x] Remove all dependency on yfinance
- [x] Make algo built on backtrader not fastquant on top of backtrader. (cut out the middle package!)
- [ ] Make a automatic date getter (get most recent weekday, if no today)
- [ ] Add a way to get Metadata on backtests.