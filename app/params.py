# A list of all parameters and their descriptions in a Dictionary object.



# Stock universe filter params
from datetime import datetime, timedelta, date


min_uni_volume = 1000000
min_uni_price = 15
max_uni_price =200

# RSI backtest Params '
# See RSI Optimizer Function 

backtest_init_cash = 1000
backtest_start_date = datetime(2020,6,1)
backtest_period_start = 3
backtest_period_stop = 34
backtest_period_step = 10

backtest_rsi_lower_start = 30
backtest_rsi_lower_stop = 41
backtest_rsi_lower_step = 5

backtest_rsi_upper_start = 64
backtest_rsi_upper_stop = 75
backtest_rsi_upper_step = 5


# Asset Acquisition rules
#the percent of the total portfolio each new security will represent
aa_pct_portfolio = .1