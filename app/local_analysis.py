from networking import cloud_object
from live_trader import most_recent_weekday, most_recent_trade_day
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
cloud_connection = cloud_object('backtests-and-positions')

# i=200
# recent_weekday_attempt = (str(most_recent_weekday(offset = i)))
# yesterdays_portfolio = cloud_connection.download_from_backtests('2020-12-24')
# print(yesterdays_portfolio)
# yesterdays_portfolio.to_csv("app/tmp/Backtests/test.csv")
# testing_df = pd.DataFrame([[1,2,3,4,6],[1,2,3,4,5]])
# cloud_connection.save_to_backtests(testing_df, "TESTING.csv")

# print(cloud_connection.download_from_backtests("2021-02-12"))

print(most_recent_trade_day(offset=-1))

#%%
from pytz import timezone
import time
import datetime as dt
from datetime import timedelta
import pandas as pd
import exchange_calendars as tc
# get trading calendar
xnys = tc.get_calendar("XNYS")
today = dt.datetime.today()+timedelta(-2)
t = dt.time(10,00)
today_base = dt.datetime.combine(today, t)

today = today_base
print(today.strftime('%Y-%m-%d-T%H'))
print(pd.Timestamp(today.strftime('%Y-%m-%d-T%H'), tz = 'US/Eastern'))
print(xnys.is_session(pd.Timestamp(today.strftime('%Y-%m-%d'))))
# %%

# %%
