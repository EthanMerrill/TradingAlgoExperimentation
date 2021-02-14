from networking import cloud_object
from live_trader import most_recent_weekday
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
cloud_connection = cloud_object('backtests-and-positions')

i=200
recent_weekday_attempt = (str(most_recent_weekday(offset = i)))
yesterdays_portfolio = cloud_connection.download_from_backtests('2020-12-24')
print(yesterdays_portfolio)
# yesterdays_portfolio.to_csv("app/tmp/Backtests/test.csv")