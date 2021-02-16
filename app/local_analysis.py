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