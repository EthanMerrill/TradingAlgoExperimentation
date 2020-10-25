# https://github.com/ChakshuGupta13/technical-indicators-backtesting/blob/master/RSI.py
#%%
from datetime import datetime
import backtrader as bt
import alpaca_backtrader_api
import keys

# class RelativeStrengthIndexStrategy(bt.SignalStrategy):
#     def __init__(self):
#         self.index = bt.ind.RelativeStrengthIndex()

#     def next(self):
#         if self.index < 40 and self.datas[0].close[0] > self.datas[-1].close[-1]:
#             self.buy(size=0.5)

#         elif self.index > 70 and self.datas[0].close[0] < self.datas[-1].close[-1]:
#             self.close(size=0.5)

#         if self.datas[0].datetime.date(0) == datetime(2020, 1, 29).date():
#             self.close()


# cerebro = bt.Cerebro(stdstats=True)
# cerebro.addobserver(bt.observers.BuySell)
# print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
# print('Starting Portfolio Cash: %.2f' % cerebro.broker.get_cash())
# cerebro.addstrategy(RelativeStrengthIndexStrategy)
# data0 = bt.feeds.YahooFinanceData(dataname='^NSEI', fromdate=datetime(2016, 1, 1),
#                                   todate=datetime(2020, 1, 30))

# cerebro.adddata(data0)
# cerebro.run()
# print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
# print('Final Portfolio Cash: %.2f' % cerebro.broker.get_cash())
# cerebro.plot()

# https://github.com/alpacahq/alpaca-backtrader-api/blob/master/sample/strategy_sma_crossover.py

# Your credentials here
ALPACA_API_KEY = keys.keys.get("alpaca")
ALPACA_SECRET_KEY = keys.keys.get("alpaca_secret")

"""
You have 3 options: 
 - backtest (IS_BACKTEST=True, IS_LIVE=False)
 - paper trade (IS_BACKTEST=False, IS_LIVE=False) 
 - live trade (IS_BACKTEST=False, IS_LIVE=True) 
"""
IS_BACKTEST = True
IS_LIVE = False
symbol = "WORK"
USE_POLYGON = True



class SmaCross1(bt.Strategy):
    def notify_fund(self, cash, value, fundvalue, shares):
        super().notify_fund(cash, value, fundvalue, shares)

    def notify_store(self, msg, *args, **kwargs):
        super().notify_store(msg, *args, **kwargs)
        self.log(msg)

    def notify_data(self, data, status, *args, **kwargs):
        super().notify_data(data, status, *args, **kwargs)
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status), *args)
        if data._getstatusname(status) == "LIVE":
            self.live_bars = True

    # list of parameters which are configurable for the strategy
    params = dict(
        pfast=10,  # period for the fast moving average
        pslow=30   # period for the slow moving average
    )

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime[0]
        dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_trade(self, trade):
        self.log("placing trade for {}. target size: {}".format(
            trade.getdataname(),
            trade.size))

    def notify_order(self, order):
        print(f"Order notification. status{order.getstatusname()}.")
        print(f"Order info. status{order.info}.")

    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

    def __init__(self):
        self.live_bars = False
        sma1 = bt.ind.SMA(self.data0, period=self.p.pfast)
        sma2 = bt.ind.SMA(self.data0, period=self.p.pslow)
        self.crossover0 = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        if not self.live_bars and not IS_BACKTEST:
            # only run code if we have live bars (today's bars).
            # ignore if we are backtesting
            return
        # if fast crosses slow to the upside
        if not self.positionsbyname[symbol].size and self.crossover0 > 0:
            self.buy(data=data0, size=5)  # enter long

        # in the market & cross to the downside
        if self.positionsbyname[symbol].size and self.crossover0 <= 0:
            self.close(data=data0)  # close long position


class BasicRSI(bt.Strategy):
    def notify_fund(self, cash, value, fundvalue, shares):
        super().notify_fund(cash, value, fundvalue, shares)

    def notify_store(self, msg, *args, **kwargs):
        super().notify_store(msg, *args, **kwargs)
        self.log(msg)

    def notify_data(self, data, status, *args, **kwargs):
        super().notify_data(data, status, *args, **kwargs)
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status), *args)
        if data._getstatusname(status) == "LIVE":
            self.live_bars = True

    params = dict(
        rsi_period,
        rsi_lowe,
        rsi_upper
    )
    # The logging function for this strategy

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime[0]
        dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_trade(self, trade):
        self.log("placing trade for {}. target size: {}".format(
            trade.getdataname(),
            trade.size))

    def notify_order(self, order):
        print(f"Order notification. status{order.getstatusname()}.")
        print(f"Order info. status{order.info}.")

    def stop(self):
        print('==================================================')
        print('Starting Value - %.2f' % self.broker.startingcash)
        print('Ending   Value - %.2f' % self.broker.getvalue())
        print('==================================================')

    # Below is where all the decisions happen 
    def __init__(self):
        self.live_bars = False
        self.RSI = bt.ind.RelativeStrengthIndex(self.data0, period=self.p.rsi_period)
        # sma2 = bt.ind.SMA(self.data0, period=self.p.pslow)
        # self.crossover0 = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        if not self.live_bars and not IS_BACKTEST:
            # only run code if we have live bars (today's bars).
            # ignore if we are backtesting
            return
        # If crosses RSI_UPPER, Sell
        if self.positionsbyname[symbol].size and self.RSI > self.p.rsi_upper:
            self.close(data=data0)  # close long position
        # If crosses RSI_Lower, buy
        if not self.positionsbyname[symbol].size and self.RSI < self.p.rsi_lower:
            self.buy(data=data0, size=5)  # enter long



if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # setup params

    # Create a cerebro entity
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BasicRSI,rsi_period = 11,rsi_lower = 34,rsi_upper = 74)

    store = alpaca_backtrader_api.AlpacaStore(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=not IS_LIVE,
        usePolygon=USE_POLYGON
    )

    DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData
    if IS_BACKTEST:
        data0 = DataFactory(dataname=symbol,
                            historical=True,
                            fromdate=datetime(2020, 1, 1),
                            todate=datetime(2020, 10, 11),
                            timeframe=bt.TimeFrame.Days)
    else:
        data0 = DataFactory(dataname=symbol,
                            historical=False,
                            timeframe=bt.TimeFrame.Days,
                            backfill_start=True,
                            )
        # or just alpaca_backtrader_api.AlpacaBroker()
        broker = store.getbroker()
        cerebro.setbroker(broker)
    cerebro.adddata(data0)

    if IS_BACKTEST:
        # backtrader broker set initial simulated cash
        cerebro.broker.setcash(100000.0)

    print('Starting Portfolio Value: {}'.format(cerebro.broker.getvalue()))
    cerebro.run()
    print('Final Portfolio Value: {}'.format(cerebro.broker.getvalue()))
    cerebro.plot()

# %%
