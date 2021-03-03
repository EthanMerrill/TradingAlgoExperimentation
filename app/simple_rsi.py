# https://github.com/ChakshuGupta13/technical-indicators-backtesting/blob/master/RSI.py
#%%
from datetime import datetime
from time import time
import backtrader as bt
import alpaca_backtrader_api
import backtrader.analyzers as btanalyzers
import backtrader.filters as btfilters
import os
import traceback
################################

# import json
# from datetime import date, timedelta
# import datetime as dt

# try:
#     with open('GOOGLE_APPLICATION_CREDENTIALS.json') as f:
#         GACdata = json.load(f)

#     with open('ALPACA_KEYS.json') as m:
#         ALPACA_DATA = json.load(m)

#     os.environ["alpaca_secret_paper"] = ALPACA_DATA["alpaca_secret_paper"]
#     os.environ["alpaca_secret_live"] = ALPACA_DATA["alpaca_secret_live"]
#     os.environ["alpaca_live"] = ALPACA_DATA["alpaca_live"]
#     os.environ["alpaca_paper"] = ALPACA_DATA["alpaca_paper"]
# except Exception as e:
#     print(f"Error loading keys from google key manager: error {e}")

    
##################################
try:

    ALPACA_API_KEY = os.environ['alpaca_paper']
    ALPACA_SECRET_KEY = os.environ['alpaca_secret_paper']
    ALPACA_API_SECRET_KEY = os.environ['alpaca_secret_paper']
except Exception as e:
    print(f"issue importing alpaca keys{e}")

"""
You have 3 options: 
 - backtest (IS_BACKTEST=True, IS_LIVE=False)
 - paper trade (IS_BACKTEST=False, IS_LIVE=False) 
 - live trade (IS_BACKTEST=False, IS_LIVE=True) 
"""
IS_BACKTEST = True
IS_LIVE = False
USE_POLYGON = True


#########################################
# # Buy and hold strat: https://www.backtrader.com/blog/2019-06-13-buy-and-hold/buy-and-hold/
# class BuyAndHold_1(bt.Strategy):
#     def start(self):
#         self.val_start = self.broker.get_cash()  # keep the starting cash

#     def nextstart(self):
#         # Buy all the available cash
#         size = int(self.broker.get_cash() / self.data)
#         self.buy(size=size)

#     def stop(self):
#         # calculate the actual returns
#         self.roi = (self.broker.get_value() / self.val_start) - 1.0
#         print('ROI:        {:.2f}%'.format(100.0 * self.roi))


##########################################
# Custom RSI Strat 
#%%
class BasicRSI(bt.Strategy):
    def notify_fund(self, cash, value, fundvalue, shares):
        super().notify_fund(cash, value, fundvalue, shares)

    def notify_store(self, msg, *args, **kwargs):
        super().notify_store(msg, *args, **kwargs)
        self.log(msg)

    def notify_data(self, data, status, *args, **kwargs):
        super().notify_data(data, status, *args, **kwargs)
        if self.p.verbose == True: 
             print('DATA NOTIF:', data._getstatusname(status), *args)
        if data._getstatusname(status) == "LIVE":
            self.live_bars = True
    # stuff that can be passed to the class: https://www.backtrader.com/docu/concepts/
    params = dict(
        verbose = False,
        data0 = "data0",
        symbol = "NA",
        sizer = "none", # Determines what amount to acquire
        rsi_period= 1,
        rsi_lower= 1,
        rsi_upper= 1,
        atrperiod=1,  # measure volatility over x days
        emaperiod=10,  # smooth out period for atr volatility
        stopfactor=2.5,  # actual stop distance for smoothed atr
        trade_time_limit = 30
    )

    # The logging function for this strategy

    def log(self, txt, dt=None):
        dt = dt or self.data.datetime[0]
        dt = bt.num2date(dt)
        print('%s, %s' % (dt.isoformat(), txt))

    def start(self):
        self.entering = 0
        self.start_val = self.broker.get_value()

    def notify_trade(self, trade):
        if self.p.verbose == True:
            self.log("placing trade for {}. target size: {}".format(
                trade.getdataname(),
                trade.size))

    def notify_order(self, order):
        # for max order duration : get the order execution bar
        self.bar_executed = len(self)
        if self.p.verbose == True:
            print(f"Order notification. status{order.getstatusname()}.")
            print(f"Order info. status{order.info}.")

    def stop(self):
        self.stop_val = self.broker.get_value()
        self.p.pnl_val = self.stop_val - self.start_val
        if self.p.verbose == True:
            print('==================================================')
            print('Starting Value - %.2f' % self.broker.startingcash)
            print('Ending   Value - %.2f' % self.broker.getvalue())
            print('==================================================')

    # Below is where all the decisions happen 
    def __init__(self):
        self.live_bars = False
        self.RSI = bt.ind.RelativeStrengthIndex(self.data0, period=self.p.rsi_period)

                # Trailing Stop Indicator
        self.stoptrailer = st = StopTrailer(atrperiod=self.p.atrperiod,
                                            emaperiod=self.p.emaperiod,
                                            stopfactor=self.p.stopfactor
                                            )
        # Exit Criteria (Stop Trail) for long / short positions
        self.exit_long = bt.ind.CrossDown(self.data,
                                          st.stop_long, plotname='Exit Long')
        self.exit_short = bt.ind.CrossUp(self.data,
                                         st.stop_short, plotname='Exit Short')
        
        if self.p.sizer is not None:
            self.sizer = self.p.sizer

    def next(self):
        if not self.live_bars and not IS_BACKTEST:
            # only run code if we have live bars (today's bars).
            # ignore if we are backtesting
            return
        # logic
        closing = None
        if self.position.size > 0:  # In the market - Long
            # self.log('Long Stop Price: {:.2f}', self.stoptrailer.stop_long[0])
            if self.exit_long:
                closing = self.close()
                    # for max position duration: calculate position duration based on current strategy length:
            self.duration = len(self) - self.bar_executed + 1
                    # for max positions duration: if order is open longer than x days, close position:
            if self.duration == self.p.trade_time_limit:
                self.close()
        # If crosses RSI_UPPER, Sell
        if self.positionsbyname[self.p.symbol].size and self.RSI > self.p.rsi_upper:
            self.close(data=self.p.data0)  # close long position
        # If crosses RSI_Lower, buy
        if not self.positionsbyname[self.p.symbol].size and self.RSI < self.p.rsi_lower:
            self.buy(data=self.p.data0)  # enter long
#########################################
#  https://www.backtrader.com/blog/2019-08-22-practical-backtesting-replication/practical-replication/
class StopTrailer(bt.Indicator):
    _nextforce = True  # force system into step by step calcs

    lines = ('stop_long', 'stop_short',)
    plotinfo = dict(subplot=False, plotlinelabels=True)

    params = dict(
        atrperiod=14,
        emaperiod=10,
        stopfactor=3,
    )

    def __init__(self):
        self.strat = self._owner  # alias for clarity

        # Volatility which determines stop distance
        atr = bt.ind.ATR(self.data, period=self.p.atrperiod)
        emaatr = bt.ind.EMA(atr, period=self.p.emaperiod)
        self.stop_dist = emaatr * self.p.stopfactor

        # Running stop price calc, applied in next according to market pos
        self.s_l = self.data - self.stop_dist
        self.s_s = self.data + self.stop_dist

    def next(self):
        # When entering the market, the stop has to be set
        if self.strat.entering > 0:  # entering long
            self.l.stop_long[0] = self.s_l[0]
        elif self.strat.entering < 0:  # entering short
            self.l.stop_short[0] = self.s_s[0]

        else:  # In the market, adjust stop only in the direction of the trade
            if self.strat.position.size > 0:
                self.l.stop_long[0] = max(self.s_l[0], self.l.stop_long[-1])
            elif self.strat.position.size < 0:
                self.l.stop_short[0] = min(self.s_s[0], self.l.stop_short[-1])


##################

def get_symbol_data(symbol, start_date, end_date):
    store = alpaca_backtrader_api.AlpacaStore(
        key_id= os.environ["alpaca_paper"],
        secret_key=os.environ["alpaca_secret_paper"],
        paper=True,
        usePolygon=False
    )
    DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData

    data0 = DataFactory(dataname=symbol,
                        historical=True,
                        fromdate=start_date,
                        todate=end_date,
                        timeframe=bt.TimeFrame.Days)

    return data0
#%%
def callable_rsi_backtest(symbol, data0, period, lower, upper,cash, trade_time_limit = 30, maxcpus = None, plot = False, verbose=False):
# Create a cerebro entity
    # Start the timer!
    start_time = time()
    
    cerebro = bt.Cerebro()
        
    # broker = store.getbroker()
    # cerebro.setbroker(broker) ######FOR some reason setting a broker screws evertything up
    #DATA
    # data0.addfilter(btfilters.SessionFiller)
    cerebro.adddata(data0)
    
    # backtrader broker set initial simulated cash
    cerebro.broker.setcash(cash)

    # Apply Total, Average, Compound and Annualized Returns calculated using a logarithmic approach
    # #ANALYZER
    cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="mysharpe")
    cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(btanalyzers.TimeDrawDown, _name="timedraw")
    cerebro.addanalyzer(btanalyzers.TimeReturn, timeframe=bt.TimeFrame.NoTimeFrame, data = data0, _name="basereturn")
    

    #STRATEGY
    if maxcpus != 1:
        cerebro.addstrategy(BasicRSI,verbose=verbose, 
                                    data0 = data0, 
                                    symbol=symbol, 
                                    rsi_period = period,    
                                    rsi_lower = lower, 
                                    rsi_upper = upper, 
                                    atrperiod = period, 
                                    emaperiod = period, 
                                    trade_time_limit = trade_time_limit,
                                    sizer = bt.sizers.AllInSizer())
    else: 
        cerebro.optstrategy(BasicRSI,verbose=verbose, 
                                    data0 = data0, 
                                    symbol=symbol,
                                    rsi_period = period, 
                                    rsi_lower = lower, 
                                    rsi_upper = upper, 
                                    atrperiod = period, 
                                    emaperiod = period, 
                                    trade_time_limit = trade_time_limit, 
                                    sizer = bt.sizers.AllInSizer())
    
    try:
        theStrats=cerebro.run(maxcpus=maxcpus)
    except Exception as e:
        if verbose ==True:
            log_traceback(e)
        return
        pass
    
    if plot == True:
        cerebro.plot()

    # Calculate total time taken by the function
    end_time = time()
    time_basic = end_time-start_time

    return theStrats, time_basic

# Helper function
# https://realpython.com/the-most-diabolical-python-antipattern/
def log_traceback(ex):
    tb_lines = traceback.format_exception(ex.__class__, ex, ex.__traceback__)
    tb_text = ''.join(tb_lines)
    # I'll let you implement the ExceptionLogger class,
    # and the timestamping.
    #NEED TO MAKE AN EXCEPTION LOGGER OR USE THE LOGGING MODULE
    print(tb_text)
    
# results.analyzers.mysharpe.get_analysis


# returnedStrats, time_elapsed = callable_rsi_backtest("PYPL",datetime(2019, 1, 1), datetime(2019, 12, 31),5, 30, [61,75,74,73,72,71,70,60,50],1000)

# strats = [x[0] for x in returnedStrats]
# for i, strat in enumerate(strats):
#     # rets = strat.analyzers.returns.get_analysis()
#     rets = list(strat.analyzers.returns.get_analysis().items())[0][1]
#     base_rets = strat.analyzers.basereturn.get_analysis()
#     base_rets = list(base_rets.items())[0][1]
#     profit = strat.p.pnl_val
#     print(i, profit, rets, base_rets)

# print(max(list(strat.analyzers.returns.get_analysis().items())[0][1] for strat in strats))
#%%
   
# if __name__ == '__main__':



#     os.environ["alpaca_secret_paper"] = ALPACA_DATA["alpaca_secret_paper"]
#     os.environ["alpaca_secret_live"] = ALPACA_DATA["alpaca_secret_live"]
#     os.environ["alpaca_live"] = ALPACA_DATA["alpaca_live"]
#     os.environ["alpaca_paper"] = ALPACA_DATA["alpaca_paper"]

#     thing = callable_rsi_backtest('aapl',  dt.datetime(2020, 6, 1), date.today(), 14, 30, 70, 1000)
#     print(thing)



    # # Create a cerebro entity
    # cerebro = bt.Cerebro()
    # cerebro.addstrategy(BasicRSI,rsi_period = 11,rsi_lower = 34,rsi_upper = 74)

    # store = alpaca_backtrader_api.AlpacaStore(
    #     key_id=ALPACA_API_KEY,
    #     secret_key=ALPACA_SECRET_KEY,
    #     paper=not IS_LIVE,
    #     usePolygon=USE_POLYGON
    # )

    # DataFactory = store.getdata  # or use alpaca_backtrader_api.AlpacaData
    # if IS_BACKTEST:
    #     data0 = DataFactory(dataname=symbol,
    #                         historical=True,
    #                         fromdate=datetime(2020, 1, 1),
    #                         todate=datetime(2020, 10, 11),
    #                         timeframe=bt.TimeFrame.Days)
    # else:
    #     data0 = DataFactory(dataname=symbol,
    #                         historical=False,
    #                         timeframe=bt.TimeFrame.Days,
    #                         backfill_start=True,
    #                         )
    #     # or just alpaca_backtrader_api.AlpacaBroker()
    #     broker = store.getbroker()
    #     cerebro.setbroker(broker)
    # cerebro.adddata(data0)

    # if IS_BACKTEST:
    #     # backtrader broker set initial simulated cash
    #     cerebro.broker.setcash(100000.0)

    #     # Apply Total, Average, Compound and Annualized Returns calculated using a logarithmic approach
    # cerebro.addanalyzer(btanalyzers.Returns, _name="returns")
    # cerebro.addanalyzer(btanalyzers.SharpeRatio, _name="mysharpe")
    # cerebro.addanalyzer(btanalyzers.DrawDown, _name="drawdown")
    # cerebro.addanalyzer(btanalyzers.TimeDrawDown, _name="timedraw")

    # print('Starting Portfolio Value: {}'.format(cerebro.broker.getvalue()))
    # cerebro.run()
    # print('Final Portfolio Value: {}'.format(cerebro.broker.getvalue()))
    # cerebro.plot()



# %%

# %%
