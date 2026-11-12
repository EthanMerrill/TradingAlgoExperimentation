"""Backtest execution realism: next-open fills, intraday stops, costs."""
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("ENVIRONMENT", "dev")

import numpy as np
import pandas as pd

from config import globalConfig  # type: ignore
from strategies.rsi import RSIStrategy


def make_bars(rows, direction="long"):
    """rows: list of (open, high, low, close). RSI crosses the entry
    threshold on bar 2's close, so the entry fills on bar 3's open."""
    dates = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=dates)
    vals = [50.0] * len(rows)
    if direction == "long":
        vals[1], vals[2] = 35.0, 25.0   # cross below 30
    else:
        vals[1], vals[2] = 65.0, 75.0   # cross above 70
    return df, pd.Series(vals, index=dates)


class RealismBase(unittest.TestCase):
    slippage = 0.0
    commission = 0.0
    sec = 0.0

    def setUp(self):
        for name, val in (("STOP_LOSS_PCT", 0.05), ("TAKE_PROFIT_PCT", 0.15),
                          ("BACKTEST_SLIPPAGE_BPS", self.slippage),
                          ("BACKTEST_COMMISSION_PER_SHARE", self.commission),
                          ("BACKTEST_SEC_FEE_BPS", self.sec)):
            p = patch.object(globalConfig, name, val, create=True)
            p.start()
            self.addCleanup(p.stop)

    def run_sim(self, rows, direction="long", cash=10000.0):
        df, rsi = make_bars(rows, direction)
        strat = RSIStrategy(14, 30, 70, max_hold_days=30, direction=direction)
        return strat._simulate_trades(df, rsi, cash), df


FLAT = (100, 100, 100, 100)


class TestFills(RealismBase):
    def test_entry_at_next_open(self):
        (trades, eq), df = self.run_sim(
            [FLAT, FLAT, FLAT, (101, 102, 100.5, 101.5), (101, 102, 100, 101),
             (101, 102, 100, 101)])
        # No exit in this window -> position open; check via equity: marked at close
        self.assertEqual(trades, [])
        # entry at 101 (bar 3 open), close 101.5 -> +0.495%
        self.assertAlmostEqual(eq[3], 10000 * (1 + (101.5 - 101) / 101), places=6)
        self.assertEqual(eq[2], 10000.0)

    def test_gap_through_stop_fills_at_open(self):
        (trades, _), df = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (90, 92, 88, 91), FLAT])
        t = trades[0]
        self.assertEqual(t['exit_reason'], 'stop_loss')
        self.assertEqual(t['entry_price'], 100)
        self.assertEqual(t['exit_price'], 90)      # open, not the 95 stop
        self.assertEqual(t['entry_date'], df.index[3])
        self.assertEqual(t['exit_date'], df.index[4])

    def test_intraday_stop_fills_at_stop(self):
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (99, 100, 94, 96), FLAT])
        self.assertEqual(trades[0]['exit_reason'], 'stop_loss')
        self.assertAlmostEqual(trades[0]['exit_price'], 95.0)

    def test_same_bar_stop_and_tp_assumes_stop(self):
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (100, 120, 90, 100), FLAT])
        self.assertEqual(trades[0]['exit_reason'], 'stop_loss')
        self.assertAlmostEqual(trades[0]['exit_price'], 95.0)

    def test_take_profit_intraday_and_gap(self):
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (101, 116, 100, 110), FLAT])
        self.assertEqual(trades[0]['exit_reason'], 'take_profit_fallback')
        self.assertAlmostEqual(trades[0]['exit_price'], 115.0)
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (120, 125, 119, 122), FLAT])
        self.assertEqual(trades[0]['exit_price'], 120)

    def test_entry_bar_protection_is_evaluated(self):
        (trades, _), df = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 90, 92), FLAT, FLAT])
        self.assertEqual(trades[0]['exit_date'], df.index[3])
        self.assertAlmostEqual(trades[0]['exit_price'], 95.0)

    def test_short_mirror(self):
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (110, 112, 108, 111), FLAT],
            direction="short")
        t = trades[0]
        self.assertEqual(t['exit_reason'], 'stop_loss')
        self.assertEqual(t['exit_price'], 110)     # gap through 105 stop
        self.assertAlmostEqual(t['return'], -0.10)
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (100, 106, 80, 100), FLAT],
            direction="short")
        self.assertAlmostEqual(trades[0]['exit_price'], 105.0)  # stop first

    def test_rsi_cross_exit_fills_next_open(self):
        rows = [FLAT, FLAT, FLAT, (100, 101, 99, 100), (101, 102, 100, 101),
                (101, 102, 100, 102), (103, 104, 102, 103), (104, 105, 103, 104)]
        df, rsi = make_bars(rows)
        rsi.iloc[4], rsi.iloc[5] = 65.0, 75.0   # cross above 70 at bar 5 close
        strat = RSIStrategy(14, 30, 70, max_hold_days=30)
        trades, _ = strat._simulate_trades(df, rsi, 10000.0)
        self.assertEqual(trades[0]['exit_reason'], 'rsi_cross')
        self.assertEqual(trades[0]['exit_date'], df.index[6])
        self.assertEqual(trades[0]['exit_price'], 103)  # bar 6 open

    def test_max_hold_exit_next_open(self):
        rows = [FLAT] * 3 + [(100, 101, 99, 100)] + [(100, 101, 99, 100)] * 5
        df, rsi = make_bars(rows)
        strat = RSIStrategy(14, 30, 70, max_hold_days=2)
        trades, _ = strat._simulate_trades(df, rsi, 10000.0)
        self.assertEqual(trades[0]['exit_reason'], 'max_hold_days')
        # entry bar 3, held 2 days at bar 5 close -> exit at bar 6 open
        self.assertEqual(trades[0]['exit_date'], df.index[6])


class TestSlippage(RealismBase):
    slippage = 10.0  # 10 bps

    def test_slippage_is_adverse(self):
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (99, 100, 94, 96), FLAT])
        t = trades[0]
        self.assertAlmostEqual(t['entry_price'], 100 * 1.001)
        self.assertAlmostEqual(t['exit_price'], (100 * 1.001 * 0.95) * 0.999)
        (trades, _), _ = self.run_sim(
            [FLAT, FLAT, FLAT, (100, 101, 99, 100), (110, 112, 108, 111), FLAT],
            direction="short")
        self.assertAlmostEqual(trades[0]['entry_price'], 100 * 0.999)  # sells lower
        self.assertGreater(trades[0]['exit_price'], 110)               # buys higher


class TestCosts(unittest.TestCase):
    ROWS = [FLAT, FLAT, FLAT, (100, 101, 99, 100), (101, 102, 100, 101),
            (101, 102, 100, 102), (103, 104, 102, 103), (104, 105, 103, 104)]

    def total_return(self, **cfg):
        base = dict(STOP_LOSS_PCT=0.05, TAKE_PROFIT_PCT=0.15,
                    BACKTEST_SLIPPAGE_BPS=0.0, BACKTEST_COMMISSION_PER_SHARE=0.0,
                    BACKTEST_SEC_FEE_BPS=0.0)
        base.update(cfg)
        patches = [patch.object(globalConfig, k, v, create=True) for k, v in base.items()]
        for p in patches:
            p.start()
        try:
            df, rsi = make_bars(self.ROWS)
            rsi.iloc[4], rsi.iloc[5] = 65.0, 75.0
            strat = RSIStrategy(14, 30, 70, max_hold_days=30)
            trades, eq = strat._simulate_trades(df, rsi, 10000.0)
            return trades, eq
        finally:
            for p in patches:
                p.stop()

    def test_costs_reduce_returns(self):
        _, eq0 = self.total_return()
        _, eq_slip = self.total_return(BACKTEST_SLIPPAGE_BPS=10.0)
        _, eq_comm = self.total_return(BACKTEST_COMMISSION_PER_SHARE=0.05)
        _, eq_sec = self.total_return(BACKTEST_SEC_FEE_BPS=5.0)
        self.assertLess(eq_slip[-1], eq0[-1])
        self.assertLess(eq_comm[-1], eq0[-1])
        self.assertLess(eq_sec[-1], eq0[-1])

    def test_equity_equals_compounded_trade_returns(self):
        # Two sequential trades, with costs enabled.
        rows = [FLAT, FLAT, FLAT, (100, 101, 99, 100), (101, 102, 100, 101),
                (101, 102, 100, 102), (103, 104, 102, 103), (104, 105, 103, 104),
                (104, 105, 103, 104), (104, 105, 103, 104), (105, 106, 104, 105),
                (105, 106, 104, 105), (106, 107, 105, 106), (106, 107, 105, 106)]
        rows += [(106, 107, 105, 106)] * 20  # satisfy min-length for backtest_with_rsi
        cfg = dict(STOP_LOSS_PCT=0.05, TAKE_PROFIT_PCT=0.15,
                   BACKTEST_SLIPPAGE_BPS=5.0, BACKTEST_COMMISSION_PER_SHARE=0.01,
                   BACKTEST_SEC_FEE_BPS=0.0278)
        patches = [patch.object(globalConfig, k, v, create=True) for k, v in cfg.items()]
        for p in patches:
            p.start()
        try:
            df, rsi = make_bars(rows)
            rsi.iloc[4], rsi.iloc[5] = 65.0, 75.0      # exit trade 1
            rsi.iloc[7], rsi.iloc[8] = 35.0, 25.0      # re-enter
            rsi.iloc[10], rsi.iloc[11] = 65.0, 75.0    # exit trade 2
            strat = RSIStrategy(14, 30, 70, max_hold_days=30)
            trades, eq = strat._simulate_trades(df, rsi, 10000.0)
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(len(trades), 2)
        compounded = 10000.0 * np.prod([1 + t['return'] for t in trades])
        self.assertAlmostEqual(eq[-1], compounded, places=6)
        # Full backtest path agrees too
        with patch.object(globalConfig, 'BACKTEST_SLIPPAGE_BPS', 5.0, create=True), \
             patch.object(globalConfig, 'BACKTEST_COMMISSION_PER_SHARE', 0.01, create=True), \
             patch.object(globalConfig, 'BACKTEST_SEC_FEE_BPS', 0.0278, create=True):
            res = strat.backtest_with_rsi(df, "T", rsi, 10000.0)
        self.assertAlmostEqual(res.total_return, compounded / 10000.0 - 1, places=9)
        self.assertEqual(res.num_trades, 2)
        self.assertEqual(res.win_rate, sum(t['return'] > 0 for t in trades) / 2)


class TestWindowEnd(RealismBase):
    slippage = 10.0   # 10 bps
    commission = 0.0
    sec = 0.0

    ROWS = [FLAT, FLAT, FLAT, (101, 102, 100.5, 101.5), (101, 102, 100, 101)] + [
        (101, 102, 100, 101)] * 18 + [(101, 102, 100, 101.2)]

    def test_open_trade_closed_at_last_close_with_flag(self):
        df, rsi = make_bars(self.ROWS)
        strat = RSIStrategy(14, 30, 70, max_hold_days=30)
        trades, eq = strat._simulate_trades(df, rsi, 10000.0, close_open_at_end=True)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t['exit_reason'], 'window_end')
        self.assertEqual(t['exit_date'], df.index[-1])
        self.assertAlmostEqual(t['exit_price'], 101.2 * (1 - 0.001))
        self.assertAlmostEqual(t['entry_price'], 101 * 1.001)
        self.assertAlmostEqual(eq[-1], 10000 * np.prod([1 + x['return'] for x in trades]), places=6)
        res = strat.backtest_with_rsi(df, "T", rsi, 10000.0, close_open_at_end=True)
        self.assertEqual(res.num_trades, 1)
        self.assertAlmostEqual(res.total_return, trades[0]['return'], places=9)
        self.assertEqual(res.win_rate, 1.0 if t['return'] > 0 else 0.0)

    def test_default_unchanged(self):
        df, rsi = make_bars(self.ROWS)
        strat = RSIStrategy(14, 30, 70, max_hold_days=30)
        trades, eq = strat._simulate_trades(df, rsi, 10000.0)
        self.assertEqual(trades, [])
        res = strat.backtest_with_rsi(df, "T", rsi, 10000.0)
        self.assertEqual(res.num_trades, 0)


if __name__ == "__main__":
    unittest.main()
