#!/usr/bin/env python3
"""Unit tests for the momentum portfolio-variant simulator."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'research'))

from momentum_rank_portfolio import simulate  # noqa: E402


def _base(n=20, m=3, px=10.0):
    close = np.full((n, m), px)
    return dict(close=close, open_=close.copy(), member=np.ones((n, m), bool),
                trend=np.ones((n, m)), score=np.tile(np.arange(m, 0, -1.0), (n, 1)),
                regime_ok=np.ones(n, bool), rebalance=np.zeros(n, bool), start_i=0,
                cost_bps=0.0, min_cash=0.0)


class TestSimulate(unittest.TestCase):
    def test_sma_mode_buys_top_score_next_open(self):
        k = _base()
        eq, trades = simulate(slots=1, exit_mode="sma", buffer=20, **k)
        self.assertEqual(trades, [])            # still held at the end
        k["close"][5:, 0] = 20.0; k["open_"][5:, 0] = 20.0
        eq, _ = simulate(slots=1, exit_mode="sma", buffer=20, **k)
        self.assertAlmostEqual(eq[-1], 200_000.0)  # bought sym 0 at 10 on day 1

    def test_sma_exit_sells_next_open(self):
        k = _base()
        k["trend"][8, 0] = 0.0
        k["open_"][9, 0] = 12.0
        _, trades = simulate(slots=1, exit_mode="sma", buffer=20, **k)
        self.assertEqual(trades[0]["exit_reason"], "rule_exit")
        self.assertEqual(trades[0]["exit_date"], 9)
        self.assertAlmostEqual(trades[0]["exit_price"], 12.0)

    def test_rank_exit_only_on_rebalance_and_beyond_buffer(self):
        k = _base(m=3)
        k["rebalance"][[0, 10]] = True
        k["score"][3:, 0] = 0.5               # sym 0 falls to rank 3 from day 3
        _, trades = simulate(slots=1, exit_mode="rank", buffer=2, **k)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_reason"], "rank_exit")
        self.assertEqual(trades[0]["exit_date"], 11)  # decided at rebalance close 10

    def test_rank_mode_keeps_holding_within_buffer(self):
        k = _base(m=3)
        k["rebalance"][[0, 10]] = True
        k["score"][3:, 0] = 1.5               # rank 2 <= buffer 2
        _, trades = simulate(slots=1, exit_mode="rank", buffer=2, **k)
        self.assertEqual(trades, [])

    def test_delisted_holding_sold_at_last_close(self):
        k = _base(n=30)
        k["close"][12:, 0] = np.nan; k["open_"][12:, 0] = np.nan
        k["close"][11, 0] = 7.0
        _, trades = simulate(slots=1, exit_mode="sma", buffer=20, **k)
        self.assertEqual(trades[0]["exit_reason"], "data_end")
        self.assertEqual(trades[0]["exit_date"], 11)
        self.assertAlmostEqual(trades[0]["exit_price"], 7.0)

    def test_regime_off_sells_and_blocks_entries(self):
        k = _base()
        k["regime_ok"][6:] = False
        eq, trades = simulate(slots=2, exit_mode="sma", buffer=20, **k)
        self.assertEqual({t["exit_reason"] for t in trades}, {"regime"})
        self.assertTrue(all(t["exit_date"] == 7 for t in trades))

    def test_cash_floor_and_slot_sizing(self):
        k = _base(m=4)
        k["min_cash"] = 0.10
        eq, _ = simulate(slots=3, exit_mode="rank", buffer=20, max_new=3,
                         **{**k, "rebalance": np.r_[True, np.zeros(19, bool)]})
        self.assertAlmostEqual(eq[-1], 100_000.0)  # flat prices, no costs


if __name__ == "__main__":
    unittest.main()
