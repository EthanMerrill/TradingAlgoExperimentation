#!/usr/bin/env python3
"""Unit tests for the backtest-ranked momentum research script."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'research'))

from momentum_rank_backtest import (  # noqa: E402
    backtest_scores, build_candidates, is_common_stock, pit_universe, rank_buckets, rule_positions)


def _panel(close: dict, n: int):
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = pd.DataFrame(close, index=idx, dtype=float)
    return {"close": c, "open": c.copy(), "volume": c * 0 + 1e6}


class TestBuildCandidates(unittest.TestCase):
    def test_delisted_holding_exits_at_last_close_not_dropped(self):
        # Uptrend, then the data stops (delisting) while still above the SMA.
        n = 40
        series = list(np.linspace(20, 40, 30)) + [np.nan] * 10
        panel = _panel({"DEAD": series}, n)
        pos = rule_positions(panel["close"], 5)
        c = build_candidates(panel, pos, panel["close"].index[0], 15, 1000)
        self.assertFalse(c.empty)
        last = c.iloc[-1]
        self.assertEqual(last["exit_reason"], "data_end")
        self.assertEqual(last["exit_date"], panel["close"].index[29])
        self.assertAlmostEqual(last["exit_price"], 40.0)

    def test_rule_exit_at_next_open_after_close_below_sma(self):
        series = [20.0 + i for i in range(20)] + [10.0] * 5
        panel = _panel({"X": series}, 25)
        pos = rule_positions(panel["close"], 5)
        c = build_candidates(panel, pos, panel["close"].index[0], 15, 1000)
        first = c.iloc[0]
        self.assertEqual(first["exit_reason"], "rule_exit")
        self.assertEqual(first["exit_date"], panel["close"].index[21])  # close 20 below -> open 21

    def test_universe_mask_gates_entries(self):
        series = [20.0 + i for i in range(30)]
        panel = _panel({"X": series}, 30)
        pos = rule_positions(panel["close"], 5)
        mask = pd.DataFrame(False, index=panel["close"].index, columns=["X"])
        mask.iloc[10, 0] = True
        c = build_candidates(panel, pos, panel["close"].index[0], 15, 1000, universe=mask)
        self.assertEqual(list(c["decision_date"]), [panel["close"].index[10]])


class TestPitUniverse(unittest.TestCase):
    def test_top_n_by_dollar_volume_and_price_band(self):
        idx = pd.bdate_range("2024-01-01", periods=80)
        close = pd.DataFrame({"A": 50.0, "B": 50.0, "C": 10.0}, index=idx)
        vol = pd.DataFrame({"A": 1e6, "B": 2e6, "C": 1e8}, index=idx)
        m = pit_universe(close, vol, top_n=1, min_price=15, max_price=1000)
        self.assertTrue(m["B"].iloc[-1])      # most liquid in band
        self.assertFalse(m["A"].iloc[-1])     # outside top 1
        self.assertFalse(m["C"].iloc[-1])     # liquid but below min price
        self.assertFalse(m.iloc[0].any())     # no history -> no membership


class TestScores(unittest.TestCase):
    def test_scores_use_only_data_before_skip(self):
        idx = pd.bdate_range("2024-01-01", periods=120)
        rng = np.random.default_rng(0)
        close = pd.DataFrame({"X": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, 120))}, index=idx)
        pos = rule_positions(close, 10)
        s1 = backtest_scores(close, pos, 60, 10, 5, 1, metric="return")
        shocked = close.copy()
        shocked.iloc[-10:] *= 3  # change only the last `skip` days
        s2 = backtest_scores(shocked, rule_positions(shocked, 10), 60, 10, 5, 1, metric="return")
        self.assertAlmostEqual(s1["X"].iloc[-1], s2["X"].iloc[-1])

    def test_rank_buckets_monotone_example(self):
        d = pd.Timestamp("2024-01-02")
        cands = pd.DataFrame({
            "entry_date": [d] * 4, "exit_date": [d + pd.Timedelta(days=10)] * 4,
            "entry_price": [10.0] * 4, "exit_price": [14.0, 12.0, 10.0, 9.0],
            "score": [4, 3, 2, 1]})
        b = rank_buckets(cands, edges=(1, 2, 3, 10**9))
        self.assertAlmostEqual(b["mean_ret"].iloc[0], 0.4)
        self.assertAlmostEqual(b["mean_ret"].iloc[-1], -0.05)


class TestCommonStock(unittest.TestCase):
    def test_funds_and_preferreds_excluded_companies_kept(self):
        for sym, name in [("SGOV", "iShares 0-3 Month Treasury Bond ETF"),
                          ("USFR", "WisdomTree Floating Rate Treasury Fund"),
                          ("BGT", "BlackRock Floating Rate Income Trust"),
                          ("GLD", "SPDR Gold Trust, SPDR Gold Shares"),
                          ("DLR.PRL", "Digital Realty Trust, Inc. Preferred")]:
            self.assertFalse(is_common_stock(sym, name), sym)
        for sym, name in [("NTRS", "Northern Trust Corporation Common Stock"),
                          ("VNO", "Vornado Realty Trust"), ("O", "Realty Income Corporation"),
                          ("GOLD", "Barrick Gold Corp"), ("RHT", "")]:
            self.assertTrue(is_common_stock(sym, name), sym)


if __name__ == "__main__":
    unittest.main()
