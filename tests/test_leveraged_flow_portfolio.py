#!/usr/bin/env python3
"""
Unit tests for the ETF-level cross-sectional flow portfolio.
"""
import os
import sys
import unittest
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.leveraged_flow_portfolio import (  # noqa: E402
    DEFAULT_AUM_USD,
    etf_flow_estimates,
    placebo_portfolio_test,
    portfolio_session_returns,
    portfolio_stats,
    split_half_stability,
)
from strategies.leveraged_rebalance_signal import (  # noqa: E402
    prepare_features,
)

_BARS_PER_DAY = 78


def _bars(days=30, base=100.0, rets=None):
    """Intraday bars: session i trades flat at base*(1+prod(rets[:i+1])).

    The day's return is a gap at the open, so ``cum_ret`` equals the day return
    at EVERY bar — including mid-afternoon decision times. (Emitting the move
    only on the final bar would leave cum_ret=0 at 15:20, which is not what
    these tests exercise.)
    """
    idx_rets = rets if rets is not None else [0.0] * days
    sessions = pd.bdate_range("2026-06-01", periods=days)
    frames = []
    level = base
    for i, day in enumerate(sessions):
        start = (pd.Timestamp(day).tz_localize("US/Eastern")
                 + pd.Timedelta(hours=9, minutes=30))
        idx = pd.date_range(start, periods=_BARS_PER_DAY, freq="5min")
        n = len(idx)
        level = level * (1.0 + idx_rets[i % len(idx_rets)])
        close = np.full(n, level)
        frames.append(pd.DataFrame({
            "open": close, "high": close * 1.0005, "low": close * 0.9995,
            "close": close, "volume": np.full(n, 1000.0),
        }, index=idx))
    return pd.concat(frames)


class TestEtfFlowEstimates(unittest.TestCase):
    def _etf_features(self, rets):
        # One synthetic ETF ("FAKE") is not in the map, so patch the map by
        # passing a features frame keyed by a real ETF ticker instead.
        from strategies.leveraged_single_stock_etfs import (
            LEVERAGED_SINGLE_STOCK_ETFS)
        real_etf = next(iter(LEVERAGED_SINGLE_STOCK_ETFS))
        return real_etf, {real_etf: prepare_features(_bars(rets=rets))}

    def test_notional_is_leverage_times_aum_times_return(self):
        from strategies.leveraged_single_stock_etfs import (
            LEVERAGED_SINGLE_STOCK_ETFS)

        etf, feats = self._etf_features([0.01] * 20)
        fund = LEVERAGED_SINGLE_STOCK_ETFS[etf]
        aum = fund.aum_usd or DEFAULT_AUM_USD
        asof_tod = 15 * 60 + 20

        flows = etf_flow_estimates(feats, asof_tod=asof_tod)
        self.assertFalse(flows.empty)
        expected = fund.leverage * aum * 0.01
        self.assertAlmostEqual(
            flows["notional"].iloc[0], expected, delta=abs(expected) * 1e-9)

    def test_uses_only_data_up_to_asof(self):
        """The flow at T must not use the session close (T+ hours)."""
        from strategies.leveraged_single_stock_etfs import (
            LEVERAGED_SINGLE_STOCK_ETFS)

        etf, feats = self._etf_features([0.0] * 20)
        features = feats[etf]
        asof_tod = 15 * 60 + 20
        # First session has no prior close (NaN); later sessions must show the
        # full day return at the as-of bar (gap-at-open generator).
        asof_rets = features.loc[features["tod"] == asof_tod, "cum_ret"]
        self.assertTrue(pd.isna(asof_rets.iloc[0]))
        self.assertTrue((asof_rets.iloc[1:] == 0.0).all())
        flows = etf_flow_estimates(feats, asof_tod=asof_tod)
        self.assertFalse(flows.empty)
        self.assertTrue(np.isfinite(flows["notional"]).all())

    def test_empty_inputs(self):
        self.assertTrue(etf_flow_estimates({}, 920).empty)


class TestPortfolioSessionReturns(unittest.TestCase):
    def _setup(self, rets_a, rets_b, top_k=2):
        features = {
            "AAA": prepare_features(_bars(rets=rets_a)),
            "BBB": prepare_features(_bars(rets=rets_b)),
        }
        # Synthetic flows: rank AAA first on up days, BBB first on down days.
        rows = []
        n = min(len(rets_a), len(rets_b))
        for i, (ra, rb) in enumerate(zip(rets_a, rets_b)):
            day = pd.bdate_range("2026-06-01", periods=n)[i].date()
            rows.append({"date": day, "underlying": "AAA",
                         "notional": 100.0, "abs_notional": 100.0, "n_funds": 1})
            rows.append({"date": day, "underlying": "BBB",
                         "notional": 50.0, "abs_notional": 50.0, "n_funds": 1})
        flows = pd.DataFrame(rows)
        return flows, features

    def test_directions_follow_flow_sign(self):
        rets_a = [0.02] * 10
        rets_b = [-0.01] * 10
        flows, features = self._setup(rets_a, rets_b)
        returns, decisions = portfolio_session_returns(
            flows, features, entry_tod=15 * 60 + 20, top_k=2,
            cost_bps=0.0)
        self.assertFalse(returns.empty)
        first = decisions[0]
        # AAA has the larger |flow| and positive flow => long.
        self.assertEqual(first.symbols[0], "AAA")
        self.assertEqual(first.directions[0], "long")
        # Gap-at-open generator: entry price == session close => return 0.
        self.assertEqual(first.returns[0], 0.0)

    def test_costs_reduce_returns(self):
        rets = [0.02] * 10
        flows, features = self._setup(rets, rets)
        free, _ = portfolio_session_returns(
            flows, features, entry_tod=15 * 60 + 20, top_k=1, cost_bps=0.0)
        costly, _ = portfolio_session_returns(
            flows, features, entry_tod=15 * 60 + 20, top_k=1, cost_bps=50.0)
        self.assertGreater(
            (free - costly).abs().sum(), 0)
        self.assertAlmostEqual(
            (free - costly).iloc[0], 2 * 50.0 / 10_000.0, places=12)

    def test_short_direction_filter(self):
        rets_a = [0.02] * 8
        rets_b = [-0.01] * 8
        flows, features = self._setup(rets_a, rets_b)
        returns, decisions = portfolio_session_returns(
            flows, features, entry_tod=15 * 60 + 20, top_k=2,
            direction_filter="short", cost_bps=0.0)
        for dec in decisions:
            self.assertTrue(all(d == "short" for d in dec.directions))

    def test_min_abs_notional_skips_small_sessions(self):
        rets = [0.02] * 6
        flows, features = self._setup(rets, rets)
        returns, _ = portfolio_session_returns(
            flows, features, entry_tod=15 * 60 + 20, top_k=1,
            min_abs_notional=1e9, cost_bps=0.0)
        self.assertTrue(returns.empty)


class TestStats(unittest.TestCase):
    def test_portfolio_stats(self):
        s = pd.Series([0.01, 0.02, -0.01, 0.01], dtype=float)
        stats = portfolio_stats(s)
        self.assertEqual(stats["n_sessions"], 4)
        self.assertAlmostEqual(stats["mean"], 0.0075)
        self.assertEqual(stats["hit_rate"], 0.75)
        self.assertGreater(stats["t_stat"], 0)

    def test_stats_empty(self):
        stats = portfolio_stats(pd.Series(dtype=float))
        self.assertIsNone(stats["mean"])

    def test_split_half(self):
        s = pd.Series([0.01] * 10 + [-0.01] * 10, dtype=float)
        stab = split_half_stability(s)
        self.assertGreater(stab["first_mean"], 0)
        self.assertLess(stab["second_mean"], 0)
        self.assertFalse(stab["consistent"])


class TestPlacebo(unittest.TestCase):
    def test_extreme_observed_mean_is_significant(self):
        """An observed mean far above the null must give a small p."""
        features = {"AAA": prepare_features(_bars(rets=[0.01] * 30))}
        flows = pd.DataFrame([
            {"date": pd.bdate_range("2026-06-01", periods=30)[i].date(),
             "underlying": "AAA", "notional": 100.0,
             "abs_notional": 100.0, "n_funds": 1}
            for i in range(30)])
        test = placebo_portfolio_test(
            flows, features, entry_tod=15 * 60 + 20, top_k=1,
            direction_filter="long", cost_bps=0.0, min_abs_notional=0.0,
            observed_mean=0.05, n_draws=200)
        self.assertIsNotNone(test["p_value"])
        self.assertLessEqual(test["p_value"], 0.05)

    def test_too_few_sessions_returns_none(self):
        flows = pd.DataFrame([
            {"date": date(2026, 6, 1), "underlying": "AAA",
             "notional": 1.0, "abs_notional": 1.0, "n_funds": 1}])
        test = placebo_portfolio_test(
            flows, {}, entry_tod=920, top_k=1, direction_filter="both",
            cost_bps=0.0, min_abs_notional=0.0, observed_mean=0.1, n_draws=50)
        self.assertIsNone(test["p_value"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
