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

    def test_notional_is_leverage_minus_one_times_aum_times_fund_return(self):
        from strategies.leveraged_single_stock_etfs import (
            LEVERAGED_SINGLE_STOCK_ETFS)

        etf, feats = self._etf_features([0.01] * 20)
        fund = LEVERAGED_SINGLE_STOCK_ETFS[etf]
        aum = fund.aum_usd or DEFAULT_AUM_USD
        asof_tod = 15 * 60 + 20

        flows = etf_flow_estimates(feats, asof_tod=asof_tod)
        self.assertFalse(flows.empty)
        # Rebalance = A*(L-1)*R_fund  (= A*L*(L-1)*r_underlying).
        expected = (fund.leverage - 1.0) * aum * 0.01
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



def _rth_session(_day):
    from datetime import time as _t
    return (_t(9, 30), _t(16, 0))


class TestFixes(unittest.TestCase):
    """Regressions for the RTH / formula / exit / placebo fixes."""

    def test_rebalance_multiplier(self):
        from strategies.leveraged_single_stock_etfs import rebalance_multiplier
        self.assertEqual(rebalance_multiplier(2.0), 2.0)
        self.assertEqual(rebalance_multiplier(-1.0), 2.0)   # inverse BUYS up
        self.assertEqual(rebalance_multiplier(-2.0), 6.0)
        self.assertEqual(rebalance_multiplier(1.0), 0.0)

    def test_inverse_fund_flow_is_positive_on_up_day(self):
        from strategies.leveraged_single_stock_etfs import (
            LEVERAGED_SINGLE_STOCK_ETFS)
        inverse = next(e for e, f in LEVERAGED_SINGLE_STOCK_ETFS.items()
                       if f.leverage == -1.0)
        # Underlying +2% => a 1x inverse fund is down 2%.
        feats = {inverse: prepare_features(_bars(rets=[-0.02] * 5))}
        flows = etf_flow_estimates(feats, asof_tod=920)
        self.assertTrue((flows["notional"] > 0).all())

    def test_rth_only_drops_extended_hours(self):
        days = pd.bdate_range("2026-06-01", periods=2)
        frames = []
        for i, day in enumerate(days):
            idx = pd.date_range(
                pd.Timestamp(day).tz_localize("US/Eastern")
                + pd.Timedelta(hours=4), periods=192, freq="5min")
            close = np.full(len(idx), 100.0 + i)
            tod = idx.hour * 60 + idx.minute
            close[tod >= 16 * 60] = 150.0  # after-hours spike
            frames.append(pd.DataFrame({
                "open": close, "high": close, "low": close, "close": close,
                "volume": np.full(len(idx), 10.0)}, index=idx))
        bars = pd.concat(frames)
        full = prepare_features(bars)
        rth = prepare_features(bars, rth_only=True, session_lookup=_rth_session)
        self.assertEqual(full["tod"].max(), 19 * 60 + 55)
        self.assertEqual(rth["tod"].min(), 9 * 60 + 30)
        self.assertEqual(rth["tod"].max(), 15 * 60 + 55)
        # Prior close is the RTH close (100), not the after-hours 150.
        day2 = rth[rth["date"] == days[1].date()]
        self.assertTrue((day2["prior_close"] == 100.0).all())

    def test_rth_only_respects_early_close_and_holidays(self):
        from datetime import time as _t
        bars = _bars(days=2)
        first = bars.index[0].tz_convert("US/Eastern").date()

        def lookup(day):
            return (_t(9, 30), _t(13, 0)) if day == first else None
        rth = prepare_features(bars, rth_only=True, session_lookup=lookup)
        self.assertEqual(set(rth["date"]), {first})
        self.assertEqual(rth["tod"].max(), 12 * 60 + 55)

    def test_flow_uses_last_bar_at_or_before_asof(self):
        from strategies.leveraged_single_stock_etfs import (
            LEVERAGED_SINGLE_STOCK_ETFS)
        etf = next(iter(LEVERAGED_SINGLE_STOCK_ETFS))
        feats = prepare_features(_bars(rets=[0.01] * 5))
        # Sparse fund: no print at 15:20 (or later) on any day.
        sparse = feats[feats["tod"] < 920]
        flows = etf_flow_estimates({etf: sparse}, asof_tod=920)
        self.assertEqual(len(flows), 29)  # 30 sessions; first has no prior close

    def test_exit_tod_uses_open_of_exit_bar(self):
        from strategies.leveraged_flow_portfolio import _price_tables
        feats = prepare_features(_bars(rets=[0.0] * 3))
        feats.loc[feats["tod"] == 955, "open"] = 123.0
        _, close_px = _price_tables({"AAA": feats}, 920, exit_tod=955)
        _, last_px = _price_tables({"AAA": feats}, 920)
        day = feats["date"].iloc[-1]
        self.assertEqual(close_px[("AAA", day)], 123.0)
        self.assertNotEqual(last_px[("AAA", day)], 123.0)

    def test_eligible_placebo_only_draws_qualifying_names(self):
        """With one eligible name per session the null equals the strategy."""
        dates = pd.bdate_range("2026-06-01", periods=30)
        features = {
            "WIN": prepare_features(_bars(rets=[0.0] * 30)),
            "LOSE": prepare_features(_bars(rets=[0.0] * 30)),
        }
        rows = []
        for d in dates:
            rows.append({"date": d.date(), "underlying": "WIN",
                         "notional": 1e8, "abs_notional": 1e8, "n_funds": 1})
            rows.append({"date": d.date(), "underlying": "LOSE",
                         "notional": -1e8, "abs_notional": 1e8, "n_funds": 1})
        flows = pd.DataFrame(rows)
        returns, _ = portfolio_session_returns(
            flows, features, entry_tod=920, top_k=1,
            direction_filter="long", cost_bps=1.0, min_abs_notional=1e7)
        test = placebo_portfolio_test(
            flows, features, entry_tod=920, top_k=1, direction_filter="long",
            cost_bps=1.0, min_abs_notional=1e7,
            observed_mean=float(returns.mean()), n_draws=50)
        self.assertAlmostEqual(test["null_mean"], float(returns.mean()))
        self.assertGreater(test["p_value"], 0.5)

    def test_adv_score(self):
        from strategies.leveraged_flow_portfolio import add_adv_score
        dates = pd.bdate_range("2026-06-01", periods=30)
        features = {"AAA": prepare_features(_bars(rets=[0.0] * 30))}
        flows = pd.DataFrame([
            {"date": d.date(), "underlying": "AAA", "notional": 1e6,
             "abs_notional": 1e6, "n_funds": 1} for d in dates])
        scored = add_adv_score(flows, features, lookback_days=4)
        # 78 bars × $100 × 1000 shares = $7.8M/day.
        self.assertAlmostEqual(scored["adv_usd"].iloc[-1], 7.8e6)
        self.assertAlmostEqual(scored["rank_score"].iloc[-1], 1e6 / 7.8e6)
        # First session has no history: dropped, never filled from the future.
        self.assertNotIn(dates[0].date(), set(scored["date"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
