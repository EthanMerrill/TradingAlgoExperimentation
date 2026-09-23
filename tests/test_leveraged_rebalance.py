#!/usr/bin/env python3
"""
Unit tests for the leveraged-ETF rebalance signal math and strategy.

Detection is exercised on synthetic intraday bars with a *planted* late-session
volume/price spike, so the expected event is known exactly.
"""
import os
import sys
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.leveraged_rebalance import LeveragedRebalanceStrategy  # noqa: E402
from strategies.leveraged_rebalance_signal import (  # noqa: E402
    detect_events,
    parse_hhmm,
    prepare_features,
    simulate_events,
)
from strategies.leveraged_single_stock_etfs import underlyings  # noqa: E402

_BARS_PER_DAY = 78          # 09:30–15:55 at 5m
_LATE_WINDOW_BARS = 12      # last hour


def _make_intraday_bars(
    days: int = 6,
    spike_day: int = None,
    spike_mult: float = 10.0,
    rise: float = 0.03,
    base_vol: float = 1000.0,
) -> pd.DataFrame:
    """Build 5m bars over ``days`` sessions, optionally planting a late spike."""
    sessions = pd.bdate_range("2026-06-01", periods=days)
    frames = []
    for i, day in enumerate(sessions):
        start = (pd.Timestamp(day).tz_localize("US/Eastern")
                 + pd.Timedelta(hours=9, minutes=30))
        idx = pd.date_range(start, periods=_BARS_PER_DAY, freq="5min")
        n = len(idx)
        close = (100.0 + np.arange(n) * 0.01).astype(float)
        volume = np.full(n, base_vol)

        if spike_day is not None and i == spike_day:
            late = np.arange(n) >= (n - _LATE_WINDOW_BARS)
            k = np.arange(1, late.sum() + 1)
            close[late] = close[late] * (1.0 + rise * k / late.sum())
            volume[late] = base_vol * spike_mult

        frames.append(pd.DataFrame({
            "open": close,
            "high": close * 1.0005,
            "low": close * 0.9995,
            "close": close,
            "volume": volume,
        }, index=idx))
    return pd.concat(frames)


class TestParseHhmm(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_hhmm("15:20"), 15 * 60 + 20)
        self.assertEqual(parse_hhmm("09:30"), 9 * 60 + 30)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_hhmm("1520")
        with self.assertRaises(ValueError):
            parse_hhmm("25:00")


class TestPrepareFeatures(unittest.TestCase):
    def test_empty_input(self):
        self.assertTrue(prepare_features(pd.DataFrame()).empty)

    def test_features_added(self):
        features = prepare_features(
            _make_intraday_bars(days=4), rvol_lookback_days=2)
        for column in ("date", "tod", "cum_ret", "rvol_base", "rvol"):
            self.assertIn(column, features.columns)
        # First session has no prior close / baseline.
        self.assertTrue(features["rvol"].isna().sum() > 0)

    def test_rvol_spike_visible(self):
        features = prepare_features(
            _make_intraday_bars(days=5, spike_day=3), rvol_lookback_days=2)
        spike_day_rows = features[features["date"]
                                  == sorted(features["date"].unique())[3]]
        self.assertGreater(spike_day_rows["rvol"].max(), 5.0)


class TestDetectEvents(unittest.TestCase):
    def test_detects_planted_spike(self):
        features = prepare_features(
            _make_intraday_bars(days=6, spike_day=3), rvol_lookback_days=2)
        events = detect_events(
            features, entry_start="15:20", entry_end="15:50",
            rvol_threshold=2.0, momentum_threshold=0.005)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["direction"], "long")
        self.assertGreater(events.iloc[0]["rvol"], 2.0)

    def test_no_spike_no_events(self):
        features = prepare_features(
            _make_intraday_bars(days=6), rvol_lookback_days=2)
        events = detect_events(
            features, entry_start="15:20", entry_end="15:50",
            rvol_threshold=2.0, momentum_threshold=0.005)
        self.assertTrue(events.empty)

    def test_direction_filter_excludes_event(self):
        features = prepare_features(
            _make_intraday_bars(days=6, spike_day=3), rvol_lookback_days=2)
        events = detect_events(
            features, entry_start="15:20", entry_end="15:50",
            rvol_threshold=2.0, momentum_threshold=0.005,
            direction_mode="short")
        self.assertTrue(events.empty)

    def test_high_threshold_suppresses_event(self):
        features = prepare_features(
            _make_intraday_bars(days=6, spike_day=3), rvol_lookback_days=2)
        events = detect_events(
            features, entry_start="15:20", entry_end="15:50",
            rvol_threshold=50.0, momentum_threshold=0.005)
        self.assertTrue(events.empty)

    def test_bad_window_raises(self):
        features = prepare_features(
            _make_intraday_bars(days=4), rvol_lookback_days=2)
        with self.assertRaises(ValueError):
            detect_events(features, entry_start="15:50", entry_end="15:20")


class TestSimulateEvents(unittest.TestCase):
    def _events(self, **kwargs):
        features = prepare_features(
            _make_intraday_bars(days=6, spike_day=3), rvol_lookback_days=2)
        return features, detect_events(
            features, entry_start="15:20", entry_end="15:50",
            rvol_threshold=2.0, momentum_threshold=0.005, **kwargs)

    def test_close_exit_profitable(self):
        features, events = self._events()
        trades = simulate_events(
            features, events, exit_mode="close", cost_bps=0.0)
        self.assertEqual(len(trades), 1)
        self.assertGreater(trades[0]["return"], 0)
        self.assertEqual(trades[0]["exit_reason"], "close")

    def test_cost_reduces_return(self):
        features, events = self._events()
        free = simulate_events(
            features, events, exit_mode="close", cost_bps=0.0)
        costly = simulate_events(
            features, events, exit_mode="close", cost_bps=50.0)
        self.assertLess(costly[0]["return"], free[0]["return"])

    def test_next_open_exit(self):
        features, events = self._events()
        trades = simulate_events(
            features, events, exit_mode="next_open", cost_bps=0.0)
        self.assertEqual(len(trades), 1)
        self.assertNotEqual(trades[0]["exit_date"], trades[0]["entry_date"])

    def test_entry_ts_is_eastern_and_keeps_detection_detail(self):
        """The trigger timestamp is ET (bars arrive UTC) and carries the signal."""
        features, events = self._events()
        trades = simulate_events(
            features, events, exit_mode="close", cost_bps=0.0)
        ts = trades[0]["entry_ts"]
        self.assertIsNotNone(ts)
        self.assertIsNotNone(ts.tzinfo)
        # ET is UTC-4 (EDT) or UTC-5 (EST).
        self.assertIn(ts.utcoffset().total_seconds(), (-4 * 3600, -5 * 3600))
        self.assertIn("rvol", trades[0])
        self.assertIn("momentum", trades[0])


class TestLeveragedRebalanceStrategy(unittest.TestCase):
    def _strategy(self):
        return LeveragedRebalanceStrategy(rvol_lookback_days=2)

    def test_universe_is_underlyings(self):
        self.assertEqual(self._strategy().symbol_universe(), underlyings())

    def test_data_timeframe_and_adjustment(self):
        from alpaca.data.enums import Adjustment
        strategy = self._strategy()
        self.assertEqual(strategy.data_timeframe, "5m")
        self.assertEqual(strategy.data_adjustment, Adjustment.SPLIT)

    def test_param_grid_direction_mapping(self):
        strategy = self._strategy()
        long_grid = strategy.get_param_grid("long")
        short_grid = strategy.get_param_grid("short")
        self.assertTrue(long_grid)
        self.assertTrue(all(p["direction_mode"] == "long" for p in long_grid))
        self.assertTrue(all(p["direction_mode"] ==
                        "short" for p in short_grid))

    def test_backtest_detects_and_trades(self):
        strategy = self._strategy()
        bars = _make_intraday_bars(days=6, spike_day=3)
        prepared = strategy.prepare(bars)
        result = strategy.backtest(
            bars, "NVDA", prepared=prepared,
            entry_start="15:20", entry_end="15:50",
            rvol_threshold=2.0, momentum_threshold=0.005,
            direction_mode="long", exit_mode="close", hold_days=0,
            cost_bps=0.0, rvol_lookback_days=2)
        self.assertEqual(result.strategy_name, "leveraged_etf_rebalance")
        self.assertEqual(result.num_trades, 1)
        self.assertGreater(result.total_return, 0)
        self.assertTrue(result.profitable)
        self.assertEqual(result.params["rvol_threshold"], 2.0)

    def test_backtest_no_spike_is_flat(self):
        strategy = self._strategy()
        bars = _make_intraday_bars(days=6)
        result = strategy.backtest(
            bars, "NVDA", prepared=strategy.prepare(bars),
            rvol_threshold=2.0, momentum_threshold=0.005,
            direction_mode="long", exit_mode="close",
            cost_bps=0.0, rvol_lookback_days=2)
        self.assertEqual(result.num_trades, 0)
        self.assertFalse(result.profitable)

    def test_single_trade_sharpe_is_finite(self):
        """A single trade must not yield a NaN Sharpe (n=1 sample std is NaN)."""
        strategy = self._strategy()
        bars = _make_intraday_bars(days=6, spike_day=3)
        result = strategy.backtest(
            bars, "NVDA", prepared=strategy.prepare(bars),
            rvol_threshold=2.0, momentum_threshold=0.005,
            direction_mode="long", exit_mode="close",
            cost_bps=0.0, rvol_lookback_days=2)
        self.assertEqual(result.num_trades, 1)
        self.assertTrue(np.isfinite(result.sharpe_ratio))
        self.assertTrue(np.isfinite(result.calmar_ratio))

    def test_prepare_reused_when_lookback_matches(self):
        strategy = self._strategy()
        bars = _make_intraday_bars(days=6, spike_day=3)
        prepared = strategy.prepare(bars)
        # Same lookback → prepared frame reused (no recompute error).
        result = strategy.backtest(
            bars, "NVDA", prepared=prepared, rvol_lookback_days=2,
            rvol_threshold=2.0, momentum_threshold=0.005,
            direction_mode="long", exit_mode="close")
        self.assertEqual(result.num_trades, 1)

    def test_registered(self):
        from strategies.registry import get_strategy, list_strategies
        self.assertIn("leveraged_etf_rebalance", list_strategies())
        self.assertIs(get_strategy("leveraged_etf_rebalance"),
                      LeveragedRebalanceStrategy)


class TestRealAlpha(unittest.TestCase):
    """Alpha must be EXCESS over an exposure-matched baseline, not raw return.

    Otherwise a long-only strategy in a drifting market reports market beta as
    alpha (the bug the walk-forward path had: ``alpha = oos_total_return``).
    """

    def _run(self, **overrides):
        strategy = LeveragedRebalanceStrategy(rvol_lookback_days=3)
        params = dict(
            entry_start="15:20", entry_end="15:50",
            rvol_threshold=2.0, momentum_threshold=0.005,
            direction_mode="long", exit_mode="close", hold_days=0,
            cost_bps=2.0, rvol_lookback_days=3)
        params.update(overrides)
        bars = _make_intraday_bars(days=8, spike_day=5)
        return strategy.backtest(
            bars, "NVDA", prepared=strategy.prepare(bars), **params)

    def test_alpha_is_per_session_excess(self):
        """alpha = strategy mean per-trade − baseline mean per-session."""
        result = self._run()
        p = result.params
        self.assertAlmostEqual(
            result.alpha,
            p["strategy_mean_return"] - p["benchmark_mean_return"],
            places=10)

    def test_benchmark_spans_all_sessions(self):
        """The baseline is 'hold the window every day', so it covers >= trades.

        Restricting it to the strategy's own sessions would be degenerate.
        """
        result = self._run()
        self.assertGreater(result.params["benchmark_trades"], 0)
        self.assertGreaterEqual(
            result.params["benchmark_trades"], result.num_trades)

    def test_alpha_is_not_raw_return(self):
        """The benchmark must actually be netted off (it is usually non-zero)."""
        result = self._run()
        if abs(result.params["benchmark_mean_return"]) > 1e-9:
            self.assertNotAlmostEqual(
                result.alpha, result.total_return, places=6)

    def test_no_selectivity_gives_no_alpha_when_timings_match(self):
        """With thresholds at zero the filter selects nothing, so alpha ≈ 0.

        The strategy then enters at the window open on every session — exactly
        the baseline — leaving no edge to attribute.
        """
        result = self._run(rvol_threshold=0.0, momentum_threshold=0.0)
        self.assertEqual(
            result.params["benchmark_trades"], result.num_trades)
        self.assertAlmostEqual(result.alpha, 0.0, places=9)


class TestPlaceboTest(unittest.TestCase):
    """The permutation test must reject skill on noise and detect a real edge."""

    def _features(self, days=40, spike_day=None, drift=0.0):
        bars = _make_intraday_bars(
            days=days, spike_day=spike_day, spike_mult=10.0, rise=0.05)
        if drift:
            # Add a constant per-session drift so one direction is systematically
            # better, giving the placebo something real to detect.
            bars = bars.copy()
            bars["close"] = bars["close"] + drift
        return prepare_features(bars, rvol_lookback_days=3)

    def test_null_distribution_is_centered_on_zero(self):
        from strategies.leveraged_rebalance_signal import placebo_alpha_test

        features = self._features()
        test = placebo_alpha_test(
            features, "15:20", "15:50", "close", 0, 0.0, "both",
            n_selected=10, observed_alpha=0.0, n_draws=300)
        self.assertIsNotNone(test["p_value"])
        self.assertAlmostEqual(test["null_mean"], 0.0, places=4)

    def test_p_value_is_bounded_and_defined(self):
        from strategies.leveraged_rebalance_signal import placebo_alpha_test

        features = self._features()
        test = placebo_alpha_test(
            features, "15:20", "15:50", "close", 0, 0.0, "both",
            n_selected=12, observed_alpha=0.01, n_draws=200)
        self.assertGreater(test["p_value"], 0.0)
        self.assertLessEqual(test["p_value"], 1.0)
        self.assertLessEqual(test["percentile"], 100.0)

    def test_extreme_alpha_is_significant(self):
        """An absurdly good alpha must sit above the whole null distribution."""
        from strategies.leveraged_rebalance_signal import placebo_alpha_test

        features = self._features()
        test = placebo_alpha_test(
            features, "15:20", "15:50", "close", 0, 0.0, "both",
            n_selected=10, observed_alpha=1.0, n_draws=200)
        self.assertLessEqual(test["p_value"], 0.05)

    def test_returns_undefined_without_enough_sessions(self):
        from strategies.leveraged_rebalance_signal import placebo_alpha_test

        features = self._features(days=2)
        test = placebo_alpha_test(
            features, "15:20", "15:50", "close", 0, 0.0, "both",
            n_selected=1, observed_alpha=0.5, n_draws=50)
        self.assertIsNone(test["p_value"])

    def test_strategy_exposes_placebo_hook(self):
        strategy = LeveragedRebalanceStrategy(rvol_lookback_days=3)
        bars = _make_intraday_bars(days=40, spike_day=30)
        test = strategy.placebo_p_value(
            bars,
            {"entry_start": "15:20", "entry_end": "15:50",
             "exit_mode": "close", "hold_days": 0, "cost_bps": 0.0,
             "direction_mode": "both", "rvol_lookback_days": 3},
            alpha=0.01, num_trades=12, n_draws=100)
        self.assertIsNotNone(test)
        self.assertIn("p_value", test)

    def test_base_strategy_placebo_defaults_to_none(self):
        from strategies.rsi import RSIStrategy
        self.assertIsNone(
            RSIStrategy.create().placebo_p_value(
                pd.DataFrame(), {}, alpha=1.0, num_trades=1))


class TestNoLookahead(unittest.TestCase):
    """Entry must never use information from a later bar than the entry bar.

    Regression test: an earlier version paired a window-open entry PRICE with a
    direction taken from a later qualifying bar. In a 30-minute window the leak
    was ≤30 min; with an afternoon-long window it could be hours.
    """

    @staticmethod
    def _features_with_late_trigger():
        """Spiking sessions stay flat until a late jump on the FINAL bar.

        Spikes land on alternating sessions so ``prior_close`` for a spiking
        session is a flat session's close — otherwise every session's
        ``prior_close`` would itself contain a spike and the day return would
        collapse to zero. Qualification therefore cannot occur before the last
        bar, which is what makes a window-open entry provably uninformed.
        """
        sessions = pd.bdate_range("2026-06-01", periods=8)
        frames = []
        for i, day in enumerate(sessions):
            start = (pd.Timestamp(day).tz_localize("US/Eastern")
                     + pd.Timedelta(hours=9, minutes=30))
            idx = pd.date_range(start, periods=_BARS_PER_DAY, freq="5min")
            n = len(idx)
            close = np.full(n, 100.0)
            volume = np.full(n, 1000.0)
            if i > 0 and i % 2 == 0:
                close[-1] = 110.0      # +10% jump only on the very last bar
                volume[-1] = 50_000.0
            frames.append(pd.DataFrame({
                "open": close, "high": close, "low": close,
                "close": close, "volume": volume,
            }, index=idx))
        return prepare_features(pd.concat(frames), rvol_lookback_days=2)

    def test_window_start_does_not_borrow_a_later_trigger(self):
        """window_start entry must not fire when qualification happens later."""
        features = self._features_with_late_trigger()
        # Wide window (12:30 → close) so any lookahead would be hours long.
        events = detect_events(
            features, entry_start="12:30", entry_end="15:55",
            rvol_threshold=2.0, momentum_threshold=0.015,
            direction_mode="both", enter_at="window_start")
        # The move does not exist at 12:30, so no session may qualify at the open.
        self.assertTrue(events.empty)

    def test_trigger_mode_entry_bar_matches_signal_bar(self):
        """In trigger mode the entry bar IS the qualifying bar."""
        features = self._features_with_late_trigger()
        events = detect_events(
            features, entry_start="12:30", entry_end="15:55",
            rvol_threshold=2.0, momentum_threshold=0.015,
            direction_mode="both", enter_at="trigger")
        self.assertFalse(events.empty)
        # Every entry must sit on the final bar of its session (15:55 ET).
        for ts in events["entry_ts"]:
            self.assertEqual(ts.hour, 15)
            self.assertEqual(ts.minute, 55)

    def test_entry_bar_price_matches_direction_information(self):
        """Reported momentum must be consistent with the entry price's day move."""
        features = self._features_with_late_trigger()
        events = detect_events(
            features, entry_start="12:30", entry_end="15:55",
            rvol_threshold=2.0, momentum_threshold=0.015,
            direction_mode="both", enter_at="trigger")
        for row in events.itertuples(index=False):
            day_ret = row.entry_price / row.prior_close - 1.0
            self.assertAlmostEqual(row.momentum, day_ret, places=9)
            self.assertEqual(
                row.direction, "long" if day_ret >= 0 else "short")


class TestWalkForwardIntegration(unittest.TestCase):
    """Walk-forward must work for this strategy, not only for RSI.

    Exercises the real ``WalkForwardValidator`` end-to-end on synthetic bars:
    optimize on each in-sample slice, then evaluate the chosen parameters on the
    following out-of-sample slice.
    """

    @staticmethod
    def _spiking_bars(sessions: int = 120, spike_every: int = 3) -> pd.DataFrame:
        """Multi-month 5m bars with a strong late-day spike every N sessions."""
        days = pd.bdate_range("2026-01-05", periods=sessions)
        frames = []
        for i, day in enumerate(days):
            start = (pd.Timestamp(day).tz_localize("US/Eastern")
                     + pd.Timedelta(hours=9, minutes=30))
            idx = pd.date_range(start, periods=_BARS_PER_DAY, freq="5min")
            n = len(idx)
            close = (100.0 + np.arange(n) * 0.01).astype(float)
            volume = np.full(n, 1000.0)
            if i > 0 and i % spike_every == 0:
                late = np.arange(n) >= (n - _LATE_WINDOW_BARS)
                k = np.arange(1, late.sum() + 1)
                close[late] = close[late] * (1.0 + 0.05 * k / late.sum())
                volume[late] = 1000.0 * 15.0
            frames.append(pd.DataFrame({
                "open": close,
                "high": close * 1.0005,
                "low": close * 0.9995,
                "close": close,
                "volume": volume,
            }, index=idx))
        return pd.concat(frames)

    def test_walk_forward_uses_strategy_params_not_rsi(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from optimizer import StrategyOptimizer
        from walk_forward import WalkForwardValidator

        cfg = SimpleNamespace(
            WF_IS_MONTHS=1, WF_OOS_MONTHS=1, WF_STEP_MONTHS=1,
            WF_MIN_WINDOWS=1, BACKTEST_INIT_CASH=10000)

        strategy = LeveragedRebalanceStrategy(rvol_lookback_days=3)
        validator = WalkForwardValidator(StrategyOptimizer(strategy=strategy))

        bars = self._spiking_bars()
        start = bars.index.min().tz_localize(None).to_pydatetime()
        end = bars.index.max().tz_localize(None).to_pydatetime()

        with patch("walk_forward.globalConfig", cfg):
            wf = validator.validate_symbol(
                "NVDA", start, end, "long", prefetched_full_data=bars)

        self.assertIsNotNone(
            wf, "walk-forward returned None for this strategy")
        self.assertEqual(wf.strategy_name, "leveraged_etf_rebalance")
        self.assertGreater(len(wf.windows), 0,
                           "no walk-forward windows formed")

        # The chosen parameters must be this strategy's, never RSI's.
        self.assertTrue(wf.best_params, "best_params was not carried through")
        self.assertIn("rvol_threshold", wf.best_params)
        self.assertIn("entry_start", wf.best_params)
        self.assertNotIn("rsi_period", wf.best_params)

        # Converted result stays strategy-tagged with strategy params.
        converted = wf.to_backtest_result()
        self.assertEqual(converted.strategy_name, "leveraged_etf_rebalance")
        self.assertIn("rvol_threshold", converted.params)

        # OOS alpha must be the mean per-session excess over the baseline, not
        # the raw OOS return (the pre-fix behavior credited drift as alpha).
        window_alphas = [w.oos_alpha for w in wf.windows if w.oos_validated]
        if window_alphas:
            self.assertAlmostEqual(
                wf.alpha, sum(window_alphas) / len(window_alphas), places=10)
        self.assertAlmostEqual(
            converted.buy_and_hold_return, wf.oos_benchmark_return, places=10)

        self.assertGreaterEqual(wf.param_stability, 0.0)
        self.assertLessEqual(wf.param_stability, 1.0)

    def test_warmup_falls_back_to_strategy_not_rsi_heuristic(self):
        """OOS warmup comes from the strategy (RSI heuristic is legacy-only)."""
        from optimizer import StrategyOptimizer
        from walk_forward import WalkForwardValidator

        strategy = LeveragedRebalanceStrategy(rvol_lookback_days=10)
        validator = WalkForwardValidator(StrategyOptimizer(strategy=strategy))
        self.assertEqual(validator._warmup_days(), strategy.warmup_days())
        self.assertNotEqual(validator._warmup_days(), 14 * 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
