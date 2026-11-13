#!/usr/bin/env python3
"""Tests for the live momentum_rank portfolio strategy and its engine/worker wiring."""
import os
import sys
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.base import LiveSignal, RebalancePlan, StrategyContext  # noqa: E402
from strategies.momentum_rank import MomentumRankStrategy  # noqa: E402
from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402

HOLIDAYS = {date(2026, 11, 26)}  # Thanksgiving (Thu)


def _fake_session(d):
    if isinstance(d, datetime):
        d = d.date()
    if d.weekday() >= 5 or d in HOLIDAYS:
        return None
    return ("09:30", "16:00")


def _pos(symbol, strategy, closed=False, entry_price=100.0, qty=10.0, entry_date=None):
    return SimpleNamespace(symbol=symbol, strategy_name=strategy, closed=closed,
                           entry_price=entry_price, quantity=qty, side="long",
                           intraday=False,
                           entry_date=entry_date or datetime.now(timezone.utc))


class TestRebalanceSchedule(unittest.TestCase):
    def setUp(self):
        p = patch("strategies.momentum_rank.get_session", side_effect=_fake_session)
        p.start()
        self.addCleanup(p.stop)
        self.s = MomentumRankStrategy()

    def test_monday_is_rebalance_day(self):
        self.assertTrue(self.s.is_rebalance_due(datetime(2026, 11, 16, 10, 0)))

    def test_midweek_is_not(self):
        self.assertFalse(self.s.is_rebalance_due(datetime(2026, 11, 18, 10, 0)))

    def test_tuesday_after_monday_holiday(self):
        HOLIDAYS.add(date(2026, 9, 7))  # Labor Day
        self.assertTrue(self.s.is_rebalance_due(datetime(2026, 9, 8, 10, 0)))

    def test_weekend_never(self):
        self.assertFalse(self.s.is_rebalance_due(datetime(2026, 11, 15, 10, 0)))


class TestRebalancePlan(unittest.TestCase):
    """Synthetic universe: S00 steadiest uptrend ... S29 noisiest."""

    DECISION = date(2026, 11, 13)   # Friday close
    TODAY = datetime(2026, 11, 16, 9, 35)  # Monday open

    def setUp(self):
        p = patch("strategies.momentum_rank.get_session", side_effect=_fake_session)
        p.start()
        self.addCleanup(p.stop)
        idx = pd.bdate_range(end=pd.Timestamp(self.DECISION), periods=420)
        rng = np.random.default_rng(0)
        cols = {}
        for k in range(30):
            r = 0.0015 + rng.normal(0, 0.004 * (k + 1), len(idx))
            cols[f"S{k:02d}"] = 50 * np.cumprod(1 + r)
        cols["BOND"] = 100 * np.cumprod(1 + np.full(len(idx), 0.0002))
        self.close = pd.DataFrame(cols, index=idx)
        names = {s: f"{s} Inc. Common Stock" for s in cols}
        names["BOND"] = "iShares 0-3 Month Treasury Bond ETF"
        self.dp = Mock()
        self.dp.get_stock_universe.return_value = pd.DataFrame(
            {"symbol": list(names), "name": list(names.values())})
        self.positions = []
        self.strategy = MomentumRankStrategy(slots=5, buffer=20)

    def _plan(self, close=None):
        ctx = StrategyContext(
            data_provider=self.dp,
            positions_manager=SimpleNamespace(positions=self.positions),
            config=SimpleNamespace(MIN_PRICE=15.0, MAX_PRICE=1e9), as_of=self.TODAY)
        with patch.object(self.strategy, "_fetch_closes",
                          return_value=self.close if close is None else close):
            return self.strategy.rebalance_plan(ctx)

    def _ranks(self):
        return self.strategy.rank_table(self.close, [c for c in self.close if c != "BOND"])

    def test_fills_slots_with_best_ranked_trend_on_names(self):
        plan = self._plan()
        table = self._ranks()
        expected = table[table["trend"] == 1.0].index[:5].tolist()
        self.assertEqual([e.symbol for e in plan.entries][:5], expected)
        self.assertEqual(plan.notes["free_slots"], 5)
        self.assertEqual(plan.exits, [])
        self.assertNotIn("BOND", [e.symbol for e in plan.entries])  # funds excluded
        self.assertTrue(all(e.strategy_name == "momentum_rank" for e in plan.entries))

    def test_exits_beyond_buffer_keeps_within(self):
        table = self._ranks()
        good, bad = table.index[2], table.index[25]
        self.positions = [_pos(good, "momentum_rank"), _pos(bad, "momentum_rank")]
        plan = self._plan()
        self.assertEqual([s for s, _ in plan.exits], [bad])
        self.assertEqual(plan.notes["free_slots"], 4)
        self.assertNotIn(good, [e.symbol for e in plan.entries])

    def test_symbol_held_by_other_strategy_not_bought(self):
        table = self._ranks()
        top = table[table["trend"] == 1.0].index[0]
        self.positions = [_pos(top, "rsi_mean_reversion")]
        plan = self._plan()
        self.assertNotIn(top, [e.symbol for e in plan.entries])
        self.assertEqual(plan.exits, [])  # never sells another strategy's position

    def test_todays_partial_bar_ignored(self):
        extra = self.close.iloc[[-1]].copy() * 0.5
        extra.index = [pd.Timestamp(self.TODAY.date())]
        with_today = pd.concat([self.close, extra])
        self.assertEqual([e.symbol for e in self._plan(with_today).entries],
                         [e.symbol for e in self._plan().entries])

    def test_stale_data_returns_none(self):
        self.assertIsNone(self._plan(self.close.iloc[:-1]))

    def test_holding_dropped_from_daily_universe_is_not_sold(self):
        table = self._ranks()
        held = table.index[1]
        self.positions = [_pos(held, "momentum_rank")]
        uni = self.dp.get_stock_universe.return_value
        self.dp.get_stock_universe.return_value = uni[uni.symbol != held]
        self.assertEqual(self._plan().exits, [])


class _EngineBase(unittest.TestCase):
    def setUp(self):
        dp = patch("trading_engine.data_provider")
        self.dp = dp.start()
        self.addCleanup(dp.stop)
        self.dp.get_account_info.return_value = {
            "cash": 100_000.0, "equity": 100_000.0, "buying_power": 100_000.0}
        self.cfg = SimpleNamespace(
            MAX_NEW_POSITIONS_PER_DAY=2, MAX_POSITIONS=10, POSITION_SIZE_PCT=0.1,
            MIN_CASH_PCT=0.05, MIN_NUM_TRADES=0, MAX_HOLD_DAYS=15,
            STRATEGIES_ENABLED=["rsi_mean_reversion", "momentum_rank"],
            STRATEGY_ALLOCATION={"rsi_mean_reversion": 0.75, "momentum_rank": 0.25})
        cp = patch("trading_engine.globalConfig", self.cfg)
        cp.start()
        self.addCleanup(cp.stop)
        self.engine = TradingEngine()
        self.pm = MagicMock()
        self.pm.positions = []
        self.pm.get_and_reconcile_positions.side_effect = lambda: self.pm.positions
        self.engine._positions_manager = self.pm


class TestExecuteRebalancePlan(_EngineBase):
    def _plan(self, symbols, exits=()):
        return RebalancePlan(
            strategy_name="momentum_rank", exits=list(exits), slots=5, min_cash_pct=0.10,
            entries=[LiveSignal(symbol=s, entry_price=100.0, composite_score=3.0,
                                strategy_name="momentum_rank", extra={"rank": i + 1})
                     for i, s in enumerate(symbols)])

    def _run(self, plan):
        bought = []

        def buy(opp, shares):
            bought.append((opp, shares))
            self.pm.positions.append(_pos(opp.symbol, opp.strategy_name, qty=shares))
            return True
        with patch.object(self.engine, "place_buy_order", side_effect=buy), \
                patch.object(self.engine, "_get_current_price", return_value=100.0), \
                patch.object(self.engine, "_force_close_position", return_value=True) as fc:
            self.engine.execute_rebalance_plan(plan)
        return bought, fc

    def test_sizes_from_strategy_budget_and_fills_slots(self):
        bought, _ = self._run(self._plan(["A", "B", "C", "D", "E", "F", "G"]))
        self.assertEqual([o.symbol for o, _ in bought], ["A", "B", "C", "D", "E"])
        # 25% of $100k, 10% sleeve cash, 5 slots -> $4,500 -> 45 shares @ $100
        self.assertTrue(all(sh == 45 for _, sh in bought))
        for opp, _ in bought:
            self.assertIsNone(opp.stop_loss_price)
            self.assertIsNone(opp.take_profit_price)
            self.assertEqual(opp.strategy_name, "momentum_rank")

    def test_ignores_rsi_slot_limits(self):
        # RSI already uses all 10 MAX_POSITIONS and both daily new slots.
        self.pm.positions = [_pos(f"R{i}", "rsi_mean_reversion") for i in range(10)]
        bought, _ = self._run(self._plan(["A", "B", "C"]))
        self.assertEqual(len(bought), 3)

    def test_exits_only_own_positions_then_refills(self):
        self.pm.positions = [_pos("OLD", "momentum_rank", qty=45),
                             _pos("X", "rsi_mean_reversion")]

        def close(summary, pos, reason):
            pos.closed = True
            return True
        with patch.object(self.engine, "place_buy_order", return_value=True) as buy, \
                patch.object(self.engine, "_get_current_price", return_value=100.0), \
                patch.object(self.engine, "_force_close_position", side_effect=close) as fc:
            self.engine.execute_rebalance_plan(
                self._plan(["X", "A"], exits=[("OLD", "rank_exit_31"), ("X", "rank_exit")]))
        self.assertEqual([c.args[1].symbol for c in fc.call_args_list], ["OLD"])
        self.assertEqual([c.args[0].symbol for c in buy.call_args_list], ["A"])  # X held by RSI

    def test_budget_caps_entries(self):
        self.pm.positions = [_pos(f"M{i}", "momentum_rank", entry_price=100.0, qty=240)
                             for i in range(1)]  # $24k of $25k used
        bought, _ = self._run(self._plan(["A"]))
        self.assertEqual(bought[0][1], 10)  # only $1k left


class TestPortfolioExemptions(_EngineBase):
    def test_no_max_hold_or_oco_refresh_for_portfolio_positions(self):
        old = datetime.now(timezone.utc) - timedelta(days=100)
        mom = _pos("MOM", "momentum_rank", entry_date=old)
        rsi = _pos("RSI", "rsi_mean_reversion", entry_date=old)
        with patch.object(self.engine, "_force_close_position", return_value=True) as fc, \
                patch.object(self.engine, "place_oco_close_order") as oco:
            self.engine.update_portfolio_orders(
                {"orders_placed": 0, "orders": []}, [mom, rsi])
        self.assertEqual([c.args[1].symbol for c in fc.call_args_list], ["RSI"])
        oco.assert_not_called()

    def test_new_positions_today_excludes_portfolio_entries(self):
        self.pm.positions = [_pos("A", "momentum_rank"), _pos("B", "momentum_rank"),
                             _pos("C", "rsi_mean_reversion")]
        self.assertEqual(self.engine._new_positions_remaining_today(), 1)

    def test_no_protective_order_without_levels(self):
        opp = TradingOpportunity(symbol="A", backtest_return=0, alpha=0, win_rate=0,
                                 entry_price=100.0, stop_loss_price=None,
                                 take_profit_price=None, strategy_name="momentum_rank")
        with patch.object(self.engine, "place_oco_close_order") as oco:
            self.assertTrue(self.engine._arm_entry_protection(opp, 10, 1))
        oco.assert_not_called()


class TestWorkerRebalance(unittest.TestCase):
    def setUp(self):
        import main  # pylint: disable=import-outside-toplevel
        self.main = main
        self.strategy = Mock()
        self.cls = Mock(execution_style="portfolio")
        self.cls.create.return_value = self.strategy
        self.algo = Mock()
        self.algo.trading_engine.execute_rebalance_plan.return_value = {"errors": []}
        for target, val in (("main.get_strategy", Mock(return_value=self.cls)),
                            ("main.globalConfig", SimpleNamespace(
                                STRATEGIES_ENABLED=["momentum_rank"]))):
            p = patch(target, val)
            p.start()
            self.addCleanup(p.stop)
        self.now = datetime(2026, 11, 16, 9, 31)

    def test_retries_until_plan_then_runs_once(self):
        done = {}
        self.strategy.is_rebalance_due.return_value = True
        self.strategy.rebalance_plan.return_value = None
        self.main._run_portfolio_rebalances(self.algo, self.now, done)
        self.assertEqual(done, {})
        self.strategy.rebalance_plan.return_value = "plan"
        self.main._run_portfolio_rebalances(self.algo, self.now, done)
        self.main._run_portfolio_rebalances(self.algo, self.now, done)
        self.algo.trading_engine.execute_rebalance_plan.assert_called_once_with("plan")
        self.assertEqual(done, {"momentum_rank": "2026-11-16"})

    def test_not_due_marks_done_without_trading(self):
        done = {}
        self.strategy.is_rebalance_due.return_value = False
        self.main._run_portfolio_rebalances(self.algo, self.now, done)
        self.algo.trading_engine.execute_rebalance_plan.assert_not_called()
        self.assertIn("momentum_rank", done)

    def test_gives_up_after_deadline(self):
        done = {}
        self.strategy.is_rebalance_due.return_value = True
        self.main._run_portfolio_rebalances(self.algo, datetime(2026, 11, 16, 12, 5), done)
        self.strategy.rebalance_plan.assert_not_called()
        self.assertIn("momentum_rank", done)

    def test_exit_failure_retried(self):
        done = {}
        self.strategy.is_rebalance_due.return_value = True
        self.strategy.rebalance_plan.return_value = "plan"
        self.algo.trading_engine.execute_rebalance_plan.return_value = {
            "errors": ["OLD: exit failed (rank_exit_30)"]}
        self.main._run_portfolio_rebalances(self.algo, self.now, done)
        self.assertEqual(done, {})


if __name__ == "__main__":
    unittest.main()
