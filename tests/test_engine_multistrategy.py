#!/usr/bin/env python3
"""
Phase C tests: strategy-aware engine dispatch, cross-strategy symbol dedup,
per-strategy capital allocation, and position strategy tagging.
"""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.base import BacktestResult, LiveSignal, Strategy  # noqa: E402
from strategies.registry import register  # noqa: E402
from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402


class _SignalStrategyA(Strategy):
    """Test strategy that emits a fixed LiveSignal for AAPL."""
    name = "test_sig_a"

    def __init__(self):
        self.composite = 5.0

    def backtest(self, data, symbol, initial_cash=10000, prepared=None, **params):
        raise NotImplementedError

    def evaluate_live_signals(self, ctx):
        return [LiveSignal(
            symbol="AAPL", direction="long", entry_price=100.0,
            stop_loss=95.0, take_profit=110.0,
            backtest_return=0.2, alpha=0.05, win_rate=0.9,
            composite_score=self.composite, num_trades=10,
            strategy_name=self.name,
        )]


class _SignalStrategyB(Strategy):
    """Same symbol as A but with a different composite score."""
    name = "test_sig_b"

    def __init__(self):
        self.composite = 8.0

    def backtest(self, data, symbol, initial_cash=10000, prepared=None, **params):
        raise NotImplementedError

    def evaluate_live_signals(self, ctx):
        return [LiveSignal(
            symbol="AAPL", direction="long", entry_price=100.0,
            stop_loss=95.0, take_profit=110.0,
            backtest_return=0.3, alpha=0.1, win_rate=0.95,
            composite_score=self.composite, num_trades=12,
            strategy_name=self.name,
        )]


class TestStrategyAwareDispatch(unittest.TestCase):
    """_identify_opportunities dispatches to strategy evaluate_live_signals."""

    def setUp(self):
        register(_SignalStrategyA)
        register(_SignalStrategyB)
        with patch('trading_engine.data_provider'):
            self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine._positions_manager.positions = []

    def _result(self, symbol, strategy_name):
        return BacktestResult(
            symbol=symbol, rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.2, buy_and_hold_return=0.1, alpha=0.05,
            num_trades=10, win_rate=0.9, avg_trade_duration=5.0,
            max_drawdown=0.05, sharpe_ratio=1.5, profitable=True,
            strategy_name=strategy_name,
        )

    def test_signal_strategy_path_used(self):
        results = [self._result("AAPL", "test_sig_a")]
        opps = self.engine.identify_buying_opportunities(results)
        self.assertEqual(len(opps), 1)
        self.assertEqual(opps[0].strategy_name, "test_sig_a")
        self.assertEqual(opps[0].entry_price, 100.0)
        # RSI-specific fields are zeroed for non-RSI strategies
        self.assertEqual(opps[0].current_rsi, 0.0)

    def test_legacy_rsi_path_still_works(self):
        results = [self._result("AAPL", "rsi_mean_reversion")]
        with patch.object(self.engine, '_get_rsi_with_previous',
                          return_value=(25.0, 35.0)), \
                patch.object(self.engine, '_get_current_price',
                             return_value=150.0), \
                patch.object(self.engine, '_compute_rsi_take_profit',
                             return_value=170.0):
            opps = self.engine.identify_buying_opportunities(results)
        self.assertEqual(len(opps), 1)
        self.assertEqual(opps[0].strategy_name, "rsi_mean_reversion")
        self.assertEqual(opps[0].current_rsi, 25.0)

    def test_cross_strategy_symbol_dedup(self):
        """Same symbol from two strategies → highest composite wins."""
        results = [
            self._result("AAPL", "test_sig_a"),
            self._result("AAPL", "test_sig_b"),
        ]
        opps = self.engine.identify_buying_opportunities(results)
        self.assertEqual(len(opps), 1)
        self.assertEqual(opps[0].strategy_name,
                         "test_sig_b")  # composite 8 > 5


class TestStrategyAllocation(unittest.TestCase):
    """Per-strategy capital budgets in position sizing."""

    def setUp(self):
        dp_patcher = patch('trading_engine.data_provider')
        mock_dp = dp_patcher.start()
        self.addCleanup(dp_patcher.stop)
        mock_dp.get_account_info.return_value = {
            'cash': 100000.0, 'equity': 100000.0, 'buying_power': 100000.0}
        self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine._positions_manager.positions = []

        self.cfg = SimpleNamespace(
            MAX_NEW_POSITIONS_PER_DAY=10,
            MAX_POSITIONS=20,
            POSITION_SIZE_PCT=0.1,
            MIN_NUM_TRADES=0,
            STRATEGIES_ENABLED=['test_sig_a', 'test_sig_b'],
            STRATEGY_ALLOCATION={'test_sig_a': 0.8, 'test_sig_b': 0.2},
        )
        patcher = patch('trading_engine.globalConfig', self.cfg)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _opp(self, symbol, strategy_name):
        return TradingOpportunity(
            symbol=symbol, current_rsi=0.0, target_rsi_lower=0,
            target_rsi_upper=0, rsi_period=14, backtest_return=0.2,
            alpha=0.05, win_rate=0.9, entry_price=100.0,
            stop_loss_price=95.0, take_profit_price=110.0,
            num_trades=10, composite_score=5.0, strategy_name=strategy_name,
        )

    def test_budgets_are_respected(self):
        """Each strategy's notional must not exceed its budget."""
        opps = [self._opp("AAA", "test_sig_a"), self._opp("BBB", "test_sig_b")]
        allocs = self.engine.calculate_position_sizes(opps)
        by_strategy = {}
        for op, shares in allocs:
            by_strategy[op.strategy_name] = by_strategy.get(
                op.strategy_name, 0) + shares * op.entry_price
        # Budgets: a → $80k, b → $20k; position size pct → $10k each
        self.assertLessEqual(by_strategy['test_sig_a'], 80000)
        self.assertLessEqual(by_strategy['test_sig_b'], 20000)
        # Both should be sized at the $10k equal-weight cap
        self.assertEqual(by_strategy['test_sig_a'], 10000)
        self.assertEqual(by_strategy['test_sig_b'], 10000)

    def test_exhausted_budget_skipped(self):
        """A strategy already at its budget gets no new allocation."""
        existing = Mock()
        existing.closed = False
        existing.side = 'long'
        existing.strategy_name = 'test_sig_b'
        existing.entry_price = 100.0
        existing.quantity = 200.0  # notional = $20k = full budget for B
        self.engine._positions_manager.positions = [existing]

        opps = [self._opp("AAA", "test_sig_a"), self._opp("BBB", "test_sig_b")]
        allocs = self.engine.calculate_position_sizes(opps)
        symbols = {op.symbol for op, _ in allocs}
        # B's budget is exhausted by the existing position → B skipped
        self.assertIn("AAA", symbols)
        self.assertNotIn("BBB", symbols)


class TestPositionTagging(unittest.TestCase):
    """New positions carry the owning strategy_name."""

    def setUp(self):
        with patch('trading_engine.data_provider'):
            self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine._positions_manager.positions = []
        self.engine._positions_manager.open_position = Mock()

    def test_place_buy_order_tags_strategy(self):
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = Mock()
        self.engine.trading_client.submit_order.return_value = Mock(id="o1")
        opp = TradingOpportunity(
            symbol="AAPL", current_rsi=0.0, target_rsi_lower=0,
            target_rsi_upper=0, rsi_period=14, backtest_return=0.2,
            alpha=0.05, win_rate=0.9, entry_price=100.0,
            stop_loss_price=95.0, take_profit_price=110.0,
            num_trades=10, strategy_name="test_sig_a",
        )
        with patch.object(self.engine, '_make_unique_client_order_id',
                          return_value="AAPL-BUY-TEST"), \
                patch('trading_engine.storage.save_orders'):
            result = self.engine.place_buy_order(opp, 5)
        self.assertTrue(result)
        created = self.engine._positions_manager.open_position.call_args[0][0]
        self.assertEqual(created.strategy_name, "test_sig_a")


if __name__ == "__main__":
    unittest.main()
