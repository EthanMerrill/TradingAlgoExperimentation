#!/usr/bin/env python3
"""Unit tests for the trading_engine module."""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
from alpaca.trading.enums import OrderSide

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from positions import Position  # noqa: E402
from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402


class TestTradingOpportunity(unittest.TestCase):
    """Test cases for the TradingOpportunity dataclass."""

    def test_trading_opportunity_creation(self):
        opportunity = TradingOpportunity(
            symbol="AAPL",
            current_rsi=25.0,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            backtest_return=0.15,
            alpha=0.05,
            win_rate=0.65,
            entry_price=150.00,
            stop_loss_price=140.00,
            take_profit_price=160.00,
            num_trades=10,
        )

        self.assertEqual(opportunity.symbol, "AAPL")
        self.assertEqual(opportunity.rsi_period, 14)
        self.assertEqual(opportunity.num_trades, 10)


class TestTradingEngine(unittest.TestCase):
    """Test cases for current TradingEngine behavior."""

    def setUp(self):
        with patch('trading_engine.data_provider'):
            self.engine = TradingEngine()
        self.engine._positions_manager = Mock()
        self.engine._positions_manager.positions = []
        self.engine._positions_manager.open_position = Mock()
        self.engine._positions_manager.close_position = Mock()

    def test_positions_manager_starts_none(self):
        """Before injection, _positions_manager is None."""
        with patch('trading_engine.data_provider'):
            engine = TradingEngine()
        self.assertIsNone(engine._positions_manager)

    def test_set_positions_manager_injects_shared_instance(self):
        """set_positions_manager injects a shared PositionsManager instance."""
        with patch('trading_engine.data_provider'):
            engine = TradingEngine()
        mock_manager = Mock()
        engine.set_positions_manager(mock_manager)
        self.assertIs(engine._positions_manager, mock_manager)

    def _result(self, symbol, alpha=0.1, win_rate=0.9, num_trades=10):
        r = Mock()
        r.symbol = symbol
        r.rsi_period = 14
        r.rsi_lower = 30
        r.rsi_upper = 70
        r.total_return = 0.2
        r.alpha = alpha
        r.win_rate = win_rate
        r.num_trades = num_trades
        r.composite_score = 5.0
        r.direction = "long"
        return r

    def _position(self, symbol="AAPL", days_ago=1):
        return Position(
            symbol=symbol,
            quantity=10.0,
            entry_price=100.0,
            current_price=101.0,
            current_rsi=40.0,
            entry_date=datetime.now() - timedelta(days=days_ago),
            alpha=0.1,
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            closed=False,
        )

    def test_identify_buying_opportunities_filters_and_cross(self):
        good = self._result("AAPL", alpha=0.2, win_rate=0.95, num_trades=12)
        bad = self._result("TSLA", alpha=-0.1, win_rate=0.95, num_trades=12)

        def rsi_side_effect(symbol, _period):
            if symbol == "AAPL":
                return (25.0, 35.0)  # Crossed below
            return (20.0, 25.0)  # No cross

        with patch.object(self.engine, '_get_rsi_with_previous', side_effect=rsi_side_effect), \
                patch.object(self.engine, '_get_current_price', return_value=150.0), \
                patch.object(self.engine, '_compute_rsi_take_profit', return_value=170.0):
            opportunities = self.engine.identify_buying_opportunities([
                                                                      good, bad])

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].symbol, "AAPL")
        self.assertEqual(opportunities[0].take_profit_price, 170.0)

    @patch('trading_engine.data_provider')
    def test_calculate_position_sizes(self, mock_data_provider):
        mock_data_provider.get_account_info.return_value = {
            'cash': 50000.0,
            'equity': 100000.0,
            'buying_power': 200000.0,
        }
        opp = TradingOpportunity(
            symbol="AAPL",
            current_rsi=25.0,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            backtest_return=0.15,
            alpha=0.05,
            win_rate=0.9,
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            num_trades=10,
        )

        allocations = self.engine.calculate_position_sizes([opp])

        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0][0].symbol, "AAPL")
        self.assertGreater(allocations[0][1], 0)

    def test_place_buy_order_dry_run_returns_true(self):
        self.engine.set_dry_run_mode(True)
        opp = TradingOpportunity(
            symbol="AAPL",
            current_rsi=25.0,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            backtest_return=0.15,
            alpha=0.05,
            win_rate=0.9,
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            num_trades=10,
        )

        result = self.engine.place_buy_order(opp, 5)

        # Dry-run simulates success (consistent with OCO/market-close).
        self.assertTrue(result)
        # No position is opened and no order is persisted in dry-run.
        self.engine._positions_manager.open_position.assert_not_called()

    def test_place_buy_order_live_success_adds_position(self):
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = Mock()
        self.engine.trading_client.submit_order.return_value = Mock(
            id="order_1")
        opp = TradingOpportunity(
            symbol="AAPL",
            current_rsi=25.0,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            backtest_return=0.15,
            alpha=0.05,
            win_rate=0.9,
            entry_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            num_trades=10,
        )

        with patch.object(self.engine, '_make_unique_client_order_id',
                          return_value="AAPL-BUY-TEST"), \
                patch('trading_engine.storage.save_orders') as mock_save_orders:
            result = self.engine.place_buy_order(opp, 5)

        self.assertTrue(result)
        self.engine._positions_manager.open_position.assert_called_once()
        mock_save_orders.assert_called_once()
        created = self.engine._positions_manager.open_position.call_args[0][0]
        self.assertEqual(created.client_order_id, "AAPL-BUY-TEST")
        self.assertEqual(created.order_id, "order_1")
        saved = mock_save_orders.call_args[0][0][0]
        self.assertEqual(saved.leg, "entry")

    def test_place_oco_close_order_dry_run(self):
        self.engine.set_dry_run_mode(True)

        result = self.engine.place_oco_close_order(
            "AAPL", 10, 95.0, 110.0, side="long")

        self.assertTrue(result)

    def test_place_market_sell_order_dry_run(self):
        self.engine.set_dry_run_mode(True)

        result = self.engine.place_market_sell_order(
            "AAPL", 10, "max_hold_days")

        self.assertTrue(result)

    def test_update_portfolio_orders_enforces_max_hold_days(self):
        expired = self._position("AAPL", days_ago=90)
        active = self._position("MSFT", days_ago=2)
        summary = {
            'orders_placed': 0,
            'positions_exited': 0,
            'opportunities_found': 0,
            'new_positions': 0,
            'errors': [],
            'dry_run': False,
        }

        with patch.object(self.engine, 'place_market_sell_order', return_value=True), \
                patch.object(self.engine, 'calculate_todays_stop_loss_and_take_profit', return_value=(90.0, 115.0)), \
                patch.object(self.engine, 'place_oco_close_order', return_value=True):
            out = self.engine.update_portfolio_orders(
                summary, [expired, active])

        self.assertEqual(out['positions_exited'], 1)
        self.assertEqual(out['orders_placed'], 1)
        self.engine._positions_manager.close_position.assert_called_once_with(
            "AAPL")

    def test_identify_shorting_opportunities_cross_above(self):
        """Short opportunity fires on RSI cross-above rsi_upper."""
        long_r = self._result("AAPL", alpha=0.1, win_rate=0.9, num_trades=10)
        long_r.direction = "long"
        short_r = self._result("AAPL", alpha=0.15, win_rate=0.9, num_trades=10)
        short_r.direction = "short"

        # RSI cross-above: current=75, previous=65 (crossed above 70)
        def rsi_side_effect(symbol, _period):
            return (75.0, 65.0)

        with patch.object(self.engine, '_get_rsi_with_previous', side_effect=rsi_side_effect), \
                patch.object(self.engine, '_get_current_price', return_value=200.0), \
                patch.object(self.engine, '_compute_rsi_cover_price', return_value=170.0):
            opportunities = self.engine.identify_shorting_opportunities([
                                                                        long_r, short_r])

        self.assertEqual(len(opportunities), 1)
        self.assertEqual(opportunities[0].symbol, "AAPL")
        self.assertEqual(opportunities[0].direction, "short")
        # Stop loss for short should be above entry: 200 * 1.05 = 210
        self.assertAlmostEqual(
            opportunities[0].stop_loss_price, 210.0, places=1)
        self.assertEqual(opportunities[0].take_profit_price, 170.0)

    def test_short_opportunity_excludes_if_no_cross(self):
        """Short opportunity should not fire if RSI hasn't crossed above rsi_upper yet."""
        results = [self._result("AAPL")]
        results[0].direction = "short"

        # RSI: current=65, previous=60 (no cross above 70)
        def rsi_side_effect(symbol, _period):
            return (65.0, 60.0)

        with patch.object(self.engine, '_get_rsi_with_previous', side_effect=rsi_side_effect), \
                patch.object(self.engine, '_get_current_price', return_value=200.0):
            opportunities = self.engine.identify_shorting_opportunities(
                results)

        self.assertEqual(len(opportunities), 0)

    def test_short_opportunity_excludes_existing_short(self):
        """Symbol with existing open short position should be excluded."""
        results = [self._result("AAPL")]
        results[0].direction = "short"
        results[0].rsi_lower = 30
        results[0].rsi_upper = 70

        # Add an existing open short position
        existing_short = self._position("AAPL")
        existing_short.side = "short"
        self.engine._positions_manager.positions = [existing_short]

        # RSI cross-above
        def rsi_side_effect(symbol, _period):
            return (75.0, 65.0)

        with patch.object(self.engine, '_get_rsi_with_previous', side_effect=rsi_side_effect), \
                patch.object(self.engine, '_get_current_price', return_value=200.0):
            opportunities = self.engine.identify_shorting_opportunities(
                results)

        self.assertEqual(len(opportunities), 0,
                         "Should exclude symbol with existing short position")

    @patch('trading_engine.data_provider')
    def test_calculate_short_position_sizes_respects_ratio(self, mock_data_provider):
        """Short position sizes respect max_short_long_ratio cap."""
        mock_data_provider.get_account_info.return_value = {
            'cash': 50000.0,
            'equity': 100000.0,
            'buying_power': 200000.0,
        }
        # Add existing short with $2000 notional (negative qty, as produced
        # by _place_order for shorts).
        existing_short = self._position("MSFT")
        existing_short.side = "short"
        existing_short.entry_price = 100.0
        existing_short.quantity = -20.0  # $2000 notional
        self.engine._positions_manager.positions = [existing_short]

        opp = TradingOpportunity(
            symbol="AAPL",
            current_rsi=75.0,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            backtest_return=0.15,
            alpha=0.05,
            win_rate=0.9,
            entry_price=200.0,
            stop_loss_price=210.0,
            take_profit_price=170.0,
            num_trades=10,
            direction="short",
        )

        with patch('trading_engine.globalConfig') as mock_cfg:
            mock_cfg.POSITION_SIZE_PCT = 0.1
            mock_cfg.MAX_SHORT_LONG_RATIO = 0.30
            mock_cfg.MAX_NEW_POSITIONS_PER_DAY = 2
            mock_cfg.MAX_POSITIONS = 10

            allocations = self.engine.calculate_short_position_sizes([opp])

        # Max short notional = 100000 * 0.30 = 30000
        # Existing = 2000, available = 28000
        # Per-position = 28000, capped at position_size_pct = 10000
        # Shares = 10000 / 200 = 50
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0][0].symbol, "AAPL")
        shares = allocations[0][1]
        notional = shares * 200.0
        self.assertLessEqual(
            notional, 10000.0, "Should respect position size cap")
        self.assertLessEqual(notional + 2000, 30000.0,
                             "Total short notional should respect cap")
        self.assertGreater(shares, 0)

    def test_calculate_todays_stop_loss_and_take_profit_long_underwater(self):
        """Fix #0: underwater long must never produce take-profit below stop-loss."""
        pos = self._position("AAPL")
        pos.entry_price = 100.0
        pos.current_price = 90.0  # underwater

        with patch.object(self.engine, '_fetch_ohlcv_once',
                          return_value=pd.DataFrame({'close': [100.0, 99.0, 98.0]})), \
                patch.object(self.engine, '_get_current_price', return_value=90.0), \
                patch('trading_engine.RSIStrategy.calculate_price_for_target_rsi',
                      return_value=97.0), \
                patch('trading_engine.globalConfig') as mock_cfg:
            mock_cfg.STOP_LOSS_PCT = 0.05
            mock_cfg.TAKE_PROFIT_PCT = 0.10
            stop, take = self.engine.calculate_todays_stop_loss_and_take_profit(
                pos)

        # stop anchored to entry*0.95 = 95.0; take must be above it (and above entry).
        self.assertGreater(take, stop, "take-profit must stay above stop-loss")
        self.assertGreater(take, pos.entry_price,
                           "take-profit must stay above entry for a long")

    def test_place_market_sell_order_short_covers_with_buy(self):
        """Fix #1: closing a short must BUY to cover, not SELL again."""
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = Mock()
        self.engine.trading_client.submit_order.return_value = Mock(
            id="order_cover_1")

        with patch.object(self.engine, '_make_unique_client_order_id',
                          return_value="AAPL-BUY-TEST"), \
                patch('trading_engine.storage.save_orders') as mock_save_orders:
            result = self.engine.place_market_sell_order(
                "AAPL", 10, "max_hold_days", side="short")

        self.assertTrue(result)
        submitted = self.engine.trading_client.submit_order.call_args[0][0]
        self.assertEqual(submitted.side, OrderSide.BUY)
        # Persisted order records a buy (cover), not a sell.
        saved = mock_save_orders.call_args[0][0][0]
        self.assertEqual(saved.side, "buy")

    def test_place_oco_close_order_refuses_inverted_long(self):
        """Fix #3: inverted long SL/TP must be refused before cancelling orders."""
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = Mock()

        # stop (110) above take-profit (90) is inverted for a long.
        result = self.engine.place_oco_close_order(
            "AAPL", 10, stop_loss_price=110.0, take_profit_price=90.0, side="long")

        self.assertFalse(result)
        self.engine.trading_client.get_orders.assert_not_called()
        self.engine.trading_client.submit_order.assert_not_called()

    def test_place_oco_close_order_refuses_inverted_short(self):
        """Fix #3: inverted short cover (cover above stop) must be refused."""
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = Mock()

        # cover (110) above stop (90) is inverted for a short.
        result = self.engine.place_oco_close_order(
            "AAPL", 10, stop_loss_price=90.0, take_profit_price=110.0, side="short")

        self.assertFalse(result)
        self.engine.trading_client.get_orders.assert_not_called()
        self.engine.trading_client.submit_order.assert_not_called()

    def test_place_short_order_dry_run_returns_true(self):
        """Dry run short order simulates success and does not open a position."""
        self.engine.set_dry_run_mode(True)
        opp = TradingOpportunity(
            symbol="AAPL",
            current_rsi=75.0,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            backtest_return=0.15,
            alpha=0.05,
            win_rate=0.9,
            entry_price=200.0,
            stop_loss_price=210.0,
            take_profit_price=170.0,
            num_trades=10,
            direction="short",
        )

        result = self.engine.place_short_order(opp, 5)
        self.assertTrue(result)
        self.engine._positions_manager.open_position.assert_not_called()

    def test_place_short_order_live_success_adds_position(self):
        """Live short order adds position with side='short'."""
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = Mock()
        self.engine.trading_client.submit_order.return_value = Mock(
            id="order_short_1")
        opp = TradingOpportunity(
            symbol="AAPL",
            current_rsi=75.0,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            backtest_return=0.15,
            alpha=0.05,
            win_rate=0.9,
            entry_price=200.0,
            stop_loss_price=210.0,
            take_profit_price=170.0,
            num_trades=10,
            direction="short",
        )

        with patch.object(self.engine, '_make_unique_client_order_id',
                          return_value="AAPL-SELL-TEST"), \
                patch('trading_engine.storage.save_orders') as mock_save_orders:
            result = self.engine.place_short_order(opp, 5)

        self.assertTrue(result)
        self.engine._positions_manager.open_position.assert_called_once()
        mock_save_orders.assert_called_once()
        # Verify the position was created with side="short"
        call_args = self.engine._positions_manager.open_position.call_args
        created_position = call_args[0][0]
        self.assertEqual(created_position.side, "short")
        self.assertEqual(created_position.client_order_id, "AAPL-SELL-TEST")
        self.assertEqual(created_position.order_id, "order_short_1")
        # Order ledger uses unsigned qty + explicit side (fix E).
        saved = mock_save_orders.call_args[0][0][0]
        self.assertEqual(saved.qty, 5.0, "order ledger qty must be unsigned")
        self.assertEqual(saved.side, "sell")

    def test_exit_opposite_position_no_conflict(self):
        """No exit when direction matches existing position."""
        existing = self._position("AAPL")
        self.engine._positions_manager.positions = [existing]

        result = self.engine._exit_opposite_position("AAPL", "long")
        self.assertFalse(result, "Should not exit when direction matches")

    def test_exit_opposite_position_exits_long(self):
        """Exit existing long when a short signal fires (no flip)."""
        existing = self._position("AAPL")
        self.engine._positions_manager.positions = [existing]
        self.engine.set_dry_run_mode(True)

        result = self.engine._exit_opposite_position("AAPL", "short")
        self.assertTrue(result, "Should exit conflicting long position")
        # Dry-run does not mutate position state.
        self.engine._positions_manager.close_position.assert_not_called()

    def test_exit_opposite_position_exits_short(self):
        """Exit existing short when a long signal fires (no flip)."""
        existing = self._position("AAPL")
        existing.side = "short"
        self.engine._positions_manager.positions = [existing]
        self.engine.set_dry_run_mode(True)

        result = self.engine._exit_opposite_position("AAPL", "long")
        self.assertTrue(result, "Should exit conflicting short position")
        # Dry-run does not mutate position state.
        self.engine._positions_manager.close_position.assert_not_called()

    def test_exit_opposite_position_uses_side_aware_close(self):
        """Exiting a short must close via BUY (cover), not SELL (fix B/C)."""
        existing = self._position("AAPL")
        existing.side = "short"
        existing.quantity = -10.0
        self.engine._positions_manager.positions = [existing]
        self.engine.set_dry_run_mode(False)

        with patch.object(self.engine, 'place_market_sell_order',
                          return_value=True) as mock_close:
            result = self.engine._exit_opposite_position("AAPL", "long")

        self.assertTrue(result)
        mock_close.assert_called_once_with(
            "AAPL", 10, "opposite_signal", side="short")
        self.engine._positions_manager.close_position.assert_called_once_with(
            "AAPL")

    def test_exit_opposite_position_fails_without_client_does_not_mark_closed(self):
        """Fix D: when broker close fails, don't mark the position closed."""
        existing = self._position("AAPL")
        self.engine._positions_manager.positions = [existing]
        self.engine.set_dry_run_mode(False)
        self.engine.trading_client = None

        result = self.engine._exit_opposite_position("AAPL", "short")
        self.assertFalse(result)
        self.engine._positions_manager.close_position.assert_not_called()

    def test_long_opportunity_does_not_exclude_existing_short(self):
        """A symbol held short still appears as a long opportunity (to be exited)."""
        results = [self._result("AAPL")]
        results[0].direction = "long"

        existing_short = self._position("AAPL")
        existing_short.side = "short"
        self.engine._positions_manager.positions = [existing_short]

        def rsi_side_effect(symbol, _period):
            return (25.0, 35.0)  # crossed below 30

        with patch.object(self.engine, '_get_rsi_with_previous', side_effect=rsi_side_effect), \
                patch.object(self.engine, '_get_current_price', return_value=150.0), \
                patch.object(self.engine, '_compute_rsi_take_profit', return_value=170.0):
            opportunities = self.engine.identify_buying_opportunities(results)

        self.assertEqual(len(opportunities), 1,
                         "existing short should not be excluded from long opportunities")

    def test_identify_purchases_exit_does_not_consume_position_slot(self):
        """Exits must not consume new-position slots (sizing sees only entries)."""
        existing_short = self._position("EXIT")
        existing_short.side = "short"
        self.engine._positions_manager.positions = [existing_short]

        exit_op = TradingOpportunity(
            symbol="EXIT", current_rsi=25.0, target_rsi_lower=30,
            target_rsi_upper=70, rsi_period=14, backtest_return=0.15,
            alpha=0.05, win_rate=0.9, entry_price=100.0,
            stop_loss_price=95.0, take_profit_price=110.0, num_trades=10,
        )
        entry_op = TradingOpportunity(
            symbol="NEW", current_rsi=24.0, target_rsi_lower=30,
            target_rsi_upper=70, rsi_period=14, backtest_return=0.2,
            alpha=0.1, win_rate=0.9, entry_price=50.0,
            stop_loss_price=47.5, take_profit_price=55.0, num_trades=10,
        )

        summary = {'opportunities_found': 0, 'new_positions': 0,
                   'orders_placed': 0, 'positions_exited': 0}

        with patch.object(self.engine, 'identify_buying_opportunities',
                          return_value=[exit_op, entry_op]), \
                patch.object(self.engine, '_exit_opposite_position',
                             return_value=True) as mock_exit, \
                patch.object(self.engine, 'calculate_position_sizes',
                             return_value=[(entry_op, 10)]) as mock_size, \
                patch.object(self.engine, 'place_buy_order', return_value=True):
            self.engine.identify_purchases(summary, [])

        # Sizing must only see the new entry, not the exit.
        sized_opps = mock_size.call_args[0][0]
        self.assertEqual([op.symbol for op in sized_opps], ["NEW"])
        mock_exit.assert_called_once_with("EXIT", "long")
        self.assertEqual(summary['positions_exited'], 1)
        self.assertEqual(summary['new_positions'], 1)

    def test_identify_and_execute_shorts_exit_does_not_consume_position_slot(self):
        """Short exits must not consume new-position slots."""
        existing_long = self._position("EXIT")
        existing_long.side = "long"
        self.engine._positions_manager.positions = [existing_long]

        exit_op = TradingOpportunity(
            symbol="EXIT", current_rsi=75.0, target_rsi_lower=30,
            target_rsi_upper=70, rsi_period=14, backtest_return=0.15,
            alpha=0.05, win_rate=0.9, entry_price=100.0,
            stop_loss_price=105.0, take_profit_price=90.0, num_trades=10,
            direction="short",
        )
        entry_op = TradingOpportunity(
            symbol="NEW", current_rsi=76.0, target_rsi_lower=30,
            target_rsi_upper=70, rsi_period=14, backtest_return=0.2,
            alpha=0.1, win_rate=0.9, entry_price=50.0,
            stop_loss_price=52.5, take_profit_price=45.0, num_trades=10,
            direction="short",
        )

        summary = {'opportunities_found': 0, 'new_positions': 0,
                   'orders_placed': 0, 'positions_exited': 0}

        with patch.object(self.engine, 'identify_shorting_opportunities',
                          return_value=[exit_op, entry_op]), \
                patch.object(self.engine, '_exit_opposite_position',
                             return_value=True) as mock_exit, \
                patch.object(self.engine, 'calculate_short_position_sizes',
                             return_value=[(entry_op, 10)]) as mock_size, \
                patch.object(self.engine, 'place_short_order', return_value=True):
            self.engine.identify_and_execute_shorts(summary, [])

        sized_opps = mock_size.call_args[0][0]
        self.assertEqual([op.symbol for op in sized_opps], ["NEW"])
        mock_exit.assert_called_once_with("EXIT", "short")
        self.assertEqual(summary['positions_exited'], 1)
        self.assertEqual(summary['new_positions'], 1)


if __name__ == '__main__':
    unittest.main()
