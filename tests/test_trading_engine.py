#!/usr/bin/env python3
"""Unit tests for the trading_engine module."""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

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

    def test_place_buy_order_dry_run_returns_false(self):
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

        # Current implementation intentionally does not mark dry-run as success.
        self.assertFalse(result)

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

        result = self.engine.place_buy_order(opp, 5)

        self.assertTrue(result)
        self.engine._positions_manager.open_position.assert_called_once()

    def test_place_oco_sell_order_dry_run(self):
        self.engine.set_dry_run_mode(True)

        result = self.engine.place_oco_sell_order("AAPL", 10, 95.0, 110.0)

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
                patch.object(self.engine, 'place_oco_sell_order', return_value=True):
            out = self.engine.update_portfolio_orders(
                summary, [expired, active])

        self.assertEqual(out['positions_exited'], 1)
        self.assertEqual(out['orders_placed'], 1)
        self.engine._positions_manager.close_position.assert_called_once_with(
            "AAPL")


if __name__ == '__main__':
    unittest.main()
