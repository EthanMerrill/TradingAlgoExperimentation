#!/usr/bin/env python3
"""
Unit tests for the trading_engine module.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from trading_engine import TradingEngine, TradingOpportunity

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestTradingOpportunity(unittest.TestCase):
    """Test cases for the TradingOpportunity dataclass."""

    def test_trading_opportunity_creation(self):
        """Test creating a TradingOpportunity object."""
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
            num_trades=10
        )

        self.assertEqual(opportunity.symbol, "AAPL")
        self.assertEqual(opportunity.current_rsi, 25.0)
        self.assertEqual(opportunity.target_rsi_lower, 30)
        self.assertEqual(opportunity.target_rsi_upper, 70)
        self.assertEqual(opportunity.rsi_period, 14)
        self.assertEqual(opportunity.backtest_return, 0.15)
        self.assertEqual(opportunity.alpha, 0.05)
        self.assertEqual(opportunity.win_rate, 0.65)
        self.assertEqual(opportunity.entry_price, 150.00)
        self.assertEqual(opportunity.stop_loss_price, 140.00)
        self.assertEqual(opportunity.take_profit_price, 160.00)
        self.assertEqual(opportunity.num_trades, 10)

    def test_trading_opportunity_default_num_trades(self):
        """Test TradingOpportunity with default num_trades."""
        opportunity = TradingOpportunity(
            symbol="TSLA",
            current_rsi=30.0,
            target_rsi_lower=25,
            target_rsi_upper=75,
            rsi_period=21,
            backtest_return=0.20,
            alpha=0.08,
            win_rate=0.70,
            entry_price=800.00,
            stop_loss_price=750.00,
            take_profit_price=850.00
        )

        self.assertEqual(opportunity.num_trades, 0)


class TestTradingEngine(unittest.TestCase):
    """Test cases for the TradingEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('trading_engine.data_provider'):
            self.trading_engine = TradingEngine()

    @patch('trading_engine.data_provider')
    def test_trading_engine_init(self, mock_data_provider):
        """Test TradingEngine initialization."""
        mock_trading_client = Mock()
        mock_data_provider.trading_client = mock_trading_client

        engine = TradingEngine()

        self.assertEqual(engine.trading_client, mock_trading_client)
        self.assertIsNone(engine._last_position_update)
        self.assertFalse(engine.dry_run)

    def test_set_dry_run_mode_true(self):
        """Test setting dry run mode to True."""
        with patch('trading_engine.logger') as mock_logger:
            self.trading_engine.set_dry_run_mode(True)

            self.assertTrue(self.trading_engine.dry_run)
            mock_logger.info.assert_called_with(
                "🌵 DRY RUN MODE ENABLED - No actual orders will be placed")

    def test_set_dry_run_mode_false(self):
        """Test setting dry run mode to False."""
        with patch('trading_engine.logger') as mock_logger:
            self.trading_engine.set_dry_run_mode(False)

            self.assertFalse(self.trading_engine.dry_run)
            mock_logger.info.assert_called_with(
                "🚀 LIVE TRADING MODE ENABLED - Orders will be placed")

    @patch('trading_engine.data_provider')
    def test_get_current_positions_exception(self, mock_data_provider):
        """Test current positions retrieval with exception."""
        mock_trading_client = Mock()
        mock_trading_client.get_all_positions.side_effect = Exception(
            "API Error")
        mock_data_provider.trading_client = mock_trading_client

        with patch('trading_engine.logger') as mock_logger:
            engine = TradingEngine()

            positions = engine.get_current_positions()

            self.assertEqual(positions, [])
            mock_logger.error.assert_called()

    def test_identify_buying_opportunities_empty_results(self):
        """Test identifying buying opportunities with empty results."""
        opportunities = self.trading_engine.identify_buying_opportunities([])

        self.assertEqual(opportunities, [])

    def test_identify_buying_opportunities_with_valid_results(self):
        """Test identifying buying opportunities with valid backtest results."""
        # Mock backtest results
        mock_result1 = Mock()
        mock_result1.symbol = "AAPL"
        mock_result1.profitable = True
        mock_result1.alpha = 0.05
        mock_result1.win_rate = 0.65
        mock_result1.rsi_period = 14
        mock_result1.rsi_lower = 30
        mock_result1.rsi_upper = 70
        mock_result1.total_return = 0.15
        mock_result1.num_trades = 10
        mock_result1.current_rsi = 25.0

        mock_result2 = Mock()
        mock_result2.symbol = "TSLA"
        mock_result2.profitable = False
        mock_result2.alpha = -0.02

        # Mock current price retrieval
        with patch('trading_engine.data_provider') as mock_data_provider:
            mock_data_provider.get_current_price.return_value = 150.00

            opportunities = self.trading_engine.identify_buying_opportunities(
                [mock_result1, mock_result2])

            # Should only include profitable opportunities
            self.assertEqual(len(opportunities), 1)
            self.assertEqual(opportunities[0].symbol, "AAPL")
            self.assertEqual(opportunities[0].current_rsi, 25.0)

    def test_calculate_position_size(self):
        """Test position size calculation."""
        entry_price = 150.00
        stop_loss_price = 140.00

        # Mock portfolio value
        with patch.object(self.trading_engine, '_get_portfolio_value', return_value=100000.00):
            position_size = self.trading_engine.calculate_position_size(
                entry_price, stop_loss_price)

            self.assertIsInstance(position_size, int)
            self.assertGreater(position_size, 0)

    def test_calculate_position_size_invalid_prices(self):
        """Test position size calculation with invalid prices."""
        entry_price = 150.00
        stop_loss_price = 160.00  # Stop loss higher than entry (invalid)

        with patch.object(self.trading_engine, '_get_portfolio_value', return_value=100000.00):
            position_size = self.trading_engine.calculate_position_size(
                entry_price, stop_loss_price)

            self.assertEqual(position_size, 0)

    def test_place_buy_order_dry_run(self):
        """Test placing buy order in dry run mode."""
        self.trading_engine.set_dry_run_mode(True)

        mock_opportunity = Mock()
        mock_opportunity.symbol = "AAPL"
        mock_opportunity.entry_price = 150.00
        mock_opportunity.stop_loss_price = 140.00
        mock_opportunity.take_profit_price = 160.00

        with patch('trading_engine.logger') as mock_logger:
            with patch.object(self.trading_engine, 'calculate_position_size', return_value=100):
                result = self.trading_engine.place_buy_order(mock_opportunity)

                self.assertTrue(result)
                mock_logger.info.assert_called()

    def test_place_buy_order_live_mode(self):
        """Test placing buy order in live mode."""
        self.trading_engine.set_dry_run_mode(False)

        mock_opportunity = Mock()
        mock_opportunity.symbol = "AAPL"
        mock_opportunity.entry_price = 150.00
        mock_opportunity.stop_loss_price = 140.00
        mock_opportunity.take_profit_price = 160.00

        # Mock trading client
        mock_order = Mock()
        mock_order.id = "order_123"
        self.trading_engine.trading_client.submit_order.return_value = mock_order

        with patch.object(self.trading_engine, 'calculate_position_size', return_value=100):
            result = self.trading_engine.place_buy_order(mock_opportunity)

            self.assertTrue(result)
            self.trading_engine.trading_client.submit_order.assert_called()

    def test_place_buy_order_zero_position_size(self):
        """Test placing buy order with zero position size."""
        mock_opportunity = Mock()
        mock_opportunity.symbol = "AAPL"

        with patch.object(self.trading_engine, 'calculate_position_size', return_value=0):
            with patch('trading_engine.logger') as mock_logger:
                result = self.trading_engine.place_buy_order(mock_opportunity)

                self.assertFalse(result)
                mock_logger.warning.assert_called()

    def test_place_sell_order_dry_run(self):
        """Test placing sell order in dry run mode."""
        self.trading_engine.set_dry_run_mode(True)

        with patch('trading_engine.logger') as mock_logger:
            result = self.trading_engine.place_sell_order("AAPL", 100)

            self.assertTrue(result)
            mock_logger.info.assert_called()

    def test_place_sell_order_live_mode(self):
        """Test placing sell order in live mode."""
        self.trading_engine.set_dry_run_mode(False)

        # Mock trading client
        mock_order = Mock()
        mock_order.id = "sell_order_123"
        self.trading_engine.trading_client.submit_order.return_value = mock_order

        result = self.trading_engine.place_sell_order("AAPL", 100)

        self.assertTrue(result)
        self.trading_engine.trading_client.submit_order.assert_called()

    def test_check_exit_conditions_no_positions(self):
        """Test checking exit conditions with no positions."""
        with patch.object(self.trading_engine, 'get_current_positions', return_value=[]):
            result = self.trading_engine.check_exit_conditions()

            self.assertEqual(result, [])

    def test_check_exit_conditions_with_positions(self):
        """Test checking exit conditions with positions."""
        # Mock position
        mock_position = Mock()
        mock_position.symbol = "AAPL"
        mock_position.quantity = 100.0
        mock_position.current_price = 155.00
        mock_position.stop_loss_price = 140.00
        mock_position.take_profit_price = 160.00
        mock_position.rsi_period = 14
        mock_position.rsi_upper = 70

        # Mock current RSI
        with patch('trading_engine.data_provider') as mock_data_provider:
            mock_data_provider.get_current_rsi.return_value = 75.0  # Above upper threshold

            with patch.object(self.trading_engine, 'get_current_positions', return_value=[mock_position]):
                exit_signals = self.trading_engine.check_exit_conditions()

                self.assertEqual(len(exit_signals), 1)
                self.assertEqual(exit_signals[0]['symbol'], "AAPL")
                self.assertEqual(exit_signals[0]['reason'], "RSI overbought")

    def test_get_portfolio_value(self):
        """Test getting portfolio value."""
        # Mock account
        mock_account = Mock()
        mock_account.portfolio_value = "150000.00"
        self.trading_engine.trading_client.get_account.return_value = mock_account

        portfolio_value = self.trading_engine._get_portfolio_value()

        self.assertEqual(portfolio_value, 150000.00)

    def test_get_portfolio_value_exception(self):
        """Test getting portfolio value with exception."""
        self.trading_engine.trading_client.get_account.side_effect = Exception(
            "API Error")

        with patch('trading_engine.logger') as mock_logger:
            portfolio_value = self.trading_engine._get_portfolio_value()

            self.assertEqual(portfolio_value, 0.0)
            mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main()
