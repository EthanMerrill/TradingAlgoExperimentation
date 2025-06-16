#!/usr/bin/env python3
"""
Unit tests for the main module.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestMainModule(unittest.TestCase):
    """Test cases for the main module functions."""

    @patch('main.utils.setup_logging')
    @patch('main.utils.is_trading_day')
    @patch('main.utils.is_market_hours')
    def test_main_execution_outside_trading_hours(self, mock_is_market_hours, mock_is_trading_day, mock_setup_logging):
        """Test main execution outside trading hours."""
        mock_is_trading_day.return_value = True
        mock_is_market_hours.return_value = False

        with patch('main.logger') as mock_logger:
            # Import and run main after mocking
            import main

            # The main function should log that it's outside trading hours
            mock_logger.info.assert_called()

    @patch('main.utils.setup_logging')
    @patch('main.utils.is_trading_day')
    @patch('main.utils.is_market_hours')
    def test_main_execution_non_trading_day(self, mock_is_market_hours, mock_is_trading_day, mock_setup_logging):
        """Test main execution on non-trading day."""
        mock_is_trading_day.return_value = False
        mock_is_market_hours.return_value = True

        with patch('main.logger') as mock_logger:
            # Import and run main after mocking
            import main

            # The main function should log that it's not a trading day
            mock_logger.info.assert_called()

    @patch('main.utils.setup_logging')
    @patch('main.utils.is_trading_day')
    @patch('main.utils.is_market_hours')
    @patch('main.StrategyBacktester')
    @patch('main.TradingEngine')
    def test_main_execution_during_trading_hours(self, mock_trading_engine_class, mock_backtester_class,
                                                 mock_is_market_hours, mock_is_trading_day, mock_setup_logging):
        """Test main execution during trading hours."""
        mock_is_trading_day.return_value = True
        mock_is_market_hours.return_value = True

        # Mock backtester
        mock_backtester = Mock()
        mock_backtester.run_parallel_backtests.return_value = []
        mock_backtester_class.return_value = mock_backtester

        # Mock trading engine
        mock_trading_engine = Mock()
        mock_trading_engine.identify_buying_opportunities.return_value = []
        mock_trading_engine_class.return_value = mock_trading_engine

        with patch('main.config') as mock_config:
            mock_config.SYMBOLS = ['AAPL', 'TSLA']
            mock_config.RSI_CONFIGS = [(14, 30, 70)]

            with patch('main.logger') as mock_logger:
                # Import and run main after mocking
                import main

                # Verify that backtesting and trading logic was called
                mock_backtester.run_parallel_backtests.assert_called()
                mock_trading_engine.identify_buying_opportunities.assert_called()

    @patch('main.utils.setup_logging')
    def test_main_execution_with_exception(self, mock_setup_logging):
        """Test main execution with exception handling."""
        with patch('main.utils.is_trading_day', side_effect=Exception("Test error")):
            with patch('main.logger') as mock_logger:
                # Import and run main after mocking
                import main

                # Should log the error
                mock_logger.error.assert_called()

    @patch('main.StrategyBacktester')
    def test_run_backtests_function(self, mock_backtester_class):
        """Test the run_backtests function."""
        # Mock backtester
        mock_backtester = Mock()
        mock_result = Mock()
        mock_result.profitable = True
        mock_backtester.run_parallel_backtests.return_value = [mock_result]
        mock_backtester_class.return_value = mock_backtester

        # Mock cloud storage upload
        with patch('main.cloud_storage') as mock_cloud_storage:
            mock_cloud_storage.upload_backtest_results.return_value = True

            with patch('main.config') as mock_config:
                mock_config.SYMBOLS = ['AAPL']
                mock_config.RSI_CONFIGS = [(14, 30, 70)]

                # Import the function
                from main import run_backtests

                results = run_backtests()

                self.assertEqual(len(results), 1)
                self.assertTrue(results[0].profitable)
                mock_cloud_storage.upload_backtest_results.assert_called()

    @patch('main.TradingEngine')
    def test_execute_trades_function(self, mock_trading_engine_class):
        """Test the execute_trades function."""
        # Mock trading engine
        mock_trading_engine = Mock()
        mock_opportunity = Mock()
        mock_opportunity.symbol = "AAPL"
        mock_trading_engine.identify_buying_opportunities.return_value = [
            mock_opportunity]
        mock_trading_engine.place_buy_order.return_value = True
        mock_trading_engine_class.return_value = mock_trading_engine

        # Mock cloud storage upload
        with patch('main.cloud_storage') as mock_cloud_storage:
            mock_cloud_storage.upload_position_entries.return_value = True

            # Import the function
            from main import execute_trades

            mock_backtest_results = [Mock()]

            execute_trades(mock_backtest_results)

            mock_trading_engine.identify_buying_opportunities.assert_called_with(
                mock_backtest_results)
            mock_trading_engine.place_buy_order.assert_called_with(
                mock_opportunity)
            mock_cloud_storage.upload_position_entries.assert_called()

    @patch('main.TradingEngine')
    def test_check_exit_signals_function(self, mock_trading_engine_class):
        """Test the check_exit_signals function."""
        # Mock trading engine
        mock_trading_engine = Mock()
        mock_exit_signal = {
            'symbol': 'AAPL',
            'quantity': 100,
            'reason': 'Take profit reached'
        }
        mock_trading_engine.check_exit_conditions.return_value = [
            mock_exit_signal]
        mock_trading_engine.place_sell_order.return_value = True
        mock_trading_engine_class.return_value = mock_trading_engine

        # Import the function
        from main import check_exit_signals

        check_exit_signals()

        mock_trading_engine.check_exit_conditions.assert_called()
        mock_trading_engine.place_sell_order.assert_called_with('AAPL', 100)

    @patch('main.config')
    def test_config_validation(self, mock_config):
        """Test configuration validation."""
        # Test with valid config
        mock_config.SYMBOLS = ['AAPL', 'TSLA']
        mock_config.RSI_CONFIGS = [(14, 30, 70), (21, 25, 75)]
        mock_config.MAX_POSITIONS = 5
        mock_config.RISK_PER_TRADE = 0.02

        # Import should work without issues
        try:
            import main
            config_valid = True
        except Exception:
            config_valid = False

        self.assertTrue(config_valid)


if __name__ == '__main__':
    unittest.main()
