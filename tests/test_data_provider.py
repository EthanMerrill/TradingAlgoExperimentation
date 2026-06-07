#!/usr/bin/env python3
"""
Unit tests for the data_provider module.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from data_provider import BarData, DataProvider, TechnicalIndicators  # noqa: E402


class TestBarData(unittest.TestCase):
    """Test cases for the BarData dataclass."""

    def test_bar_data_creation(self):
        """Test creating a BarData object."""
        bar_data = BarData(
            symbol="AAPL",
            timestamp=datetime(2025, 6, 14, 10, 30, 0),
            open=150.00,
            high=152.50,
            low=149.50,
            close=151.00,
            volume=1000000
        )

        self.assertEqual(bar_data.symbol, "AAPL")
        self.assertEqual(bar_data.open, 150.00)
        self.assertEqual(bar_data.high, 152.50)
        self.assertEqual(bar_data.low, 149.50)
        self.assertEqual(bar_data.close, 151.00)
        self.assertEqual(bar_data.volume, 1000000)


class TestDataProvider(unittest.TestCase):
    """Test cases for the DataProvider class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'base_url': 'https://paper-api.alpaca.markets'
        }

    @patch('data_provider.globalConfig')
    def test_data_provider_init_with_credentials(self, mock_config):
        """Test DataProvider initialization with valid credentials."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.StockHistoricalDataClient') as mock_historical, \
                patch('data_provider.TradingClient') as mock_trading:

            data_provider = DataProvider()

            self.assertIsNotNone(data_provider.historical_client)
            self.assertIsNotNone(data_provider.trading_client)
            mock_historical.assert_called_once()
            mock_trading.assert_called_once()

    @patch('data_provider.globalConfig')
    def test_data_provider_init_without_credentials(self, mock_config):
        """Test DataProvider initialization without credentials."""
        mock_config.get_alpaca_config.return_value = {
            'api_key': '', 'secret_key': ''}

        with patch('data_provider.logger') as mock_logger:
            data_provider = DataProvider()

            self.assertIsNone(data_provider.historical_client)
            self.assertIsNone(data_provider.trading_client)
            mock_logger.warning.assert_called()

    @patch('data_provider.globalConfig')
    def test_get_historical_data_success(self, mock_config):
        """Test successful historical data retrieval."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        # Mock the Alpaca response
        mock_bar = Mock()
        mock_bar.timestamp = datetime(2025, 6, 14, 10, 30, 0)
        mock_bar.open = 150.0
        mock_bar.high = 152.5
        mock_bar.low = 149.5
        mock_bar.close = 151.0
        mock_bar.volume = 1000000

        mock_response = {'AAPL': [mock_bar]}

        with patch('data_provider.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_stock_bars.return_value = mock_response
            mock_client_class.return_value = mock_client

            data_provider = DataProvider()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            result = data_provider.get_single_stock_bars(
                'AAPL', start_date, end_date)

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]['close'], 151.0)

    @patch('data_provider.globalConfig')
    def test_get_historical_data_no_client(self, mock_config):
        """Test historical data retrieval when client is None."""
        mock_config.get_alpaca_config.return_value = {
            'api_key': '', 'secret_key': ''}

        data_provider = DataProvider()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        result = data_provider.get_single_stock_bars(
            'AAPL', start_date, end_date)

        self.assertTrue(result.empty)

    @patch('data_provider.globalConfig')
    def test_get_historical_data_exception(self, mock_config):
        """Test historical data retrieval with exception."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_stock_bars.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_client

            with patch('data_provider.logger') as mock_logger:
                data_provider = DataProvider()

                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                result = data_provider.get_single_stock_bars(
                    'AAPL', start_date, end_date)

                self.assertTrue(result.empty)
                mock_logger.error.assert_called()

    @patch('data_provider.globalConfig')
    def test_get_multiple_stocks_data(self, mock_config):
        """Test getting data for multiple stocks."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        # Mock the Alpaca response for multiple symbols
        mock_bar_aapl = Mock()
        mock_bar_aapl.timestamp = datetime(2025, 6, 14, 10, 30, 0)
        mock_bar_aapl.close = 151.0

        mock_bar_tsla = Mock()
        mock_bar_tsla.timestamp = datetime(2025, 6, 14, 10, 30, 0)
        mock_bar_tsla.close = 250.0

        with patch('data_provider.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            # Return different responses per call
            mock_client.get_stock_bars.side_effect = [
                {'AAPL': [mock_bar_aapl]},
                {'TSLA': [mock_bar_tsla]},
            ]
            mock_client_class.return_value = mock_client

            data_provider = DataProvider()

            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            result_aapl = data_provider.get_single_stock_bars(
                'AAPL', start_date, end_date)
            result_tsla = data_provider.get_single_stock_bars(
                'TSLA', start_date, end_date)

            self.assertIsInstance(result_aapl, pd.DataFrame)
            self.assertIsInstance(result_tsla, pd.DataFrame)
            self.assertEqual(result_aapl.iloc[0]['close'], 151.0)
            self.assertEqual(result_tsla.iloc[0]['close'], 250.0)

    @patch('data_provider.globalConfig')
    def test_get_current_price(self, mock_config):
        """Test getting current price for a symbol."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        # Mock the snapshot response
        mock_snapshot = Mock()
        mock_snapshot.latest_trade.price = 151.50
        mock_response = {'AAPL': mock_snapshot}

        with patch('data_provider.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_stock_snapshots.return_value = mock_response
            mock_client_class.return_value = mock_client

            data_provider = DataProvider()

            result = data_provider.get_current_price('AAPL')

            self.assertEqual(result, 151.50)

    @patch('data_provider.globalConfig')
    def test_get_current_price_no_data(self, mock_config):
        """Test getting current price when no data is available."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            mock_client.get_stock_snapshots.return_value = {}
            mock_client_class.return_value = mock_client

            data_provider = DataProvider()

            result = data_provider.get_current_price('INVALID')

            self.assertIsNone(result)

    @patch('data_provider.globalConfig')
    def test_get_market_snapshot(self, mock_config):
        """Test getting current snapshot for symbols."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.StockHistoricalDataClient') as mock_client_class:
            mock_client = Mock()
            # get_current_snapshot calls get_stock_snapshot (singular), returns dict
            mock_client.get_stock_snapshot.return_value = {
                'AAPL': {'latest_trade': {'price': 151.50, 'size': 100}}}
            mock_client_class.return_value = mock_client

            data_provider = DataProvider()

            result = data_provider.get_current_snapshot('AAPL')

            self.assertIsInstance(result, dict)
            self.assertIn('price', result)
            self.assertEqual(result['price'], 151.50)


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for the TechnicalIndicators class."""

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        # Create sample price data
        prices = pd.Series([100, 105, 103, 108, 110, 107,
                           112, 115, 113, 118, 120, 117, 122, 125, 123])

        rsi = TechnicalIndicators.calculate_rsi(prices, period=14)

        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(prices))
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))

    def test_calculate_rsi_insufficient_data(self):
        """Test RSI calculation with insufficient data."""
        prices = pd.Series([100, 105, 103])  # Only 3 data points

        rsi = TechnicalIndicators.calculate_rsi(prices, period=14)

        # Should return NaN for insufficient data
        self.assertTrue(rsi.isna().all())

    def test_calculate_moving_average(self):
        """Test moving average calculation."""
        prices = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])

        ma = TechnicalIndicators.calculate_moving_average(prices, window=5)

        self.assertIsInstance(ma, pd.Series)
        self.assertEqual(len(ma), len(prices))

        # Check that the 5th value is the average of first 5 values
        expected_5th = (10 + 12 + 14 + 16 + 18) / 5
        self.assertAlmostEqual(ma.iloc[4], expected_5th)

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = pd.Series([100, 102, 104, 103, 105, 107,
                           106, 108, 110, 109, 111, 113, 112, 114, 116])

        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
            prices, window=10, num_std=2)

        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)

        # Upper band should be above middle, middle above lower
        valid_data = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_data.any():
            self.assertTrue(all(upper[valid_data] >= middle[valid_data]))
            self.assertTrue(all(middle[valid_data] >= lower[valid_data]))

    def test_calculate_macd(self):
        """Test MACD calculation."""
        prices = pd.Series(range(100, 150))  # Trending upward

        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(
            prices)

        self.assertIsInstance(macd_line, pd.Series)
        self.assertIsInstance(signal_line, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
        self.assertEqual(len(macd_line), len(prices))

    def test_calculate_stochastic(self):
        """Test Stochastic oscillator calculation."""
        high = pd.Series([105, 107, 109, 108, 110, 112, 111, 113, 115, 114])
        low = pd.Series([95, 97, 99, 98, 100, 102, 101, 103, 105, 104])
        close = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])

        k_percent, d_percent = TechnicalIndicators.calculate_stochastic(
            high, low, close)

        self.assertIsInstance(k_percent, pd.Series)
        self.assertIsInstance(d_percent, pd.Series)

        # Stochastic should be between 0 and 100
        valid_k = k_percent.dropna()
        valid_d = d_percent.dropna()
        if len(valid_k) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_k))
        if len(valid_d) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_d))


if __name__ == '__main__':
    unittest.main()
