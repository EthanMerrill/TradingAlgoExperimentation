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

from data_provider import DataProvider, TechnicalIndicators  # noqa: E402


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

    @patch('data_provider.globalConfig')
    @patch('data_provider.time.sleep')
    def test_filter_symbols_by_max_volume(self, _mock_sleep, mock_config):
        """Test universe filter excludes symbols above max volume."""
        mock_config.get_alpaca_config.return_value = self.mock_config
        mock_config.PAPER_TRADE = True
        mock_config.API_RATE_LIMIT_DELAY = 0.0
        mock_config.MIN_PRICE = 8.0
        mock_config.MAX_PRICE = 350.0
        mock_config.MIN_VOLUME = 20
        mock_config.MAX_VOLUME = 2000000
        mock_config.MAX_MARKET_CAP = None

        snapshots = {
            'AAPL': {
                'latest_trade': {'price': 150.0},
                'previous_daily_bar': {'volume': 2500000},
            },
            'MSFT': {
                'latest_trade': {'price': 300.0},
                'previous_daily_bar': {'volume': 1000000},
            },
        }

        with patch('data_provider.StockHistoricalDataClient') as mock_historical_class, \
                patch('data_provider.TradingClient') as mock_trading_class:
            mock_historical = Mock()
            mock_historical.get_stock_snapshot.return_value = snapshots
            mock_historical_class.return_value = mock_historical
            mock_trading_class.return_value = Mock()

            data_provider = DataProvider()
            filtered = data_provider._filter_symbols_by_price(['AAPL', 'MSFT'])

            self.assertEqual(filtered, ['MSFT'])

    @patch('data_provider.globalConfig')
    @patch('data_provider.time.sleep')
    def test_filter_symbols_by_max_market_cap_missing_cap_passes(self, _mock_sleep, mock_config):
        """Test market-cap filter only excludes symbols with known cap above the max."""
        mock_config.get_alpaca_config.return_value = self.mock_config
        mock_config.PAPER_TRADE = True
        mock_config.API_RATE_LIMIT_DELAY = 0.0
        mock_config.MIN_PRICE = 8.0
        mock_config.MAX_PRICE = 350.0
        mock_config.MIN_VOLUME = 20
        mock_config.MAX_VOLUME = 5000000
        mock_config.MAX_MARKET_CAP = 200000000000

        snapshots = {
            'AAPL': {
                'latest_trade': {'price': 150.0},
                'previous_daily_bar': {'volume': 1200000},
            },
            'MSFT': {
                'latest_trade': {'price': 300.0},
                'previous_daily_bar': {'volume': 1000000},
            },
            'AMD': {
                'latest_trade': {'price': 110.0},
                'previous_daily_bar': {'volume': 1400000},
            },
        }
        symbol_market_caps = {
            'AAPL': 250000000000,
            'AMD': 150000000000,
        }

        with patch('data_provider.StockHistoricalDataClient') as mock_historical_class, \
                patch('data_provider.TradingClient') as mock_trading_class:
            mock_historical = Mock()
            mock_historical.get_stock_snapshot.return_value = snapshots
            mock_historical_class.return_value = mock_historical
            mock_trading_class.return_value = Mock()

            data_provider = DataProvider()
            filtered = data_provider._filter_symbols_by_price(
                ['AAPL', 'MSFT', 'AMD'], symbol_market_caps=symbol_market_caps)

            self.assertEqual(filtered, ['MSFT', 'AMD'])


class TestDataProviderOrderHistory(unittest.TestCase):
    """Test cases for the DataProvider order history methods."""

    def setUp(self):
        self.mock_config = {
            'api_key': 'test_key',
            'secret_key': 'test_secret',
            'base_url': 'https://paper-api.alpaca.markets'
        }

    def _make_mock_order(self, symbol, side, filled_qty, filled_avg_price,
                         submitted_at, filled_at=None, order_type='market', status='filled'):
        """Helper to create a mock Alpaca order object."""
        order = Mock()
        order.symbol = symbol
        order.side = side
        order.filled_qty = str(filled_qty)
        order.filled_avg_price = str(filled_avg_price)
        order.submitted_at = submitted_at
        order.filled_at = filled_at or submitted_at
        order.type = order_type
        order.status = status
        return order

    @patch('data_provider.globalConfig')
    def test_get_filled_orders_for_symbol_success(self, mock_config):
        """Test successfully retrieving filled orders for a symbol."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.TradingClient') as mock_trading_class:
            mock_trading = Mock()
            now = datetime.now()
            mock_trading.get_orders.return_value = [
                self._make_mock_order(
                    'AAPL', 'buy', 10, 150.25,
                    now - timedelta(days=5)
                ),
                self._make_mock_order(
                    'AAPL', 'sell', 5, 155.00,
                    now - timedelta(days=2)
                ),
            ]
            mock_trading_class.return_value = mock_trading

            data_provider = DataProvider()
            result = data_provider.get_filled_orders_for_symbol('AAPL')

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertIn('symbol', result.columns)
            self.assertIn('side', result.columns)
            self.assertIn('filled_qty', result.columns)
            self.assertIn('filled_avg_price', result.columns)
            self.assertIn('submitted_at', result.columns)
            self.assertEqual(result.iloc[0]['symbol'], 'AAPL')
            self.assertEqual(result.iloc[0]['side'], 'buy')
            self.assertEqual(result.iloc[0]['filled_qty'], 10.0)

            mock_trading.get_orders.assert_called_once()
            # Verify the filter was constructed correctly
            from alpaca.trading.requests import GetOrdersRequest
            call_kwargs = mock_trading.get_orders.call_args.kwargs
            self.assertIn('filter', call_kwargs)
            filter_req = call_kwargs['filter']
            self.assertIsInstance(filter_req, GetOrdersRequest)
            self.assertEqual(filter_req.status, 'closed')
            self.assertEqual(filter_req.symbols, ['AAPL'])
            self.assertEqual(filter_req.limit, 50)
            self.assertEqual(filter_req.direction, 'desc')

    @patch('data_provider.globalConfig')
    def test_get_filled_orders_for_symbol_empty(self, mock_config):
        """Test retrieving filled orders when none exist."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.TradingClient') as mock_trading_class:
            mock_trading = Mock()
            mock_trading.get_orders.return_value = []
            mock_trading_class.return_value = mock_trading

            data_provider = DataProvider()
            result = data_provider.get_filled_orders_for_symbol('UNKNOWN')

            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)

    @patch('data_provider.globalConfig')
    def test_get_filled_orders_for_symbol_no_client(self, mock_config):
        """Test retrieving filled orders when trading client is None."""
        mock_config.get_alpaca_config.return_value = {
            'api_key': '', 'secret_key': ''}

        data_provider = DataProvider()
        result = data_provider.get_filled_orders_for_symbol('AAPL')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    @patch('data_provider.globalConfig')
    def test_get_filled_orders_for_symbol_normalizes_enum_side(self, mock_config):
        """Enum-valued side/type/status must be normalized to lowercase strings."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        class _FakeEnum:
            value = "BUY"

        order = Mock()
        order.symbol = "AAPL"
        order.side = _FakeEnum
        order.filled_qty = "10"
        order.filled_avg_price = "150.25"
        order.submitted_at = datetime.now()
        order.filled_at = datetime.now()
        order.type = _FakeEnum
        order.status = _FakeEnum

        with patch('data_provider.TradingClient') as mock_trading_class:
            mock_trading = Mock()
            mock_trading.get_orders.return_value = [order]
            mock_trading_class.return_value = mock_trading

            data_provider = DataProvider()
            result = data_provider.get_filled_orders_for_symbol('AAPL')

        self.assertEqual(result.iloc[0]['side'], "buy")
        self.assertEqual(result.iloc[0]['order_type'], "buy")
        self.assertEqual(result.iloc[0]['status'], "buy")

    @patch('data_provider.globalConfig')
    def test_get_entry_order_for_symbol_long(self, mock_config):
        """Test finding entry order for a long position."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.TradingClient') as mock_trading_class:
            mock_trading = Mock()
            now = datetime(2026, 5, 15, 10, 30, 0)
            mock_trading.get_orders.return_value = [
                self._make_mock_order(
                    'AAPL', 'sell', 10, 165.00,
                    now - timedelta(days=1)
                ),
                self._make_mock_order(
                    'AAPL', 'buy', 10, 150.25,
                    now - timedelta(days=10)
                ),
            ]
            mock_trading_class.return_value = mock_trading

            data_provider = DataProvider()
            result = data_provider.get_entry_order_for_symbol(
                'AAPL', side='long')

            self.assertIsNotNone(result)
            submitted_at, price = result
            self.assertIsInstance(submitted_at, datetime)
            self.assertEqual(price, 150.25)

    @patch('data_provider.globalConfig')
    def test_get_entry_order_for_symbol_short(self, mock_config):
        """Test finding entry order for a short position."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.TradingClient') as mock_trading_class:
            mock_trading = Mock()
            now = datetime(2026, 5, 15, 10, 30, 0)
            mock_trading.get_orders.return_value = [
                self._make_mock_order(
                    'TSLA', 'buy', 5, 200.00,
                    now - timedelta(days=3)
                ),
                self._make_mock_order(
                    'TSLA', 'sell', 5, 210.00,
                    now - timedelta(days=8)
                ),
            ]
            mock_trading_class.return_value = mock_trading

            data_provider = DataProvider()
            result = data_provider.get_entry_order_for_symbol(
                'TSLA', side='short')

            self.assertIsNotNone(result)
            submitted_at, price = result
            self.assertIsInstance(submitted_at, datetime)
            self.assertEqual(price, 210.00)

    @patch('data_provider.globalConfig')
    def test_get_entry_order_for_symbol_no_match(self, mock_config):
        """Test finding entry order when no matching order exists."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.TradingClient') as mock_trading_class:
            mock_trading = Mock()
            mock_trading.get_orders.return_value = []
            mock_trading_class.return_value = mock_trading

            data_provider = DataProvider()
            result = data_provider.get_entry_order_for_symbol(
                'NONE', side='long')

            self.assertIsNone(result)

    @patch('data_provider.globalConfig')
    def test_get_entry_order_for_symbol_returns_buy_not_sell(self, mock_config):
        """Test long entry ignores sell orders."""
        mock_config.get_alpaca_config.return_value = self.mock_config

        with patch('data_provider.TradingClient') as mock_trading_class:
            mock_trading = Mock()
            now = datetime(2026, 5, 15, 10, 30, 0)
            mock_trading.get_orders.return_value = [
                self._make_mock_order(
                    'AAPL', 'sell', 10, 155.00,
                    now - timedelta(days=1)
                ),
            ]
            mock_trading_class.return_value = mock_trading

            data_provider = DataProvider()
            result = data_provider.get_entry_order_for_symbol(
                'AAPL', side='long')

            self.assertIsNone(result)


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
