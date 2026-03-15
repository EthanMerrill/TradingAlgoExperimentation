#!/usr/bin/env python3
"""
Unit tests for the PositionsManager class.
"""
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from positions import Position, PositionsManager

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestPosition(unittest.TestCase):
    """Test cases for the Position dataclass."""

    def test_position_creation(self):
        """Test creating a Position object."""
        position = Position(
            symbol="AAPL",
            quantity=100.0,
            entry_price=150.00,
            current_price=155.00,
            entry_date=datetime(2025, 6, 14),
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70,
            stop_loss_price=140.00,
            take_profit_price=160.00
        )

        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.quantity, 100.0)
        self.assertEqual(position.entry_price, 150.00)
        self.assertEqual(position.current_price, 155.00)
        self.assertEqual(position.rsi_period, 14)
        self.assertEqual(position.rsi_lower, 30)
        self.assertEqual(position.rsi_upper, 70)
        self.assertEqual(position.stop_loss_price, 140.00)
        self.assertEqual(position.take_profit_price, 160.00)

    def test_position_optional_fields(self):
        """Test creating a Position with optional fields as None."""
        position = Position(
            symbol="TSLA",
            quantity=50.0,
            entry_price=800.00,
            current_price=850.00,
            entry_date=datetime(2025, 6, 14),
            rsi_period=14,
            rsi_lower=30,
            rsi_upper=70
        )

        self.assertIsNone(position.stop_loss_price)
        self.assertIsNone(position.take_profit_price)


class TestPositionsManager(unittest.TestCase):
    """Test cases for the PositionsManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_cloud_storage = Mock()
        self.mock_data_provider = Mock()
        self.positions_manager = PositionsManager(
            self.mock_cloud_storage, self.mock_data_provider)

    def test_init(self):
        """Test PositionsManager initialization."""
        self.assertEqual(self.positions_manager.cloud_storage,
                         self.mock_cloud_storage)

    def test_get_positions_from_alpaca_placeholder(self):
        """Test the _get_positions_from_alpaca placeholder method."""
        result = self.positions_manager._get_positions_from_alpaca()
        self.assertEqual(result, [])

    @patch('positions.cloud_storage')
    def test_get_positions_from_google_cloud_no_files(self, mock_cloud_storage_module):
        """Test getting positions when no files exist in cloud storage."""
        # Mock cloud_storage.list_position_files() to return empty list
        mock_cloud_storage_module.list_position_files.return_value = []

        with patch('positions.logger') as mock_logger:
            result = self.positions_manager._get_positions_from_google_cloud()

            self.assertEqual(result, {})
            mock_logger.warning.assert_called_with(
                "No position files found in cloud storage")

    @patch('positions.cloud_storage')
    def test_get_positions_from_google_cloud_success(self, mock_cloud_storage_module):
        """Test successfully getting positions from cloud storage."""
        # Mock the cloud storage methods
        mock_cloud_storage_module.list_position_files.return_value = [
            'positions_20250614.csv', 'positions_20250613.csv']
        mock_df = pd.DataFrame({
            'symbol': ['AAPL', 'TSLA'],
            'quantity': [100, 50],
            'entry_price': [150.0, 800.0]
        })
        mock_cloud_storage_module.load_position_entries.return_value = mock_df

        with patch('positions.logger') as mock_logger:
            result = self.positions_manager._get_positions_from_google_cloud()

            # Should return the DataFrame
            pd.testing.assert_frame_equal(result, mock_df)
            mock_logger.info.assert_called_with(
                "Loading positions from %s", 'positions_20250614.csv')

    @patch('positions.cloud_storage')
    def test_get_positions_from_google_cloud_empty_file(self, mock_cloud_storage_module):
        """Test getting positions when the file is empty."""
        mock_cloud_storage_module.list_position_files.return_value = [
            'positions_20250614.csv']
        mock_cloud_storage_module.load_position_entries.return_value = pd.DataFrame()

        with patch('positions.logger') as mock_logger:
            result = self.positions_manager._get_positions_from_google_cloud()

            self.assertEqual(result, {})
            mock_logger.warning.assert_called_with(
                "No data found in %s", 'positions_20250614.csv')

    @patch('positions.cloud_storage')
    def test_get_positions_from_google_cloud_exception(self, mock_cloud_storage_module):
        """Test exception handling in getting positions from cloud storage."""
        mock_cloud_storage_module.list_position_files.side_effect = Exception(
            "Cloud storage error")

        with patch('positions.logger') as mock_logger:
            result = self.positions_manager._get_positions_from_google_cloud()

            self.assertEqual(result, [])
            mock_logger.error.assert_called_with(
                "Error loading positions from cloud storage: %s", "Cloud storage error")

    def test_get_and_reconcile_positions_no_alpaca_positions(self):
        """Test reconciliation when there are no Alpaca positions."""
        with patch.object(self.positions_manager, '_get_positions_from_alpaca', return_value=[]):
            with patch('positions.logger') as mock_logger:
                result = self.positions_manager.get_and_reconcile_positions()

                self.assertEqual(result, [])
                mock_logger.warning.assert_called_with(
                    "No positions found in Alpaca")

    def test_get_and_reconcile_positions_no_cloud_positions(self):
        """Test reconciliation when there are no cloud positions."""
        mock_alpaca_positions = [Mock()]

        with patch.object(self.positions_manager, '_get_positions_from_alpaca', return_value=mock_alpaca_positions):
            with patch.object(self.positions_manager, '_get_positions_from_google_cloud', return_value=[]):
                with patch('positions.logger') as mock_logger:
                    result = self.positions_manager.get_and_reconcile_positions()

                    self.assertEqual(result, [])
                    mock_logger.warning.assert_called_with(
                        "No positions found in Google Cloud Storage")


if __name__ == '__main__':
    unittest.main()
