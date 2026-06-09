#!/usr/bin/env python3
"""
Unit tests for the cloud_storage module.
"""
import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from cloud_storage import CloudStorage  # noqa: E402


class TestCloudStorage(unittest.TestCase):
    """Test cases for the CloudStorage class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.GCS_BUCKET_NAME = 'test-bucket'

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_cloud_storage_init_success(self, mock_import_module, mock_config):
        """Test CloudStorage initialization with valid credentials."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_import_module.import_module.return_value = mock_storage_module

        cloud_storage = CloudStorage()

        self.assertIsNotNone(cloud_storage.client)
        self.assertIsNotNone(cloud_storage.bucket)
        mock_import_module.import_module.assert_called_once_with(
            "google.cloud.storage")
        mock_client.bucket.assert_called_once_with('test-bucket')

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_cloud_storage_init_failure(self, mock_import_module, mock_config):
        """Test CloudStorage initialization with invalid credentials."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_import_module.import_module.side_effect = Exception(
            "Authentication failed")

        with patch('cloud_storage.logger') as mock_logger:
            cloud_storage = CloudStorage()

            self.assertIsNone(cloud_storage.client)
            self.assertIsNone(cloud_storage.bucket)
            mock_logger.error.assert_called()

    def test_round_floats_dict(self):
        """Test _round_floats method with dictionary."""
        cloud_storage = CloudStorage()

        data = {
            'float_val': 1.23456,
            'int_val': 42,
            'str_val': 'test',
            'numpy_float': np.float64(2.34567)
        }

        result = cloud_storage._round_floats(data)

        self.assertEqual(result['float_val'], 1.23)
        self.assertEqual(result['int_val'], 42)
        self.assertEqual(result['str_val'], 'test')
        self.assertEqual(result['numpy_float'], 2.35)

    def test_round_floats_dataframe(self):
        """Test _round_floats method with DataFrame."""
        cloud_storage = CloudStorage()

        df = pd.DataFrame({
            'col1': [1.23456, 2.34567],
            'col2': [3.45678, 4.56789]
        })

        result = cloud_storage._round_floats(df)

        self.assertAlmostEqual(result.iloc[0, 0], 1.23, places=2)
        self.assertAlmostEqual(result.iloc[0, 1], 3.46, places=2)

    def test_round_floats_list(self):
        """Test _round_floats method with list."""
        cloud_storage = CloudStorage()

        data = [
            {'value': 1.23456},
            {'value': 2.34567}
        ]

        result = cloud_storage._round_floats(data)

        self.assertEqual(result[0]['value'], 1.23)
        self.assertEqual(result[1]['value'], 2.35)

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_upload_backtest_results_success(self, mock_import_module, mock_config):
        """Test successful backtest results upload."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_config.get_environment_path.return_value = 'dev/Backtests'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_bucket.blob.return_value = mock_blob
        mock_import_module.import_module.return_value = mock_storage_module

        cloud_storage = CloudStorage()

        # Mock backtest results (save_backtest_results uses attribute access)
        mock_results = [
            Mock(symbol='AAPL', total_return=0.15, alpha=0.05,
                 rsi_period=14, rsi_lower=30, rsi_upper=70,
                 buy_and_hold_return=0.10, num_trades=5, win_rate=0.6,
                 avg_trade_duration=10.5, max_drawdown=0.08, sharpe_ratio=1.2,
                 profitable=True, current_rsi=45.0),
            Mock(symbol='TSLA', total_return=0.20, alpha=0.08,
                 rsi_period=14, rsi_lower=30, rsi_upper=70,
                 buy_and_hold_return=0.12, num_trades=3, win_rate=0.67,
                 avg_trade_duration=7.0, max_drawdown=0.10, sharpe_ratio=0.9,
                 profitable=True, current_rsi=40.0)
        ]

        success = cloud_storage.save_backtest_results(mock_results)

        self.assertTrue(success)
        mock_blob.upload_from_string.assert_called()

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_upload_backtest_results_no_client(self, mock_import_module, mock_config):
        """Test backtest results upload when client is None."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_import_module.import_module.side_effect = Exception("No client")

        cloud_storage = CloudStorage()

        success = cloud_storage.save_backtest_results([])

        self.assertFalse(success)

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_upload_position_entries_success(self, mock_import_module, mock_config):
        """Test successful position entries upload."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_config.get_environment_path.return_value = 'dev/Positions'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_bucket.blob.return_value = mock_blob
        mock_import_module.import_module.return_value = mock_storage_module

        cloud_storage = CloudStorage()

        # save_positions accepts list of Position objects
        from positions import Position
        from datetime import datetime
        mock_positions = [
            Position(symbol='AAPL', quantity=10.0, entry_price=150.0,
                     current_price=151.0, current_rsi=45.0,
                     entry_date=datetime.now(), alpha=0.05, rsi_period=14,
                     rsi_lower=30, rsi_upper=70, stop_loss_price=140.0,
                     take_profit_price=160.0, closed=False),
            Position(symbol='TSLA', quantity=5.0, entry_price=800.0,
                     current_price=810.0, current_rsi=40.0,
                     entry_date=datetime.now(), alpha=0.08, rsi_period=14,
                     rsi_lower=30, rsi_upper=70, stop_loss_price=750.0,
                     take_profit_price=850.0, closed=False)
        ]

        success = cloud_storage.save_positions(mock_positions)

        self.assertTrue(success)
        mock_blob.upload_from_string.assert_called()

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_list_backtest_files_success(self, mock_import_module, mock_config):
        """Test successful listing of backtest files."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_config.get_environment_path.return_value = 'dev/Backtests'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob1 = Mock()
        mock_blob1.name = 'dev/Backtests/backtest_20250614.csv'
        mock_blob2 = Mock()
        mock_blob2.name = 'dev/Backtests/backtest_20250613.csv'
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_import_module.import_module.return_value = mock_storage_module

        cloud_storage = CloudStorage()

        files = cloud_storage.list_backtest_files()

        self.assertEqual(len(files), 2)
        self.assertIn('backtest_20250614.csv', files)
        self.assertIn('backtest_20250613.csv', files)

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_list_backtest_files_no_client(self, mock_import_module, mock_config):
        """Test listing backtest files when client is None."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_import_module.import_module.side_effect = Exception("No client")

        cloud_storage = CloudStorage()

        files = cloud_storage.list_backtest_files()

        self.assertEqual(files, [])

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_list_position_files_success(self, mock_import_module, mock_config):
        """Test successful listing of position files."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_config.get_environment_path.return_value = 'dev/Positions'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob1 = Mock()
        mock_blob1.name = 'dev/Positions/positions_20250614.csv'
        mock_blob2 = Mock()
        mock_blob2.name = 'dev/Positions/positions_20250613.csv'
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_import_module.import_module.return_value = mock_storage_module

        cloud_storage = CloudStorage()

        files = cloud_storage.list_position_files()

        self.assertEqual(len(files), 2)
        self.assertIn('positions_20250614.csv', files)
        self.assertIn('positions_20250613.csv', files)

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_load_backtest_results_success(self, mock_import_module, mock_config):
        """Test successful loading of backtest results."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_config.get_environment_path.return_value = 'dev/Backtests'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_bucket.blob.return_value = mock_blob
        mock_import_module.import_module.return_value = mock_storage_module

        # Mock CSV data matching BacktestResult fields
        csv_data = (
            "symbol,rsi_period,rsi_lower,rsi_upper,total_return,buy_and_hold_return,"
            "alpha,num_trades,win_rate,avg_trade_duration,max_drawdown,sharpe_ratio,"
            "profitable,current_rsi,trade_details\n"
            "AAPL,14,30,70,0.15,0.10,0.05,5,0.6,10.5,0.08,1.2,True,45.0,\n"
            "TSLA,14,30,70,0.20,0.12,0.08,3,0.67,7.0,0.10,0.9,True,40.0,\n"
        )
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = csv_data

        cloud_storage = CloudStorage()

        results = cloud_storage.load_backtest_results('test_file.csv')

        self.assertEqual(len(results), 2)
        # Results are BacktestResult objects, not dicts
        self.assertEqual(results[0].symbol, 'AAPL')
        self.assertEqual(results[1].symbol, 'TSLA')
        self.assertEqual(results[0].total_return, 0.15)

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_load_backtest_results_file_not_found(self, mock_import_module, mock_config):
        """Test loading backtest results when file doesn't exist."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_bucket.blob.return_value = mock_blob
        mock_import_module.import_module.return_value = mock_storage_module

        # Mock file not found
        mock_blob.download_as_text.side_effect = Exception("File not found")

        with patch('cloud_storage.logger') as mock_logger:
            cloud_storage = CloudStorage()

            results = cloud_storage.load_backtest_results(
                'nonexistent_file.json')

            self.assertEqual(results, [])
            mock_logger.error.assert_called()

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_load_position_entries_success(self, mock_import_module, mock_config):
        """Test successful loading of position entries."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_bucket.blob.return_value = mock_blob
        mock_import_module.import_module.return_value = mock_storage_module

        # Mock CSV data
        csv_data = "symbol,entry_price,stop_loss_price\nAAPL,150.0,140.0\nTSLA,800.0,750.0"
        mock_blob.download_as_text.return_value = csv_data

        cloud_storage = CloudStorage()

        result = cloud_storage.load_position_entries('test_file.csv')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['symbol'], 'AAPL')
        self.assertEqual(result.iloc[1]['symbol'], 'TSLA')

    @patch('cloud_storage.globalConfig')
    @patch('cloud_storage.importlib')
    def test_load_position_entries_parses_dates(self, mock_import_module, mock_config):
        """Regression: load_position_entries must parse entry_date/exit_date as datetime,
        not leave them as strings — otherwise ``datetime - entry_date`` raises TypeError."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_storage_module = Mock()
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage_module.Client.return_value = mock_client
        mock_bucket.blob.return_value = mock_blob
        mock_import_module.import_module.return_value = mock_storage_module

        csv_data = (
            "symbol,entry_price,entry_date,exit_date\n"
            "AAPL,150.0,2025-06-07,\n"
            "TSLA,800.0,2025-06-01,2025-06-05"
        )
        mock_blob.download_as_text.return_value = csv_data

        cloud_storage = CloudStorage()
        result = cloud_storage.load_position_entries('test_file.csv')

        from datetime import datetime
        self.assertIsInstance(result.iloc[0]['entry_date'], datetime,
                              "entry_date must be datetime, not str — parse_dates may be missing from read_csv")
        self.assertIsInstance(result.iloc[1]['exit_date'], datetime)


if __name__ == '__main__':
    unittest.main()
