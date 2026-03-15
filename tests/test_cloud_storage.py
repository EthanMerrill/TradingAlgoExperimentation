#!/usr/bin/env python3
"""
Unit tests for the cloud_storage module.
"""
import io
import json
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from cloud_storage import CloudStorage

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestCloudStorage(unittest.TestCase):
    """Test cases for the CloudStorage class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.GCS_BUCKET_NAME = 'test-bucket'

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_cloud_storage_init_success(self, mock_client_class, mock_config):
        """Test CloudStorage initialization with valid credentials."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        cloud_storage = CloudStorage()

        self.assertIsNotNone(cloud_storage.client)
        self.assertIsNotNone(cloud_storage.bucket)
        mock_client.bucket.assert_called_once_with('test-bucket')

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_cloud_storage_init_failure(self, mock_client_class, mock_config):
        """Test CloudStorage initialization with invalid credentials."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client_class.side_effect = Exception("Authentication failed")

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

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_upload_backtest_results_success(self, mock_client_class, mock_config):
        """Test successful backtest results upload."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client

        cloud_storage = CloudStorage()

        # Mock backtest results
        mock_results = [
            Mock(symbol='AAPL', total_return=0.15, alpha=0.05),
            Mock(symbol='TSLA', total_return=0.20, alpha=0.08)
        ]

        # Mock the to_dict method for each result
        for i, result in enumerate(mock_results):
            result.to_dict.return_value = {
                'symbol': result.symbol,
                'total_return': result.total_return,
                'alpha': result.alpha
            }

        success = cloud_storage.upload_backtest_results(mock_results)

        self.assertTrue(success)
        mock_blob.upload_from_string.assert_called()

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_upload_backtest_results_no_client(self, mock_client_class, mock_config):
        """Test backtest results upload when client is None."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client_class.side_effect = Exception("No client")

        cloud_storage = CloudStorage()

        success = cloud_storage.upload_backtest_results([])

        self.assertFalse(success)

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_upload_position_entries_success(self, mock_client_class, mock_config):
        """Test successful position entries upload."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client

        cloud_storage = CloudStorage()

        # Mock trading opportunities
        mock_opportunities = [
            Mock(symbol='AAPL', entry_price=150.0, stop_loss_price=140.0),
            Mock(symbol='TSLA', entry_price=800.0, stop_loss_price=750.0)
        ]

        success = cloud_storage.upload_position_entries(mock_opportunities)

        self.assertTrue(success)
        mock_blob.upload_from_string.assert_called()

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_list_backtest_files_success(self, mock_client_class, mock_config):
        """Test successful listing of backtest files."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob1 = Mock()
        mock_blob1.name = 'backtests/backtest_20250614.json'
        mock_blob2 = Mock()
        mock_blob2.name = 'backtests/backtest_20250613.json'
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        cloud_storage = CloudStorage()

        files = cloud_storage.list_backtest_files()

        self.assertEqual(len(files), 2)
        self.assertIn('backtest_20250614.json', files)
        self.assertIn('backtest_20250613.json', files)

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_list_backtest_files_no_client(self, mock_client_class, mock_config):
        """Test listing backtest files when client is None."""
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client_class.side_effect = Exception("No client")

        cloud_storage = CloudStorage()

        files = cloud_storage.list_backtest_files()

        self.assertEqual(files, [])

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_list_position_files_success(self, mock_client_class, mock_config):
        """Test successful listing of position files."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob1 = Mock()
        mock_blob1.name = 'positions/positions_20250614.csv'
        mock_blob2 = Mock()
        mock_blob2.name = 'positions/positions_20250613.csv'
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        cloud_storage = CloudStorage()

        files = cloud_storage.list_position_files()

        self.assertEqual(len(files), 2)
        self.assertIn('positions_20250614.csv', files)
        self.assertIn('positions_20250613.csv', files)

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_load_backtest_results_success(self, mock_client_class, mock_config):
        """Test successful loading of backtest results."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client

        # Mock JSON data
        test_data = [
            {
                'symbol': 'AAPL',
                'total_return': 0.15,
                'alpha': 0.05
            },
            {
                'symbol': 'TSLA',
                'total_return': 0.20,
                'alpha': 0.08
            }
        ]
        mock_blob.download_as_text.return_value = json.dumps(test_data)

        cloud_storage = CloudStorage()

        results = cloud_storage.load_backtest_results('test_file.json')

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['symbol'], 'AAPL')
        self.assertEqual(results[1]['symbol'], 'TSLA')

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_load_backtest_results_file_not_found(self, mock_client_class, mock_config):
        """Test loading backtest results when file doesn't exist."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client

        # Mock file not found
        mock_blob.download_as_text.side_effect = Exception("File not found")

        with patch('cloud_storage.logger') as mock_logger:
            cloud_storage = CloudStorage()

            results = cloud_storage.load_backtest_results(
                'nonexistent_file.json')

            self.assertEqual(results, [])
            mock_logger.error.assert_called()

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_load_position_entries_success(self, mock_client_class, mock_config):
        """Test successful loading of position entries."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_client_class.return_value = mock_client

        # Mock CSV data
        csv_data = "symbol,entry_price,stop_loss_price\nAAPL,150.0,140.0\nTSLA,800.0,750.0"
        mock_blob.download_as_text.return_value = csv_data

        cloud_storage = CloudStorage()

        result = cloud_storage.load_position_entries('test_file.csv')

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['symbol'], 'AAPL')
        self.assertEqual(result.iloc[1]['symbol'], 'TSLA')

    @patch('cloud_storage.config')
    @patch('cloud_storage.storage.Client')
    def test_delete_old_files_success(self, mock_client_class, mock_config):
        """Test successful deletion of old files."""
        # Setup mocks
        mock_config.GCS_BUCKET_NAME = 'test-bucket'
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob1 = Mock()
        mock_blob1.name = 'backtests/backtest_20250601.json'
        mock_blob1.time_created = datetime(2025, 6, 1)
        mock_blob2 = Mock()
        mock_blob2.name = 'backtests/backtest_20250614.json'
        mock_blob2.time_created = datetime(2025, 6, 14)
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_client.bucket.return_value = mock_bucket
        mock_client_class.return_value = mock_client

        cloud_storage = CloudStorage()

        deleted_count = cloud_storage.delete_old_files(days_to_keep=7)

        self.assertGreaterEqual(deleted_count, 0)
        # The old file should be deleted
        mock_blob1.delete.assert_called_once()


if __name__ == '__main__':
    unittest.main()
