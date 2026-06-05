#!/usr/bin/env python3
"""Regression tests for position reconciliation return values."""

import os
import sys
import unittest
from unittest.mock import Mock

import pandas as pd

# Add app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from positions import PositionsManager  # noqa: E402


class TestPositionsReconcileRegression(unittest.TestCase):
    """Ensure reconciliation always returns a list (never None)."""

    def test_reconcile_returns_empty_list_when_no_positions(self):
        mock_cloud_storage = Mock()
        mock_data_provider = Mock()

        mock_data_provider.get_current_positions_df.return_value = pd.DataFrame()
        mock_cloud_storage.get_latest_positions_df.return_value = pd.DataFrame()

        manager = PositionsManager(mock_cloud_storage, mock_data_provider)

        result = manager.get_and_reconcile_positions()

        self.assertIsInstance(result, list)
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
