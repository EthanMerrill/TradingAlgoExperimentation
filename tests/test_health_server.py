#!/usr/bin/env python3
"""Unit tests for the Flask health server and dashboard endpoints."""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

# Add app path before imports.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from health_server import create_app, _df_row_to_dict  # noqa: E402


class TestHealthEndpoint(unittest.TestCase):
    """Tests for the unauthenticated /health endpoint."""

    def setUp(self):
        self.app = create_app(shared_state={
            'last_result': {
                'status': 'success',
                'trading_summary': {'trades': 3, 'pnl': 150.0},
                'backtest_count': 42,
                'duration': 123.4,
            },
        })
        self.client = self.app.test_client()

    def test_health_returns_200(self):
        """GET /health should return 200 OK."""
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)

    def test_health_returns_json(self):
        """GET /health should return JSON with expected keys."""
        resp = self.client.get('/health')
        data = resp.get_json()
        self.assertEqual(data['status'], 'idle')
        self.assertEqual(data['last_run_status'], 'success')
        self.assertEqual(data['last_run_backtest_count'], 42)
        self.assertEqual(data['last_run_duration_seconds'], 123.4)

    def test_health_no_auth_required(self):
        """GET /health should work without any Authorization header."""
        resp = self.client.get('/health')
        self.assertEqual(resp.status_code, 200)

    def test_health_with_no_result_shows_running(self):
        """GET /health without shared_state should show 'running' status."""
        app = create_app(shared_state=None)
        client = app.test_client()
        resp = client.get('/health')
        data = resp.get_json()
        self.assertEqual(data['status'], 'running')


class TestApiPositionsEndpoint(unittest.TestCase):
    """Tests for the authenticated /api/positions endpoint."""

    def _build_df(self, rows):
        """Build a DataFrame with the storage schema from a list of dicts."""
        return pd.DataFrame(rows)

    def _make_open_row(self):
        return {
            'symbol': 'AAPL',
            'shares': 100.0,
            'entry_price': 150.0,
            'current_price': 155.0,
            'current_rsi': 45.0,
            'entry_date': pd.Timestamp('2025-06-14'),
            'alpha': 0.12,
            'rsi_period': 14,
            'rsi_lower': 30,
            'rsi_upper': 70,
            'stop_loss_price': 142.5,
            'take_profit_price': 172.5,
            'closed': False,
            'exit_date': pd.NaT,
            'exit_price': np.nan,
            'realized_return': np.nan,
        }

    def _make_closed_row(self):
        return {
            'symbol': 'TSLA',
            'shares': 50.0,
            'entry_price': 800.0,
            'current_price': 850.0,
            'current_rsi': 40.0,
            'entry_date': pd.Timestamp('2025-06-10'),
            'alpha': 0.08,
            'rsi_period': 14,
            'rsi_lower': 30,
            'rsi_upper': 70,
            'stop_loss_price': np.nan,
            'take_profit_price': np.nan,
            'closed': True,
            'exit_date': pd.Timestamp('2025-06-15'),
            'exit_price': 820.0,
            'realized_return': 0.025,
        }

    def setUp(self):
        self.open_row = self._make_open_row()
        self.closed_row = self._make_closed_row()
        self.full_df = self._build_df([self.open_row, self.closed_row])

        self.mock_storage = Mock()
        self.mock_storage.list_position_files.return_value = [
            'positions_20250710.csv']
        self.mock_storage.get_latest_position_file.return_value = 'positions_20250710.csv'
        self.mock_storage.load_position_entries.return_value = self.full_df

        self.app = create_app(storage_backend=self.mock_storage)
        self.client = self.app.test_client()

        self._orig_password = os.environ.get('DASHBOARD_PASSWORD')
        os.environ['DASHBOARD_PASSWORD'] = 'testpass'

    def tearDown(self):
        if self._orig_password is not None:
            os.environ['DASHBOARD_PASSWORD'] = self._orig_password
        else:
            os.environ.pop('DASHBOARD_PASSWORD', None)

    def _auth_headers(self):
        import base64
        credentials = base64.b64encode(b'admin:testpass').decode('utf-8')
        return {'Authorization': f'Basic {credentials}'}

    # ── Auth tests ──

    def test_positions_returns_401_without_auth(self):
        """GET /api/positions without auth should return 401."""
        resp = self.client.get('/api/positions')
        self.assertEqual(resp.status_code, 401)

    def test_positions_returns_401_with_wrong_password(self):
        """GET /api/positions with wrong password should return 401."""
        import base64
        creds = base64.b64encode(b'admin:wrongpass').decode('utf-8')
        resp = self.client.get('/api/positions',
                               headers={'Authorization': f'Basic {creds}'})
        self.assertEqual(resp.status_code, 401)

    def test_positions_returns_401_with_wrong_username(self):
        """GET /api/positions with wrong username should return 401."""
        import base64
        creds = base64.b64encode(b'user:testpass').decode('utf-8')
        resp = self.client.get('/api/positions',
                               headers={'Authorization': f'Basic {creds}'})
        self.assertEqual(resp.status_code, 401)

    def test_positions_returns_200_with_valid_auth(self):
        """GET /api/positions with valid auth should return 200."""
        resp = self.client.get('/api/positions', headers=self._auth_headers())
        self.assertEqual(resp.status_code, 200)

    def test_positions_returns_503_when_password_unset(self):
        """GET /api/positions should return 503 when DASHBOARD_PASSWORD is unset."""
        os.environ.pop('DASHBOARD_PASSWORD', None)
        resp = self.client.get('/api/positions')
        self.assertEqual(resp.status_code, 503)
        data = resp.get_json()
        self.assertIn('DASHBOARD_PASSWORD', data.get('error', ''))

    # ── Data tests ──

    def test_positions_returns_array(self):
        """GET /api/positions should return a JSON array."""
        resp = self.client.get('/api/positions', headers=self._auth_headers())
        data = resp.get_json()
        self.assertIsInstance(data, list)

    def test_positions_returns_all(self):
        """GET /api/positions should return all positions."""
        resp = self.client.get('/api/positions', headers=self._auth_headers())
        data = resp.get_json()
        self.assertEqual(len(data), 2)

    def test_positions_open_and_closed_both_present(self):
        """All positions should be present regardless of open/closed status."""
        resp = self.client.get('/api/positions', headers=self._auth_headers())
        data = resp.get_json()
        symbols = [row['symbol'] for row in data]
        self.assertIn('AAPL', symbols)
        self.assertIn('TSLA', symbols)

    def test_positions_has_expected_fields(self):
        """Each position should have the expected fields."""
        resp = self.client.get('/api/positions',
                               headers=self._auth_headers())
        data = resp.get_json()
        pos = next(p for p in data if p['symbol'] == 'AAPL')
        self.assertEqual(pos['symbol'], 'AAPL')
        self.assertEqual(pos['quantity'], 100.0)
        self.assertEqual(pos['entry_price'], 150.0)
        self.assertEqual(pos['current_price'], 155.0)
        self.assertEqual(pos['current_rsi'], 45.0)
        self.assertEqual(pos['side'], 'long')
        self.assertFalse(pos['closed'])

    def test_positions_closed_has_exit_fields(self):
        """Closed positions should include exit-related fields."""
        resp = self.client.get('/api/positions',
                               headers=self._auth_headers())
        data = resp.get_json()
        # Find the closed position (TSLA)
        pos = next(p for p in data if p['symbol'] == 'TSLA')
        self.assertTrue(pos['closed'])
        self.assertEqual(pos['symbol'], 'TSLA')
        self.assertIsNotNone(pos['exit_date'])
        self.assertEqual(pos['exit_price'], 820.0)
        self.assertEqual(pos['realized_return'], 0.025)

    def test_positions_serializes_datetimes(self):
        """Datetime fields should be serialized as ISO 8601 strings."""
        resp = self.client.get('/api/positions',
                               headers=self._auth_headers())
        data = resp.get_json()
        pos = data[0]
        self.assertIsInstance(pos['entry_date'], str)
        # Should be parseable back to datetime
        parsed = datetime.fromisoformat(pos['entry_date'])
        self.assertIsInstance(parsed, datetime)
        parsed = datetime.fromisoformat(pos['entry_date'])
        self.assertIsInstance(parsed, datetime)

    def test_positions_none_storage_returns_503(self):
        """When storage_backend is None, return 503."""
        app = create_app(storage_backend=None)
        client = app.test_client()
        os.environ['DASHBOARD_PASSWORD'] = 'testpass'
        import base64
        creds = base64.b64encode(b'admin:testpass').decode('utf-8')
        resp = client.get('/api/positions',
                          headers={'Authorization': f'Basic {creds}'})
        self.assertEqual(resp.status_code, 503)


class TestDashboardHtmlEndpoint(unittest.TestCase):
    """Tests for the dashboard HTML endpoint."""

    def setUp(self):
        # Frontend dir isn't available during tests, but we can test auth
        self._orig_password = os.environ.get('DASHBOARD_PASSWORD')

    def tearDown(self):
        if self._orig_password is not None:
            os.environ['DASHBOARD_PASSWORD'] = self._orig_password
        else:
            os.environ.pop('DASHBOARD_PASSWORD', None)

    def test_index_returns_401_without_auth(self):
        """GET / without auth should return 401."""
        os.environ['DASHBOARD_PASSWORD'] = 'testpass'
        app = create_app()
        client = app.test_client()
        resp = client.get('/')
        self.assertEqual(resp.status_code, 401)

    def test_index_returns_503_when_password_unset(self):
        """GET / should return 503 when DASHBOARD_PASSWORD is unset."""
        os.environ.pop('DASHBOARD_PASSWORD', None)
        app = create_app()
        client = app.test_client()
        resp = client.get('/')
        self.assertEqual(resp.status_code, 503)


class TestDfRowToDict(unittest.TestCase):
    """Tests for the _df_row_to_dict helper."""

    def test_normalizes_shares_to_quantity(self):
        """Storage column 'shares' should be mapped to 'quantity'."""
        row = pd.Series({
            'symbol': 'AAPL', 'shares': 100.0, 'entry_price': 150.0,
            'current_price': 155.0, 'current_rsi': 45.0,
            'entry_date': pd.Timestamp('2025-06-14'),
            'alpha': 0.12, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': 142.5, 'take_profit_price': 172.5,
            'closed': False,
            'exit_date': pd.NaT, 'exit_price': np.nan, 'realized_return': np.nan,
        })
        result = _df_row_to_dict(row)
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['quantity'], 100.0)
        self.assertNotIn('shares', result)
        self.assertEqual(result['side'], 'long')
        self.assertIsInstance(result['entry_date'], str)

    def test_derives_short_side(self):
        """Negative shares should derive side='short'."""
        row = pd.Series({
            'symbol': 'TSLA', 'shares': -50.0, 'entry_price': 800.0,
            'current_price': 780.0, 'current_rsi': 65.0,
            'entry_date': pd.Timestamp('2025-06-14'),
            'alpha': 0.08, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': np.nan, 'take_profit_price': np.nan,
            'closed': False,
            'exit_date': pd.NaT, 'exit_price': np.nan, 'realized_return': np.nan,
        })
        result = _df_row_to_dict(row)
        self.assertEqual(result['side'], 'short')
        self.assertEqual(result['quantity'], -50.0)

    def test_preserves_existing_side(self):
        """If 'side' is already in the row, keep it."""
        row = pd.Series({
            'symbol': 'MSFT', 'shares': 200.0, 'side': 'long',
            'entry_price': 400.0, 'current_price': 420.0, 'current_rsi': 55.0,
            'entry_date': pd.Timestamp('2025-05-01'),
            'alpha': 0.15, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': np.nan, 'take_profit_price': np.nan,
            'closed': True, 'exit_date': pd.Timestamp('2025-06-01'),
            'exit_price': 430.0, 'realized_return': 0.075,
        })
        result = _df_row_to_dict(row)
        self.assertEqual(result['side'], 'long')
        self.assertTrue(result['closed'])

    def test_handles_nan_values(self):
        """NaN values should be converted to None."""
        row = pd.Series({
            'symbol': 'XYZ', 'shares': 10.0, 'entry_price': 50.0,
            'current_price': 55.0, 'current_rsi': 30.0,
            'entry_date': pd.Timestamp('2025-01-01'),
            'alpha': np.nan, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': np.nan, 'take_profit_price': np.nan,
            'closed': False, 'exit_date': pd.NaT,
            'exit_price': np.nan, 'realized_return': np.nan,
        })
        result = _df_row_to_dict(row)
        self.assertIsNone(result['alpha'])
        self.assertIsNone(result['stop_loss_price'])
        self.assertIsNone(result['exit_price'])

    def test_adds_exit_reason_when_missing(self):
        """exit_reason should default to None when absent (GCS doesn't store it)."""
        row = pd.Series({
            'symbol': 'ABC', 'shares': 1.0, 'entry_price': 10.0,
            'current_price': 12.0, 'current_rsi': 50.0,
            'entry_date': pd.Timestamp('2025-01-01'),
            'alpha': 0.0, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': np.nan, 'take_profit_price': np.nan,
            'closed': False, 'exit_date': pd.NaT,
            'exit_price': np.nan, 'realized_return': np.nan,
        })
        result = _df_row_to_dict(row)
        self.assertIsNone(result['exit_reason'])

    def test_closed_string_to_bool_true(self):
        """CSV string 'True' should become Python bool True."""
        row = pd.Series({
            'symbol': 'AAPL', 'shares': 100.0, 'entry_price': 150.0,
            'current_price': 155.0, 'current_rsi': 45.0,
            'entry_date': pd.Timestamp('2025-06-14'),
            'alpha': 0.12, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': 142.5, 'take_profit_price': 172.5,
            'closed': 'True', 'exit_date': pd.NaT,
            'exit_price': np.nan, 'realized_return': np.nan,
        })
        result = _df_row_to_dict(row)
        self.assertIs(result['closed'], True)
        self.assertIsInstance(result['closed'], bool)

    def test_closed_string_to_bool_false(self):
        """CSV string 'False' should become Python bool False."""
        row = pd.Series({
            'symbol': 'AAPL', 'shares': 100.0, 'entry_price': 150.0,
            'current_price': 155.0, 'current_rsi': 45.0,
            'entry_date': pd.Timestamp('2025-06-14'),
            'alpha': 0.12, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': 142.5, 'take_profit_price': 172.5,
            'closed': 'False', 'exit_date': pd.NaT,
            'exit_price': np.nan, 'realized_return': np.nan,
        })
        result = _df_row_to_dict(row)
        self.assertIs(result['closed'], False)
        self.assertIsInstance(result['closed'], bool)

    def test_closed_int_to_bool(self):
        """Int 0/1 values should become bool."""
        row = pd.Series({
            'symbol': 'AAPL', 'shares': 100.0, 'entry_price': 150.0,
            'current_price': 155.0, 'current_rsi': 45.0,
            'entry_date': pd.Timestamp('2025-06-14'),
            'alpha': 0.12, 'rsi_period': 14, 'rsi_lower': 30, 'rsi_upper': 70,
            'stop_loss_price': 142.5, 'take_profit_price': 172.5,
            'closed': 1, 'exit_date': pd.NaT,
            'exit_price': np.nan, 'realized_return': np.nan,
        })
        result = _df_row_to_dict(row)
        self.assertIs(result['closed'], True)


if __name__ == '__main__':
    unittest.main()
