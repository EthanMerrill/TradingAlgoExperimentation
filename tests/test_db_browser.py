#!/usr/bin/env python3
"""Tests for the dashboard DB-browser endpoints + storage browse methods."""
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestPostgresDbBrowse(unittest.TestCase):
    """PostgresStorage.db_list_tables / db_fetch_table."""

    @classmethod
    def setUpClass(cls):
        if "asyncpg" not in sys.modules:
            sys.modules["asyncpg"] = Mock()

    def setUp(self):
        from storage.postgres import PostgresStorage
        self.PostgresStorage = PostgresStorage

        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        conn.execute = AsyncMock(return_value="OK")
        conn.executemany = AsyncMock(return_value=None)

        class _FakeAcquire:
            def __init__(self, c):
                self._c = c
            async def __aenter__(self):
                return self._c
            async def __aexit__(self, *args):
                pass

        pool = Mock()
        pool.acquire = Mock(return_value=_FakeAcquire(conn))
        self._conn = conn
        self._pool = pool

    def _connected(self):
        s = self.PostgresStorage.__new__(self.PostgresStorage)
        s._dsn = "postgresql://u:p@h/db"
        s._env = "dev"
        s._connected = True
        s._pool = self._pool
        return s

    def _disconnected(self):
        s = self.PostgresStorage.__new__(self.PostgresStorage)
        s._dsn = ""
        s._env = "dev"
        s._connected = False
        s._pool = None
        return s

    def test_disconnected_disables_browse(self):
        s = self._disconnected()
        self.assertFalse(s.db_browse_enabled())
        self.assertEqual(s.db_list_tables(), [])
        with self.assertRaises(ValueError):
            s.db_fetch_table("backtest_results")

    def test_db_list_tables(self):
        self._conn.fetch = AsyncMock(return_value=[
            dict(table_name="backtest_results"),
            dict(table_name="orders"),
            dict(table_name="position_snapshots"),
        ])
        s = self._connected()
        self.assertTrue(s.db_browse_enabled())
        self.assertEqual(
            s.db_list_tables(),
            ["backtest_results", "orders", "position_snapshots"])

    def test_db_fetch_table_rejects_unknown(self):
        # db_list_tables returns nothing → the name is not allowlisted
        self._conn.fetch = AsyncMock(return_value=[])
        s = self._connected()
        with self.assertRaises(ValueError):
            s.db_fetch_table("users")

    def test_db_fetch_table_success(self):
        # Sequential fetch responses: list tables → count → rows
        self._conn.fetch = AsyncMock(side_effect=[
            [dict(table_name="backtest_results")],
            [dict(n=2)],
            [dict(symbol="AAPL", total_return=0.15,
                  created_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))],
        ])
        s = self._connected()
        result = s.db_fetch_table("backtest_results", limit=10, offset=0)
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["columns"], ["symbol", "total_return", "created_at"])
        self.assertEqual(result["rows"][0]["symbol"], "AAPL")
        # datetimes must be JSON-safe (ISO strings)
        self.assertIsInstance(result["rows"][0]["created_at"], str)
        self.assertEqual(result["limit"], 10)
        self.assertEqual(result["offset"], 0)


class TestHealthDbEndpoints(unittest.TestCase):
    """/api/db/tables + /api/db/table/<name> (auth required)."""

    def setUp(self):
        self._orig_password = os.environ.get('DASHBOARD_PASSWORD')
        os.environ['DASHBOARD_PASSWORD'] = 'testpass'

    def tearDown(self):
        if self._orig_password is not None:
            os.environ['DASHBOARD_PASSWORD'] = self._orig_password
        else:
            os.environ.pop('DASHBOARD_PASSWORD', None)

    def _auth_headers(self):
        import base64
        creds = base64.b64encode(b'admin:testpass').decode('utf-8')
        return {'Authorization': f'Basic {creds}'}

    def _make_app(self, storage_backend):
        from health_server import create_app
        return create_app(storage_backend=storage_backend,
                          shared_state={'last_result': None})

    def test_tables_requires_auth(self):
        from health_server import create_app
        app = create_app(storage_backend=Mock())
        resp = app.test_client().get('/api/db/tables')
        self.assertEqual(resp.status_code, 401)

    def test_tables_unsupported_backend(self):
        storage_backend = Mock()
        storage_backend.db_browse_enabled.return_value = False
        client = self._make_app(storage_backend).test_client()
        resp = client.get('/api/db/tables', headers=self._auth_headers())
        self.assertEqual(resp.status_code, 501)
        data = resp.get_json()
        self.assertFalse(data['enabled'])

    def test_tables_supported(self):
        storage_backend = Mock()
        storage_backend.db_browse_enabled.return_value = True
        storage_backend.db_list_tables.return_value = ['backtest_results', 'orders']
        client = self._make_app(storage_backend).test_client()
        resp = client.get('/api/db/tables', headers=self._auth_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['enabled'])
        self.assertEqual(data['tables'], ['backtest_results', 'orders'])

    def test_table_unsupported_backend(self):
        storage_backend = Mock()
        storage_backend.db_browse_enabled.return_value = False
        client = self._make_app(storage_backend).test_client()
        resp = client.get('/api/db/table/backtest_results',
                          headers=self._auth_headers())
        self.assertEqual(resp.status_code, 501)

    def test_table_success(self):
        storage_backend = Mock()
        storage_backend.db_browse_enabled.return_value = True
        storage_backend.db_fetch_table.return_value = {
            'table': 'backtest_results', 'columns': ['symbol'],
            'rows': [{'symbol': 'AAPL'}], 'total': 1,
            'limit': 100, 'offset': 0,
        }
        client = self._make_app(storage_backend).test_client()
        resp = client.get('/api/db/table/backtest_results',
                          headers=self._auth_headers())
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['rows'], [{'symbol': 'AAPL'}])

    def test_table_unknown_returns_400(self):
        storage_backend = Mock()
        storage_backend.db_browse_enabled.return_value = True
        storage_backend.db_fetch_table.side_effect = ValueError("Unknown table")
        client = self._make_app(storage_backend).test_client()
        resp = client.get('/api/db/table/nope',
                          headers=self._auth_headers())
        self.assertEqual(resp.status_code, 400)

    def test_table_invalid_limit_returns_400(self):
        storage_backend = Mock()
        storage_backend.db_browse_enabled.return_value = True
        client = self._make_app(storage_backend).test_client()
        resp = client.get('/api/db/table/backtest_results?limit=abc',
                          headers=self._auth_headers())
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
