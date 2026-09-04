#!/usr/bin/env python3
"""Unit tests for the PostgresStorage backend."""
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


# -- mock pool helpers --------------------------------------------------------

def _make_mock_pool():
    """Return (conn, pool) where conn is an AsyncMock and pool.acquire()
    returns a real async context manager (so ``async with`` works)."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock(return_value="OK")
    conn.executemany = AsyncMock(return_value=None)

    class _FakeAcquire:
        def __init__(self, c): self._c = c
        async def __aenter__(self): return self._c
        async def __aexit__(self, *args): pass

    pool = Mock()
    pool.acquire = Mock(return_value=_FakeAcquire(conn))
    return conn, pool


# -- tests --------------------------------------------------------------------

class TestPostgresStorage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if "asyncpg" not in sys.modules:
            sys.modules["asyncpg"] = Mock()

    def setUp(self):
        from storage.postgres import PostgresStorage
        self.PostgresStorage = PostgresStorage
        self._conn, self._pool = _make_mock_pool()

    def _connected(self, env="dev"):
        s = self.PostgresStorage.__new__(self.PostgresStorage)
        s._dsn = "postgresql://u:p@h/db"
        s._env = env
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

    # --- init (real constructor) ---

    @patch("config.globalConfig")
    def test_init_missing_database_url(self, cfg):
        cfg.DATABASE_URL = ""
        cfg.ENVIRONMENT = "dev"
        s = self.PostgresStorage()
        self.assertFalse(s._connected)
        self.assertIsNone(s._pool)

    @patch("config.globalConfig")
    def test_init_success(self, cfg):
        cfg.DATABASE_URL = "postgresql://u:p@h/db"
        cfg.ENVIRONMENT = "qa"
        sys.modules["asyncpg"].create_pool = AsyncMock(return_value=self._pool)
        s = self.PostgresStorage(database_url="postgresql://u:p@h/db")
        self.assertTrue(s._connected)
        self._conn.execute.assert_called()

    # --- save_backtest_results ---

    def test_save_bt_disconnected(self):
        self.assertFalse(self._disconnected().save_backtest_results([]))

    def test_save_bt_success(self):
        from strategy import BacktestResult
        s = self._connected()
        r = [BacktestResult(symbol="AAPL", rsi_period=14, rsi_lower=30,
             rsi_upper=70, total_return=0.15, buy_and_hold_return=0.1, alpha=0.05,
             num_trades=5, win_rate=0.6, avg_trade_duration=10.5, max_drawdown=0.08,
             sharpe_ratio=1.2, profitable=True, current_rsi=45.0)]
        self.assertTrue(s.save_backtest_results(r, "20250610_170000"))
        self._conn.executemany.assert_called()

    def test_backtest_result_to_dict_sanitizes_nan(self):
        # Non-finite floats must be normalized to None at write time so they
        # never reach the DB (and later the browser) as NaN/Infinity.
        import math
        from storage.backend import backtest_result_to_dict
        from strategy import BacktestResult
        r = BacktestResult(symbol="AAPL", rsi_period=14, rsi_lower=30,
             rsi_upper=70, total_return=math.nan,
             buy_and_hold_return=math.inf, alpha=0.05, num_trades=5,
             win_rate=0.6, avg_trade_duration=-math.inf,
             max_drawdown=0.08, sharpe_ratio=math.nan, profitable=True,
             current_rsi=math.nan)
        d = backtest_result_to_dict(r)
        self.assertIsNone(d["total_return"])
        self.assertIsNone(d["buy_and_hold_return"])
        self.assertIsNone(d["avg_trade_duration"])
        self.assertIsNone(d["sharpe_ratio"])
        self.assertIsNone(d["current_rsi"])
        # finite values are preserved
        self.assertEqual(d["alpha"], 0.05)
        self.assertEqual(d["win_rate"], 0.6)
        self.assertEqual(d["max_drawdown"], 0.08)

    # --- load_backtest_results ---

    def test_load_bt_disconnected(self):
        self.assertEqual(self._disconnected(
        ).load_backtest_results("f.csv"), [])

    def test_load_bt_success(self):
        self._conn.fetch = AsyncMock(return_value=[dict(
            symbol="AAPL", rsi_period=14, rsi_lower=30, rsi_upper=70,
            total_return=0.15, buy_and_hold_return=0.1, alpha=0.05,
            num_trades=5, win_rate=0.6, avg_trade_duration=10.5,
            max_drawdown=0.08, sharpe_ratio=1.2, calmar_ratio=1.5,
            composite_score=2.0, direction="long", profitable=True, current_rsi=45.0)])
        results = self._connected().load_backtest_results(
            "backtest_results_20250610_170000.csv")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].symbol, "AAPL")

    def test_load_bt_empty(self):
        self._conn.fetch = AsyncMock(return_value=[])
        self.assertEqual(self._connected().load_backtest_results("x.csv"), [])

    # --- save_positions ---

    def test_save_pos_disconnected(self):
        self.assertFalse(self._disconnected().save_positions([]))

    def test_save_pos_empty(self):
        self.assertTrue(self._connected().save_positions([]))

    def test_save_pos_success(self):
        from positions import Position
        s = self._connected()
        pos = [Position(symbol="AAPL", quantity=10.0, entry_price=150.0,
                        current_price=151.0, current_rsi=45.0,
                        entry_date=datetime.now(timezone.utc), alpha=0.05,
                        rsi_period=14, rsi_lower=30, rsi_upper=70,
                        stop_loss_price=140.0, take_profit_price=160.0, closed=False)]
        self.assertTrue(s.save_positions(pos, timestamp="20250610_170000"))
        self._conn.executemany.assert_called()

    # --- save_orders ---

    def test_save_orders_disconnected(self):
        self.assertFalse(self._disconnected().save_orders([]))

    def test_save_orders_success_upsert(self):
        from order import Order
        s = self._connected()
        orders = [Order(client_order_id="AAPL-BUY-1", symbol="AAPL",
                        side="buy", qty=5.0, status="new", leg="entry")]
        self.assertTrue(s.save_orders(orders))
        self._conn.executemany.assert_called()
        sql = self._conn.executemany.call_args[0][0]
        self.assertIn("ON CONFLICT (environment, client_order_id)", sql)

    def test_save_orders_empty(self):
        self.assertTrue(self._connected().save_orders([]))

    # --- load_orders ---

    def test_load_orders_disconnected(self):
        self.assertEqual(self._disconnected().load_orders(), [])

    def test_load_orders_success(self):
        self._conn.fetch = AsyncMock(return_value=[dict(
            client_order_id="AAPL-BUY-1", order_id="o1", symbol="AAPL",
            side="buy", qty=5.0, order_type="market", order_class="bracket",
            status="new", stop_price=None, limit_price=None,
            submitted_at=None, filled_at=None, leg="entry")])
        orders = self._connected().load_orders(symbol="AAPL")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].symbol, "AAPL")
        self.assertEqual(orders[0].client_order_id, "AAPL-BUY-1")
        self.assertEqual(orders[0].status, "new")

    # --- save_metadata ---

    def test_save_md_disconnected(self):
        self.assertFalse(self._disconnected().save_metadata({}))

    def test_save_md_success(self):
        s = self._connected()
        self.assertTrue(s.save_metadata(
            {"equity": 10000.1, "n": 3}, timestamp="t"))
        self._conn.execute.assert_called()

    # --- list_backtest_files ---

    def test_list_bt_disconnected(self):
        self.assertEqual(self._disconnected().list_backtest_files(), [])

    def test_list_bt_success(self):
        self._conn.fetch = AsyncMock(return_value=[
            {"run_timestamp": "20250614_120000"}, {"run_timestamp": "20250613_110000"}])
        files = self._connected().list_backtest_files()
        self.assertEqual(len(files), 2)
        self.assertIn("backtest_results_20250614_120000.csv", files)

    # --- list_position_files ---

    def test_list_pos_success(self):
        self._conn.fetch = AsyncMock(
            return_value=[{"snapshot_timestamp": "20250614_120000"}])
        files = self._connected().list_position_files()
        self.assertEqual(files, ["positions_20250614_120000.csv"])

    # --- load_position_entries ---

    def test_load_pos_success(self):
        self._conn.fetch = AsyncMock(return_value=[dict(
            symbol="AAPL", shares=10.0, entry_price=150.0, current_price=151.0,
            current_rsi=45.0, entry_date=datetime(2025, 6, 10, tzinfo=timezone.utc),
            rsi_period=14, rsi_lower=30, rsi_upper=70, alpha=0.05,
            stop_loss_price=140.0, take_profit_price=160.0, closed=False,
            exit_date=None, exit_price=None, realized_return=None, side="long")])
        df = self._connected().load_position_entries("positions_20250610_170000.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "AAPL")

    def test_load_pos_empty(self):
        self._conn.fetch = AsyncMock(return_value=[])
        self.assertTrue(self._connected().load_position_entries("x.csv").empty)

    def test_load_pos_parses_dates(self):
        self._conn.fetch = AsyncMock(return_value=[dict(
            symbol="AAPL", shares=10.0, entry_price=150.0, current_price=151.0,
            current_rsi=45.0, entry_date=datetime(2025, 6, 7, tzinfo=timezone.utc),
            rsi_period=14, rsi_lower=30, rsi_upper=70, alpha=0.05,
            stop_loss_price=None, take_profit_price=None, closed=False,
            exit_date=None, exit_price=None, realized_return=None, side="long")])
        df = self._connected().load_position_entries("positions_20250610_170000.csv")
        self.assertIsInstance(df.iloc[0]["entry_date"], datetime)

    # --- get_latest_position_file ---

    def test_latest_pos_file(self):
        self._conn.fetch = AsyncMock(
            return_value=[{"snapshot_timestamp": "20250614_120000"}])
        self.assertEqual(self._connected().get_latest_position_file(),
                         "positions_20250614_120000.csv")

    def test_latest_pos_file_none(self):
        self._conn.fetch = AsyncMock(return_value=[])
        self.assertIsNone(self._connected().get_latest_position_file())

    # --- get_latest_positions_df ---

    def test_latest_df_open(self):
        cnt = [0]

        async def side(*a, **kw):
            cnt[0] += 1
            if cnt[0] == 1:
                return [{"snapshot_timestamp": "t"}]
            return [dict(symbol="AAPL", shares=10.0, entry_price=150.0,
                         current_price=151.0, current_rsi=45.0,
                         entry_date=datetime(2025, 6, 10, tzinfo=timezone.utc),
                         rsi_period=14, rsi_lower=30, rsi_upper=70, alpha=0.05,
                         stop_loss_price=140.0, take_profit_price=160.0, closed=False,
                         exit_date=None, exit_price=None, realized_return=None, side="long")]
        self._conn.fetch = AsyncMock(side_effect=side)
        df = self._connected().get_latest_positions_df(openPosition=True)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "AAPL")

    def test_latest_df_closed(self):
        cnt = [0]

        async def side(*a, **kw):
            cnt[0] += 1
            if cnt[0] == 1:
                return [{"snapshot_timestamp": "t"}]
            return [
                dict(symbol="AAPL", shares=10.0, entry_price=150.0,
                     current_price=151.0, current_rsi=45.0,
                     entry_date=datetime(2025, 6, 10, tzinfo=timezone.utc),
                     rsi_period=14, rsi_lower=30, rsi_upper=70, alpha=0.05,
                     stop_loss_price=140.0, take_profit_price=160.0, closed=False,
                     exit_date=None, exit_price=None, realized_return=None, side="long"),
                dict(symbol="TSLA", shares=5.0, entry_price=800.0,
                     current_price=810.0, current_rsi=40.0,
                     entry_date=datetime(2025, 6, 9, tzinfo=timezone.utc),
                     rsi_period=14, rsi_lower=30, rsi_upper=70, alpha=0.08,
                     stop_loss_price=750.0, take_profit_price=850.0, closed=True,
                     exit_date=datetime(2025, 6, 14, tzinfo=timezone.utc),
                     exit_price=820.0, realized_return=0.025, side="long"),
            ]
        self._conn.fetch = AsyncMock(side_effect=side)
        df = self._connected().get_latest_positions_df(openPosition=False)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "TSLA")

    # --- ABC compliance ---

    def test_is_storage_backend(self):
        from storage.backend import StorageBackend
        self.assertTrue(issubclass(self.PostgresStorage, StorageBackend))

    # --- _filename_to_timestamp ---

    def test_filename_to_timestamp(self):
        from storage.postgres import _filename_to_timestamp
        self.assertEqual(_filename_to_timestamp("backtest_results_20250610_170343.csv"),
                         "20250610_170343")
        self.assertEqual(_filename_to_timestamp("positions_20250610_170343.csv"),
                         "20250610_170343")


if __name__ == "__main__":
    unittest.main()
