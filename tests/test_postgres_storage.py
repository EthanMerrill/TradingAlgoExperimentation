#!/usr/bin/env python3
"""Unit tests for the PostgresStorage backend."""
import json
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
        from strategies.base import BacktestResult
        s = self._connected()
        r = [BacktestResult(symbol="AAPL", params={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70, "current_rsi": 45.0}, total_return=0.15, buy_and_hold_return=0.1, alpha=0.05,
             num_trades=5, win_rate=0.6, avg_trade_duration=10.5, max_drawdown=0.08,
             sharpe_ratio=1.2, profitable=True)]
        self.assertTrue(s.save_backtest_results(r, "20250610_170000"))
        self._conn.executemany.assert_called()

    # --- prune_backtest_results ---

    def test_prune_disconnected(self):
        self.assertEqual(self._disconnected().prune_backtest_results(30), 0)

    def test_prune_disabled_when_zero_or_negative(self):
        s = self._connected()
        self.assertEqual(s.prune_backtest_results(0), 0)
        self.assertEqual(s.prune_backtest_results(-1), 0)
        # A disabled retention must not even hit the database.
        self._conn.fetch.assert_not_called()

    def test_prune_deletes_and_returns_count(self):
        s = self._connected()
        self._conn.fetch = AsyncMock(return_value=[{"n": 7}])
        self.assertEqual(s.prune_backtest_results(30), 7)
        sql = self._conn.fetch.call_args[0][0]
        self.assertIn("DELETE FROM backtest_results", sql)
        self.assertIn("RETURNING 1", sql)

    def test_prune_is_environment_scoped(self):
        s = self._connected(env="qa")
        self._conn.fetch = AsyncMock(return_value=[{"n": 1}])
        s.prune_backtest_results(30)
        args = self._conn.fetch.call_args[0][1:]
        self.assertEqual(args[0], "qa")

    def test_prune_uses_current_window_cutoff(self):
        from datetime import datetime, timedelta
        from storage.backend import retention_cutoff_timestamp
        s = self._connected()
        self._conn.fetch = AsyncMock(return_value=[{"n": 0}])
        s.prune_backtest_results(30)
        cutoff = self._conn.fetch.call_args[0][2]
        expected = retention_cutoff_timestamp(30)
        # Allow for a second boundary crossing between the two calls.
        self.assertIn(
            cutoff,
            {expected,
             retention_cutoff_timestamp(
                 30, now=datetime.now() + timedelta(seconds=1)),
             retention_cutoff_timestamp(30, now=datetime.now() - timedelta(seconds=1))},
        )

    def test_prune_swallows_db_errors(self):
        s = self._connected()
        self._conn.fetch = AsyncMock(side_effect=RuntimeError("boom"))
        self.assertEqual(s.prune_backtest_results(30), 0)

    def test_backtest_result_to_dict_sanitizes_nan(self):
        # Non-finite floats must be normalized to None at write time so they
        # never reach the DB (and later the browser) as NaN/Infinity.
        import math
        from storage.backend import backtest_result_to_dict
        from strategies.base import BacktestResult
        r = BacktestResult(symbol="AAPL",
                           params={"rsi_period": 14, "rsi_lower": 30,
                                   "rsi_upper": 70, "current_rsi": math.nan},
                           total_return=math.nan,
                           buy_and_hold_return=math.inf, alpha=0.05, num_trades=5,
                           win_rate=0.6, avg_trade_duration=-math.inf,
                           max_drawdown=0.08, sharpe_ratio=math.nan, profitable=True)
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
            symbol="AAPL", params={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70},
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

    def test_position_schema_includes_exit_reason(self):
        """exit_reason must be a first-class, persisted position column."""
        from storage.postgres import _POSITION_COLS, _DDL_POSITION_SNAPSHOTS
        self.assertIn("exit_reason", _POSITION_COLS)
        self.assertIn("exit_reason", _DDL_POSITION_SNAPSHOTS)
        # Also needs a migration so pre-existing tables gain the column.
        self.assertIn(
            "ADD COLUMN IF NOT EXISTS exit_reason", _DDL_POSITION_SNAPSHOTS)

    def test_save_positions_persists_exit_reason(self):
        """The exit_reason value must reach the INSERT tuple (not dropped)."""
        from positions import Position
        from storage.postgres import _POSITION_COLS
        s = self._connected()
        pos = [Position(
            symbol="AAPL", quantity=10.0, entry_price=150.0,
            current_price=151.0, current_rsi=45.0,
            entry_date=datetime.now(timezone.utc), alpha=0.05,
            rsi_period=14, rsi_lower=30, rsi_upper=70,
            closed=True, exit_price=160.0, realized_return=0.0667,
            exit_reason="oco_take_profit")]
        self.assertTrue(s.save_positions(pos, timestamp="20250610_170000"))

        rows = self._conn.executemany.call_args[0][1]
        # Tuple shape is (snapshot_timestamp, environment, *_POSITION_COLS)
        idx = _POSITION_COLS.index("exit_reason") + 2
        self.assertEqual(rows[0][idx], "oco_take_profit")

    # --- position_metadata / closed_trades ---

    def test_new_tables_in_ddl(self):
        from storage.postgres import _ALL_DDL
        self.assertIn("CREATE TABLE IF NOT EXISTS position_metadata", _ALL_DDL)
        self.assertIn("UNIQUE (environment, symbol)", _ALL_DDL)
        self.assertIn("CREATE TABLE IF NOT EXISTS closed_trades", _ALL_DDL)
        self.assertIn("UNIQUE (environment, dedup_key)", _ALL_DDL)
        self.assertIn("exit_price_estimated", _ALL_DDL)

    @patch("config.globalConfig")
    def test_init_runs_idempotent_migration_for_environment(self, cfg):
        cfg.DATABASE_URL = "postgresql://u:p@h/db"
        cfg.ENVIRONMENT = "qa"
        sys.modules["asyncpg"].create_pool = AsyncMock(return_value=self._pool)
        self.PostgresStorage(database_url="postgresql://u:p@h/db")
        sqls = [c[0][0] for c in self._conn.execute.call_args_list]
        seeds = [q for q in sqls if "INSERT INTO position_metadata" in q
                 or "INSERT INTO closed_trades" in q]
        self.assertEqual(len(seeds), 2)
        for q in seeds:
            # seeds only into an empty table, from the latest snapshot
            self.assertIn("NOT EXISTS (SELECT 1 FROM", q)
            self.assertIn("MAX(snapshot_timestamp)", q)
            self.assertIn("ON CONFLICT", q)
        env_args = [c[0][1] for c in self._conn.execute.call_args_list
                    if len(c[0]) > 1]
        self.assertEqual(env_args, ["qa", "qa"])
        meta_sql = next(q for q in seeds if "INTO position_metadata" in q)
        self.assertIn("COALESCE(closed, FALSE) = FALSE", meta_sql)
        closed_sql = next(q for q in seeds if "INTO closed_trades" in q)
        self.assertIn("closed IS TRUE", closed_sql)
        self.assertIn("(exit_price IS NULL)", closed_sql)

    @patch("config.globalConfig")
    def test_migration_failure_does_not_block_startup(self, cfg):
        cfg.DATABASE_URL = "postgresql://u:p@h/db"
        cfg.ENVIRONMENT = "dev"

        async def _exec(sql, *args):
            if args:
                raise RuntimeError("boom")
            return "OK"
        self._conn.execute = AsyncMock(side_effect=_exec)
        sys.modules["asyncpg"].create_pool = AsyncMock(return_value=self._pool)
        s = self.PostgresStorage(database_url="postgresql://u:p@h/db")
        self.assertTrue(s._connected)

    def _meta_row(self, symbol="AAPL", **kw):
        row = {"symbol": symbol, "strategy_name": "rsi_mean_reversion",
               "rsi_period": 14, "stop_loss_price": 95.0,
               "entry_date": datetime(2026, 1, 2, 15, 0),
               "intraday": False, "quantity": 5.0, "entry_price": 100.0}
        row.update(kw)
        return row

    def test_save_position_metadata_disconnected_and_empty(self):
        self.assertFalse(
            self._disconnected().save_position_metadata([self._meta_row()]))
        self.assertTrue(self._connected().save_position_metadata([]))

    def test_save_position_metadata_upserts_by_env_symbol(self):
        from storage.backend import POSITION_METADATA_FIELDS
        s = self._connected(env="dev")
        self.assertTrue(s.save_position_metadata([
            self._meta_row(), self._meta_row("MSFT", strategy_name=None)]))
        sql, rows = self._conn.executemany.call_args[0]
        self.assertIn("INSERT INTO position_metadata", sql)
        self.assertIn("ON CONFLICT (environment, symbol) DO UPDATE", sql)
        self.assertIn("updated_at = NOW()", sql)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), len(POSITION_METADATA_FIELDS) + 1)
        self.assertEqual(rows[0][0], "dev")
        self.assertEqual(rows[0][1], "AAPL")
        # naive entry_date localised to UTC for TIMESTAMPTZ
        idx = POSITION_METADATA_FIELDS.index("entry_date") + 1
        self.assertIsNotNone(rows[0][idx].tzinfo)
        # missing strategy falls back to 'unattributed'
        sidx = POSITION_METADATA_FIELDS.index("strategy_name") + 1
        self.assertEqual(rows[1][sidx], "unattributed")

    def test_load_position_metadata(self):
        self.assertIsNone(self._disconnected().load_position_metadata())
        self._conn.fetch = AsyncMock(return_value=[{"symbol": "AAPL"}])
        self.assertEqual(self._connected().load_position_metadata(),
                         [{"symbol": "AAPL"}])
        sql, env = self._conn.fetch.call_args[0]
        self.assertIn("FROM position_metadata WHERE environment = $1", sql)
        self.assertEqual(env, "dev")

    def test_load_position_metadata_error_is_none_not_empty(self):
        self._conn.fetch = AsyncMock(side_effect=RuntimeError("db down"))
        self.assertIsNone(self._connected().load_position_metadata())

    def test_delete_position_metadata(self):
        self.assertFalse(self._disconnected().delete_position_metadata(["A"]))
        self.assertTrue(self._connected().delete_position_metadata([]))
        s = self._connected(env="qa")
        self.assertTrue(s.delete_position_metadata(["AAPL", "MSFT"]))
        sql, env, symbols = self._conn.execute.call_args[0]
        self.assertIn("DELETE FROM position_metadata", sql)
        self.assertEqual((env, symbols), ("qa", ["AAPL", "MSFT"]))

    def _trade(self, **kw):
        t = {"symbol": "AAPL", "side": "long", "quantity": 5.0,
             "entry_price": 100.0, "exit_price": 110.0,
             "entry_date": datetime(2026, 1, 2, 15, 0),
             "exit_date": datetime(2026, 1, 5, 15, 0),
             "realized_return": 0.1, "exit_reason": "oco_take_profit",
             "exit_price_estimated": False, "strategy_name": "x",
             "entry_client_order_id": "cid-1", "exit_order_id": "o2"}
        t.update(kw)
        return t

    def test_save_closed_trade_inserts_once_by_dedup_key(self):
        from storage.backend import CLOSED_TRADE_FIELDS
        self.assertFalse(self._disconnected().save_closed_trade(self._trade()))
        s = self._connected(env="dev")
        self.assertTrue(s.save_closed_trade(self._trade()))
        args = self._conn.execute.call_args[0]
        sql = args[0]
        self.assertIn("INSERT INTO closed_trades", sql)
        self.assertIn("ON CONFLICT (environment, dedup_key) DO NOTHING", sql)
        self.assertEqual(args[1], "dev")
        self.assertEqual(args[2], "cid-1")            # dedup_key
        self.assertEqual(len(args), 3 + len(CLOSED_TRADE_FIELDS))
        eidx = 3 + CLOSED_TRADE_FIELDS.index("exit_date")
        self.assertIsNotNone(args[eidx].tzinfo)

    def test_closed_trade_key_without_client_order_id(self):
        from storage.backend import closed_trade_key
        k1 = closed_trade_key(self._trade(entry_client_order_id=None))
        k2 = closed_trade_key(self._trade(
            entry_client_order_id="nan", exit_price=1.0,
            exit_date=datetime(2030, 1, 1)))
        self.assertEqual(k1, k2)       # exit data does not affect the key
        self.assertEqual(k1, "AAPL|long|2026-01-02T15:00:00")
        self.assertNotEqual(k1, closed_trade_key(self._trade(
            entry_client_order_id=None, side="short")))

    def test_save_closed_trade_error_returns_false(self):
        self._conn.execute = AsyncMock(side_effect=RuntimeError("x"))
        self.assertFalse(self._connected().save_closed_trade(self._trade()))

    def test_load_closed_trades(self):
        self.assertIsNone(self._disconnected().load_closed_trades())
        self._conn.fetch = AsyncMock(return_value=[{"symbol": "AAPL"}])
        self.assertEqual(self._connected().load_closed_trades(50),
                         [{"symbol": "AAPL"}])
        sql, env, limit = self._conn.fetch.call_args[0]
        self.assertIn("FROM closed_trades WHERE environment = $1", sql)
        self.assertIn("ORDER BY exit_date DESC NULLS LAST", sql)
        self.assertEqual((env, limit), ("dev", 50))
        self._conn.fetch = AsyncMock(side_effect=RuntimeError("x"))
        self.assertIsNone(self._connected().load_closed_trades())

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
            params={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70}, alpha=0.05,
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
            params={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70}, alpha=0.05,
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
                         params={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70}, alpha=0.05,
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
                     params={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70}, alpha=0.05,
                     stop_loss_price=140.0, take_profit_price=160.0, closed=False,
                     exit_date=None, exit_price=None, realized_return=None, side="long"),
                dict(symbol="TSLA", shares=5.0, entry_price=800.0,
                     current_price=810.0, current_rsi=40.0,
                     entry_date=datetime(2025, 6, 9, tzinfo=timezone.utc),
                     params={"rsi_period": 14, "rsi_lower": 30, "rsi_upper": 70}, alpha=0.08,
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


# --- strategy_daily_performance ---

    def test_strategy_performance_schema_in_ddl(self):
        """The DDL must create strategy_daily_performance with upsert key."""
        from storage import postgres
        ddl = postgres._ALL_DDL
        self.assertIn("CREATE TABLE IF NOT EXISTS strategy_daily_performance",
                      ddl)
        self.assertIn("uq_sdp_env_date_strategy", ddl)

    def test_save_strategy_performance_connected(self):
        s = self._connected(env="dev")
        records = [{
            "snapshot_date": "2026-09-22",
            "strategy_name": "leveraged_flow_portfolio",
            "equity": 100000.0,
            "allocation_weight": 0.15,
            "budget_notional": 15000.0,
            "open_positions": 3,
            "open_market_value": 14000.0,
            "unrealized_pnl": 120.5,
            "realized_pnl": -30.0,
        }]
        self.assertTrue(s.save_strategy_performance(records, "2026-09-22"))
        self._conn.executemany.assert_called_once()
        sql = self._conn.executemany.call_args[0][0]
        rows = self._conn.executemany.call_args[0][1]
        self.assertIn(
            "ON CONFLICT (environment, snapshot_date, strategy_name)", sql)
        # environment + snapshot_date are the first two params of each row
        self.assertEqual(rows[0][0], "dev")
        self.assertEqual(rows[0][1], "2026-09-22")
        # strategy_name follows snapshot_date (regression: snapshot_date used
        # to be inserted twice, which Postgres rejects).
        self.assertEqual(rows[0][2], "leveraged_flow_portfolio")
        # The INSERT column list must not repeat any column, and the number of
        # placeholders/row values must match the column count.
        insert_cols = sql.split("(", 1)[1].split(")", 1)[0]
        col_names = [c.strip() for c in insert_cols.split(",")]
        self.assertEqual(
            len(col_names), len(set(col_names)),
            f"duplicate column in INSERT: {col_names}")
        values_clause = sql.split("VALUES", 1)[1].split("ON CONFLICT", 1)[0]
        self.assertEqual(values_clause.count("$"), len(col_names))
        self.assertEqual(len(rows[0]), len(col_names))

    def test_save_strategy_performance_empty_records(self):
        s = self._connected()
        self.assertTrue(s.save_strategy_performance([], "2026-09-22"))
        self._conn.executemany.assert_not_called()

    def test_save_strategy_performance_disconnected(self):
        s = self._disconnected()
        self.assertFalse(s.save_strategy_performance(
            [{"snapshot_date": "2026-09-22",
              "strategy_name": "rsi_mean_reversion"}]))

    def test_load_strategy_performance_connected(self):
        s = self._connected(env="dev")
        self._conn.fetch = AsyncMock(return_value=[
            {"snapshot_date": "2026-09-22",
             "strategy_name": "rsi_mean_reversion",
             "equity": 100000.0, "allocation_weight": 0.85,
             "budget_notional": 85000.0, "open_positions": 2,
             "open_market_value": 5000.0, "unrealized_pnl": 12.0,
             "realized_pnl": 3.0},
        ])
        rows = s.load_strategy_performance()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["snapshot_date"], "2026-09-22")
        sql = self._conn.fetch.call_args[0][0]
        self.assertIn("FROM strategy_daily_performance", sql)
        self.assertIn("ORDER BY snapshot_date", sql)

    def test_load_strategy_performance_filtered(self):
        s = self._connected(env="dev")
        self._conn.fetch = AsyncMock(return_value=[])
        s.load_strategy_performance("leveraged_flow_portfolio")
        args = self._conn.fetch.call_args[0]
        self.assertEqual(args[1], "dev")
        self.assertEqual(args[2], "leveraged_flow_portfolio")

    def test_load_strategy_performance_disconnected(self):
        s = self._disconnected()
        self.assertEqual(s.load_strategy_performance(), [])

    def test_normalize_strategy_performance_record(self):
        from storage.backend import normalize_strategy_performance_record
        d = normalize_strategy_performance_record({
            "snapshot_date": "2026-09-22",
            "strategy_name": "rsi_mean_reversion",
            "open_positions": 2.7,
            "unrealized_pnl": float("nan"),
        })
        self.assertEqual(d["snapshot_date"], "2026-09-22")
        self.assertEqual(d["open_positions"], 2)
        self.assertIsNone(d["unrealized_pnl"])

    # --- load_session_metadata (Run Activity → Run history) ---

    def test_load_session_metadata_parses_jsonb(self):
        s = self._connected(env="dev")
        self._conn.fetch = AsyncMock(return_value=[
            {"timestamp": "20260930_141200",
             "created_at": datetime(2026, 9, 30, 14, 12, tzinfo=timezone.utc),
             "metadata": json.dumps({
                 "backtest_count": 3,
                 "results_summary": {"orders_placed": 2},
             })},
        ])
        rows = s.load_session_metadata(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timestamp"], "20260930_141200")
        self.assertEqual(rows[0]["metadata"]["backtest_count"], 3)
        self.assertEqual(
            rows[0]["metadata"]["results_summary"]["orders_placed"], 2)
        # created_at is normalized to an ISO string
        self.assertTrue(rows[0]["created_at"].startswith("2026-09-30T14:12"))

    def test_load_session_metadata_accepts_dict_and_tolerates_bad_json(self):
        s = self._connected()
        self._conn.fetch = AsyncMock(return_value=[
            {"timestamp": "a", "created_at": None,
             "metadata": {"ok": True}},          # already-decoded dict
            {"timestamp": "b", "created_at": None,
             "metadata": "not json"},            # malformed
            {"timestamp": "c", "created_at": None,
             "metadata": None},                  # missing
        ])
        rows = s.load_session_metadata()
        self.assertEqual(rows[0]["metadata"], {"ok": True})
        self.assertEqual(rows[1]["metadata"], {})
        self.assertEqual(rows[2]["metadata"], {})

    def test_load_session_metadata_query_and_filters(self):
        s = self._connected(env="qa")
        self._conn.fetch = AsyncMock(return_value=[])
        s.load_session_metadata(limit=25, timestamp="20260930_141200")
        sql = self._conn.fetch.call_args[0][0]
        args = self._conn.fetch.call_args[0][1:]
        self.assertIn("FROM session_metadata", sql)
        self.assertIn("ORDER BY timestamp DESC", sql)
        self.assertIn("LIMIT", sql)
        self.assertEqual(args[0], "qa")
        self.assertEqual(args[1], "20260930_141200")
        self.assertEqual(args[2], 25)

    def test_load_session_metadata_disconnected(self):
        self.assertEqual(self._disconnected().load_session_metadata(), [])

    def test_load_session_metadata_swallows_db_errors(self):
        s = self._connected()
        self._conn.fetch = AsyncMock(side_effect=RuntimeError("boom"))
        self.assertEqual(s.load_session_metadata(), [])

    # -- pending_signals -----------------------------------------------------

    def _pending_signal(self, symbol="AAPL", direction="long"):
        return {
            "symbol": symbol,
            "direction": direction,
            "strategy_name": "rsi_mean_reversion",
            "opportunity": {"symbol": symbol, "entry_price": 100.0},
        }

    def test_save_pending_signals_disconnected(self):
        self.assertFalse(
            self._disconnected().save_pending_signals([self._pending_signal()]))

    def test_save_pending_signals_empty(self):
        self.assertTrue(self._connected().save_pending_signals([]))

    def test_save_pending_signals_upserts_by_env_symbol_direction(self):
        s = self._connected(env="dev")
        self.assertTrue(s.save_pending_signals([
            self._pending_signal(), self._pending_signal("TSLA", "short")]))
        sql = self._conn.executemany.call_args[0][0]
        rows = self._conn.executemany.call_args[0][1]
        self.assertIn("INSERT INTO pending_signals", sql)
        self.assertIn("ON CONFLICT (environment, symbol, direction)", sql)
        self.assertIn("status = 'queued'", sql)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "dev")          # environment
        self.assertEqual(rows[0][1], "AAPL")
        self.assertEqual(rows[0][2], "long")
        self.assertEqual(json.loads(rows[0][4]), {"symbol": "AAPL",
                                                  "entry_price": 100.0})
        self.assertEqual(rows[1][2], "short")

    def test_save_pending_signals_serializes_datetime_values(self):
        """Non-JSON-safe values (datetimes) in the opportunity are serialized
        via the default encoder instead of raising."""
        s = self._connected()
        sig = self._pending_signal()
        sig["opportunity"]["queued_check"] = datetime(2026, 10, 31, 10, 0)
        self.assertTrue(s.save_pending_signals([sig]))
        row = self._conn.executemany.call_args[0][1][0]
        self.assertEqual(json.loads(row[4])["queued_check"],
                         "2026-10-31T10:00:00")

    def test_load_pending_signals_disconnected(self):
        self.assertEqual(self._disconnected().load_pending_signals(), [])

    def test_load_pending_signals_scopes_env_and_status(self):
        s = self._connected(env="qa")
        self._conn.fetch = AsyncMock(return_value=[{
            "id": 7, "symbol": "AAPL", "direction": "long",
            "strategy_name": "rsi_mean_reversion", "status": "queued",
            "reason": None, "queued_at": datetime(2026, 10, 31, 10, 0),
            "opportunity": '{"symbol": "AAPL"}',
        }])
        signals = s.load_pending_signals()
        sql = self._conn.fetch.call_args[0][0]
        args = self._conn.fetch.call_args[0][1:]
        self.assertIn("FROM pending_signals", sql)
        self.assertIn("environment = $1", sql)
        self.assertIn("status = $2", sql)
        self.assertIn("ORDER BY queued_at ASC", sql)
        self.assertEqual(args[0], "qa")
        self.assertEqual(args[1], "queued")
        self.assertEqual(signals[0]["id"], 7)
        self.assertEqual(signals[0]["opportunity"], {"symbol": "AAPL"})

    def test_resolve_pending_signals_disconnected(self):
        self.assertFalse(self._disconnected().resolve_pending_signals([1], "submitted"))

    def test_resolve_pending_signals_builds_env_scoped_update(self):
        s = self._connected(env="qa")
        self.assertTrue(s.resolve_pending_signals([1, 2], "expired",
                                                  reason="stale"))
        sql = self._conn.execute.call_args[0][0]
        args = self._conn.execute.call_args[0][1:]
        self.assertIn("UPDATE pending_signals", sql)
        self.assertIn("environment = $1", sql)
        self.assertIn("resolved_at", sql)
        self.assertEqual(args[0], "qa")
        self.assertEqual(args[1], "expired")
        self.assertEqual(args[2], "stale")
        self.assertEqual(list(args[3:]), [1, 2])

    def test_resolve_pending_signals_empty_ids_noop(self):
        s = self._connected()
        self.assertTrue(s.resolve_pending_signals([], "submitted"))
        self._conn.execute.assert_not_called()

    def test_resolve_pending_signals_swallows_db_errors(self):
        s = self._connected()
        self._conn.execute = AsyncMock(side_effect=RuntimeError("boom"))
        self.assertFalse(s.resolve_pending_signals([1], "submitted"))


if __name__ == "__main__":
    unittest.main()
