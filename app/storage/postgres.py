"""
Postgres storage backend implementing the StorageBackend ABC.

Uses asyncpg with a synchronous bridge (asyncio.run) so callers don't need
to change.  Schema is auto-created on first use (DDL is idempotent).

Requires DATABASE_URL env var when STORAGE_BACKEND=postgres.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import asyncpg
import pandas as pd

from storage.backend import StorageBackend, backtest_result_to_dict, dict_to_backtest_result, normalize_position_for_save

if TYPE_CHECKING:
    from config import Config
    from strategy import BacktestResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchronous bridge helpers
# ---------------------------------------------------------------------------


def _sync(coro):
    """Run an async coroutine and return its result synchronously."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # We're inside an already-running event loop — run in a background thread.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ---------------------------------------------------------------------------
# DDL (idempotent CREATE IF NOT EXISTS)
# ---------------------------------------------------------------------------

_DDL_BACKTEST_RESULTS = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id              SERIAL PRIMARY KEY,
    run_timestamp   TEXT        NOT NULL,
    environment     TEXT        NOT NULL,
    symbol          TEXT        NOT NULL,
    rsi_period      INTEGER     NOT NULL,
    rsi_lower       INTEGER     NOT NULL,
    rsi_upper       INTEGER     NOT NULL,
    total_return        DOUBLE PRECISION,
    buy_and_hold_return DOUBLE PRECISION,
    alpha               DOUBLE PRECISION,
    num_trades          INTEGER,
    win_rate            DOUBLE PRECISION,
    avg_trade_duration  DOUBLE PRECISION,
    max_drawdown        DOUBLE PRECISION,
    sharpe_ratio        DOUBLE PRECISION,
    calmar_ratio        DOUBLE PRECISION DEFAULT 0.0,
    composite_score     DOUBLE PRECISION DEFAULT 0.0,
    direction           TEXT            DEFAULT 'long',
    profitable          BOOLEAN,
    current_rsi         DOUBLE PRECISION,
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_bt_timestamp
    ON backtest_results (run_timestamp, environment);
"""

_DDL_POSITION_SNAPSHOTS = """
CREATE TABLE IF NOT EXISTS position_snapshots (
    id                  SERIAL PRIMARY KEY,
    snapshot_timestamp  TEXT        NOT NULL,
    environment         TEXT        NOT NULL,
    symbol              TEXT        NOT NULL,
    shares              DOUBLE PRECISION,
    entry_price         DOUBLE PRECISION,
    current_price       DOUBLE PRECISION,
    current_rsi         DOUBLE PRECISION,
    entry_date          TIMESTAMPTZ,
    rsi_period          INTEGER,
    rsi_lower           INTEGER,
    rsi_upper           INTEGER,
    alpha               DOUBLE PRECISION,
    stop_loss_price     DOUBLE PRECISION,
    take_profit_price   DOUBLE PRECISION,
    closed              BOOLEAN     DEFAULT FALSE,
    exit_date           TIMESTAMPTZ,
    exit_price          DOUBLE PRECISION,
    realized_return     DOUBLE PRECISION,
    side                TEXT        DEFAULT 'long',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ps_timestamp
    ON position_snapshots (snapshot_timestamp, environment);
"""

_DDL_SESSION_METADATA = """
CREATE TABLE IF NOT EXISTS session_metadata (
    id          SERIAL PRIMARY KEY,
    timestamp   TEXT        NOT NULL,
    environment TEXT        NOT NULL,
    metadata    JSONB       NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sm_timestamp
    ON session_metadata (timestamp, environment);
"""

_ALL_DDL = _DDL_BACKTEST_RESULTS + _DDL_POSITION_SNAPSHOTS + _DDL_SESSION_METADATA

# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------

_BACKTEST_COLS = [
    "symbol", "rsi_period", "rsi_lower", "rsi_upper",
    "total_return", "buy_and_hold_return", "alpha",
    "num_trades", "win_rate", "avg_trade_duration",
    "max_drawdown", "sharpe_ratio", "calmar_ratio",
    "composite_score", "direction", "profitable", "current_rsi",
]

_POSITION_COLS = [
    "symbol", "shares", "entry_price", "current_price", "current_rsi",
    "entry_date", "rsi_period", "rsi_lower", "rsi_upper", "alpha",
    "stop_loss_price", "take_profit_price", "closed",
    "exit_date", "exit_price", "realized_return", "side",
]

# ---------------------------------------------------------------------------
# PostgresStorage
# ---------------------------------------------------------------------------


class PostgresStorage(StorageBackend):
    """Storage backend backed by a Postgres database via asyncpg."""

    def __init__(self, database_url: Optional[str] = None):
        # type: ignore # pylint: disable=import-outside-toplevel
        from config import globalConfig

        self._dsn = database_url or getattr(globalConfig, "DATABASE_URL", "")
        self._env = globalConfig.ENVIRONMENT
        self._connected = False
        self._pool = None

        if not self._dsn:
            logger.error(
                "DATABASE_URL not set — PostgresStorage will be unavailable. "
                "Set DATABASE_URL env var or switch STORAGE_BACKEND to 'gcs'."
            )
            return

        try:
            _sync(self._init_pool_and_schema())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "Failed to initialise Postgres pool / schema: %s", exc)
            self._connected = False

    # -- pool & schema -------------------------------------------------------

    async def _init_pool_and_schema(self) -> None:
        # import asyncpg  # pylint: disable=import-outside-toplevel

        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=1,
            max_size=4,
            command_timeout=30,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(_ALL_DDL)
        self._connected = True
        logger.info(
            "Postgres pool connected & schema ensured (env=%s)", self._env)

    async def _fetch(self, query: str, *args) -> List[asyncpg.Record]:
        if not self._connected or self._pool is None:
            return []
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def _execute(self, query: str, *args) -> str:
        if not self._connected or self._pool is None:
            return "NOT_CONNECTED"
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    # -- save_backtest_results -----------------------------------------------

    def save_backtest_results(
        self, results: "List[BacktestResult]", timestamp: Optional[str] = None
    ) -> bool:
        if not self._connected:
            logger.error(
                "Postgres not connected — cannot save backtest results")
            return False

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        rows: List[tuple] = []
        for r in results:
            d = backtest_result_to_dict(r)
            rows.append((
                timestamp, self._env,
                d["symbol"],
                d["rsi_period"], d["rsi_lower"], d["rsi_upper"],
                d["total_return"], d["buy_and_hold_return"],
                d["alpha"], d["num_trades"], d["win_rate"],
                d["avg_trade_duration"], d["max_drawdown"],
                d["sharpe_ratio"], d["calmar_ratio"],
                d["composite_score"], d["direction"], d["profitable"],
                d["current_rsi"],
            ))

        col_placeholders = ", ".join(
            f"${i}" for i in range(1, len(_BACKTEST_COLS) + 3)
        )
        sql = (
            "INSERT INTO backtest_results "
            "(run_timestamp, environment, " + ", ".join(_BACKTEST_COLS) + ") "
            "VALUES (" + col_placeholders + ")"
        )

        try:
            _sync(self._execute_many(sql, rows))
            logger.info(
                "Saved %d backtest results to Postgres (run=%s)", len(
                    results), timestamp
            )
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error saving backtest results to Postgres: %s", exc)
            return False

    async def _execute_many(self, sql: str, rows: List[tuple]) -> None:
        if not self._connected or self._pool is None:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(sql, rows)

    # -- load_backtest_results -----------------------------------------------

    def load_backtest_results(self, filename: str) -> "List[BacktestResult]":
        if not self._connected:
            logger.error(
                "Postgres not connected — cannot load backtest results")
            return []

        ts = _filename_to_timestamp(filename)
        sql = (
            "SELECT " + ", ".join(_BACKTEST_COLS) + " "
            "FROM backtest_results "
            "WHERE run_timestamp = $1 AND environment = $2"
        )

        try:
            rows = _sync(self._fetch(sql, ts, self._env))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error loading backtest results from Postgres: %s", exc)
            return []

        if not rows:
            logger.error("No backtest results found for timestamp %s", ts)
            return []

        results: List[BacktestResult] = []
        for row in rows:
            results.append(dict_to_backtest_result(dict(row)))

        logger.info(
            "Loaded %d backtest results from Postgres (run=%s)", len(results), ts)
        return results

    # -- save_positions ------------------------------------------------------

    def save_positions(
        self,
        positions_data,
        _run_number: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        if not self._connected:
            logger.error("Postgres not connected — cannot save positions")
            return False

        # Normalise to list-of-dicts (same logic as GcsStorage)
        if isinstance(positions_data, list):
            if not positions_data:
                return True  # Nothing to save
            rows_list: List[Dict[str, Any]] = [
                normalize_position_for_save(pos) for pos in positions_data
            ]
        elif isinstance(positions_data, pd.DataFrame):
            rows_list = cast(List[Dict[str, Any]], positions_data.where(
                pd.notna(positions_data), None
            ).to_dict(orient="records"))
        else:
            logger.error("Unsupported positions_data type: %s",
                         type(positions_data))
            return False

        if not rows_list:
            return True

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        col_placeholders = ", ".join(
            f"${i}" for i in range(1, len(_POSITION_COLS) + 3)
        )
        sql = (
            "INSERT INTO position_snapshots "
            "(snapshot_timestamp, environment, " + ", ".join(_POSITION_COLS) + ") "
            "VALUES (" + col_placeholders + ")"
        )

        tuples: List[tuple] = []
        for d in rows_list:
            tup = (timestamp, self._env)
            for col in _POSITION_COLS:
                val = d.get(col)
                # Round numeric columns
                if isinstance(val, float):
                    val = round(val, 2)
                tup += (val,)
            tuples.append(tup)

        try:
            _sync(self._execute_many(sql, tuples))
            logger.info("Saved %d positions to Postgres (snapshot=%s)",
                        len(tuples), timestamp)
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error saving positions to Postgres: %s", exc)
            return False

    # -- save_metadata -------------------------------------------------------

    def save_metadata(
        self, metadata: dict, timestamp: Optional[str] = None
    ) -> bool:
        if not self._connected:
            logger.error("Postgres not connected — cannot save metadata")
            return False

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Round floats in metadata
        clean = _round_dict(metadata)
        clean["timestamp"] = timestamp

        try:
            import json
            _sync(self._execute(
                "INSERT INTO session_metadata (timestamp, environment, metadata) "
                "VALUES ($1, $2, $3)",
                timestamp, self._env, json.dumps(clean),
            ))
            logger.info(
                "Saved session metadata to Postgres (ts=%s)", timestamp)
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error saving metadata to Postgres: %s", exc)
            return False

    # -- list_backtest_files -------------------------------------------------

    def list_backtest_files(self) -> List[str]:
        if not self._connected:
            return []

        try:
            rows = _sync(self._fetch(
                "SELECT DISTINCT run_timestamp FROM backtest_results "
                "WHERE environment = $1 ORDER BY run_timestamp DESC",
                self._env,
            ))
            # Return in the same format as GCS: backtest_results_{timestamp}.csv
            return [f"backtest_results_{r['run_timestamp']}.csv" for r in rows]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error listing backtest files from Postgres: %s", exc)
            return []

    # -- list_position_files -------------------------------------------------

    def list_position_files(self) -> List[str]:
        if not self._connected:
            return []

        try:
            rows = _sync(self._fetch(
                "SELECT DISTINCT snapshot_timestamp FROM position_snapshots "
                "WHERE environment = $1 ORDER BY snapshot_timestamp DESC",
                self._env,
            ))
            return [f"positions_{r['snapshot_timestamp']}.csv" for r in rows]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error listing position files from Postgres: %s", exc)
            return []

    # -- load_position_entries -----------------------------------------------

    def load_position_entries(self, filename: str) -> pd.DataFrame:
        if not self._connected:
            logger.error("Postgres not connected — cannot load positions")
            return pd.DataFrame()

        ts = _filename_to_timestamp(filename)
        sql = (
            "SELECT " + ", ".join(_POSITION_COLS) + " "
            "FROM position_snapshots "
            "WHERE snapshot_timestamp = $1 AND environment = $2"
        )

        try:
            rows = _sync(self._fetch(sql, ts, self._env))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "Error loading position entries from Postgres: %s", exc)
            return pd.DataFrame()

        if not rows:
            logger.error("Position snapshot %s not found", ts)
            return pd.DataFrame()

        df = pd.DataFrame([dict(r) for r in rows])
        # Parse date columns
        for col in ("entry_date", "exit_date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

        logger.info(
            "Loaded %d position entries from Postgres (snapshot=%s)", len(df), ts)
        return df

    # -- get_latest_position_file --------------------------------------------

    def get_latest_position_file(self) -> Optional[str]:
        if not self._connected:
            return None

        try:
            rows = _sync(self._fetch(
                "SELECT snapshot_timestamp FROM position_snapshots "
                "WHERE environment = $1 "
                "ORDER BY snapshot_timestamp DESC LIMIT 1",
                self._env,
            ))
            if rows:
                return f"positions_{rows[0]['snapshot_timestamp']}.csv"
            return None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error getting latest position file: %s", exc)
            return None

    # -- get_latest_positions_df ---------------------------------------------

    def get_latest_positions_df(self, openPosition: bool = True) -> pd.DataFrame:
        latest_file = self.get_latest_position_file()
        if not latest_file:
            logger.warning("No position snapshots found in Postgres")
            return pd.DataFrame()

        df = self.load_position_entries(latest_file)
        if df.empty:
            return df

        if "closed" in df.columns:
            if openPosition:
                return df[df["closed"] != True]
            else:
                return df[df["closed"] == True]
        else:
            if not openPosition:
                logger.warning(
                    "Position snapshot '%s' is missing 'closed' column — "
                    "returning empty DataFrame for closed positions query.",
                    latest_file,
                )
                return pd.DataFrame()
            logger.info(
                "Position snapshot '%s' is missing 'closed' column — "
                "treating all rows as open (legacy format).",
                latest_file,
            )
            return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filename_to_timestamp(filename: str) -> str:
    """Extract the YYYYMMDD_HHMMSS timestamp from a GCS-style filename.

    'backtest_results_20250610_170343.csv' -> '20250610_170343'
    'positions_20250610_170343.csv'          -> '20250610_170343'
    """
    # Strip prefix (e.g. 'backtest_results_' or 'positions_') and '.csv' suffix
    core = filename.rsplit(".csv", 1)[0]  # remove .csv
    # Find the last underscore — everything after it is the time part
    parts = core.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
        # format: backtest_results_20250610_170343
        # We need the last two segments: _20250610_170343
        # Actually, the prefix can be multiple words like backtest_results
        # So let's find the pattern: YYYYMMDD_HHMMSS at the end
        import re
        m = re.search(r"(\d{8}_\d{6})$", core)
        if m:
            return m.group(1)
    return core


def _to_utc(dt) -> Optional[datetime]:
    """Convert a datetime to UTC-aware, or return None."""
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _round_dict(d: dict) -> dict:
    """Round all float values in a dict to 2 decimal places."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = round(v, 2)
        elif isinstance(v, dict):
            out[k] = _round_dict(v)
        elif isinstance(v, list):
            out[k] = [
                _round_dict(item) if isinstance(item, dict)
                else round(item, 2) if isinstance(item, float)
                else item
                for item in v
            ]
        else:
            out[k] = v
    return out
