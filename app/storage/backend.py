"""
Abstract storage backend interface.
All persistence operations (GCS, Postgres, etc.) must implement this ABC.
"""
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from config import Config
    from strategy import BacktestResult

logger = logging.getLogger(__name__)

# Ordered field list shared by both GCS and Postgres backends.
BACKTEST_FIELDS = [
    "symbol",
    "rsi_period",
    "rsi_lower",
    "rsi_upper",
    "total_return",
    "buy_and_hold_return",
    "alpha",
    "num_trades",
    "win_rate",
    "avg_trade_duration",
    "max_drawdown",
    "sharpe_ratio",
    "calmar_ratio",
    "composite_score",
    "direction",
    "profitable",
    "current_rsi",
    "strategy_name",
    "params",
]

POSITION_FIELDS = [
    "symbol",
    "shares",
    "entry_price",
    "current_price",
    "current_rsi",
    "entry_date",
    "rsi_period",
    "rsi_lower",
    "rsi_upper",
    "alpha",
    "stop_loss_price",
    "take_profit_price",
    "closed",
    "exit_date",
    "exit_price",
    "realized_return",
    "side",
    "order_id",
    "client_order_id",
    "strategy_name",
    "intraday",
]


def _safe_round(value: Any, ndigits: int = 2) -> Any:
    """Round a numeric value safely (passes through non-numerics like Mock)."""
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


def _serialize_params(params: Any) -> Optional[str]:
    """Serialize strategy params to a JSON string (None when empty/unserializable).

    Defensive: tolerates Mock objects (storage tests) and non-serializable
    values by returning None instead of raising.
    """
    if params is None:
        return None
    try:
        return json.dumps(params) if params else None
    except (TypeError, ValueError):
        return None


def backtest_result_to_dict(result: "BacktestResult") -> Dict[str, Any]:
    """Convert a BacktestResult to a flat, rounded dict for serialization."""
    return {
        "symbol": result.symbol,
        "rsi_period": result.rsi_period,
        "rsi_lower": result.rsi_lower,
        "rsi_upper": result.rsi_upper,
        "total_return": _safe_round(result.total_return),
        "buy_and_hold_return": _safe_round(result.buy_and_hold_return),
        "alpha": _safe_round(result.alpha),
        "num_trades": result.num_trades,
        "win_rate": _safe_round(result.win_rate),
        "avg_trade_duration": _safe_round(result.avg_trade_duration),
        "max_drawdown": _safe_round(result.max_drawdown),
        "sharpe_ratio": _safe_round(result.sharpe_ratio),
        "calmar_ratio": _safe_round(result.calmar_ratio),
        "composite_score": _safe_round(result.composite_score),
        "direction": result.direction,
        "profitable": result.profitable,
        "current_rsi": _safe_round(result.current_rsi),
        "strategy_name": result.strategy_name,
        "params": _serialize_params(result.params),
    }


def dict_to_backtest_result(d: Dict[str, Any]) -> "BacktestResult":
    """Reconstruct a BacktestResult from a flat dict (CSV row / DB row)."""
    from strategy import BacktestResult  # pylint: disable=import-outside-toplevel

    return BacktestResult(
        symbol=str(d["symbol"]),
        rsi_period=int(d["rsi_period"]),
        rsi_lower=int(d["rsi_lower"]),
        rsi_upper=int(d["rsi_upper"]),
        total_return=float(d["total_return"]) if d.get(
            "total_return") is not None else 0.0,
        buy_and_hold_return=float(d["buy_and_hold_return"]) if d.get(
            "buy_and_hold_return") is not None else 0.0,
        alpha=float(d["alpha"]) if d.get("alpha") is not None else 0.0,
        num_trades=int(d["num_trades"]) if d.get(
            "num_trades") is not None else 0,
        win_rate=float(d["win_rate"]) if d.get(
            "win_rate") is not None else 0.0,
        avg_trade_duration=float(d["avg_trade_duration"]) if d.get(
            "avg_trade_duration") is not None else 0.0,
        max_drawdown=float(d["max_drawdown"]) if d.get(
            "max_drawdown") is not None else 0.0,
        sharpe_ratio=float(d["sharpe_ratio"]) if d.get(
            "sharpe_ratio") is not None else 0.0,
        calmar_ratio=float(d.get("calmar_ratio", 0)),
        composite_score=float(d.get("composite_score", 0)),
        direction=str(d.get("direction", "long")),
        profitable=bool(d["profitable"]),
        current_rsi=float(d["current_rsi"]) if d.get("current_rsi") is not None and not (
            isinstance(d.get("current_rsi"), float) and pd.isna(d["current_rsi"])) else None,
        strategy_name=str(d.get("strategy_name", "rsi_mean_reversion")),
        params=_deserialize_params(d.get("params")),
    )


def _deserialize_params(raw: Any) -> Dict[str, Any]:
    """Deserialize the ``params`` column (JSON string or dict) safely."""
    if raw is None:
        return {}
    if isinstance(raw, float) and pd.isna(raw):
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(raw, dict):
        return raw
    return {}


def normalize_position_for_save(pos: Any) -> Dict[str, Any]:
    """Normalize a single Position object into a flat dict for serialization.

    Computes exit_price and realized_return when the position is closed,
    and returns all POSITION_FIELDS populated.
    """
    exit_price = getattr(pos, "exit_price", None)
    if exit_price is None and getattr(pos, "closed", False):
        exit_price = getattr(pos, "current_price", None)

    realized_return = getattr(pos, "realized_return", None)
    if realized_return is None and getattr(pos, "closed", False) and getattr(pos, "entry_price", None):
        if exit_price is not None:
            ep = pos.entry_price
            side = getattr(pos, "side", "long")
            if side == "short":
                realized_return = (ep - exit_price) / ep
            else:
                realized_return = (exit_price - ep) / ep

    return {
        "symbol": pos.symbol,
        "shares": pos.quantity,
        "entry_price": pos.entry_price,
        "current_price": pos.current_price,
        "current_rsi": pos.current_rsi,
        "entry_date": pos.entry_date if isinstance(pos.entry_date, datetime) else pos.entry_date,
        "rsi_period": pos.rsi_period,
        "rsi_lower": pos.rsi_lower,
        "rsi_upper": pos.rsi_upper,
        "alpha": pos.alpha,
        "stop_loss_price": pos.stop_loss_price,
        "take_profit_price": pos.take_profit_price,
        "closed": getattr(pos, "closed", False),
        "exit_date": pos.exit_date if isinstance(getattr(pos, "exit_date", None), datetime) else getattr(pos, "exit_date", None),
        "exit_price": exit_price,
        "realized_return": realized_return,
        "side": getattr(pos, "side", "long"),
        "order_id": getattr(pos, "order_id", None),
        "client_order_id": getattr(pos, "client_order_id", None),
        "strategy_name": getattr(pos, "strategy_name", "rsi_mean_reversion"),
        "intraday": bool(getattr(pos, "intraday", False)),
    }


ORDER_FIELDS = [
    "client_order_id",
    "order_id",
    "symbol",
    "side",
    "qty",
    "order_type",
    "order_class",
    "status",
    "stop_price",
    "limit_price",
    "submitted_at",
    "filled_at",
    "leg",
]


def _parse_optional_datetime(value: Any) -> Optional[datetime]:
    """Parse a possibly-datetime/None/NaN value into a datetime or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def order_to_dict(order: Any) -> Dict[str, Any]:
    """Convert an Order to a flat, serializable dict."""
    return {
        "client_order_id": getattr(order, "client_order_id", None),
        "order_id": getattr(order, "order_id", None),
        "symbol": getattr(order, "symbol", None),
        "side": getattr(order, "side", None),
        "qty": _safe_round(getattr(order, "qty", None)),
        "order_type": getattr(order, "order_type", None),
        "order_class": getattr(order, "order_class", None),
        "status": getattr(order, "status", None),
        "stop_price": _safe_round(getattr(order, "stop_price", None)),
        "limit_price": _safe_round(getattr(order, "limit_price", None)),
        "submitted_at": getattr(order, "submitted_at", None),
        "filled_at": getattr(order, "filled_at", None),
        "leg": getattr(order, "leg", None),
    }


def dict_to_order(d: Dict[str, Any]) -> "Order":
    """Reconstruct an Order from a flat dict (CSV row / DB row)."""
    from order import Order  # pylint: disable=import-outside-toplevel

    def _opt_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        return float(v)

    return Order(
        client_order_id=str(d.get("client_order_id") or ""),
        symbol=str(d.get("symbol") or ""),
        side=str(d.get("side") or ""),
        qty=float(d.get("qty") or 0),
        order_type=str(d.get("order_type") or "market"),
        order_class=str(d.get("order_class") or "simple"),
        status=str(d.get("status") or "new"),
        order_id=d.get("order_id") or None,
        stop_price=_opt_float(d.get("stop_price")),
        limit_price=_opt_float(d.get("limit_price")),
        submitted_at=_parse_optional_datetime(d.get("submitted_at")),
        filled_at=_parse_optional_datetime(d.get("filled_at")),
        leg=d.get("leg") or None,
    )


class StorageBackend(ABC):
    """Abstract base class for all storage backends (GCS, Postgres, etc.)."""

    @abstractmethod
    def save_backtest_results(
        self, results: "List[BacktestResult]", timestamp: Optional[str] = None
    ) -> bool:
        """Save backtest results to the backend."""

    @abstractmethod
    def load_backtest_results(self, filename: str) -> "List[BacktestResult]":
        """Load backtest results from the backend."""

    @abstractmethod
    def save_positions(
        self,
        positions_data,
        _run_number: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Save positions snapshot to the backend."""

    @abstractmethod
    def save_metadata(
        self, metadata: dict, timestamp: Optional[str] = None
    ) -> bool:
        """Save session metadata to the backend."""

    @abstractmethod
    def list_backtest_files(self) -> List[str]:
        """List all backtest file identifiers in the backend."""

    @abstractmethod
    def list_position_files(self) -> List[str]:
        """List all position file identifiers in the backend."""

    @abstractmethod
    def load_position_entries(self, filename: str) -> pd.DataFrame:
        """Load position entries from a specific file in the backend."""

    @abstractmethod
    def get_latest_position_file(self) -> Optional[str]:
        """Get the most recent position file identifier."""

    @abstractmethod
    def get_latest_positions_df(self, openPosition: bool = True) -> pd.DataFrame:
        """Get the most recent position DataFrame."""

    def save_orders(self, orders, timestamp: Optional[str] = None) -> bool:
        """Persist broker orders. Default no-op; backends may override."""
        return False

    def load_orders(
        self, symbol: Optional[str] = None, status: Optional[str] = None
    ) -> List[Any]:
        """Load persisted orders. Default returns an empty list."""
        return []

    def get_open_orders_stored(self, symbol: Optional[str] = None) -> List[Any]:
        """Load persisted orders that are not in a terminal status."""
        return [
            o for o in self.load_orders(symbol)
            if not getattr(o, "is_terminal", False)
        ]

    @staticmethod
    def create(config: "Config") -> "StorageBackend":
        """Factory: return the correct StorageBackend for the active config."""
        backend = getattr(config, "STORAGE_BACKEND", "gcs")

        if backend == "gcs":
            from storage.gcs import GcsStorage  # pylint: disable=import-outside-toplevel
            return GcsStorage()

        if backend == "postgres":
            from storage.postgres import (  # pylint: disable=import-outside-toplevel
                PostgresStorage,
            )
            return PostgresStorage()

        raise ValueError(
            f"Unknown STORAGE_BACKEND '{backend}'. "
            f"Expected 'gcs' or 'postgres'."
        )

    # ------------------------------------------------------------------
    # Optional DB-browse support (used by the dashboard "Database" tab).
    #
    # Concrete methods with safe defaults so backends that don't support
    # relational browsing (e.g. GCS) and test mocks need no changes.
    # ------------------------------------------------------------------

    def db_browse_enabled(self) -> bool:
        """True if the backend supports browsing tables (dashboard DB tab)."""
        return False

    def db_list_tables(self) -> List[str]:
        """List browsable table names. Default: none."""
        return []

    def db_fetch_table(
        self, table: str, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        """Fetch a page of rows from ``table``.

        Returns a dict with ``rows``, ``columns``, ``total``, ``limit``,
        ``offset``. Raises ValueError for unknown tables.
        """
        raise NotImplementedError(
            "This storage backend does not support table browsing")
