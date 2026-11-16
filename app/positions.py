"""Position tracking.

The BROKER (Alpaca) is the single source of truth for which positions are
OPEN and for their symbol, signed quantity (negative = short), average entry
price, current price and market value.  Local storage can never add, remove or
resize an open position.

Local storage holds only what the broker cannot know:

* ``position_metadata`` -- one row per open symbol: strategy attribution, RSI
  parameters, the *intended* SL/TP levels and the entry order ids.  Joined onto
  broker positions by symbol.  A broker position with no row is adopted as
  ``unattributed`` with default parameters.
* ``closed_trades`` -- one row per completed round trip, written exactly once
  when the engine closes a position, or when reconcile finds metadata for a
  symbol the broker no longer holds (exit price resolved from order history; an
  estimate is a last resort and is flagged ``exit_price_estimated``).
* ``position_snapshots`` -- a per-session snapshot of broker state + metadata
  for the dashboard/history.  Never read to decide what is open.

If the broker positions fetch fails, reconcile raises ``PositionsFetchError``:
nothing is closed and nothing is written.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import globalConfig  # type: ignore
from storage.backend import closed_trade_key
from utils import parse_dt

logger = logging.getLogger(__name__)

UNATTRIBUTED = "unattributed"


class PositionsFetchError(RuntimeError):
    """The broker positions fetch failed; the open set is unknown."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value) -> Optional[datetime]:
    """Coerce a datetime/Timestamp/str into a tz-aware UTC datetime (or None)."""
    dt = parse_dt(value)
    if dt is None or (not isinstance(dt, datetime)) or pd.isna(dt):
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _num(value, default=None):
    """float(value), or ``default`` for None/NaN/unparseable."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(out) else out


def _text(value) -> Optional[str]:
    """Stripped string, or None for None/NaN/'nan'-style placeholders."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text.lower() in ("", "nan", "nat", "none", "<na>") else text


def _realized_return(side: str, entry: Optional[float],
                     exit_price: Optional[float]) -> Optional[float]:
    if not entry or entry <= 0 or exit_price is None:
        return None
    diff = (entry - exit_price) if side == "short" else (exit_price - entry)
    return diff / entry


@dataclass
class Position:
    """Current (or closed) position information."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    current_rsi: float
    entry_date: datetime
    alpha: float
    rsi_period: int
    rsi_lower: int
    rsi_upper: int
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    realized_return: Optional[float] = None
    exit_reason: Optional[str] = None
    closed: bool = False
    side: str = field(init=False, default="long")
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    # Owning strategy (registry key); 'unattributed' for broker positions we
    # have no metadata for.
    strategy_name: str = "rsi_mean_reversion"
    # Intraday (bar-loop) positions are opened and closed within one session;
    # their exits are managed by BarLoopEngine, not the daily OCO refresh.
    intraday: bool = False
    composite_score: float = 0.0
    # True when exit_price is an estimate rather than a confirmed fill.
    exit_price_estimated: bool = False

    def __post_init__(self):
        """Derive side from quantity: negative qty = short position."""
        if self.quantity < 0:
            object.__setattr__(self, 'side', 'short')


class PositionsManager:
    """Joins broker positions with stored metadata and archives closed trades."""

    # Calls to get_and_reconcile_positions() within this window return the
    # cached open set (avoids duplicate Alpaca calls within a cycle).
    RECONCILIATION_CACHE_TTL_SECONDS = 60
    # Closed trades kept in memory (feeds the engine's "opened today" cap).
    CLOSED_TRADES_LOOKBACK = 500
    # A just-opened position may not be visible in the broker's positions
    # endpoint for a moment; within this window a missing broker position is
    # not treated as a close.
    OPEN_VISIBILITY_GRACE_SECONDS = 120
    # After the engine closes a position, how long the broker may keep
    # reporting it before we conclude the close did not happen and treat the
    # position (with its ORIGINAL metadata) as open again.
    CLOSE_CONFIRM_GRACE_SECONDS = 600

    def __init__(self, storage_backend, data_provider_instance):
        self.storage_backend = storage_backend
        self.data_provider = data_provider_instance
        self.positions: List[Position] = []
        self._reconciled_at: Optional[float] = None
        self._cached_open_positions: Optional[List[Position]] = None
        self._session_snapshot_timestamp: Optional[str] = None
        # Last metadata rows read/written, by symbol (fallback if the store
        # becomes unreadable).  ``_metadata_ok`` is False after a failed read.
        self._metadata: Dict[str, dict] = {}
        self._metadata_ok = True
        # symbol -> (Position, time.time()) opened this process, awaiting
        # broker visibility.
        self._pending_opens: Dict[str, Tuple[Position, float]] = {}
        # False after a failed broker fetch until the next successful one;
        # the engine refuses new entries while False.
        self.reconcile_ok = True
        # symbol -> {"pos": closed Position, "ts": time, "exit_order_id": id}
        # for closes issued by the engine and not yet confirmed by the broker.
        # Nothing is written to closed_trades / deleted from metadata until
        # the broker no longer holds the symbol.  (In-memory on purpose: after
        # a restart the metadata still exists, so reconcile either archives it
        # from order history or keeps it attached to a still-open position.)
        self._closing: Dict[str, dict] = {}
        # Symbols already looked up this process (attribution recovery /
        # exit repair), so a miss is not re-queried on every reconcile.
        self._attribution_tried: set = set()
        self._exit_repair_tried: set = set()

    # ------------------------------------------------------------------
    # Session / snapshots
    # ------------------------------------------------------------------

    def begin_session(self) -> None:
        """Start a persistence session: all snapshot saves this session upsert
        into one snapshot timestamp."""
        self._session_snapshot_timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S")

    def _session_timestamp(self) -> str:
        if self._session_snapshot_timestamp is None:
            self.begin_session()
        return self._session_snapshot_timestamp  # type: ignore[return-value]

    def persist_positions(self) -> None:
        """Persist open-position metadata and this session's open snapshot.

        Metadata edits made in memory (e.g. refreshed SL/TP) are written back;
        skipped when the last metadata read failed, so defaults can never
        overwrite real attribution.  Closed trades are written at close time,
        not here.
        """
        open_positions = [p for p in self.positions if not p.closed]
        if not open_positions:
            logger.info("persist_positions: no open positions to save")
            return
        if self._metadata_ok:
            self._save_metadata(open_positions)
        self.storage_backend.save_positions(
            open_positions, timestamp=self._session_timestamp())

    # ------------------------------------------------------------------
    # Metadata <-> Position
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_row(p: Position) -> dict:
        return {
            "symbol": p.symbol,
            "strategy_name": p.strategy_name or UNATTRIBUTED,
            "rsi_period": p.rsi_period,
            "rsi_lower": p.rsi_lower,
            "rsi_upper": p.rsi_upper,
            "alpha": p.alpha,
            "composite_score": p.composite_score,
            "stop_loss_price": p.stop_loss_price,
            "take_profit_price": p.take_profit_price,
            "intraday": bool(p.intraday),
            "entry_client_order_id": p.client_order_id,
            "entry_order_id": p.order_id,
            "entry_date": p.entry_date,
            "current_rsi": p.current_rsi,
            "side": p.side,
            "quantity": p.quantity,
            "entry_price": p.entry_price,
        }

    def _save_metadata(self, positions: List[Position]) -> bool:
        rows = [self._metadata_row(p) for p in positions]
        try:
            ok = bool(self.storage_backend.save_position_metadata(rows))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error saving position metadata: %s", exc)
            ok = False
        if ok:
            self._metadata.update({r["symbol"]: r for r in rows})
        else:
            logger.error("Could not persist metadata for %s",
                         [r["symbol"] for r in rows])
        return ok

    @staticmethod
    def _position_from_broker(row, meta: dict) -> Position:
        """Broker row + metadata -> Position.  Symbol, quantity, entry price
        and current price come ONLY from the broker row."""
        qty = _num(row.get("qty"), 0.0)
        entry_price = _num(row.get("avg_entry_price"), 0.0)
        side = "short" if qty < 0 else "long"
        pct_sl, pct_tp = globalConfig.STOP_LOSS_PCT, globalConfig.TAKE_PROFIT_PCT
        sl = _num(meta.get("stop_loss_price"))
        tp = _num(meta.get("take_profit_price"))
        if entry_price > 0:
            if sl is None:
                sl = entry_price * (1 + pct_sl if side == "short" else 1 - pct_sl)
            if tp is None:
                tp = entry_price * (1 - pct_tp if side == "short" else 1 + pct_tp)
        current = _num(row.get("current_price"), 0.0) or entry_price
        return Position(
            symbol=str(row["symbol"]),
            quantity=qty,
            entry_price=entry_price,
            current_price=current,
            current_rsi=_num(meta.get("current_rsi"), 0.0),
            entry_date=_aware(meta.get("entry_date")) or _utc_now(),
            alpha=_num(meta.get("alpha"), 0.0),
            rsi_period=int(_num(meta.get("rsi_period"), 14)),
            rsi_lower=int(_num(meta.get("rsi_lower"), 30)),
            rsi_upper=int(_num(meta.get("rsi_upper"), 70)),
            stop_loss_price=sl,
            take_profit_price=tp,
            order_id=_text(meta.get("entry_order_id")),
            client_order_id=_text(meta.get("entry_client_order_id")),
            strategy_name=_text(meta.get("strategy_name")) or UNATTRIBUTED,
            intraday=bool(meta.get("intraday") or False),
            composite_score=_num(meta.get("composite_score"), 0.0),
        )

    def _adopt(self, row, broker_levels) -> dict:
        """Default metadata for a broker position we have no record of."""
        symbol = str(row["symbol"])
        side = "short" if _num(row.get("qty"), 0.0) < 0 else "long"
        entry_date = _utc_now()
        try:
            info = self.data_provider.get_entry_order_for_symbol(
                symbol, side=side)
            if info is not None:
                found = _aware(info[0])
                if found is not None:
                    entry_date = found
        except Exception:  # pylint: disable=broad-exception-caught
            pass  # Mock/unknown shapes or API errors: keep "now"
        sl, tp = broker_levels.get(symbol, (None, None))
        logger.warning(
            "Broker position %s (%s) has no stored metadata — adopting as "
            "'%s' (entry_date=%s, broker SL/TP=%s/%s)",
            symbol, side, UNATTRIBUTED, entry_date, sl, tp)
        return {"symbol": symbol, "strategy_name": UNATTRIBUTED,
                "entry_date": entry_date, "stop_loss_price": sl,
                "take_profit_price": tp}

    def _recover_attribution(self, row) -> Optional[dict]:
        """Metadata recovered from the position snapshot written when the
        engine opened this broker position (same symbol, side and entry price).

        Covers positions whose metadata write failed at entry.  Broker SL/TP
        legs already in ``meta`` win; snapshot levels only fill gaps.  Looked
        up at most once per symbol per process; None if nothing matches.
        """
        symbol = str(row["symbol"])
        if symbol in self._attribution_tried:
            return None
        self._attribution_tried.add(symbol)
        finder = getattr(self.storage_backend, 'find_snapshot_attribution', None)
        qty = _num(row.get("qty"), 0.0)
        entry_price = _num(row.get("avg_entry_price"), 0.0)
        if finder is None or not entry_price:
            return None
        try:
            snap = finder(symbol, "short" if qty < 0 else "long", entry_price)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Attribution lookup failed for %s: %s", symbol, exc)
            return None
        if not isinstance(snap, dict):
            return None
        strategy = _text(snap.get("strategy_name"))
        if not strategy or strategy == UNATTRIBUTED:
            return None
        found = {
            "strategy_name": strategy,
            "entry_client_order_id": _text(snap.get("client_order_id")),
            "entry_order_id": _text(snap.get("order_id")),
            "entry_date": _aware(snap.get("entry_date")),
            "rsi_period": snap.get("rsi_period"),
            "rsi_lower": snap.get("rsi_lower"),
            "rsi_upper": snap.get("rsi_upper"),
            "alpha": snap.get("alpha"),
            "current_rsi": snap.get("current_rsi"),
            "intraday": bool(snap.get("intraday") or False),
        }
        found = {k: v for k, v in found.items() if v is not None}
        logger.warning(
            "%s: recovered attribution '%s' (entry order %s) from the entry "
            "snapshot — its metadata had not been saved", symbol, strategy,
            found.get("entry_client_order_id"))
        return found

    # ------------------------------------------------------------------
    # Broker helpers
    # ------------------------------------------------------------------

    def _fetch_broker_positions(self) -> pd.DataFrame:
        try:
            df = self.data_provider.get_current_positions_df(
                raise_on_error=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error(
                "BROKER POSITIONS FETCH FAILED (%s) — open set unknown; not "
                "reconciling, nothing closed or written", exc)
            raise PositionsFetchError(str(exc)) from exc
        if not isinstance(df, pd.DataFrame):
            raise PositionsFetchError(
                "positions fetch returned %s" % type(df).__name__)
        if df.empty or 'symbol' not in df.columns:
            return pd.DataFrame(columns=['symbol', 'qty', 'avg_entry_price',
                                         'current_price', 'market_value'])
        return df

    def _broker_sl_tp_by_symbol(
        self,
    ) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """symbol -> (stop_loss, take_profit) from the broker's live protective
        legs.  Best-effort: ``{}`` when unavailable."""
        getter = getattr(self.data_provider, 'get_open_orders', None)
        if getter is None:
            return {}
        try:
            orders_df = getter()
            if orders_df is None or getattr(orders_df, 'empty', True):
                return {}
            result: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
            for _, row in orders_df.iterrows():
                symbol = str(row.get('symbol'))
                leg = str(row.get('leg_type') or '')
                sl, tp = result.get(symbol, (None, None))
                if leg == 'stop_loss':
                    sl = _num(row.get('stop_price'), sl)
                elif leg == 'take_profit':
                    tp = _num(row.get('limit_price'), tp)
                result[symbol] = (sl, tp)
            return result
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Could not build broker SL/TP map: %s", exc)
            return {}

    def _cancel_orphan_orders(self, live_symbols: set) -> None:
        """Cancel active broker orders whose symbol has no open position.

        A resting sell on a flat symbol can open an unintended short.  Gated by
        ``globalConfig.CANCEL_ORPHAN_ORDERS``; best-effort, never raises.
        """
        if not getattr(globalConfig, 'CANCEL_ORPHAN_ORDERS', True):
            return
        getter = getattr(self.data_provider, 'get_active_orders', None)
        cancel = getattr(self.data_provider, 'cancel_order', None)
        if getter is None or cancel is None:
            return
        try:
            orders_df = getter()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Orphan-order check could not fetch orders: %s", e)
            return
        if not isinstance(orders_df, pd.DataFrame) or orders_df.empty \
                or 'symbol' not in orders_df.columns:
            return
        orphans = orders_df[~orders_df['symbol'].astype(str).isin(live_symbols)]
        for _, row in orphans.iterrows():
            order_id = row.get('order_id')
            if order_id and cancel(str(order_id)):
                logger.warning(
                    "Cancelled ORPHAN order %s (%s %s) for %s — no live "
                    "position in Alpaca", order_id, row.get('order_type'),
                    row.get('side'), row.get('symbol'))

    # ------------------------------------------------------------------
    # Reconcile
    # ------------------------------------------------------------------

    def get_and_reconcile_positions(self) -> List[Position]:
        """Return OPEN positions built from the broker, joined with metadata.

        Also archives metadata whose symbol the broker no longer holds as a
        closed trade, and refreshes ``self.positions`` (open + recent closed).
        Raises ``PositionsFetchError`` (leaving state untouched) if the broker
        fetch fails.
        """
        now = datetime.now().timestamp()
        if (self._reconciled_at is not None
                and self._cached_open_positions is not None
                and (now - self._reconciled_at)
                < self.RECONCILIATION_CACHE_TTL_SECONDS):
            return self._cached_open_positions

        try:
            broker = self._fetch_broker_positions()
        except PositionsFetchError:
            self.reconcile_ok = False
            raise
        self.reconcile_ok = True
        live = {str(r["symbol"]): r for _, r in broker.iterrows()
                if _num(r.get("qty"), 0.0) != 0}
        self._cancel_orphan_orders(set(live))

        held_out = self._settle_closing(live)
        stored = self._load_metadata()
        prior_open = {p.symbol: p for p in self.positions if not p.closed}
        pending = self._live_pending(live)

        if self._metadata_ok:
            # Metadata whose side contradicts the broker's is from an earlier
            # round trip on this symbol: archive it, then adopt fresh below.
            for symbol, row in live.items():
                meta_side = _text(stored.get(symbol, {}).get("side"))
                broker_side = "short" if _num(row.get("qty"), 0.0) < 0 else "long"
                if meta_side and meta_side != broker_side:
                    logger.warning(
                        "%s: stored metadata side=%s but broker is %s — stale "
                        "metadata; archiving it and adopting the broker "
                        "position fresh", symbol, meta_side, broker_side)
                    self._archive_closed([symbol], stored, prior_open)
                    stored.pop(symbol, None)
            gone = [s for s in stored if s not in live and s not in pending
                    and s not in self._closing]
            self._archive_closed(gone, stored, prior_open)

        broker_levels = None
        to_save: List[Position] = []
        open_positions: List[Position] = []
        for symbol, row in live.items():
            if symbol in held_out:
                continue
            meta = stored.get(symbol)
            adopted = meta is None
            if adopted:
                if broker_levels is None:
                    broker_levels = self._broker_sl_tp_by_symbol()
                meta = self._adopt(row, broker_levels)
            recovered = False
            if (_text(meta.get("strategy_name")) or UNATTRIBUTED) == UNATTRIBUTED:
                found = self._recover_attribution(row)
                if found:
                    meta = {**meta, **found}
                    recovered = True
            pos = self._position_from_broker(row, meta)
            open_positions.append(pos)
            if adopted or recovered or self._informational_changed(meta, pos):
                to_save.append(pos)
        open_positions.extend(pending.values())

        if to_save and self._metadata_ok:
            self._save_metadata(to_save)
        elif not self._metadata_ok:
            self._metadata = {p.symbol: self._metadata_row(p)
                              for p in open_positions}

        self.positions = open_positions + self._closed_positions()
        logger.info(
            "Reconciliation complete: open=%d (%s), closed_in_memory=%d",
            len(open_positions), sorted(p.symbol for p in open_positions),
            len(self.positions) - len(open_positions))

        self._reconciled_at = datetime.now().timestamp()
        self._cached_open_positions = open_positions
        return open_positions

    @staticmethod
    def _informational_changed(meta: dict, pos: Position) -> bool:
        """Has the broker-side size/price/side drifted from the stored copy?"""
        return (
            (_text(meta.get("side")) or "long") != pos.side
            or abs((_num(meta.get("quantity"), 0.0)) - pos.quantity) > 1e-9
            or abs((_num(meta.get("entry_price"), 0.0)) - pos.entry_price) > 1e-6)

    def _load_metadata(self) -> Dict[str, dict]:
        """Stored metadata by symbol; falls back to the last known copy (and
        disables metadata/close writes) when the store is unreadable."""
        rows = self.storage_backend.load_position_metadata()
        if not isinstance(rows, list):
            logger.error(
                "Position metadata UNREADABLE — using last known copy; "
                "metadata writes and trade closes are suspended this cycle")
            self._metadata_ok = False
            return dict(self._metadata)
        self._metadata_ok = True
        self._metadata = {str(r["symbol"]): dict(r) for r in rows}
        return dict(self._metadata)

    def _live_pending(self, live: dict) -> Dict[str, Position]:
        """Just-opened positions not yet visible at the broker (within grace)."""
        now = datetime.now().timestamp()
        pending: Dict[str, Position] = {}
        for symbol, (pos, opened) in list(self._pending_opens.items()):
            if symbol in live or now - opened > self.OPEN_VISIBILITY_GRACE_SECONDS:
                del self._pending_opens[symbol]
            else:
                logger.warning(
                    "%s opened %.0fs ago but not yet in broker positions — "
                    "keeping the pending entry for now", symbol, now - opened)
                pending[symbol] = pos
        return pending

    def _closed_positions(self) -> List[Position]:
        """Recent closed trades (DB) plus any closed this process that the DB
        does not (yet) have, so the daily-opened count never drops a trade."""
        rows = self.storage_backend.load_closed_trades(
            self.CLOSED_TRADES_LOOKBACK)
        previous = [p for p in self.positions if p.closed]
        if not isinstance(rows, list):
            logger.error("closed_trades unreadable — keeping in-memory closed")
            return previous
        closed = [self._closed_from_row(r) for r in rows]
        keys = {closed_trade_key(self._trade_row(p)) for p in closed}
        closed.extend(p for p in previous
                      if closed_trade_key(self._trade_row(p)) not in keys)
        return closed

    @staticmethod
    def _closed_from_row(r: dict) -> Position:
        side = _text(r.get("side")) or "long"
        qty = abs(_num(r.get("quantity"), 0.0))
        entry = _num(r.get("entry_price"), 0.0)
        exit_price = _num(r.get("exit_price"))
        return Position(
            symbol=str(r["symbol"]),
            quantity=-qty if side == "short" else qty,
            entry_price=entry,
            current_price=exit_price if exit_price is not None else entry,
            current_rsi=0.0,
            entry_date=_aware(r.get("entry_date")) or _utc_now(),
            alpha=0.0, rsi_period=14, rsi_lower=30, rsi_upper=70,
            exit_date=_aware(r.get("exit_date")),
            exit_price=exit_price,
            realized_return=_num(r.get("realized_return")),
            exit_reason=_text(r.get("exit_reason")),
            closed=True,
            client_order_id=_text(r.get("entry_client_order_id")),
            order_id=_text(r.get("exit_order_id")),
            strategy_name=_text(r.get("strategy_name")) or UNATTRIBUTED,
            exit_price_estimated=bool(r.get("exit_price_estimated")),
        )

    @staticmethod
    def _trade_row(p: Position) -> dict:
        """closed_trades row for a closed Position (exit_order_id excluded:
        Position.order_id is the ENTRY order)."""
        return {
            "symbol": p.symbol, "side": p.side,
            "quantity": abs(p.quantity), "entry_price": p.entry_price,
            "exit_price": p.exit_price, "entry_date": p.entry_date,
            "exit_date": p.exit_date, "realized_return": p.realized_return,
            "exit_reason": p.exit_reason,
            "exit_price_estimated": p.exit_price_estimated,
            "strategy_name": p.strategy_name,
            "entry_client_order_id": p.client_order_id,
            "exit_order_id": None,
        }

    # ------------------------------------------------------------------
    # Closing
    # ------------------------------------------------------------------

    @staticmethod
    def _order_time(row) -> Optional[datetime]:
        """Naive-UTC time of an order row: filled_at, else submitted_at."""
        for col in ('filled_at', 'submitted_at'):
            try:
                val = row.get(col)
            except AttributeError:
                val = None
            if val is None or not pd.notna(val):
                continue
            try:
                ts = pd.Timestamp(val)
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is not None:
                ts = ts.tz_convert('UTC').tz_localize(None)
            return ts.to_pydatetime()
        return None

    def _select_close_order(self, orders_df: pd.DataFrame,
                            close_order_side: str, entry_order_side: str,
                            entry_client_order_id: Optional[str]):
        """Pick the filled close-side order that closed the position.

        With the entry fill found in history: the EARLIEST close-side fill at
        or after it (never an older round trip's exit; None if there is none).
        Otherwise the most recent filled close-side order.  Returns a row or
        None.
        """
        close_orders = orders_df[
            (orders_df['side'] == close_order_side) &
            (orders_df['filled_qty'] > 0)]
        if close_orders.empty:
            return None

        entry_time = None
        if entry_client_order_id:
            entry = self._find_fill_by_client_order_id(
                entry_client_order_id, entry_order_side, orders_df)
            if entry is not None:
                entry_time = self._order_time({'filled_at': entry})

        rows = [(self._order_time(r), i)
                for i, (_, r) in enumerate(close_orders.iterrows())]
        if entry_time is not None:
            after = [(t, i) for t, i in rows if t is not None and t >= entry_time]
            return close_orders.iloc[min(after)[1]] if after else None
        timed = [(t, i) for t, i in rows if t is not None]
        if timed:
            return close_orders.iloc[max(timed)[1]]
        return close_orders.iloc[0]

    @staticmethod
    def _find_fill_by_client_order_id(client_order_id: str, side: str,
                                      orders_df: pd.DataFrame):
        """filled_at of the filled ``side`` order matching client_order_id
        (fallback: order_id), or None."""
        if 'client_order_id' not in orders_df.columns:
            return None
        df = orders_df[(orders_df['side'] == side)
                       & (orders_df['filled_qty'] > 0)]
        cid = str(client_order_id)
        matched = df[df['client_order_id'].astype(str) == cid]
        if matched.empty and 'order_id' in df.columns:
            matched = df[df['order_id'].astype(str) == cid]
        return None if matched.empty else matched.iloc[0].get('filled_at')

    def _find_close_fill(self, symbol: str, side: str,
                         entry_client_order_id: Optional[str]):
        """(close_order_row | None, entry_ever_filled) from order history.

        Raises if the history lookup itself fails.
        """
        orders_df = self.data_provider.get_filled_orders_for_symbol(
            symbol, limit=50)
        if (not isinstance(orders_df, pd.DataFrame) or orders_df.empty
                or 'side' not in orders_df.columns):
            return None, True  # no usable history: cannot say "never opened"
        entry_side = "sell" if side == "short" else "buy"
        close_side = "buy" if side == "short" else "sell"
        close = self._select_close_order(
            orders_df, close_side, entry_side, entry_client_order_id)
        entered = not orders_df[(orders_df['side'] == entry_side)
                                & (orders_df['filled_qty'] > 0)].empty
        return close, entered

    def _write_closed_trade(self, trade: dict) -> bool:
        """Persist a closed trade once (dedup-keyed), then drop its metadata."""
        try:
            saved = bool(self.storage_backend.save_closed_trade(trade))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error saving closed trade %s: %s",
                         trade.get("symbol"), exc)
            saved = False
        if not saved:
            logger.error("Closed trade for %s NOT persisted (metadata kept; "
                         "will retry on the next reconcile)", trade["symbol"])
            return False
        try:
            self.storage_backend.delete_position_metadata([trade["symbol"]])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error deleting metadata for %s: %s",
                         trade["symbol"], exc)
        self._metadata.pop(trade["symbol"], None)
        return True

    def _finalize_close(self, symbol: str) -> bool:
        """Broker no longer holds ``symbol``: write the engine's close."""
        rec = self._closing[symbol]
        trade = self._trade_row(rec["pos"])
        trade["exit_order_id"] = rec["exit_order_id"]
        if self._write_closed_trade(trade):
            del self._closing[symbol]
            return True
        return False

    def _settle_closing(self, live: dict) -> set:
        """Resolve engine-issued closes against the broker's positions.

        * broker no longer holds the symbol -> finalize (write closed trade,
          delete metadata);
        * broker still holds it within the grace window -> hold it out of the
          open set (the close may simply not have settled; avoids a double
          exit) without touching its metadata;
        * still held after the grace window -> the close did not happen: drop
          the closing record and the in-memory closed copy so the broker
          position is open again with its original metadata.
        Returns the symbols to hold out of the open set.
        """
        held_out = set()
        now = datetime.now().timestamp()
        for symbol in list(self._closing):
            rec = self._closing[symbol]
            if symbol not in live:
                self._finalize_close(symbol)
            elif now - rec["ts"] > self.CLOSE_CONFIRM_GRACE_SECONDS:
                logger.warning(
                    "%s: close issued %.0fs ago but the broker STILL holds "
                    "it — treating it as open again (original metadata "
                    "kept)", symbol, now - rec["ts"])
                self.positions = [p for p in self.positions
                                  if p is not rec["pos"]]
                del self._closing[symbol]
            else:
                logger.warning(
                    "%s: close issued but broker still lists it — holding it "
                    "out of the open set until the close settles", symbol)
                held_out.add(symbol)
        return held_out

    def _archive_closed(self, symbols: List[str], stored: Dict[str, dict],
                        prior_open: Dict[str, Position]) -> None:
        """Write closed_trades for metadata symbols the broker no longer holds."""
        for symbol in symbols:
            meta = stored[symbol]
            qty = _num(meta.get("quantity"), 0.0)
            side = _text(meta.get("side")) or ("short" if qty < 0 else "long")
            entry_price = _num(meta.get("entry_price"), 0.0)
            cid = _text(meta.get("entry_client_order_id"))
            try:
                close, entered = self._find_close_fill(symbol, side, cid)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Order history lookup failed for %s (%s) — "
                               "deferring its close", symbol, exc)
                continue
            if not entered:
                logger.warning(
                    "%s has metadata but no filled entry in order history — "
                    "never opened; dropping metadata, no trade recorded",
                    symbol)
                self.storage_backend.delete_position_metadata([symbol])
                self._metadata.pop(symbol, None)
                continue

            sl = _num(meta.get("stop_loss_price"))
            tp = _num(meta.get("take_profit_price"))
            exit_order_id = None
            estimated = close is None
            if close is not None:
                exit_price = float(close['filled_avg_price'])
                exit_date = (_aware(self._order_time(close)) or _utc_now())
                exit_order_id = _text(close.get('order_id'))
                if sl is not None and tp is not None:
                    reason = ("oco_stop_loss"
                              if abs(exit_price - sl) <= abs(exit_price - tp)
                              else "oco_take_profit")
                else:
                    reason = "oco_filled"
            else:
                exit_price = self._estimate_exit_price(
                    symbol, prior_open.get(symbol), entry_price)
                exit_date, reason = _utc_now(), "broker_closed"
                logger.warning(
                    "%s: no close fill found in order history — recording "
                    "ESTIMATED exit $%.2f", symbol, exit_price)

            self._write_closed_trade({
                "symbol": symbol, "side": side, "quantity": abs(qty),
                "entry_price": entry_price, "exit_price": exit_price,
                "entry_date": _aware(meta.get("entry_date")),
                "exit_date": exit_date,
                "realized_return": _realized_return(side, entry_price, exit_price),
                "exit_reason": reason, "exit_price_estimated": estimated,
                "strategy_name": _text(meta.get("strategy_name")) or UNATTRIBUTED,
                "entry_client_order_id": cid, "exit_order_id": exit_order_id,
            })

    def _estimate_exit_price(self, symbol: str, prior: Optional[Position],
                             entry_price: float) -> float:
        """Last resort: last seen price, else live snapshot, else entry."""
        if prior is not None and prior.current_price and prior.current_price > 0:
            return float(prior.current_price)
        try:
            snap = self.data_provider.get_current_snapshot(symbol)
            price = _num(snap.get('price')) if isinstance(snap, dict) else None
            if price and price > 0:
                return price
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return entry_price

    # Symbols whose order history is fetched per repair pass (API budget).
    EXIT_REPAIR_MAX_SYMBOLS = 25
    # Order-history depth for the repair; closed orders include every
    # cancelled OCO refresh, so the default 50 does not reach far back.
    EXIT_REPAIR_ORDER_LIMIT = 500
    # A recorded entry_date can be slightly after the real fill.
    EXIT_REPAIR_ENTRY_SLACK = pd.Timedelta(hours=1)
    # A larger implied move means the recorded entry is not comparable to the
    # fill (e.g. a split since entry: ETHA's 1-for-3) -- leave it estimated.
    EXIT_REPAIR_MAX_ABS_RETURN = 0.5

    @staticmethod
    def _naive_utc(value) -> Optional[datetime]:
        dt = _aware(value)
        return None if dt is None else (
            dt.astimezone(timezone.utc).replace(tzinfo=None))

    def repair_estimated_exits(self) -> int:
        """Replace estimated exits in closed_trades with the broker's fills.

        For each trade flagged ``exit_price_estimated`` the exit is the
        EARLIEST close-side fill at or after its entry (the entry fill when its
        client order id is in history, else the recorded entry_date) and
        before the next trade on the same symbol/side, never reusing a fill
        already claimed by another trade.  Trades with no such fill (or whose
        entry order never filled) stay estimated.  Each symbol's history is
        fetched at most once per process.  Returns the number repaired.
        """
        updater = getattr(self.storage_backend, 'update_closed_trade_exit', None)
        if updater is None:
            return 0
        rows = self.storage_backend.load_closed_trades(
            self.CLOSED_TRADES_LOOKBACK)
        if not isinstance(rows, list):
            return 0
        by_symbol: Dict[str, List[dict]] = {}
        for r in rows:
            by_symbol.setdefault(str(r.get("symbol")), []).append(r)
        targets = [s for s, trades in by_symbol.items()
                   if s not in self._exit_repair_tried
                   and any(t.get("exit_price_estimated") for t in trades)]
        repaired = 0
        for symbol in targets[:self.EXIT_REPAIR_MAX_SYMBOLS]:
            self._exit_repair_tried.add(symbol)
            try:
                orders = self.data_provider.get_filled_orders_for_symbol(
                    symbol, limit=self.EXIT_REPAIR_ORDER_LIMIT)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Exit repair: order history for %s failed: %s",
                               symbol, exc)
                continue
            if (not isinstance(orders, pd.DataFrame) or orders.empty
                    or 'side' not in orders.columns):
                continue
            repaired += self._repair_symbol_exits(by_symbol[symbol], orders,
                                                  updater)
        if repaired:
            logger.info("Exit repair: %d closed trade(s) now use confirmed "
                        "fills", repaired)
        return repaired

    def _repair_symbol_exits(self, trades: List[dict], orders: pd.DataFrame,
                             updater) -> int:
        used = {_text(t.get("exit_order_id")) for t in trades
                if not t.get("exit_price_estimated")} - {None}
        repaired = 0
        for side in ("long", "short"):
            seq = sorted(
                (t for t in trades
                 if (_text(t.get("side")) or "long") == side
                 and self._naive_utc(t.get("entry_date")) is not None),
                key=lambda t: self._naive_utc(t.get("entry_date")))
            entry_side = "sell" if side == "short" else "buy"
            close_side = "buy" if side == "short" else "sell"
            fills = orders[(orders['side'] == close_side)
                           & (orders['filled_qty'] > 0)]
            timed = sorted(
                ((self._order_time(r), r) for _, r in fills.iterrows()),
                key=lambda x: (x[0] is None, x[0] or datetime.min))
            for i, trade in enumerate(seq):
                if not trade.get("exit_price_estimated"):
                    continue
                cid = _text(trade.get("entry_client_order_id"))
                lower = self._naive_utc(trade.get("entry_date")) \
                    - self.EXIT_REPAIR_ENTRY_SLACK
                if cid:
                    filled_at = self._find_fill_by_client_order_id(
                        cid, entry_side, orders)
                    if filled_at is None:
                        continue          # entry never filled: leave it
                    lower = self._order_time({'filled_at': filled_at}) or lower
                upper = (self._naive_utc(seq[i + 1].get("entry_date"))
                         if i + 1 < len(seq) else None)
                match = next(
                    ((t, r) for t, r in timed
                     if t is not None and t >= lower
                     and (upper is None or t < upper)
                     and _text(r.get('order_id')) not in used), None)
                if match is None:
                    continue
                when, fill = match
                price = _num(fill.get('filled_avg_price'))
                if not price or price <= 0:
                    continue
                oid = _text(fill.get('order_id'))
                entry_price = _num(trade.get("entry_price"))
                rr = _realized_return(side, entry_price, price)
                if rr is not None and abs(rr) > self.EXIT_REPAIR_MAX_ABS_RETURN:
                    logger.warning(
                        "Exit repair: %s %s entry $%.2f vs fill $%.2f implies "
                        "%+.0f%% — entry likely not comparable (split?); "
                        "left estimated", trade.get("symbol"), side,
                        entry_price, price, rr * 100)
                    used.add(oid)     # still this trade's exit: don't reuse
                    continue
                ok = updater(
                    closed_trade_key(trade), price,
                    when.replace(tzinfo=timezone.utc), rr, oid)
                if ok:
                    used.add(oid)
                    repaired += 1
                    logger.info(
                        "Exit repair: %s %s entry $%.2f -> exit $%.2f (order "
                        "%s, %s)", trade.get("symbol"), side, entry_price or 0,
                        price, oid, when)
        return repaired

    def close_position(self, symbol: str, exit_price: Optional[float] = None):
        """Record the close of an open position (engine-initiated).

        ``exit_price`` should be the broker's confirmed fill.  Without it the
        price is looked up in order history, then estimated from the last
        known price (flagged ``exit_price_estimated``).  Writes one
        closed_trades row, deletes the metadata, and keeps the position in
        memory as closed (so it still counts toward today's entries).
        """
        pos = next((p for p in self.positions
                    if p.symbol == symbol and not p.closed), None)
        if pos is None:
            logger.warning(
                "Position with symbol %s not found in current open positions.",
                symbol)
            return

        price = _num(exit_price)
        exit_date, exit_order_id, estimated = _utc_now(), None, False
        if price is None:
            try:
                close, _ = self._find_close_fill(
                    symbol, pos.side, pos.client_order_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch close fill for %s: %s",
                               symbol, exc)
                close = None
            if close is not None:
                price = float(close['filled_avg_price'])
                exit_date = _aware(self._order_time(close)) or exit_date
                exit_order_id = _text(close.get('order_id'))
        if price is None:
            price = pos.current_price if pos.current_price and \
                pos.current_price > 0 else pos.entry_price
            estimated = True
            logger.warning("Exit price for %s ESTIMATED at $%.2f", symbol, price)

        pos.closed = True
        pos.exit_date = exit_date
        pos.exit_price = price
        pos.exit_price_estimated = estimated
        pos.realized_return = _realized_return(pos.side, pos.entry_price, price)
        pos.exit_reason = pos.exit_reason or "manual"
        self._pending_opens.pop(symbol, None)

        # Not final until the broker confirms the symbol is gone (see
        # _settle_closing); verify right away in the common case.
        self._closing[symbol] = {"pos": pos, "ts": datetime.now().timestamp(),
                                 "exit_order_id": exit_order_id}
        try:
            still_held = symbol in set(
                self._fetch_broker_positions()["symbol"].astype(str))
        except PositionsFetchError:
            still_held = True    # unknown: leave pending, reconcile settles
        if not still_held:
            self._finalize_close(symbol)
        logger.info("Closed %s position for %s: exit=%.2f, realized_return=%s",
                    pos.side, symbol, price, pos.realized_return)
        self.invalidate_reconciliation_cache()

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------

    def open_position(self, position: Position):
        """Record metadata for a position the engine just opened (confirmed
        fill).  The broker still decides whether it is open: the in-memory
        entry lets the engine see it this cycle and is replaced from broker
        data on the next reconcile."""
        self.positions = [p for p in self.positions
                          if not (p.symbol == position.symbol and not p.closed)]
        self.positions.append(position)
        self._pending_opens[position.symbol] = (
            position, datetime.now().timestamp())
        if not self._save_metadata([position]):
            logger.error(
                "Metadata for %s NOT saved — it will be adopted as "
                "'%s' by the next reconcile", position.symbol, UNATTRIBUTED)
        self.invalidate_reconciliation_cache()
        logger.info("Opened new position for %s (%s, qty=%s)", position.symbol,
                    position.strategy_name, position.quantity)

    def invalidate_reconciliation_cache(self) -> None:
        """Force a fresh reconcile on the next get_and_reconcile_positions()."""
        self._reconciled_at = None
        self._cached_open_positions = None
