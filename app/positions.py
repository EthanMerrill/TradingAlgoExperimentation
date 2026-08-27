import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from config import globalConfig  # type: ignore
from utils import parse_dt

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Current position information."""
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

    def __post_init__(self):
        """Derive side from quantity: negative qty = short position."""
        if self.quantity < 0:
            object.__setattr__(self, 'side', 'short')


class PositionsManager:
    """
    Tracks state of the positions in the trading engine.
    This class is responsible for managing position entries, saving them to cloud storage,
    and providing methods to retrieve and analyze position data.
    """

    # Seconds before a fresh reconciliation is forced.  Calls to
    # get_and_reconcile_positions() within this window return a cached
    # result, avoiding duplicate Alpaca API calls within the same cycle.
    RECONCILIATION_CACHE_TTL_SECONDS = 60

    def __init__(self, storage_backend, data_provider_instance):
        self.storage_backend = storage_backend
        self.data_provider = data_provider_instance
        # Initialize as empty list of Position objects
        self.positions: List[Position] = []
        # Reconciliation cache to avoid duplicate API calls within a cycle
        self._reconciled_at: Optional[float] = None
        self._cached_open_positions: Optional[List[Position]] = None

    def _persist_positions(self, positions_df: pd.DataFrame):
        """Save a positions DataFrame while preserving historical closed rows.

        Each ``save_positions`` call in this class originally wrote only
        open + newly-closed positions, silently dropping every closed
        position from all prior cycles.  This wrapper loads the existing
        closed rows from the latest file and merges them in so the full
        history survives.
        """
        historical_closed = self.storage_backend.get_latest_positions_df(False)
        if not historical_closed.empty:
            # Only keep columns that exist in the incoming DataFrame so
            # schema changes don't cause concat mismatches.
            common_cols = [
                c for c in positions_df.columns if c in historical_closed.columns]
            # Drop rows already present in positions_df (by symbol) to
            # avoid duplicating positions that just closed this cycle.
            if 'symbol' in common_cols and not positions_df.empty:
                incoming_symbols = set(positions_df['symbol'].astype(str))
                historical_closed = historical_closed[
                    ~historical_closed['symbol'].astype(
                        str).isin(incoming_symbols)
                ]
            if not historical_closed.empty:
                positions_df = pd.concat(
                    [positions_df, historical_closed[common_cols]],
                    ignore_index=True,
                )
        self.storage_backend.save_positions(positions_df)

    def persist_positions(self) -> None:
        """Persist the full in-memory position state (open + closed).

        Single save path: builds a DataFrame from ``self.positions`` and
        delegates to ``_persist_positions`` so historical closed rows are
        preserved.  Replaces direct ``storage.save_positions(positions)``
        calls that silently dropped closed positions.
        """
        if not self.positions:
            return
        # pylint: disable=import-outside-toplevel
        from storage.backend import normalize_position_for_save

        rows = [normalize_position_for_save(p) for p in self.positions]
        self._persist_positions(pd.DataFrame(rows))

    @staticmethod
    def _df_row_to_position(row: pd.Series, closed: bool = None,
                            exit_date_default=None) -> Position:
        """Convert a single DataFrame row (from cloud / closed / newly_closed) to a Position.

        Args:
            row: DataFrame row with position columns.
            closed: Override for closed status. If None (default), reads from row.
            exit_date_default: Default value for exit_date (passed to parse_dt).
        """
        if closed is None:
            closed = row['closed'] if 'closed' in row else False
        exit_date = parse_dt(row['exit_date'], default=exit_date_default)

        return Position(
            symbol=row['symbol'],
            quantity=float(row['shares']) if 'shares' in row else 0,
            entry_price=float(row['entry_price']
                              ) if 'entry_price' in row else 0,
            current_price=float(row['current_price']
                                ) if 'current_price' in row else 0,
            entry_date=parse_dt(row['entry_date'], default=datetime.now()),
            current_rsi=float(row['current_rsi']
                              ) if 'current_rsi' in row else 0.0,
            rsi_period=int(row['rsi_period']) if 'rsi_period' in row else 14,
            rsi_lower=int(row['rsi_lower']) if 'rsi_lower' in row else 30,
            rsi_upper=int(row['rsi_upper']) if 'rsi_upper' in row else 70,
            alpha=float(row['alpha']) if 'alpha' in row and pd.notna(
                row['alpha']) else 0.0,
            stop_loss_price=float(row['stop_loss_price']) if 'stop_loss_price' in row and pd.notna(
                row['stop_loss_price']) else None,
            take_profit_price=float(row['take_profit_price']) if 'take_profit_price' in row and pd.notna(
                row['take_profit_price']) else None,
            exit_price=float(row['exit_price']) if 'exit_price' in row and pd.notna(
                row['exit_price']) else None,
            realized_return=float(row['realized_return']) if 'realized_return' in row and pd.notna(
                row['realized_return']) else None,
            exit_reason=str(row['exit_reason']) if 'exit_reason' in row and pd.notna(
                row['exit_reason']) and row['exit_reason'] is not None else None,
            closed=closed,
            exit_date=exit_date,
            order_id=str(row['order_id']) if 'order_id' in row and pd.notna(
                row['order_id']) and row['order_id'] is not None else None,
            client_order_id=str(row['client_order_id']) if 'client_order_id' in row and pd.notna(
                row['client_order_id']) and row['client_order_id'] is not None else None,
        )

    def get_and_reconcile_positions(self) -> List[Position]:
        """
        Retrieves positions from cloud storage and alpaca and updates prices
        Returns a list of open Position objects.
        """

        # Short-circuit: return cached result if reconciliation was done
        # recently (avoids duplicate Alpaca API calls within a cycle).
        now = datetime.now().timestamp()
        if (
            self._reconciled_at is not None
            and self._cached_open_positions is not None
            and (now - self._reconciled_at) < self.RECONCILIATION_CACHE_TTL_SECONDS
        ):
            logger.debug(
                "Reconciliation cache hit — returning %d cached open positions (%.1fs ago)",
                len(self._cached_open_positions), now - self._reconciled_at,
            )
            return self._cached_open_positions

        alpaca_positions = self.data_provider.get_current_positions_df()
        cloud_positions = self.storage_backend.get_latest_positions_df(True)
        alpaca_symbols_list = (
            sorted(alpaca_positions['symbol'].astype(str).tolist())
            if not alpaca_positions.empty and 'symbol' in alpaca_positions.columns
            else []
        )
        cloud_symbols_list = (
            sorted(cloud_positions['symbol'].astype(str).tolist())
            if not cloud_positions.empty and 'symbol' in cloud_positions.columns
            else []
        )
        logger.info(
            "Reconciliation start: alpaca_open_rows=%d, cloud_open_rows=%d, alpaca_symbols=%s, cloud_symbols=%s",
            len(alpaca_positions), len(
                cloud_positions), alpaca_symbols_list, cloud_symbols_list
        )

        # Normalize cloud schema when there is no prior snapshot but Alpaca has open positions.
        if cloud_positions.empty:
            if not alpaca_positions.empty:
                logger.info(
                    "Cloud positions are empty. Initializing with Alpaca positions.")
                cloud_positions = pd.DataFrame({
                    'symbol': alpaca_positions['symbol'],
                    'shares': pd.to_numeric(
                        alpaca_positions.get('qty', 0), errors='coerce').fillna(0.0),
                    'entry_price': pd.to_numeric(
                        alpaca_positions.get('avg_entry_price', 0), errors='coerce').fillna(0.0),
                    'current_price': pd.to_numeric(
                        alpaca_positions.get('current_price', 0), errors='coerce').fillna(0.0),
                    'position_value': pd.to_numeric(
                        alpaca_positions.get('market_value', 0), errors='coerce').fillna(0.0),
                    'current_rsi': 0.0,
                    'entry_date': pd.Timestamp(datetime.now()).floor('s'),
                    'rsi_period': 14,
                    'rsi_lower': 30,
                    'rsi_upper': 70,
                    'alpha': 0.0,
                    'stop_loss_price': np.nan,
                    'take_profit_price': np.nan,
                    'exit_date': pd.NaT,
                    'exit_price': np.nan,
                    'realized_return': np.nan,
                    'exit_reason': None,
                    'closed': False,
                })
                # Mark that we just created the cloud snapshot from Alpaca so
                # we can enrich these rows using order history/backtests
                initialized_from_alpaca = True
            else:
                cloud_positions = pd.DataFrame()
                initialized_from_alpaca = False
        else:
            initialized_from_alpaca = False

        # If we initialized the cloud positions from Alpaca (no prior snapshot),
        # attempt to enrich each row with order-history-derived entry_date/price
        # and constrained backtests to avoid look-ahead bias.
        if 'initialized_from_alpaca' in locals() and initialized_from_alpaca:
            logger.info(
                "Enriching initialized Alpaca positions with order history/backtests")
            try:
                from optimizer import StrategyOptimizer
                optimizer = StrategyOptimizer()
            except (ImportError, AttributeError) as e:
                logger.warning(
                    "Could not initialize strategy optimizer for enrichment: %s", e)
                optimizer = None

            for idx, row in cloud_positions.iterrows():
                symbol = row['symbol']
                # Derive side from quantity sign (negative = short)
                qty = float(row.get('shares', 0) or 0)
                position_side = "short" if qty < 0 else "long"

                entry_date = parse_dt(
                    row.get('entry_date'), default=datetime.now())
                entry_price = float(row.get('entry_price', 0) or 0)
                order_info = None
                try:
                    order_info = self.data_provider.get_entry_order_for_symbol(
                        symbol, side=position_side)
                except (ValueError, RuntimeError, AttributeError):
                    order_info = None

                if order_info is not None:
                    try:
                        order_submitted_at, order_price = order_info
                        entry_date = order_submitted_at
                        entry_price = order_price
                    except (TypeError, ValueError):
                        # Unexpected return shape (e.g. a Mock); ignore and fall back
                        order_info = None

                # Default params
                current_rsi = 0.0
                rsi_period = 14
                rsi_lower = 30
                rsi_upper = 70
                alpha = 0.0

                if optimizer is not None:
                    try:
                        start_date = globalConfig.BACKTEST_START_DATE
                        if order_info is not None:
                            end_date = order_info[0] - timedelta(days=1)
                        else:
                            end_date = datetime.now() - timedelta(minutes=20)

                        if end_date > start_date:
                            backtest_result = optimizer.optimize_symbol(
                                symbol, start_date, end_date, direction=position_side
                            )
                            if backtest_result is not None:
                                current_rsi = float(
                                    backtest_result.current_rsi) if backtest_result.current_rsi is not None else 0.0
                                rsi_period = int(backtest_result.rsi_period)
                                rsi_lower = int(backtest_result.rsi_lower)
                                rsi_upper = int(backtest_result.rsi_upper)
                                alpha = float(backtest_result.alpha)
                        else:
                            logger.warning(
                                "%s: Entry date %s is too old for backtest window "
                                "(start=%s). Using default RSI parameters.",
                                symbol, entry_date, start_date
                            )
                    except (ValueError, TypeError, KeyError, RuntimeError) as e:
                        logger.warning(
                            "Backtest enrichment failed for %s: %s", symbol, e)

                # write enriched values back into the DataFrame (ensure datetime precision)
                try:
                    cloud_positions.at[idx, 'entry_date'] = pd.Timestamp(
                        entry_date).floor('s')
                except (ValueError, TypeError):
                    cloud_positions.at[idx, 'entry_date'] = entry_date
                cloud_positions.at[idx, 'entry_price'] = entry_price
                cloud_positions.at[idx, 'current_rsi'] = current_rsi
                cloud_positions.at[idx, 'rsi_period'] = rsi_period
                cloud_positions.at[idx, 'rsi_lower'] = rsi_lower
                cloud_positions.at[idx, 'rsi_upper'] = rsi_upper
                cloud_positions.at[idx, 'alpha'] = alpha

        # Reconcile positions from both sources
        newly_closed_positions = pd.DataFrame()
        if not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            alpaca_symbols = set(alpaca_positions['symbol']) if (
                not alpaca_positions.empty and 'symbol' in alpaca_positions.columns
            ) else set()
            cloud_symbols = set(cloud_positions['symbol'])
            logger.info(
                "Reconciliation symbol sets: alpaca_open=%d, cloud_open=%d",
                len(alpaca_symbols), len(cloud_symbols)
            )

            # Add Alpaca-only symbols as open positions (broker is source of truth for open holdings).
            alpaca_only_symbols = alpaca_symbols - cloud_symbols
            if alpaca_only_symbols:
                logger.warning(
                    "Found %d symbols in Alpaca that are not in cloud storage: %s. Adding as open positions.",
                    len(alpaca_only_symbols), alpaca_only_symbols)

                optimizer = None
                try:
                    # Local import avoids unnecessary startup cost and potential import cycles.
                    from optimizer import StrategyOptimizer
                    optimizer = StrategyOptimizer()
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        "Could not initialize strategy optimizer for Alpaca-only positions: %s", e)

                new_rows = []
                for symbol in alpaca_only_symbols:
                    alpaca_row = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol].iloc[0]

                    # Derive side from quantity sign (negative = short)
                    alpaca_qty = float(alpaca_row.get('qty', 0) or 0)
                    position_side = "short" if alpaca_qty < 0 else "long"

                    # Try to find the entry order from Alpaca order history
                    # to determine the real entry date and price, avoiding
                    # look-ahead bias in backtest enrichment.
                    entry_date = datetime.now()
                    entry_price = float(alpaca_row.get(
                        'avg_entry_price', 0) or 0)
                    order_info = None
                    try:
                        order_info = self.data_provider.get_entry_order_for_symbol(
                            symbol, side=position_side)
                    except (ValueError, RuntimeError, AttributeError):
                        order_info = None

                    if order_info is not None:
                        try:
                            order_submitted_at, order_price = order_info
                            logger.info(
                                "Found order history for Alpaca-only %s: submitted_at=%s, price=%.2f",
                                symbol, order_submitted_at, order_price
                            )
                            entry_date = order_submitted_at
                            entry_price = order_price
                        except (TypeError, ValueError):
                            order_info = None

                    current_rsi = 0.0
                    rsi_period = 14
                    rsi_lower = 30
                    rsi_upper = 70
                    alpha = 0.0
                    composite_score = 0.0

                    if optimizer is not None:
                        try:
                            start_date = globalConfig.BACKTEST_START_DATE

                            # If we have a real entry date from order history,
                            # end the backtest the day before submission to
                            # eliminate look-ahead bias.
                            if order_info is not None:
                                entry_submitted = order_info[0]
                                end_date = entry_submitted - timedelta(days=1)
                                logger.info(
                                    "%s: Running constrained backtest [%s to %s] based on order submitted_at",
                                    symbol, start_date.date() if hasattr(start_date, 'date') else start_date,
                                    end_date.date() if hasattr(end_date, 'date') else end_date
                                )
                            else:
                                end_date = datetime.now() - timedelta(minutes=20)
                                logger.info(
                                    "%s: No order history found. Running full-range backtest [%s to %s]",
                                    symbol,
                                    start_date.date() if hasattr(start_date, 'date') else start_date,
                                    end_date.date() if hasattr(end_date, 'date') else end_date
                                )

                            # Skip backtest if the entry is so old that
                            # the backtest window would be empty or negative.
                            if end_date <= start_date:
                                logger.warning(
                                    "%s: Entry date %s is too old for backtest window (start=%s). "
                                    "Using default RSI parameters.",
                                    symbol, entry_date, start_date
                                )
                            else:
                                backtest_result = optimizer.optimize_symbol(
                                    symbol, start_date, end_date, direction=position_side
                                )
                                if backtest_result is not None:
                                    current_rsi = float(
                                        backtest_result.current_rsi) if backtest_result.current_rsi is not None else 0.0
                                    rsi_period = int(
                                        backtest_result.rsi_period)
                                    rsi_lower = int(backtest_result.rsi_lower)
                                    rsi_upper = int(backtest_result.rsi_upper)
                                    alpha = float(backtest_result.alpha)
                                    composite_score = float(
                                        backtest_result.composite_score)
                        except (ValueError, TypeError, KeyError, RuntimeError) as e:
                            logger.warning(
                                "Backtest enrichment failed for Alpaca-only symbol %s: %s", symbol, e)

                    new_rows.append({
                        'symbol': symbol,
                        'shares': float(alpaca_row.get('qty', 0) or 0),
                        'entry_price': entry_price,
                        'current_price': float(alpaca_row.get('current_price', 0) or 0),
                        'position_value': float(alpaca_row.get('market_value', 0) or 0),
                        'current_rsi': current_rsi,
                        'entry_date': pd.Timestamp(entry_date).floor('s'),
                        'rsi_period': rsi_period,
                        'rsi_lower': rsi_lower,
                        'rsi_upper': rsi_upper,
                        'alpha': alpha,
                        'composite_score': composite_score,
                        'stop_loss_price': (
                            (entry_price * (1 + globalConfig.STOP_LOSS_PCT))
                            if entry_price > 0 and position_side == 'short'
                            else (entry_price * (1 - globalConfig.STOP_LOSS_PCT))
                            if entry_price > 0 else np.nan),
                        'take_profit_price': (
                            (entry_price * (1 - globalConfig.TAKE_PROFIT_PCT))
                            if entry_price > 0 and position_side == 'short'
                            else (entry_price * (1 + globalConfig.TAKE_PROFIT_PCT))
                            if entry_price > 0 else np.nan),
                        'exit_date': pd.NaT,
                        'exit_price': np.nan,
                        'realized_return': np.nan,
                        'exit_reason': None,
                        'closed': False,
                    })
                cloud_positions = pd.concat(
                    [cloud_positions, pd.DataFrame(new_rows)], ignore_index=True)
                logger.info(
                    "Added %d Alpaca-only symbols to in-memory positions snapshot",
                    len(new_rows)
                )

            # Any symbol still in cloud open positions but not in Alpaca is treated as closed.
            cloud_only_symbols = set(
                cloud_positions['symbol']) - alpaca_symbols
            if cloud_only_symbols:
                logger.warning(
                    "Found %d symbols in cloud storage that are not in Alpaca: %s. Marking as closed",
                    len(cloud_only_symbols), cloud_only_symbols)
                newly_closed_positions = cloud_positions[
                    cloud_positions['symbol'].isin(cloud_only_symbols)
                ].copy()
                # Mark cloud-only symbols as closed
                for symbol in cloud_only_symbols:
                    symbol_mask = cloud_positions['symbol'] == symbol

                    # Determine position side from quantity sign (negative = short)
                    cloud_qty = float(cloud_positions.loc[symbol_mask, 'shares'].values[0]) \
                        if 'shares' in cloud_positions.columns else 0.0
                    position_side = "short" if cloud_qty < 0 else "long"
                    # Close-order side: long→sell, short→buy
                    close_order_side = "sell" if position_side == "long" else "buy"
                    entry_order_side = "buy" if position_side == "long" else "sell"

                    stop_loss = cloud_positions.loc[symbol_mask, 'stop_loss_price'].values[0] \
                        if 'stop_loss_price' in cloud_positions.columns else np.nan
                    take_profit = cloud_positions.loc[symbol_mask, 'take_profit_price'].values[0] \
                        if 'take_profit_price' in cloud_positions.columns else np.nan
                    current_price = cloud_positions.loc[symbol_mask, 'current_price'].values[0] \
                        if 'current_price' in cloud_positions.columns else np.nan
                    entry_price = cloud_positions.loc[symbol_mask, 'entry_price'].values[0] \
                        if 'entry_price' in cloud_positions.columns else np.nan

                    exit_price_val = None
                    exit_date_val = None
                    exit_reason_val = None

                    # --- Step 0: exact fill by client_order_id/order_id ---
                    client_order_id = None
                    if 'client_order_id' in cloud_positions.columns:
                        cid_vals = cloud_positions.loc[
                            symbol_mask, 'client_order_id']
                        if len(cid_vals) and pd.notna(cid_vals.values[0]):
                            client_order_id = str(cid_vals.values[0])
                    matched = self._find_fill_by_client_order_id(
                        symbol, client_order_id)
                    if matched is not None and matched[0] and matched[0] > 0:
                        exit_price_val = matched[0]
                        filled_at = matched[2]
                        if pd.notna(filled_at):
                            exit_date_val = filled_at if isinstance(
                                filled_at, datetime) else datetime.fromisoformat(str(filled_at))
                            if exit_date_val.tzinfo is not None:
                                exit_date_val = exit_date_val.replace(
                                    tzinfo=None)
                        exit_reason_val = "matched_by_client_order_id"
                        logger.info(
                            "Reconcile %s: matched fill by client_order_id=%s at $%.2f",
                            symbol, client_order_id, exit_price_val
                        )

                    # --- Step 1: Check Alpaca order history for a real fill ---
                    try:
                        orders_df = self.data_provider.get_filled_orders_for_symbol(
                            symbol, limit=20)
                        if not orders_df.empty and 'side' in orders_df.columns:
                            # Check for a FILLED close order
                            close_orders = orders_df[
                                (orders_df['side'] == close_order_side) &
                                (orders_df['filled_qty'] > 0)
                            ]
                            if exit_price_val is None and not close_orders.empty:
                                latest_close = close_orders.iloc[0]
                                filled_price = float(
                                    latest_close['filled_avg_price'])
                                filled_at = latest_close.get('submitted_at')
                                if pd.notna(filled_at):
                                    exit_date_val = filled_at if isinstance(
                                        filled_at, datetime) else datetime.fromisoformat(str(filled_at))
                                    if exit_date_val.tzinfo is not None:
                                        exit_date_val = exit_date_val.replace(
                                            tzinfo=None)
                                exit_price_val = filled_price

                                # Determine whether it hit stop_loss or take_profit
                                if pd.notna(stop_loss) and pd.notna(take_profit):
                                    dist_stop = abs(filled_price - stop_loss)
                                    dist_take = abs(filled_price - take_profit)
                                    if dist_stop <= dist_take:
                                        exit_reason_val = "oco_stop_loss"
                                    else:
                                        exit_reason_val = "oco_take_profit"
                                else:
                                    exit_reason_val = "oco_filled"
                                logger.info(
                                    "Reconcile %s: found filled %s order at $%.2f (%s) — exit_reason=%s",
                                    symbol, close_order_side, filled_price,
                                    filled_at, exit_reason_val
                                )

                            # Check if any entry order ever existed
                            if exit_price_val is None:
                                entry_orders = orders_df[
                                    (orders_df['side'] == entry_order_side) &
                                    (orders_df['filled_qty'] > 0)
                                ]
                                if entry_orders.empty:
                                    exit_reason_val = "failed_to_open"
                                    exit_price_val = 0.0
                                    logger.warning(
                                        "Reconcile %s: no filled %s order found in history — "
                                        "marking as failed_to_open",
                                        symbol, entry_order_side
                                    )
                    except Exception as e:
                        logger.warning(
                            "Reconcile %s: order history lookup failed: %s", symbol, e)

                    # --- Step 2: Fallback to OCO approximation ---
                    if exit_price_val is None:
                        if (pd.notna(stop_loss) and pd.notna(take_profit)
                                and pd.notna(current_price) and current_price > 0):
                            if abs(current_price - stop_loss) <= abs(current_price - take_profit):
                                chosen = "stop_loss"
                                exit_price_val = stop_loss
                            else:
                                chosen = "take_profit"
                                exit_price_val = take_profit
                            logger.info(
                                "Reconcile OCO-fallback for %s: using %s=%.2f "
                                "(stop_loss=%.2f, take_profit=%.2f, current=%.2f, entry=%.2f)",
                                symbol, chosen, exit_price_val, stop_loss, take_profit,
                                current_price, entry_price
                            )
                        elif pd.notna(current_price) and current_price > 0:
                            exit_price_val = current_price
                        elif pd.notna(entry_price) and entry_price > 0:
                            exit_price_val = entry_price
                        else:
                            exit_price_val = 0.0

                    if exit_reason_val is None:
                        exit_reason_val = 'broker_closed'

                    cloud_positions.loc[symbol_mask,
                                        'exit_price'] = exit_price_val

                    if 'exit_date' not in cloud_positions.columns:
                        cloud_positions['exit_date'] = pd.NaT
                    cloud_positions.loc[symbol_mask,
                                        'exit_date'] = pd.Timestamp(
                                            exit_date_val if exit_date_val is not None
                                            else datetime.now()
                    ).floor('s')

                    if 'entry_price' in cloud_positions.columns:
                        if 'realized_return' not in cloud_positions.columns:
                            cloud_positions['realized_return'] = np.nan
                        entry_prices = pd.to_numeric(
                            cloud_positions.loc[symbol_mask, 'entry_price'], errors='coerce')
                        valid_mask = entry_prices > 0
                        if position_side == "short":
                            cloud_positions.loc[symbol_mask, 'realized_return'] = np.where(
                                valid_mask,
                                (entry_prices - exit_price_val) / entry_prices,
                                np.nan
                            )
                        else:
                            cloud_positions.loc[symbol_mask, 'realized_return'] = np.where(
                                valid_mask,
                                (exit_price_val - entry_prices) / entry_prices,
                                np.nan
                            )

                    cloud_positions.loc[cloud_positions['symbol']
                                        == symbol, 'closed'] = True
                    cloud_positions.loc[cloud_positions['symbol']
                                        == symbol, 'exit_reason'] = exit_reason_val
                    cloud_positions['closed'] = cloud_positions['closed'].astype(
                        bool)

                logger.info(
                    "Marked %d positions as closed in cloud storage", len(cloud_only_symbols))

                # Keep the updated closed rows so they can be re-added to self.positions
                # and persisted in the next positions snapshot save.
                newly_closed_positions = cloud_positions[
                    cloud_positions['symbol'].isin(cloud_only_symbols)
                ].copy()
        # Update cloud open positions with current broker values for symbols that are currently open in Alpaca.
        if not alpaca_positions.empty and not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            for index, row in cloud_positions.iterrows():
                symbol = row['symbol']
                if symbol in alpaca_positions['symbol'].values:
                    cloud_positions.at[index, 'current_price'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'current_price'].values[0]
                    cloud_positions.at[index, 'shares'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'qty'].values[0]
                    cloud_positions.at[index, 'position_value'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'market_value'].values[0]
                    # Alpaca is the source of truth for live position values.
                    # Always overwrite entry_price, shares, and current_price
                    # from the broker's current state.
                    cloud_positions.at[index, 'entry_price'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'avg_entry_price'].values[0]
                    # Ensure any datetime-like columns remain compatible when
                    # overwriting values coming from Alpaca.
                    if 'entry_date' in cloud_positions.columns:
                        try:
                            parsed = parse_dt(
                                cloud_positions.at[index, 'entry_date'])
                            if parsed is not None:
                                cloud_positions.at[index, 'entry_date'] = pd.Timestamp(
                                    parsed).floor('s')
                        except (ValueError, TypeError):
                            # Fall back to leaving the existing value if parsing fails
                            pass
        # Convert cloud positions DataFrame to a list of Position objects
        self.positions = []
        if not cloud_positions.empty:
            for _, row in cloud_positions.iterrows():
                if 'closed' in row and row['closed']:
                    continue  # Skip closed positions
                self.positions.append(self._df_row_to_position(row))

        # add the closed positions
        closed_positions = self.storage_backend.get_latest_positions_df(False)
        if not closed_positions.empty:
            for _, row in closed_positions.iterrows():
                self.positions.append(self._df_row_to_position(row))

        if not newly_closed_positions.empty:
            for _, row in newly_closed_positions.iterrows():
                self.positions.append(self._df_row_to_position(
                    row, closed=True, exit_date_default=datetime.now()))

        open_positions = [pos for pos in self.positions if not pos.closed]
        closed_positions_count = len(self.positions) - len(open_positions)
        logger.info(
            "Reconciliation complete: open_positions=%d, closed_positions=%d, total_in_memory=%d",
            len(open_positions), closed_positions_count, len(self.positions)
        )

        # Cache the result so subsequent calls within the same cycle
        # short-circuit and avoid duplicate Alpaca API calls.
        self._reconciled_at = datetime.now().timestamp()
        self._cached_open_positions = open_positions

        return open_positions

    def _find_fill_by_client_order_id(
        self, symbol: str, client_order_id: Optional[str]
    ) -> Optional[tuple]:
        """Find a filled order by client_order_id (fallback: order_id).

        Returns (filled_avg_price, filled_qty, filled_at) or None.
        Used to deterministically match a position to its close fill instead
        of guessing by side + most-recent order.
        """
        if not client_order_id:
            return None
        try:
            orders_df = self.data_provider.get_filled_orders_for_symbol(
                symbol, limit=50)
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        if orders_df.empty or 'client_order_id' not in orders_df.columns:
            return None

        cid = str(client_order_id)
        matched = orders_df[orders_df['client_order_id'].astype(str) == cid]
        if matched.empty and 'order_id' in orders_df.columns:
            matched = orders_df[orders_df['order_id'].astype(str) == cid]
        if matched.empty:
            return None

        row = matched.iloc[0]
        price = row.get('filled_avg_price')
        qty = row.get('filled_qty')
        filled_at = row.get('filled_at')
        try:
            price = float(price) if price is not None and pd.notna(price) else None
            qty = float(qty) if qty is not None and pd.notna(qty) else None
        except (TypeError, ValueError):
            return None
        return (price, qty, filled_at)

    def close_position(self, symbol: str):
        """
        Close a position by symbol.

        Marks the position as closed in-place (no longer removes from the list)
        so the realized return is preserved for subsequent cloud saves.
        """
        # Find the position in self.positions
        position_found = False
        target_position = None
        for i, position in enumerate(self.positions):
            if position.symbol == symbol and not position.closed:
                target_position = position
                position_found = True
                break

        if not position_found:
            logger.warning(
                "Position with symbol %s not found in current open positions.", symbol)
            return

        position_side = target_position.side

        # Determine which order side represents the close:
        #   Long positions  are closed with sell orders
        #   Short positions are closed with buy  orders (cover)
        close_order_side = "sell" if position_side == "long" else "buy"

        # Resolve the actual fill price for this close, preferring an exact
        # match by client_order_id/order_id and falling back to the
        # most-recent close-side heuristic.
        filled_exit_price = None
        matched = self._find_fill_by_client_order_id(
            symbol, getattr(target_position, 'client_order_id', None))
        if matched is not None and matched[0] and matched[0] > 0:
            filled_exit_price = matched[0]
            logger.info(
                "Found close fill for %s by client_order_id=%s: price=%.2f, qty=%s",
                symbol, getattr(target_position, 'client_order_id', None),
                filled_exit_price, matched[1]
            )

        if filled_exit_price is None:
            try:
                orders_df = self.data_provider.get_filled_orders_for_symbol(
                    symbol, limit=50)
                if not orders_df.empty and 'side' in orders_df.columns:
                    close_orders = orders_df[orders_df['side'] == close_order_side]
                    close_orders = close_orders[close_orders['filled_qty'] > 0]
                    if not close_orders.empty:
                        latest_close = close_orders.iloc[0]
                        filled_exit_price = latest_close['filled_avg_price']
                        logger.info(
                            "Found filled close order for %s (side=%s): price=%.2f, qty=%.2f",
                            symbol, close_order_side, filled_exit_price,
                            latest_close['filled_qty']
                        )
            except Exception as e:
                logger.warning(
                    "Could not fetch filled close price for %s: %s. Will use fallback.",
                    symbol, e
                )

        # Determine exit_price:
        #   1. Actual fill from Alpaca order history (best)
        #   2. OCO target (stop_loss or take_profit) closest to current_price
        #   3. current_price
        #   4. entry_price (last resort)
        if filled_exit_price is not None:
            exit_price = filled_exit_price
        else:
            exit_price = self._determine_exit_price_for_position(
                target_position, symbol
            )

        # Calculate realized return — formula differs for longs vs shorts.
        #   Long:  (exit - entry) / entry  → profit when exit > entry
        #   Short: (entry - exit) / entry  → profit when exit < entry (covered lower)
        if target_position.entry_price and target_position.entry_price > 0:
            if position_side == "short":
                realized_return = (
                    target_position.entry_price - exit_price
                ) / target_position.entry_price
            else:
                realized_return = (
                    exit_price - target_position.entry_price
                ) / target_position.entry_price
        else:
            realized_return = None

        # Mark the position closed in-place (keep in self.positions so
        # the end-of-session save picks up the realized return).
        target_position.closed = True
        target_position.exit_date = datetime.now()
        target_position.exit_price = exit_price
        target_position.realized_return = realized_return
        target_position.exit_reason = target_position.exit_reason or "manual"

        # Update cloud positions snapshot and persist immediately.
        cloud_positions = self.storage_backend.get_latest_positions_df(True)
        if not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            symbol_mask = cloud_positions['symbol'] == symbol

            cloud_positions.loc[symbol_mask, 'exit_price'] = exit_price
            cloud_positions.loc[symbol_mask, 'exit_date'] = datetime.now()
            cloud_positions.loc[symbol_mask, 'closed'] = True
            cloud_positions['closed'] = cloud_positions['closed'].astype(bool)

            if 'entry_price' in cloud_positions.columns:
                entry_prices = pd.to_numeric(
                    cloud_positions.loc[symbol_mask, 'entry_price'], errors='coerce')
                if position_side == "short":
                    cloud_positions.loc[symbol_mask, 'realized_return'] = np.where(
                        entry_prices > 0,
                        (entry_prices - exit_price) / entry_prices,
                        np.nan
                    )
                else:
                    cloud_positions.loc[symbol_mask, 'realized_return'] = np.where(
                        entry_prices > 0,
                        (exit_price - entry_prices) / entry_prices,
                        np.nan
                    )

            if 'exit_reason' not in cloud_positions.columns:
                cloud_positions['exit_reason'] = None
            cloud_positions.loc[symbol_mask,
                                'exit_reason'] = target_position.exit_reason

            self._persist_positions(cloud_positions)

        logger.info(
            "Closed %s position for %s: exit=%.2f, realized_return=%.4f",
            position_side, symbol, exit_price,
            realized_return if realized_return is not None else 0.0
        )

        # Invalidate reconciliation cache so the next reconcile reflects
        # the closed position without an extra Alpaca round-trip.
        self.invalidate_reconciliation_cache()

    def _determine_exit_price_for_position(self, position, symbol: str) -> float:
        """
        Determine the most likely exit price when the actual fill cannot
        be fetched from Alpaca order history.

        Tries OCO targets (stop_loss / take_profit) first — picking whichever
        is closer to current_price — then falls back to current_price or entry_price.
        """
        stop_loss = position.stop_loss_price
        take_profit = position.take_profit_price
        current_price = position.current_price
        entry_price = position.entry_price

        # Try OCO target approximation
        if (stop_loss is not None and take_profit is not None
                and current_price is not None and current_price > 0):
            dist_to_stop = abs(current_price - stop_loss)
            dist_to_take = abs(current_price - take_profit)
            if dist_to_stop <= dist_to_take:
                chosen = "stop_loss"
                exit_price = stop_loss
            else:
                chosen = "take_profit"
                exit_price = take_profit
            logger.info(
                "OCO-fallback exit price for %s: using %s=%.2f "
                "(stop_loss=%.2f, take_profit=%.2f, current=%.2f, entry=%.2f)",
                symbol, chosen, exit_price, stop_loss, take_profit,
                current_price, entry_price
            )
            return exit_price

        # Fall back to current_price or entry_price
        if current_price is not None and current_price > 0:
            logger.info(
                "Fallback exit price for %s: using current_price=%.2f "
                "(no OCO targets available)",
                symbol, current_price
            )
            return current_price
        elif entry_price is not None and entry_price > 0:
            logger.warning(
                "Fallback exit price for %s: using entry_price=%.2f "
                "(last resort — may report 0%% return)",
                symbol, entry_price
            )
            return entry_price
        else:
            logger.error(
                "Could not determine any exit price for %s", symbol
            )
            return 0.0

    def open_position(self, position: Position):
        """
        Adds a new position to the current positions.
        Args:
            position (Position): The position to add.
        """
        # Check if the position already exists
        for existing_position in self.positions:
            if existing_position.symbol == position.symbol and not existing_position.closed:
                logger.warning(
                    "Position for %s already exists. Use update_position to modify it.", position.symbol)
                return

        # Add the new position
        self.positions.append(position)
        # Invalidate reconciliation cache so the next reconcile picks up
        # the newly opened position from the broker.
        self.invalidate_reconciliation_cache()
        # Save the updated positions to cloud storage
        # self.storage_backend.save_positions(self.positions) SAVE AT END
        logger.info(
            "Opened new position for %s. Full saved positions: %s", position.symbol, self.positions)

    def invalidate_reconciliation_cache(self) -> None:
        """Clear the reconciliation cache, forcing a fresh reconcile on next call.

        Call this after any operation that changes positions (open, close, etc.)
        so subsequent get_and_reconcile_positions() calls see the latest state.
        """
        self._reconciled_at = None
        self._cached_open_positions = None
