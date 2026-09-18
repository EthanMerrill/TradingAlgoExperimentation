"""
Trading execution module.
Handles order placement, position management, and portfolio updates.
"""
import logging
import time
from dataclasses import dataclass
# pylint: disable=broad-exception-caught
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (OrderClass, OrderSide, OrderType,
                                  QueryOrderStatus, TimeInForce)
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest,
                                     MarketOrderRequest, StopLossRequest,
                                     TakeProfitRequest)
from data_provider import TechnicalIndicators, data_provider
from storage import storage
from order import Order, generate_client_order_id
from positions import Position, PositionsManager
from utils import ensure_utc
from strategies.base import StrategyContext
from strategies.registry import get_strategy
from strategy import BacktestResult, RSIStrategy

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)

# Alpaca order statuses that mean the order is no longer reserving shares.
# NOTE: 'pending_cancel' is intentionally EXCLUDED — the qty stays held until
# the cancel actually settles, which is the whole race we are guarding against.
_QTY_RELEASING_STATUSES = frozenset({
    'filled', 'canceled', 'cancelled', 'expired', 'replaced',
    'done_for_day', 'stopped', 'rejected',
})

# Alpaca statuses that mean the order no longer exists / holds no qty.
_TERMINAL_ORDER_STATUSES = frozenset({
    'filled', 'canceled', 'cancelled', 'expired', 'rejected', 'suspended',
    'done_for_day', 'stopped', 'replaced',
})

# Statuses where an existing protective order CANNOT be promptly cancelled and
# replaced:
#   * 'held'           — order is queued for the next market open; the cancel
#                        does not settle until the session opens, so the shares
#                        stay reserved (`held_for_orders`) and a replacement is
#                        rejected with 40310000.
#   * 'pending_cancel' — a cancel is already in flight; cancelling again fails
#                        with 42210000 and the qty is still reserved.
#   * 'pending_replace'— a replace is already in flight.
# Trying to cancel-and-replace in these states is what stripped protective
# orders off positions, so we skip the refresh entirely and keep what we have.
_UNREPLACEABLE_ORDER_STATUSES = frozenset({
    'held', 'pending_cancel', 'pending_replace',
})


def _status_str(value: Any) -> str:
    """Normalize an Alpaca status enum (or raw value) to a lowercase string."""
    if value is None:
        return ''
    return str(getattr(value, 'value', value)).lower()


def _is_insufficient_qty_error(exc: Exception) -> bool:
    """True when Alpaca rejects an order because held qty is not yet released.

    Matches the ``40310000`` API error code ("insufficient qty available for
    order").  This fires when a position's shares are still reserved by a
    just-cancelled protective order that has not settled yet.
    """
    if str(getattr(exc, 'code', '')) == '40310000':
        return True
    text = str(exc).lower()
    return 'insufficient qty' in text or '40310000' in text


def _strategy_is_bar_loop(strategy_name: str) -> bool:
    """True if a registered strategy runs on the bar loop (intraday)."""
    try:
        return get_strategy(strategy_name).execution_style == "bar_loop"
    except ValueError:
        return False


@dataclass
class TradingOpportunity:
    """Trading opportunity based on strategy results."""
    symbol: str
    current_rsi: float
    target_rsi_lower: int
    target_rsi_upper: int
    rsi_period: int
    backtest_return: float
    alpha: float
    win_rate: float
    entry_price: float
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    num_trades: int = 0  # Number of trades in backtest for this symbol
    # Cross-symbol Z-score (alpha + sharpe + calmar, normalised)
    composite_score: float = 0.0
    direction: str = "long"  # "long" or "short"
    # Owning strategy (registry key). Defaults to the legacy RSI strategy so
    # existing callers/positions stay backward compatible.
    strategy_name: str = "rsi_mean_reversion"
    # Intraday (bar-loop) position — entry/exit managed by BarLoopEngine.
    intraday: bool = False


class TradingEngine:
    """Main trading execution engine."""

    def __init__(self):
        self.trading_client: Optional[TradingClient] = data_provider.trading_client
        # PositionsManager is injected after construction via
        # set_positions_manager() so that TradingAlgorithm and
        # TradingEngine share a single source of position state.
        self._positions_manager: Optional[PositionsManager] = None
        self._last_position_update: Optional[datetime] = None
        self.dry_run: bool = False
        # Per-cycle OHLCV cache: avoids redundant API calls when multiple
        # methods (price, RSI, take-profit) need data for the same symbol.
        # Keyed by symbol, cleared at the start of each run cycle.
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}

    def set_positions_manager(self, manager: PositionsManager) -> None:
        """Inject a shared PositionsManager instance (single source of state)."""
        self._positions_manager = manager

    def set_dry_run_mode(self, dry_run: bool) -> None:
        """Enable or disable dry run mode."""
        self.dry_run = dry_run
        if dry_run:
            logger.info(
                "🌵 DRY RUN MODE ENABLED - No actual orders will be placed")
        else:
            logger.info("🚀 LIVE TRADING MODE ENABLED - Orders will be placed")

    def _identify_opportunities(
        self, backtest_results: List[BacktestResult], direction: str
    ) -> List[TradingOpportunity]:
        """Unified, strategy-aware opportunity identification.

        Groups backtest results by ``strategy_name``. Non-RSI registered
        strategies are asked for live signals via their ``evaluate_live_signals``
        hook; the legacy RSI cross logic (engine-native) handles
        ``rsi_mean_reversion`` results. After merging, shared filters apply:
        composite-score sort, alpha/win-rate/trade minimums, existing-position
        dedup, and cross-strategy symbol dedup (highest composite wins).

        Args:
            backtest_results: List of backtest results
            direction: "long" or "short"

        Returns:
            List of trading opportunities sorted by composite_score desc.
        """
        grouped: Dict[str, List[BacktestResult]] = {}
        for result in backtest_results:
            name = getattr(result, "strategy_name", "rsi_mean_reversion")
            grouped.setdefault(name, []).append(result)

        opportunities: List[TradingOpportunity] = []
        for name, results in grouped.items():
            try:
                strategy_cls = get_strategy(name)
            except ValueError:
                strategy_cls = None
                logger.warning(
                    "Backtest results reference unknown strategy '%s' — "
                    "falling back to the legacy RSI opportunity path", name)

            if strategy_cls is not None and strategy_cls.execution_style == "bar_loop":
                # Bar-loop strategies are evaluated by BarLoopEngine on bar
                # close during RTH — the daily session does not trade them.
                logger.debug(
                    "Strategy '%s' is bar_loop — entries managed by BarLoopEngine", name)
                continue

            if strategy_cls is not None and name != "rsi_mean_reversion":
                # Strategy-provided live signals (new framework path).
                try:
                    ctx = StrategyContext(
                        data_provider=data_provider,
                        positions_manager=self._positions_manager,
                        config=globalConfig,
                        as_of=datetime.now(),
                        ohlcv_cache=self._ohlcv_cache,
                        strategy_results=list(results),
                    )
                    signals = strategy_cls().evaluate_live_signals(ctx) or []
                    opportunities.extend(
                        self._signals_to_opportunities(signals, direction, name))
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.error(
                        "Error evaluating %s signals for strategy '%s': %s",
                        direction, name, e)
                    continue
            else:
                # Legacy RSI cross path (rsi_mean_reversion + unknown names).
                opportunities.extend(
                    self._rsi_opportunities(results, direction))

        # Shared post-loop filtering
        priority = {
            s: i for i, s in enumerate(
                getattr(globalConfig, "STRATEGIES_ENABLED", None) or [])
        }
        opportunities.sort(key=lambda x: (
            -x.composite_score, priority.get(x.strategy_name, 999)))
        opportunities = [op for op in opportunities if op.alpha > 0]
        opportunities = [
            op for op in opportunities if op.win_rate >= globalConfig.MIN_WIN_RATE]
        opportunities = [
            op for op in opportunities if op.num_trades >= globalConfig.MIN_NUM_TRADES]

        # Cross-strategy overlap policy: a symbol may only be traded once.
        # The list is already sorted by (composite_score desc, config order),
        # so the first occurrence per symbol wins.
        seen_symbols: set = set()
        deduped: List[TradingOpportunity] = []
        for op in opportunities:
            if op.symbol in seen_symbols:
                continue
            seen_symbols.add(op.symbol)
            deduped.append(op)
        opportunities = deduped

        # Existing-position dedup: exclude only same-direction open positions.
        # Opposite-direction holdings are handled as exits (no flip logic).
        if direction == "long":
            current_symbols = {
                pos.symbol for pos in self._positions_manager.positions
                if not pos.closed and getattr(pos, 'side', 'long') == 'long'}
        else:
            current_symbols = {
                pos.symbol for pos in self._positions_manager.positions
                if not pos.closed and getattr(pos, 'side', 'long') == 'short'}
        opportunities = [
            op for op in opportunities if op.symbol not in current_symbols]

        return opportunities

    def _rsi_opportunities(
        self, backtest_results: List[BacktestResult], direction: str
    ) -> List[TradingOpportunity]:
        """Legacy RSI cross-detection opportunity path (engine-native)."""
        is_long = direction == "long"
        opportunities: List[TradingOpportunity] = []

        for result in backtest_results:
            try:
                # Direction filter
                if not is_long and result.direction != "short":
                    continue

                current_rsi, previous_rsi = self._get_rsi_with_previous(
                    result.symbol, result.rsi_period)
                if current_rsi is None:
                    continue

                # Cross-detection
                if is_long:
                    is_cross = current_rsi < result.rsi_lower and (
                        previous_rsi is None or previous_rsi >= result.rsi_lower
                    )
                else:
                    is_cross = current_rsi > result.rsi_upper and (
                        previous_rsi is None or previous_rsi <= result.rsi_upper
                    )

                if previous_rsi is None:
                    logger.debug(
                        "Previous RSI unavailable for %s; using level check as fallback", result.symbol)

                if is_cross:
                    current_price = self._get_current_price(result.symbol)
                    if current_price is None:
                        continue

                    entry_price = round(current_price, 2)
                    if is_long:
                        stop_loss_price = round(
                            entry_price * (1 - globalConfig.STOP_LOSS_PCT), 2)
                        take_profit_price = self._compute_rsi_take_profit(
                            result.symbol, result.rsi_upper, result.rsi_period, entry_price)
                    else:
                        stop_loss_price = round(
                            entry_price * (1 + globalConfig.STOP_LOSS_PCT), 2)
                        take_profit_price = self._compute_rsi_cover_price(
                            result.symbol, result.rsi_lower, result.rsi_period, entry_price)

                    opportunity = TradingOpportunity(
                        symbol=result.symbol,
                        current_rsi=round(current_rsi, 2),
                        target_rsi_lower=result.rsi_lower,
                        target_rsi_upper=result.rsi_upper,
                        rsi_period=result.rsi_period,
                        backtest_return=round(result.total_return, 2),
                        alpha=round(result.alpha, 2),
                        win_rate=round(result.win_rate, 2),
                        entry_price=entry_price,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        num_trades=result.num_trades,
                        composite_score=round(result.composite_score, 2),
                        direction="long" if is_long else "short",
                        strategy_name=getattr(
                            result, "strategy_name", "rsi_mean_reversion"),
                    )
                    opportunities.append(opportunity)

            except Exception as e:
                logger.error("Error evaluating %s opportunity for %s: %s",
                             direction, result.symbol, e)
                continue

        return opportunities

    def _signals_to_opportunities(
        self, signals: List[Any], direction: str, strategy_name: str
    ) -> List[TradingOpportunity]:
        """Convert strategy-emitted LiveSignals into TradingOpportunities.

        Strategy-specific fields live in ``signal.extra``; the common fields
        map 1:1 onto TradingOpportunity (RSI-specific fields are zeroed for
        non-RSI strategies).
        """
        opportunities: List[TradingOpportunity] = []
        for sig in signals:
            try:
                if getattr(sig, "direction", "long") != direction:
                    continue
                entry_price = getattr(sig, "entry_price", None)
                if entry_price is None:
                    logger.debug(
                        "Signal for %s has no entry price — skipping", sig.symbol)
                    continue
                opportunities.append(TradingOpportunity(
                    symbol=sig.symbol,
                    current_rsi=0.0,
                    target_rsi_lower=0,
                    target_rsi_upper=0,
                    rsi_period=14,
                    backtest_return=round(
                        getattr(sig, "backtest_return", 0.0), 2),
                    alpha=round(getattr(sig, "alpha", 0.0), 2),
                    win_rate=round(getattr(sig, "win_rate", 0.0), 2),
                    entry_price=round(float(entry_price), 2),
                    stop_loss_price=(
                        round(float(sig.stop_loss), 2)
                        if getattr(sig, "stop_loss", None) is not None else None),
                    take_profit_price=(
                        round(float(sig.take_profit), 2)
                        if getattr(sig, "take_profit", None) is not None else None),
                    num_trades=int(getattr(sig, "num_trades", 0)),
                    composite_score=round(
                        getattr(sig, "composite_score", 0.0), 2),
                    direction=direction,
                    strategy_name=strategy_name,
                    intraday=_strategy_is_bar_loop(strategy_name),
                ))
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Error converting %s signal for strategy '%s': %s",
                    direction, strategy_name, e)
                continue
        return opportunities

    def identify_buying_opportunities(self, backtest_results: List[BacktestResult]) -> List[TradingOpportunity]:
        """Identify current buying opportunities based on backtest results."""
        return self._identify_opportunities(backtest_results, "long")

    def identify_shorting_opportunities(self, backtest_results: List[BacktestResult]) -> List[TradingOpportunity]:
        """Identify current short-selling opportunities based on backtest results."""
        return self._identify_opportunities(backtest_results, "short")

    def _strategy_budgets(self, equity: float) -> Dict[str, float]:
        """Return per-strategy capital budgets: ``strategy_name -> notional cap``.

        Weights come from ``globalConfig.STRATEGY_ALLOCATION`` (normalized so
        explicit weights take priority; unweighted enabled strategies split the
        leftover equally). With no weights at all, enabled strategies split
        equity evenly. Strategies with no budget entry get the full equity
        (legacy single-strategy behavior).
        """
        enabled = list(getattr(
            globalConfig, "STRATEGIES_ENABLED", None) or ["rsi_mean_reversion"])
        if not enabled:
            enabled = ["rsi_mean_reversion"]
        alloc = getattr(globalConfig, "STRATEGY_ALLOCATION", None) or {}
        explicit = {
            s: float(alloc[s])
            for s in enabled
            if alloc.get(s) is not None and float(alloc[s]) > 0
        }
        total_explicit = sum(explicit.values())
        missing = [s for s in enabled if s not in explicit]
        leftover_each = 0.0
        if missing:
            if total_explicit < 1.0:
                leftover_each = (1.0 - total_explicit) / len(missing)
        if not explicit:
            # No weights configured — even split across enabled strategies.
            even = 1.0 / len(enabled)
            return {s: equity * even for s in enabled}
        return {s: equity * (explicit.get(s, leftover_each)) for s in enabled}

    def _strategy_notional_used(self) -> Dict[str, float]:
        """Notional currently deployed per strategy (open positions only)."""
        used: Dict[str, float] = {}
        for pos in (self._positions_manager.positions or []):
            if getattr(pos, "closed", False):
                continue
            name = getattr(pos, "strategy_name", "rsi_mean_reversion")
            used[name] = used.get(name, 0.0) + \
                pos.entry_price * abs(pos.quantity)
        return used

    def calculate_position_sizes(self, opportunities: List[TradingOpportunity]) -> List[Tuple[TradingOpportunity, int]]:
        """
        Calculate position sizes for trading opportunities.

        Args:
            opportunities: List of trading opportunities

        Returns:
            List of(opportunity, shares) tuples
        """
        try:
            account_info = data_provider.get_account_info()
            current_positions = [
                pos for pos in self._positions_manager.positions if not pos.closed]

            if not account_info:
                logger.warning(
                    "Account info not available - cannot calculate position sizes")
                return []

            cash = account_info['cash']
            equity = account_info['equity']
            buying_power = account_info.get('buying_power', 0)
            logger.info(
                "Cash available: $%.2f, Equity: $%.2f, Buying Power: $%.2f", cash, equity, buying_power)

            # Check if we have enough buying power to trade
            if buying_power <= 0:
                logger.info("Insufficient buying power available")
                return []

            # Calculate how many new positions we can take
            current_position_count = len(current_positions)
            max_new_positions = min(
                globalConfig.MAX_NEW_POSITIONS_PER_DAY,
                globalConfig.MAX_POSITIONS - current_position_count
            )

            if max_new_positions <= 0:
                logger.info("No new positions allowed")
                return []

            # Select top opportunities up to max new positions
            selected_opportunities = opportunities[:max_new_positions]

            # Per-strategy capital budgets (Phase C multi-strategy allocation).
            budgets = self._strategy_budgets(equity)
            strategy_used = self._strategy_notional_used()

            # Calculate position size for each opportunity
            position_allocations = []

            for opportunity in selected_opportunities:
                # Strategy budget cap: never exceed the strategy's allocated
                # notional (existing open positions + this new position).
                budget = budgets.get(opportunity.strategy_name, equity)
                available = budget - strategy_used.get(
                    opportunity.strategy_name, 0.0)
                if available <= 0:
                    logger.info(
                        "Skipping %s (%s): strategy budget exhausted "
                        "($%.2f remaining)",
                        opportunity.symbol, opportunity.strategy_name, available)
                    continue

                # Equal weight allocation, capped by the strategy budget
                position_value = min(
                    equity * globalConfig.POSITION_SIZE_PCT,
                    available,
                )
                shares = int(position_value / opportunity.entry_price)

                if shares > 0:
                    position_allocations.append((opportunity, shares))
                    strategy_used[opportunity.strategy_name] = (
                        strategy_used.get(opportunity.strategy_name, 0.0)
                        + shares * opportunity.entry_price
                    )

            return position_allocations

        except Exception as e:
            logger.error("Error calculating position sizes: %s", e)
            return []

    def calculate_short_position_sizes(self, opportunities: List[TradingOpportunity]) -> List[Tuple[TradingOpportunity, int]]:
        """
        Calculate position sizes for short-selling opportunities.

        Enforces max_short_long_ratio: current + new short notional must not exceed
        equity * MAX_SHORT_LONG_RATIO.

        Args:
            opportunities: List of short trading opportunities

        Returns:
            List of (opportunity, shares) tuples
        """
        try:
            account_info = data_provider.get_account_info()
            current_positions = [
                pos for pos in self._positions_manager.positions if not pos.closed]

            if not account_info:
                logger.warning(
                    "Account info not available - cannot calculate short position sizes")
                return []

            equity = account_info['equity']

            # Calculate total notional value of existing short positions.
            # quantity is negative for shorts, so use abs() — otherwise the
            # cap is INCREASED by the existing short size instead of reduced.
            current_short_notional = sum(
                pos.entry_price * abs(pos.quantity)
                for pos in current_positions
                if getattr(pos, 'side', 'long') == 'short'
            )

            max_short_notional = equity * globalConfig.MAX_SHORT_LONG_RATIO
            available_short_notional = max_short_notional - current_short_notional

            logger.info(
                "Short leverage: existing=%d, notional=$%.2f, max=$%.2f, available=$%.2f",
                sum(1 for p in current_positions if getattr(
                    p, 'side', 'long') == 'short'),
                current_short_notional, max_short_notional, available_short_notional
            )

            if available_short_notional <= 0:
                logger.info(
                    "Short notional cap reached — no additional shorts allowed")
                return []

            # Calculate how many new short positions we can take
            current_position_count = len(current_positions)
            max_new_positions = min(
                globalConfig.MAX_NEW_POSITIONS_PER_DAY,
                globalConfig.MAX_POSITIONS - current_position_count
            )

            if max_new_positions <= 0:
                logger.info("No new short positions allowed (position cap)")
                return []

            selected_opportunities = opportunities[:max_new_positions]

            if not selected_opportunities:
                return []

            # Per-strategy capital budgets (Phase C multi-strategy allocation).
            budgets = self._strategy_budgets(equity)
            strategy_used = self._strategy_notional_used()

            position_allocations = []

            # Distribute remaining short capacity evenly
            per_position_notional = available_short_notional / \
                len(selected_opportunities)

            for opportunity in selected_opportunities:
                # Strategy budget cap (same rule as longs)
                budget = budgets.get(opportunity.strategy_name, equity)
                available = budget - strategy_used.get(
                    opportunity.strategy_name, 0.0)
                if available <= 0:
                    logger.info(
                        "Skipping short %s (%s): strategy budget exhausted "
                        "($%.2f remaining)",
                        opportunity.symbol, opportunity.strategy_name, available)
                    continue

                # Cap each to the per_position_notional, the strategy budget,
                # or a percentage of equity, whichever is smaller
                position_value = min(
                    per_position_notional,
                    equity * globalConfig.POSITION_SIZE_PCT,
                    available,
                )
                shares = int(position_value / opportunity.entry_price)

                if shares > 0:
                    position_allocations.append((opportunity, shares))
                    strategy_used[opportunity.strategy_name] = (
                        strategy_used.get(opportunity.strategy_name, 0.0)
                        + shares * opportunity.entry_price
                    )
                    logger.info(
                        "Short alloc for %s: %d shares @ $%.2f = $%.2f notional",
                        opportunity.symbol, shares, opportunity.entry_price,
                        shares * opportunity.entry_price
                    )

            return position_allocations

        except Exception as e:
            logger.error("Error calculating short position sizes: %s", e)
            return []

    def _place_order(self, opportunity: TradingOpportunity, shares: int, side: OrderSide, quantity_sign: int,
                     label: str, profit_label: str) -> bool:
        """Unified order placement for long (buy) and short (sell) orders.

        Args:
            opportunity: Trading opportunity
            shares: Number of shares
            side: OrderSide.BUY or OrderSide.SELL
            quantity_sign: 1 for long, -1 for short
            label: Human-readable label (e.g. "buy", "SHORT")
            profit_label: Human-readable profit label (e.g. "Take profit", "Cover target")

        Returns:
            True if order was placed successfully
        """
        order_success = False
        client_order_id: Optional[str] = None
        placed_order_id: Optional[str] = None
        try:
            if self.dry_run:
                logger.info("🔍 DRY RUN: Would place %s order for %d shares of %s at $%.2f",
                            label, shares, opportunity.symbol, opportunity.entry_price)
                logger.info("🔍 DRY RUN: Stop loss: $%.2f, %s: $%.2f",
                            opportunity.stop_loss_price, profit_label, opportunity.take_profit_price)
                logger.info("🔍 DRY RUN: Position value: $%.2f",
                            shares * opportunity.entry_price)
                return True
            else:
                if self.trading_client is None:
                    logger.error(
                        "Trading client not available - cannot place order")
                    return False

                client_order_id = self._make_unique_client_order_id(
                    generate_client_order_id(
                        opportunity.symbol,
                        "BUY" if side == OrderSide.BUY else "SELL",
                        datetime.now(),
                    )
                )

                order_request = MarketOrderRequest(
                    symbol=opportunity.symbol,
                    qty=shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(
                        stop_price=opportunity.stop_loss_price),
                    take_profit=TakeProfitRequest(
                        limit_price=opportunity.take_profit_price),
                    client_order_id=client_order_id,
                )

                order = self.trading_client.submit_order(order_request)
                placed_order_id = getattr(order, 'id', None)
                logger.info("Order placed successfully: %s", placed_order_id)
                logger.info("%s order for %d shares of %s at $%.2f",
                            label.title(), shares, opportunity.symbol, opportunity.entry_price)
                logger.info("Stop loss: $%.2f, %s: $%.2f",
                            opportunity.stop_loss_price, profit_label, opportunity.take_profit_price)

                order_success = True

        except Exception as e:
            error_msg = "Error placing %s order for %s: %s" % (
                label, opportunity.symbol, e)
            if self.dry_run:
                error_msg = "🔍 DRY RUN: " + error_msg
            logger.error(error_msg)

        if order_success:
            try:
                new_position = Position(
                    symbol=opportunity.symbol,
                    quantity=float(shares) * quantity_sign,
                    entry_price=opportunity.entry_price,
                    current_price=opportunity.entry_price,
                    current_rsi=opportunity.current_rsi,
                    entry_date=datetime.now(),
                    alpha=opportunity.alpha,
                    rsi_period=opportunity.rsi_period,
                    rsi_lower=opportunity.target_rsi_lower,
                    rsi_upper=opportunity.target_rsi_upper,
                    stop_loss_price=opportunity.stop_loss_price,
                    take_profit_price=opportunity.take_profit_price,
                    closed=False,
                    exit_date=None,
                    order_id=placed_order_id,
                    client_order_id=client_order_id,
                    strategy_name=opportunity.strategy_name,
                    intraday=opportunity.intraday,
                )

                self._positions_manager.open_position(new_position)
            except Exception as e:
                logger.error("Error adding position to positions manager for %s: %s",
                             opportunity.symbol, e)

            # Persist the order to the ledger (idempotent via client_order_id).
            try:
                if client_order_id:
                    storage.save_orders([
                        Order(
                            client_order_id=client_order_id,
                            order_id=placed_order_id,
                            symbol=opportunity.symbol,
                            side="buy" if side == OrderSide.BUY else "sell",
                            qty=float(shares),
                            order_type="market",
                            order_class="bracket",
                            status="new",
                            stop_price=opportunity.stop_loss_price,
                            limit_price=opportunity.take_profit_price,
                            submitted_at=datetime.now(),
                            leg="entry",
                        )
                    ])
            except Exception as e:
                logger.error("Error saving order to storage for %s: %s",
                             opportunity.symbol, e)
        elif not self.dry_run:
            logger.warning(
                "Order for %s did not succeed — position NOT added to manager", opportunity.symbol)

        return order_success

    def place_buy_order(self, opportunity: TradingOpportunity, shares: int) -> bool:
        """Place a buy order for a trading opportunity."""
        return self._place_order(opportunity, shares, OrderSide.BUY, 1, "buy", "Take profit")

    def place_short_order(self, opportunity: TradingOpportunity, shares: int) -> bool:
        """Place a short-sell order for a trading opportunity."""
        return self._place_order(opportunity, shares, OrderSide.SELL, -1, "SHORT", "Cover target")

    def _find_open_position(self, symbol: str) -> Optional[Position]:
        """Return the open position for a symbol, or None."""
        return next(
            (pos for pos in self._positions_manager.positions
             if pos.symbol == symbol and not pos.closed),
            None
        )

    def _has_opposite_position(self, symbol: str, direction: str) -> bool:
        """True if there is an open position in the opposite direction."""
        existing = self._find_open_position(symbol)
        if existing is None:
            return False
        return getattr(existing, 'side', 'long') != direction

    def _exit_opposite_position(self, symbol: str, new_direction: str) -> bool:
        """
        Exit an open position held in the opposite direction of ``new_direction``.

        Replaces the old flip logic: when a signal fires in the opposite
        direction of an existing position, we close that position instead of
        opening a new opposite position.

        Args:
            symbol: Stock symbol
            new_direction: The direction we are considering ("long" or "short")

        Returns:
            True if an opposite position was closed; False if there was none
            or the close failed.
        """
        existing = self._find_open_position(symbol)
        if existing is None:
            return False

        existing_side = getattr(existing, 'side', 'long')
        if existing_side == new_direction:
            return False  # Same direction — nothing to exit

        logger.info(
            "🔄 Exiting %s: closing existing %s position (signal in opposite direction)",
            symbol, existing_side
        )

        # Use the side-aware market close so the exit is persisted to the
        # order ledger, records the real fill price, and only marks the
        # position closed when the broker close succeeds.
        if self.close_position_at_market(existing, "opposite_signal"):
            return True

        logger.error("Failed to exit opposite position for %s", symbol)
        return False

    # -- position exit helpers ----------------------------------------------

    @staticmethod
    def _is_stop_breached(side: str, stop_loss_price: Any,
                          current_price: Any) -> bool:
        """True when a protective stop is already at/through the market price.

        Stops are anchored to the *entry* price, so a position that has moved
        further than ``STOP_LOSS_PCT`` against us produces a stop that has
        already been triggered.  Alpaca will not hold such a level as an OCO
        leg — it cancels the sibling take-profit leg and leaves a standalone
        stop — so the position keeps running past its risk limit instead of
        exiting.  Such positions must be closed at market instead.

        Non-numeric/unknown prices return False (treated as "not breached") so
        a missing quote never triggers an unintended liquidation.
        """
        if stop_loss_price is None or current_price is None:
            return False
        try:
            stop = float(stop_loss_price)
            current = float(current_price)
        except (TypeError, ValueError):
            return False

        if side == "short":
            # A short's stop sits ABOVE entry; it is breached once the market
            # rises to (or past) it.
            return current >= stop
        # A long's stop sits BELOW entry; it is breached once the market
        # falls to (or past) it.
        return current <= stop

    def _release_shares_for_exit(self, symbol: str, shares: float) -> bool:
        """Free a position's shares so a market exit can be submitted.

        A live stop/TP order reserves the shares, so a market close is rejected
        with 40310000 until those orders are cancelled.  Unlike an OCO
        *refresh* (which must keep protection when it cannot replace it), an
        exit is terminal — cancelling the protective orders is the correct
        move, and the position is about to be flat anyway.
        """
        if self.dry_run:
            return True

        try:
            blocking = self._get_active_orders_for_symbol(symbol)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Could not list orders blocking exit of %s: %s", symbol, e)
            return True

        if not blocking:
            return True  # nothing reserved — the shares are already free

        cancelled = self._cancel_open_orders_for_symbol(symbol)
        if cancelled and not self._wait_for_orders_cancelled(cancelled):
            logger.warning(
                "Cancel of %s for %s did not settle in time — attempting the "
                "exit anyway", cancelled, symbol)
        self._wait_for_position_qty_available(symbol, abs(float(shares)))
        return True

    def _force_close_position(self, session_summary: Dict[str, Any],
                              position: Position, reason: str) -> bool:
        """Exit ``position`` at market, recording the exit ``reason``.

        Shared by the max-hold-day and breached-stop paths.  Returns True when
        the close order was accepted.
        """
        side = getattr(position, 'side', 'long')
        shares = abs(position.quantity)

        # Release any protective orders holding the shares, otherwise the
        # market exit is rejected with 40310000 "insufficient qty".
        self._release_shares_for_exit(position.symbol, shares)

        if not self.close_position_at_market(position, reason):
            return False

        session_summary['positions_exited'] += 1
        self._record_order(
            session_summary, symbol=position.symbol,
            action='CLOSE', shares=shares,
            order_type='exit',
            strategy=getattr(position, 'strategy_name', None),
            reason=reason)
        return True

    def close_position_at_market(self, position: Position, reason: str) -> bool:
        """Place a market close for ``position``, wait for the fill, record it.

        Waits (bounded) for the broker to report ``filled_avg_price`` and hands
        that price to ``PositionsManager.close_position``.  Without this the
        realized return was computed from a *heuristic* exit price (the OCO
        stop/take-profit target), which materially misstated P&L — e.g. a
        position that actually filled at $14.19 was recorded at the $15.17 stop.

        Falls back to the previous behaviour (``exit_price=None``) when the fill
        cannot be confirmed, so a slow/absent fill never loses the exit.
        """
        symbol = position.symbol
        side = getattr(position, 'side', 'long')
        shares = abs(position.quantity)

        order_id = self.place_market_close_order(symbol, shares, reason, side)
        if order_id is None:
            logger.error(
                "Failed to close %s at market (%s)", symbol, reason)
            return False

        fill_price = self._wait_for_order_fill(order_id)
        if fill_price is not None:
            logger.info(
                "Confirmed exit fill for %s: %d shares @ $%.2f (order %s)",
                symbol, shares, fill_price, order_id)

        if not self.dry_run:
            position.exit_reason = reason
            if fill_price is not None:
                self._positions_manager.close_position(
                    symbol, exit_price=fill_price)
            else:
                self._positions_manager.close_position(symbol)
        return True

    def calculate_todays_stop_loss_and_take_profit(self, position: Position) -> Tuple[float, float]:
        """
        Calculate today's stop loss and take profit / cover prices.

        For long positions: stop_loss below entry, take_profit above entry (RSI-implied via rsi_upper).
        For short positions: stop_loss above entry, take_profit (cover) below entry (RSI-implied via rsi_lower).

        Args:
            position: Current position
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        try:
            side = getattr(position, 'side', 'long')

            # Get historical data for RSI calculation (use cache)
            data = self._fetch_ohlcv_once(
                position.symbol, position.rsi_period * 3)

            if side == "short":
                # Short: target RSI lower bound for cover price
                target_rsi = position.rsi_lower
                target_price = RSIStrategy.calculate_price_for_target_rsi(
                    data, target_rsi, position.rsi_period
                )

                if target_price is not None:
                    logger.info("Calculated cover price for %s based on RSI=%d: $%.2f",
                                position.symbol, target_rsi, target_price)
                else:
                    logger.warning(
                        "Could not calculate RSI cover price for %s", position.symbol)
                    default_stop = position.entry_price * \
                        (1 + globalConfig.STOP_LOSS_PCT)
                    default_take = position.entry_price * \
                        (1 - globalConfig.TAKE_PROFIT_PCT)
                    return default_stop, default_take

                current_price = self._get_current_price(position.symbol)
                if current_price is None:
                    default_stop = position.entry_price * \
                        (1 + globalConfig.STOP_LOSS_PCT)
                    default_take = position.entry_price * \
                        (1 - globalConfig.TAKE_PROFIT_PCT)
                    return default_stop, default_take

                # Cover price must be BELOW entry (profitable short); validate.
                if target_price >= position.entry_price:
                    take_profit_price = round(
                        position.entry_price * (1 - globalConfig.TAKE_PROFIT_PCT), 2)
                else:
                    take_profit_price = round(target_price, 2)

                stop_loss_price = round(
                    position.entry_price * (1 + globalConfig.STOP_LOSS_PCT), 2)

                logger.info("Short %s: stop loss=$%.2f, cover=$%.2f",
                            position.symbol, stop_loss_price, take_profit_price)
                return stop_loss_price, take_profit_price

            else:
                # Long: original logic — target RSI upper bound for take-profit
                target_price = RSIStrategy.calculate_price_for_target_rsi(
                    data, position.rsi_upper, position.rsi_period
                )

                if target_price is not None:
                    logger.info("Calculated target price for %s based on $%.2f RSI: $%.2f",
                                position.symbol, position.rsi_upper, target_price)
                else:
                    logger.warning(
                        "Could not calculate RSI target price for %s", position.symbol)
                    default_stop = (position.entry_price * (1 - globalConfig.STOP_LOSS_PCT)
                                    if position.stop_loss_price is None
                                    else position.stop_loss_price)
                    default_take = (position.entry_price * (1 + globalConfig.TAKE_PROFIT_PCT)
                                    if position.take_profit_price is None
                                    else position.take_profit_price)
                    return default_stop, default_take

                current_price = self._get_current_price(position.symbol)
                if current_price is None:
                    default_stop = (position.entry_price * (1 - globalConfig.STOP_LOSS_PCT)
                                    if position.stop_loss_price is None
                                    else position.stop_loss_price)
                    default_take = (position.entry_price * (1 + globalConfig.TAKE_PROFIT_PCT)
                                    if position.take_profit_price is None
                                    else position.take_profit_price)
                    return default_stop, default_take

                if target_price <= position.entry_price:
                    # RSI target at/below entry: anchor the take-profit to
                    # entry, not current price. Anchoring to current price
                    # can push the take-profit below the entry-anchored
                    # stop-loss when the position is underwater.
                    take_profit_price = round(
                        position.entry_price * (1 + globalConfig.TAKE_PROFIT_PCT), 2)
                elif target_price <= current_price:
                    take_profit_price = round(current_price * (1.0005), 2)
                else:
                    take_profit_price = round(target_price, 2)

                stop_loss_price = round(
                    position.entry_price * (1 - globalConfig.STOP_LOSS_PCT), 2)

                # Defensive: never return an inverted long SL/TP pair.
                if take_profit_price <= stop_loss_price:
                    take_profit_price = round(
                        position.entry_price * (1 + globalConfig.TAKE_PROFIT_PCT), 2)

                logger.info("Calculated new stop loss: $%.2f and take profit: $%.2f for %s",
                            stop_loss_price, take_profit_price, position.symbol)
                return stop_loss_price, take_profit_price

        except Exception as e:
            logger.error(
                "Error calculating stop loss and take profit for %s: %s",
                position.symbol, e)
            side = getattr(position, 'side', 'long')
            if side == "short":
                default_stop = position.entry_price * \
                    (1 + globalConfig.STOP_LOSS_PCT)
                default_take = position.entry_price * \
                    (1 - globalConfig.TAKE_PROFIT_PCT)
            else:
                default_stop = (position.entry_price * (1 - globalConfig.STOP_LOSS_PCT)
                                if position.stop_loss_price is None
                                else position.stop_loss_price)
                default_take = (position.entry_price * (1 + globalConfig.TAKE_PROFIT_PCT)
                                if position.take_profit_price is None
                                else position.take_profit_price)
            return default_stop, default_take

    # -- protective-order helpers (cancel / await-qty-release) ---------------

    def _get_active_orders_for_symbol(self, symbol: str) -> List[Any]:
        """Return every non-terminal order for ``symbol``.

        Deliberately queries ``QueryOrderStatus.ALL`` rather than ``.OPEN``:
        Alpaca's OPEN filter OMITS ``held`` orders — exactly the orders that
        reserve a position's shares while the market is closed.  Missing them
        meant the cancel/replace logic chased stale orders, never released the
        qty, and left positions without protection.
        """
        if self.trading_client is None:
            return []

        try:
            orders = self.trading_client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.ALL, limit=500))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Error fetching orders for %s: %s", symbol, e)
            return []

        # Guard against a non-sequence response (e.g. a mock/None) so callers
        # can iterate the result unconditionally.
        if not isinstance(orders, (list, tuple)):
            return []

        active: List[Any] = []
        for order in orders:
            order_symbol = getattr(order, 'symbol', None) or (
                order.get('symbol') if isinstance(order, dict) else None)
            if order_symbol != symbol:
                continue
            if _status_str(getattr(order, 'status', None)) in _TERMINAL_ORDER_STATUSES:
                continue
            active.append(order)
        return active

    def _cancel_open_orders_for_symbol(self, symbol: str) -> List[str]:
        """Cancel every active order for ``symbol``; return the cancelled order ids.

        Returns the ids so the caller can wait for them to actually settle and
        release the shares they were holding.
        """
        if self.trading_client is None:
            return []

        cancelled: List[str] = []
        for order in self._get_active_orders_for_symbol(symbol):
            order_id = getattr(order, 'id', None) or (
                order.get('id') if isinstance(order, dict) else None)
            if not order_id:
                continue
            try:
                logger.info(
                    "Cancelling existing order %s for %s", order_id, symbol)
                self.trading_client.cancel_order_by_id(order_id)
                cancelled.append(str(order_id))
            except Exception as e:  # pylint: disable=broad-exception-caught
                # A cancel race ("order already canceled"/"pending cancel") is
                # expected when a previous cycle already requested the cancel.
                logger.warning(
                    "Cancel failed for order %s (%s): %s", order_id, symbol, e)

        return cancelled

    def _wait_for_orders_cancelled(self, order_ids: List[str],
                                   timeout: float = 20.0,
                                   poll_interval: float = 0.5) -> bool:
        """Block until every order id reports a qty-releasing status (bounded).

        Returns True if all orders settled before ``timeout``.  Orders that
        cannot be polled are dropped from the wait so a single API hiccup
        cannot wedge the whole trading cycle — the caller's submit-retry loop
        remains the final safety net.
        """
        if not order_ids or self.trading_client is None:
            return True

        deadline = time.monotonic() + timeout
        pending = {str(oid) for oid in order_ids}

        while pending and time.monotonic() < deadline:
            for order_id in list(pending):
                try:
                    order = self.trading_client.get_order_by_id(order_id)
                    status = _status_str(getattr(order, 'status', None))
                except Exception as e:  # pylint: disable=broad-exception-caught
                    logger.warning(
                        "Could not poll order %s for cancel status: %s",
                        order_id, e)
                    pending.discard(order_id)
                    continue

                if status in _QTY_RELEASING_STATUSES:
                    pending.discard(order_id)

            if pending:
                time.sleep(poll_interval)

        if pending:
            logger.warning(
                "Timed out (%.0fs) waiting for orders to release qty: %s",
                timeout, sorted(pending))
            return False
        return True

    def _get_position_qty_available(self, symbol: str) -> Optional[float]:
        """Return broker-reported tradeable qty for ``symbol``.

        ``None`` means "unknown" (no position, transient error, or the SDK
        does not expose the field) — callers keep polling / fall through.
        """
        if self.trading_client is None:
            return None
        try:
            position = self.trading_client.get_open_position(symbol)
        except Exception:  # pylint: disable=broad-exception-caught
            # Most commonly: no open position for this symbol.
            return None

        raw = getattr(position, 'qty_available', None)
        if raw is None:
            return None
        try:
            return abs(float(raw))
        except (TypeError, ValueError):
            return None

    def _wait_for_position_qty_available(self, symbol: str, shares: float,
                                         timeout: float = 20.0,
                                         poll_interval: float = 0.5) -> bool:
        """Block until ``symbol`` reports at least ``shares`` tradeable shares.

        This is the authoritative signal that a just-cancelled protective order
        has released the position's shares and a replacement can be submitted.
        """
        required = abs(float(shares))
        deadline = time.monotonic() + timeout
        last_available: Optional[float] = None

        while time.monotonic() < deadline:
            available = self._get_position_qty_available(symbol)
            if available is not None:
                last_available = available
                if available >= required - 1e-6:
                    return True
            time.sleep(poll_interval)

        logger.warning(
            "Timed out (%.0fs) waiting for tradeable qty on %s: "
            "need %.0f, available %s",
            timeout, symbol, required,
            'unknown' if last_available is None else f'{last_available:.0f}')
        return False

    def _submit_oco_with_retry(self, oco_order: Any, symbol: str, shares: float,
                               attempts: int = 4, base_delay: float = 2.0) -> Any:
        """Submit an OCO order, retrying while Alpaca still holds the shares.

        A cancel can settle asynchronously, so even after our explicit wait the
        broker may momentarily still report ``insufficient qty``.  Each retry
        re-waits for the release (with a growing budget) before resubmitting.

        Raises the final error if every attempt fails.
        """
        if self.trading_client is None:
            raise RuntimeError(
                "Trading client not available - cannot submit order")

        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return self.trading_client.submit_order(oco_order)
            except Exception as e:  # pylint: disable=broad-exception-caught
                if not _is_insufficient_qty_error(e):
                    raise
                last_error = e
                logger.warning(
                    "OCO submit for %s still blocked by held qty "
                    "(attempt %d/%d): %s", symbol, attempt, attempts, e)
                if attempt == attempts:
                    break
                self._wait_for_position_qty_available(
                    symbol, shares, timeout=base_delay * attempt + 5.0)
                time.sleep(base_delay)

        assert last_error is not None
        raise last_error

    def place_oco_close_order(self, symbol: str, shares: float, stop_loss_price: float, take_profit_price: float, side: str = "long") -> bool:
        """
        Place an OCO (One Cancels Other) close order for an existing position.

        Supports both long and short positions:
        - Long: places a SELL OCO (take-profit above entry, stop-loss below entry)
        - Short: places a BUY OCO (cover at take-profit below entry, stop-loss above entry)

        Args:
            symbol: Stock symbol to close
            shares: Number of shares to close
            stop_loss_price: Stop loss price (below entry for long, above entry for short)
            take_profit_price: Take profit / cover price (above entry for long, below entry for short)
            side: Position side — "long" or "short" (default: "long")

        Returns:
            True if order was placed successfully
        """
        try:
            if self.dry_run:
                # Dry run mode - simulate order placement
                action = "sell" if side == "long" else "buy (cover)"
                logger.info(
                    "🔍 DRY RUN: Would place OCO %s order for %d shares of %s (%s)",
                    action, shares, symbol, side)
                logger.info("🔍 DRY RUN: Stop loss at $%.2f, Take profit at $%.2f",
                            stop_loss_price, take_profit_price)
                return True

            # Validate SL/TP orientation BEFORE cancelling existing orders.
            # An inverted pair would be rejected by Alpaca, but only after we
            # have already cancelled the (still-valid) protective orders —
            # leaving the position exposed. Refuse early instead.
            if side == "short":
                if take_profit_price >= stop_loss_price:
                    logger.error(
                        "Refusing OCO for %s: short cover $%.2f must be below stop $%.2f",
                        symbol, take_profit_price, stop_loss_price)
                    return False
            else:
                if stop_loss_price >= take_profit_price:
                    logger.error(
                        "Refusing OCO for %s: long stop $%.2f must be below take-profit $%.2f",
                        symbol, stop_loss_price, take_profit_price)
                    return False

            # Get current price for validation
            current_price = self._get_current_price(symbol)
            if current_price is None:
                logger.error("Could not get current price for %s", symbol)
                return False

            # Refresh the protective OCO.  Ordering matters: a cancel is only
            # safe if the replacement can actually be submitted, otherwise the
            # position ends up with NO protection (the bug this guards against).
            try:
                if self.trading_client is None:
                    logger.error(
                        "Trading client not available - cannot get orders")
                    return False

                required_shares = abs(float(shares))
                available = self._get_position_qty_available(symbol)
                shares_are_free = (
                    available is not None and available >= required_shares - 1e-6)

                if not shares_are_free:
                    # The shares are reserved by existing order(s). Refuse to
                    # cancel orders that cannot be replaced right now —
                    # cancelling a `held` (market-closed) or already
                    # `pending_cancel` order does not release the qty, so the
                    # replacement would be rejected and the position would be
                    # left unprotected.
                    blockers = [
                        str(getattr(o, 'id', None) or '?')
                        for o in self._get_active_orders_for_symbol(symbol)
                        if _status_str(getattr(o, 'status', None))
                        in _UNREPLACEABLE_ORDER_STATUSES
                    ]
                    if blockers:
                        logger.warning(
                            "Skipping OCO refresh for %s: existing protective "
                            "order(s) %s cannot be cancelled/replaced now "
                            "(market closed or cancel already pending; "
                            "available qty=%s of %.0f). Keeping current "
                            "protection.", symbol, blockers, available,
                            required_shares)
                        return False

                    cancelled_ids = self._cancel_open_orders_for_symbol(symbol)
                    if cancelled_ids and not self._wait_for_orders_cancelled(
                            cancelled_ids):
                        logger.warning(
                            "Aborting OCO refresh for %s: cancelled order(s) "
                            "%s never settled — not resubmitting, so any "
                            "remaining protection stays in place.",
                            symbol, cancelled_ids)
                        return False
            except Exception as e:
                logger.warning(
                    "Error cancelling existing orders for %s: %s", symbol, e)

            # Confirm the broker reports the shares as tradeable before
            # relying on the submit-retry loop below.
            self._wait_for_position_qty_available(symbol, shares)

            # Determine order side based on position direction.
            # Long → SELL to close; Short → BUY to cover.
            if side == "short":
                order_side = OrderSide.BUY
                action_label = "buy (cover)"
                # For buy stop-limit, limit_price must be >= stop_price
                # (buy at limit_price or better means equal or lower)
                stop_limit_buffer = round(stop_loss_price * 1.005, 2)
            else:
                order_side = OrderSide.SELL
                action_label = "sell"
                # For sell stop-limit, limit_price must be <= stop_price
                # (sell at limit_price or better means equal or higher)
                stop_limit_buffer = round(stop_loss_price * 0.995, 2)

            # Create OCO order according to Alpaca documentation
            # OCO orders must be limit orders with take_profit and stop_loss parameters
            client_order_id = self._make_unique_client_order_id(
                generate_client_order_id(
                    symbol,
                    "BUY" if order_side == OrderSide.BUY else "SELL",
                    datetime.now(),
                )
            )
            oco_order = LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=order_side,
                type=OrderType.LIMIT,  # Must be limit for OCO
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.OCO,
                # For OCO orders, the take-profit leg's limit_price IS the
                # primary limit; do NOT set a top-level limit_price or the
                # API will treat this as a plain limit order and drop the
                # stop-loss leg.
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(
                    stop_price=stop_loss_price,
                    # Stop-limit order with small buffer
                    limit_price=stop_limit_buffer
                ),
                client_order_id=client_order_id,
            )

            # Submit the order (retries while the broker still holds the qty)
            if self.trading_client is None:
                logger.error(
                    "Trading client not available - cannot submit order")
                return False

            order = self._submit_oco_with_retry(oco_order, symbol, shares)
            placed_order_id = getattr(order, 'id', None)
            logger.info("Order placed successfully: %s", placed_order_id)

            logger.info(
                "OCO %s order placed for %d shares of %s (%s)",
                action_label, shares, symbol, side)
            logger.info("Take profit limit: $%.2f, Stop loss: $%.2f",
                        take_profit_price, stop_loss_price)

            try:
                storage.save_orders([
                    Order(
                        client_order_id=client_order_id,
                        order_id=placed_order_id,
                        symbol=symbol,
                        side="buy" if order_side == OrderSide.BUY else "sell",
                        qty=float(shares),
                        order_type="limit",
                        order_class="oco",
                        status="new",
                        stop_price=stop_loss_price,
                        limit_price=take_profit_price,
                        submitted_at=datetime.now(),
                        leg="oco",
                    )
                ])
            except Exception as e:
                logger.error("Error saving OCO order to storage for %s: %s",
                             symbol, e)

            return True

        except Exception as e:
            action = "buy (cover)" if side == "short" else "sell"
            error_msg = "Error placing OCO %s order for %s: %s" % (
                action, symbol, e)
            if self.dry_run:
                error_msg = "🔍 DRY RUN: " + error_msg
            logger.error(error_msg)
            return False

    def place_market_sell_order(self, symbol: str, shares: float, reason: str = "manual", side: str = "long") -> bool:
        """
        Place a simple market close order (used for max-hold-day forced exits).

        Boolean convenience wrapper around :meth:`place_market_close_order`.
        Callers that need the broker order id (to wait for the fill price)
        should call that method directly.

        Args:
            symbol: Stock symbol to close
            shares: Number of shares to close (absolute value)
            reason: Human-readable reason for the exit (for logging)
            side: Position side — "long" (default) or "short"

        Returns:
            True if the order was placed successfully
        """
        return self.place_market_close_order(
            symbol, shares, reason, side) is not None

    def place_market_close_order(self, symbol: str, shares: float, reason: str = "manual", side: str = "long") -> Optional[str]:
        """
        Place a market close order and return the broker order id.

        Closes a long with a market SELL, or a short with a market BUY (cover).
        Returning the order id lets the caller wait for ``filled_avg_price`` so
        the realized return reflects the *actual* fill rather than a heuristic
        (see ``positions.close_position``).

        Args:
            symbol: Stock symbol to close
            shares: Number of shares to close (absolute value)
            reason: Human-readable reason for the exit (for logging)
            side: Position side — "long" (default) or "short"

        Returns:
            The broker order id, or None if the order could not be placed.
        """
        is_short = side == "short"
        order_side = OrderSide.BUY if is_short else OrderSide.SELL
        action = "buy (cover)" if is_short else "sell"
        side_tag = "BUY" if is_short else "SELL"
        side_label = "buy" if is_short else "sell"

        try:
            if self.dry_run:
                logger.info(
                    "🔍 DRY RUN: Would place market %s for %d shares of %s (reason: %s)",
                    action, shares, symbol, reason)
                # Sentinel so the dry-run path still reports success; there is
                # no real order to poll for a fill.
                return f"dry-run-{symbol}"

            if self.trading_client is None:
                logger.error(
                    "Trading client not available — cannot place %s order for %s",
                    action, symbol)
                return None

            client_order_id = self._make_unique_client_order_id(
                generate_client_order_id(symbol, side_tag, datetime.now())
            )

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            )
            order = self.trading_client.submit_order(order_request)
            placed_order_id = getattr(order, 'id', None)
            logger.info("Market %s order placed for %d shares of %s (reason: %s) — order %s",
                        action, shares, symbol, reason, placed_order_id)

            try:
                storage.save_orders([
                    Order(
                        client_order_id=client_order_id,
                        order_id=placed_order_id,
                        symbol=symbol,
                        side=side_label,
                        qty=float(shares),
                        order_type="market",
                        order_class="simple",
                        status="new",
                        submitted_at=datetime.now(),
                        leg="market_exit",
                    )
                ])
            except Exception as e:
                logger.error("Error saving market close order to storage for %s: %s",
                             symbol, e)

            return str(placed_order_id) if placed_order_id is not None else None

        except Exception as e:
            logger.error(
                "Error placing market %s order for %s: %s", action, symbol, e)
            return None

    def _wait_for_order_fill(self, order_id: Optional[str],
                             timeout: float = 15.0,
                             poll_interval: float = 0.5) -> Optional[float]:
        """Wait (bounded) for a close order to fill; return its average fill price.

        Market orders normally fill in well under a second, but the fill is not
        immediately visible on the submit response.  Polling gives the realized
        return a real fill price instead of a heuristic fallback.

        Returns ``None`` when the order has not filled within ``timeout`` (or
        cannot be polled), so callers fall back to their previous behaviour.
        """
        if not order_id or self.trading_client is None or self.dry_run:
            return None

        deadline = time.monotonic() + timeout
        last_status = None
        while time.monotonic() < deadline:
            try:
                order = self.trading_client.get_order_by_id(order_id)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Could not poll order %s for fill: %s", order_id, e)
                return None

            status = _status_str(getattr(order, 'status', None))
            last_status = status
            filled_price = getattr(order, 'filled_avg_price', None)
            if status == 'filled' and filled_price is not None:
                try:
                    return float(filled_price)
                except (TypeError, ValueError):
                    return None

            if status in _TERMINAL_ORDER_STATUSES:
                # Cancelled/expired/rejected — it will never fill.
                logger.warning(
                    "Close order %s reached terminal status '%s' without a fill",
                    order_id, status)
                return None

            time.sleep(poll_interval)

        logger.warning(
            "Close order %s not filled within %.0fs (last status=%s); "
            "falling back to estimated exit price",
            order_id, timeout, last_status)
        return None

    def update_portfolio_orders(self, session_summary: Dict[str, Any], current_positions: List[Position]) -> Dict[str, Any]:
        """
        Update existing positions with today's stop loss and take profit orders.
        Enforces max hold days — positions held beyond MAX_HOLD_DAYS are force-closed.
        Args:
            session_summary: Dictionary to store session summary
            current_positions: List of current positions
        Returns:
            Updated session summary with orders placed
        """
        # First pass: force-close positions that have exceeded max hold days.
        # This matches the backtest's max-hold-day exit for backtest/live parity.
        # Intraday positions are excluded — their exits are bar-engine-managed.
        daily_positions = [
            p for p in current_positions if not getattr(p, 'intraday', False)]
        now = datetime.now(timezone.utc)
        positions_to_close = []
        for position in daily_positions:
            days_held = (now - ensure_utc(position.entry_date)).days
            if days_held >= globalConfig.MAX_HOLD_DAYS:
                logger.info(
                    "⏰ Position %s held for %d days (max: %d) — force closing",
                    position.symbol, days_held, globalConfig.MAX_HOLD_DAYS)
                if self._force_close_position(
                        session_summary, position, "max_hold_days"):
                    positions_to_close.append(position.symbol)

        # Second pass: update remaining open positions with new OCO orders.
        active_positions = [
            p for p in daily_positions if p.symbol not in positions_to_close
        ]
        for position in active_positions:
            # Calculate today's stop loss and take profit based on current price
            position.stop_loss_price, position.take_profit_price = self.calculate_todays_stop_loss_and_take_profit(
                position)

            pos_side = getattr(position, 'side', 'long')

            # Breached-stop guard: the stop is anchored to the entry price, so
            # a position that has moved past STOP_LOSS_PCT already has a
            # triggered stop. Submitting it provides no working protection —
            # Alpaca cancels the OCO take-profit leg and keeps only an
            # already-live stop — so the position would run past its risk
            # limit. Exit at market instead.
            try:
                current_price = self._get_current_price(position.symbol)
            except Exception:  # pylint: disable=broad-exception-caught
                current_price = None

            if self._is_stop_breached(pos_side, position.stop_loss_price,
                                      current_price):
                logger.warning(
                    "⛔ %s %s stop already breached (stop $%.2f vs market "
                    "$%.2f) — force closing instead of placing an invalid stop",
                    pos_side, position.symbol, float(position.stop_loss_price),
                    float(current_price))
                self._force_close_position(
                    session_summary, position, "stop_loss_breached")
                continue

            if self.dry_run:
                logger.info("🔍 DRY RUN: Would update stop loss for %s to $%.2f and take profit to $%.2f",
                            position.symbol, position.stop_loss_price, position.take_profit_price)
            else:
                # Place OCO close order with updated stop loss and take profit
                if self.place_oco_close_order(position.symbol, abs(position.quantity), position.stop_loss_price, position.take_profit_price, side=pos_side):
                    session_summary['orders_placed'] += 1
                    self._record_order(
                        session_summary, symbol=position.symbol,
                        action='OCO', shares=abs(position.quantity),
                        order_type='oco',
                        strategy=getattr(position, 'strategy_name', None),
                        reason='update_sl_tp')
        return session_summary

    def identify_purchases(self, session_summary: Dict[str, Any], backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """
        Identify new buying opportunities and place orders.
        Args:
            session_summary: Dictionary to store session summary
            backtest_results: List of backtest results
        Returns:
            Updated session summary with new opportunities and orders placed
        """
        # Identify buying opportunities
        opportunities = self.identify_buying_opportunities(backtest_results)
        session_summary['opportunities_found'] = len(opportunities)
        return self._execute_purchases(session_summary, opportunities)

    def _execute_purchases(self, session_summary: Dict[str, Any], opportunities: List[TradingOpportunity]) -> Dict[str, Any]:
        """Execute long entries for a list of opportunities (shared by the daily
        session and the bar loop). Handles opposite-position exits, sizing,
        and order placement."""
        # Partition: opposite-direction holdings are exits, not new entries.
        # Exits do NOT consume new-position slots.
        exit_symbols = {
            op.symbol for op in opportunities
            if self._has_opposite_position(op.symbol, "long")
        }
        for op in opportunities:
            if op.symbol in exit_symbols:
                if self._exit_opposite_position(op.symbol, "long"):
                    session_summary['positions_exited'] += 1
                    self._record_order(
                        session_summary, symbol=op.symbol, action='COVER',
                        order_type='exit', strategy=op.strategy_name,
                        reason='opposite_signal')

        entry_opportunities = [
            op for op in opportunities if op.symbol not in exit_symbols
        ]

        # Calculate position sizes (new entries only)
        position_allocations = self.calculate_position_sizes(
            entry_opportunities)

        if position_allocations:
            logger.info("📥 Found %d new buying opportunities:",
                        len(position_allocations))
            total_investment = 0
            for i, (opportunity, shares) in enumerate(position_allocations, 1):
                position_value = shares * opportunity.entry_price
                total_investment += position_value
                logger.info("   %d. %s: %d shares @ $%.2f = $%.2f",
                            i, opportunity.symbol, shares, opportunity.entry_price, position_value)
                logger.info("      RSI: %.1f, Alpha: %.3f, Win Rate: %.1f%%",
                            opportunity.current_rsi, opportunity.alpha, opportunity.win_rate * 100)
            logger.info("   Total investment: $%.2f", total_investment)

            # Execute buy orders
            for opportunity, shares in position_allocations:
                if self.place_buy_order(opportunity, shares):
                    session_summary['orders_placed'] += 1
                    session_summary['new_positions'] += 1
                    self._record_order(
                        session_summary, symbol=opportunity.symbol,
                        action='BUY', shares=shares,
                        price=opportunity.entry_price, order_type='entry',
                        strategy=opportunity.strategy_name)
        return session_summary

    def identify_and_execute_shorts(self, session_summary: Dict[str, Any], backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """
        Identify new short-selling opportunities and place orders.

        Args:
            session_summary: Dictionary to store session summary
            backtest_results: List of backtest results

        Returns:
            Updated session summary with new short opportunities and orders placed
        """
        # Identify short-selling opportunities
        short_opportunities = self.identify_shorting_opportunities(
            backtest_results)
        logger.info("Found %d short-selling opportunities",
                    len(short_opportunities))
        return self._execute_shorts(session_summary, short_opportunities)

    def _execute_shorts(self, session_summary: Dict[str, Any], short_opportunities: List[TradingOpportunity]) -> Dict[str, Any]:
        """Execute short entries for a list of opportunities (shared by the daily
        session and the bar loop)."""
        # Partition: opposite-direction holdings are exits, not new entries.
        exit_symbols = {
            op.symbol for op in short_opportunities
            if self._has_opposite_position(op.symbol, "short")
        }
        for op in short_opportunities:
            if op.symbol in exit_symbols:
                if self._exit_opposite_position(op.symbol, "short"):
                    session_summary['positions_exited'] += 1
                    self._record_order(
                        session_summary, symbol=op.symbol, action='SELL',
                        order_type='exit', strategy=op.strategy_name,
                        reason='opposite_signal')

        entry_opportunities = [
            op for op in short_opportunities if op.symbol not in exit_symbols
        ]

        # Calculate position sizes (respects leverage cap; new entries only)
        position_allocations = self.calculate_short_position_sizes(
            entry_opportunities)

        if position_allocations:
            logger.info("📉 Found %d new short-selling opportunities:",
                        len(position_allocations))
            total_notional = 0
            for i, (opportunity, shares) in enumerate(position_allocations, 1):
                position_value = shares * opportunity.entry_price
                total_notional += position_value
                logger.info("   %d. %s: %d shares @ $%.2f = $%.2f (short)",
                            i, opportunity.symbol, shares, opportunity.entry_price, position_value)
                logger.info("      RSI: %.1f, Alpha: %.3f, Win Rate: %.1f%%",
                            opportunity.current_rsi, opportunity.alpha, opportunity.win_rate * 100)
            logger.info("   Total short notional: $%.2f", total_notional)

            # Execute short orders
            for opportunity, shares in position_allocations:
                if self.place_short_order(opportunity, shares):
                    session_summary['orders_placed'] += 1
                    session_summary['new_positions'] += 1
                    self._record_order(
                        session_summary, symbol=opportunity.symbol,
                        action='SHORT', shares=shares,
                        price=opportunity.entry_price, order_type='entry',
                        strategy=opportunity.strategy_name)
        return session_summary

    def _record_order(self, session_summary: Dict[str, Any], *, symbol: str,
                      action: str, shares=None, price=None, order_type: str = "entry",
                      strategy: Optional[str] = None, reason: Optional[str] = None) -> None:
        """Append a structured order event for dashboard feedback.

        Centralises the shape of the ``orders`` list so the frontend can render
        "what orders were placed this session" consistently.

        Args:
            session_summary: The running session summary (mutated in place).
            symbol: Ticker.
            action: Human-readable action label (``BUY``, ``SHORT``, ``OCO``,
                ``CLOSE``, ``COVER``, ``SELL``).
            shares: Quantity (None when not applicable).
            price: Fill/limit/stop reference price (None when not applicable).
            order_type: ``entry``, ``oco``, or ``exit``.
            strategy: Owning strategy registry key, if known.
            reason: Exit/order reason, if applicable (e.g. ``max_hold_days``).
        """
        session_summary.setdefault('orders', []).append({
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'type': order_type,
            'strategy': strategy,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        })

    def execute_trading_session(self, backtest_results: List[BacktestResult]) -> Dict[str, Any]:
        """
        Execute a complete trading session.

        Args:
            backtest_results: Results from strategy backtesting

        Returns:
            Dictionary with session summary
        """
        session_summary = {
            'timestamp': datetime.now(),
            'opportunities_found': 0,
            'new_positions': 0,
            'orders_placed': 0,
            'positions_exited': 0,
            'errors': [],
            'dry_run': self.dry_run,
            # Structured per-order events for dashboard feedback.
            'orders': [],
        }

        try:
            logger.info("Starting trading session...")
            # Begin a new persistence session so every save this session (the
            # immediate save on each open plus the end-of-session save) upserts
            # into one snapshot.
            self._positions_manager.begin_session()
            # Refresh positions once at the beginning of the session
            positions = self._positions_manager.get_and_reconcile_positions()

            if not positions:
                logger.info("No current positions found")
            else:
                # filter for only open positions
                open_positions = [
                    pos for pos in positions if not pos.closed]
                self.update_portfolio_orders(session_summary, open_positions)
                logger.info(
                    "Updated existing (open) positions with new stop loss and take profit orders")

            # Identify new buying opportunities
            if not backtest_results:
                logger.warning(
                    "No backtest results available - cannot identify buying opportunities")
            else:
                self.identify_purchases(session_summary, backtest_results)

                # Identify short-selling opportunities (when enabled)
                if globalConfig.ENABLE_SHORT_SELLING:
                    logger.info(
                        "📉 Short selling enabled — checking for short opportunities...")
                    self.identify_and_execute_shorts(
                        session_summary, backtest_results)

            # save updated positions to storage (always persist so that
            # positions reconciled from the broker are not lost on cycles with
            # no new backtest results).
            if not self.dry_run:
                self._positions_manager.persist_positions()
            else:
                logger.info(
                    "Dry run mode: Skipping positions save to storage")

            logger.info("Trading session complete: %s", session_summary)

        except Exception as e:
            error_msg = "Error in trading session (Partial execution to positions): %s" % e
            logger.error(error_msg)
            # save updated positions to storage
            if not self.dry_run:
                self._positions_manager.persist_positions()
            else:
                logger.info(
                    "Dry run mode: Skipping positions save to storage")
            session_summary['errors'].append(error_msg)

        # Refresh persisted order statuses from the broker (best-effort).
        self._refresh_order_statuses()

        return session_summary

    def _refresh_order_statuses(self) -> None:
        """Refresh persisted order statuses from the broker.

        Reads non-terminal orders from the ledger, fetches their current
        status from Alpaca, and upserts any changes back via save_orders.
        """
        try:
            open_orders = storage.get_open_orders_stored()
            if not open_orders:
                return
            cids = [o.client_order_id for o in open_orders if o.client_order_id]
            if not cids:
                return
            status_map = data_provider.get_order_status_map(cids)
            updated = []
            for o in open_orders:
                new_status = status_map.get(o.client_order_id)
                if new_status and new_status != o.status:
                    o.status = new_status
                    updated.append(o)
            if updated:
                storage.save_orders(updated)
                logger.info(
                    "Refreshed %d order statuses from Alpaca", len(updated))
        except Exception as e:
            logger.error("Error refreshing order statuses: %s", e)

    def _get_broker_order_by_client_id(self, client_order_id: str):
        """Fetch an order by client_order_id via the engine's client.

        Returns the order object, or None when not found (404) or on any
        lookup failure.
        """
        if self.trading_client is None or not client_order_id:
            return None
        try:
            return self.trading_client.get_order_by_client_id(client_order_id)
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    def _make_unique_client_order_id(
        self, base: str, max_attempts: int = 20
    ) -> str:
        """Return a client_order_id not already present at the broker."""
        if self._get_broker_order_by_client_id(base) is None:
            return base
        for i in range(1, max_attempts + 1):
            candidate = f"{base}-{i}"
            if self._get_broker_order_by_client_id(candidate) is None:
                return candidate
        logger.warning(
            "Could not find a free client_order_id after %d attempts; "
            "returning base '%s'", max_attempts, base,
        )
        return base

    def _clear_ohlcv_cache(self) -> None:
        """Clear the per-cycle OHLCV cache. Call at the start of each run cycle."""
        self._ohlcv_cache.clear()

    def _fetch_ohlcv_once(self, symbol: str, min_lookback_days: int) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol, caching per cycle.

        Multiple methods (_get_current_price, _get_rsi_with_previous,
        _compute_rsi_take_profit) previously made independent API calls
        for the same symbol.  This cache eliminates those redundant fetches.

        Args:
            symbol: Stock symbol
            min_lookback_days: Minimum calendar days of data needed

        Returns:
            DataFrame with OHLCV data (may be empty on failure)
        """
        cached = self._ohlcv_cache.get(symbol)
        if cached is not None and len(cached) >= 3:
            return cached

        end_date = datetime.now() - timedelta(minutes=20)
        start_date = end_date - timedelta(days=max(min_lookback_days, 14))

        try:
            data = data_provider.get_single_stock_bars(
                symbol, start_date, end_date)
            if not data.empty:
                self._ohlcv_cache[symbol] = data
            return data
        except Exception as e:
            logger.error("Error fetching OHLCV for %s: %s", symbol, e)
            return pd.DataFrame()

    def _get_rsi_with_previous(self, symbol: str, period: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current and previous RSI values for cross-detection.

        Returns:
            Tuple of (current_rsi, previous_rsi). Either may be None if unavailable.
        """
        try:
            data = self._fetch_ohlcv_once(symbol, period * 3)

            if data.empty or len(data) < period + 1:
                return None, None

            rsi = TechnicalIndicators.calculate_rsi(data, period)
            if rsi.empty or len(rsi) < 2:
                return None, None

            return float(rsi.iloc[-1]), float(rsi.iloc[-2])

        except Exception as e:
            logger.error(
                "Error getting RSI with previous for %s: %s", symbol, e)
            return None, None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            data = self._fetch_ohlcv_once(symbol, 7)

            if data.empty:
                return None

            # Get the most recent close price available
            return data['close'].iloc[-1]

        except Exception as e:
            logger.error("Error getting current price for %s: %s", symbol, e)
            return None

    def _compute_rsi_target_price(self, symbol: str, rsi_target: int, rsi_period: int, entry_price: float, direction: str) -> float:
        """
        Compute RSI-implied target price for backtest/live parity.

        For longs (direction="long"): target must be ABOVE entry; fallback = entry * (1 + PCT).
        For shorts (direction="short"): target must be BELOW entry; fallback = entry * (1 - PCT).

        Args:
            symbol: Stock symbol
            rsi_target: Target RSI threshold (rsi_upper for longs, rsi_lower for shorts)
            rsi_period: RSI calculation period
            entry_price: Current entry price (used as fallback basis)
            direction: "long" or "short"

        Returns:
            Target price (rounded to 2 decimal places)
        """
        is_long = direction == "long"
        label = "take-profit" if is_long else "cover"
        fallback_mult = 1 + globalConfig.TAKE_PROFIT_PCT if is_long else 1 - \
            globalConfig.TAKE_PROFIT_PCT
        validation_ok = (lambda tp: tp > entry_price) if is_long else (
            lambda tp: tp < entry_price)
        fallback_label = f"fixed {globalConfig.TAKE_PROFIT_PCT * 100:.1f}%" + (
            " take-profit" if is_long else " below entry")

        try:
            data = self._fetch_ohlcv_once(symbol, rsi_period * 3)

            if data.empty or len(data) < rsi_period + 1:
                logger.warning(
                    "Insufficient data for RSI %s calculation for %s. Falling back to %s.",
                    label, symbol, fallback_label
                )
                return round(entry_price * fallback_mult, 2)

            target_price = RSIStrategy.calculate_price_for_target_rsi(
                data, rsi_target, rsi_period
            )

            if target_price is None or not validation_ok(target_price):
                logger.info(
                    "RSI-implied %s for %s (target RSI=%d) invalid vs entry. Falling back to %s.",
                    label, symbol, rsi_target, fallback_label
                )
                return round(entry_price * fallback_mult, 2)

            logger.info(
                "RSI-implied %s for %s: $%.2f (RSI target: %d, entry: $%.2f)",
                label, symbol, target_price, rsi_target, entry_price
            )
            return round(target_price, 2)

        except Exception as e:
            logger.error(
                "Error computing RSI %s for %s: %s. Falling back to fixed percentage.",
                label, symbol, e
            )
            return round(entry_price * fallback_mult, 2)

    def _compute_rsi_take_profit(self, symbol: str, rsi_upper: int, rsi_period: int, entry_price: float) -> float:
        """RSI-implied take-profit for long positions (entry → target above entry)."""
        return self._compute_rsi_target_price(symbol, rsi_upper, rsi_period, entry_price, "long")

    def _compute_rsi_cover_price(self, symbol: str, rsi_lower: int, rsi_period: int, entry_price: float) -> float:
        """RSI-implied cover price for short positions (entry → target below entry)."""
        return self._compute_rsi_target_price(symbol, rsi_lower, rsi_period, entry_price, "short")
