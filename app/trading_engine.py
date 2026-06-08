"""
Trading execution module.
Handles order placement, position management, and portfolio updates.
"""
import logging
import time
from dataclasses import dataclass
# pylint: disable=broad-exception-caught
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (OrderClass, OrderSide, OrderType,
                                  QueryOrderStatus, TimeInForce)
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest,
                                     MarketOrderRequest, StopLossRequest,
                                     TakeProfitRequest)
from cloud_storage import cloud_storage
from data_provider import TechnicalIndicators, data_provider
from positions import Position, PositionsManager
from strategy import BacktestResult, RSIStrategy

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)


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
    stop_loss_price: float
    take_profit_price: float
    num_trades: int = 0  # Number of trades in backtest for this symbol
    composite_score: float = 0.0  # Composite score (alpha% + sharpe + calmar)
    direction: str = "long"  # "long" or "short"


class TradingEngine:
    """Main trading execution engine."""

    def __init__(self):
        self.trading_client: Optional[TradingClient] = data_provider.trading_client
        self._positions_manager: PositionsManager = PositionsManager(
            cloud_storage, data_provider)
        self._last_position_update: Optional[datetime] = None
        self.dry_run: bool = False

    def set_dry_run_mode(self, dry_run: bool) -> None:
        """Enable or disable dry run mode."""
        self.dry_run = dry_run
        if dry_run:
            logger.info(
                "🌵 DRY RUN MODE ENABLED - No actual orders will be placed")
        else:
            logger.info("🚀 LIVE TRADING MODE ENABLED - Orders will be placed")

    def identify_buying_opportunities(self, backtest_results: List[BacktestResult]) -> List[TradingOpportunity]:
        """
        Identify current buying opportunities based on backtest results.

        Args:
            backtest_results: List of profitable backtest results

        Returns:
            List of current trading opportunities
        """
        opportunities = []

        for result in backtest_results:
            try:
                # Get current and previous RSI for cross-detection (backtest parity).
                # The backtest requires RSI to cross below rsi_lower (i.e., was above,
                # now below), not just be below it.
                current_rsi, previous_rsi = self._get_rsi_with_previous(
                    result.symbol, result.rsi_period)
                if current_rsi is None:
                    continue

                # Cross-below check: current RSI must be below rsi_lower AND
                # previous RSI must have been at or above rsi_lower.
                # Falls back to level check if previous RSI is unavailable.
                is_cross_below = current_rsi < result.rsi_lower and (
                    previous_rsi is None or previous_rsi >= result.rsi_lower
                )
                if previous_rsi is None:
                    logger.debug(
                        "Previous RSI unavailable for %s; using level check as fallback", result.symbol)

                if is_cross_below:
                    # Get current price
                    current_price = self._get_current_price(result.symbol)

                    if current_price is None:
                        continue

                    # Calculate stop loss and take profit prices once for initial entry.
                    entry_price = round(current_price, 2)
                    stop_loss_price = round(
                        entry_price * (1 - globalConfig.STOP_LOSS_PCT), 2)

                    # Compute RSI-implied take-profit price for backtest/live parity.
                    # Uses the same calculate_price_for_target_rsi() as the backtest.
                    take_profit_price = self._compute_rsi_take_profit(
                        result.symbol, result.rsi_upper, result.rsi_period, entry_price
                    )

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
                        composite_score=round(result.composite_score, 2)
                    )

                    opportunities.append(opportunity)

            except Exception as e:
                logger.error(
                    "Error evaluating opportunity for %s: %s", result.symbol, e)
                continue
        # Opportunity filtering and sorting
        # Sort by alpha (best opportunities first)
        opportunities.sort(key=lambda x: x.alpha, reverse=True)
        # Filter out opportunities with negative alpha
        opportunities = [op for op in opportunities if op.alpha > 0]
        # Filter out opportunities with low win rate
        opportunities = [
            op for op in opportunities if op.win_rate >= globalConfig.MIN_WIN_RATE]
        # Filter opportunities with less than 2 trades (needs to be at least _slightly_ repeatable)
        opportunities = [
            op for op in opportunities if op.num_trades >= globalConfig.MIN_NUM_TRADES]
        # Remove symbols that are already in current positions
        # current_positions = self.refresh_positions()  # Refresh once at the start in main
        current_symbols = {
            pos.symbol for pos in self._positions_manager.positions if not pos.closed}
        opportunities = [
            op for op in opportunities if op.symbol not in current_symbols]

        return opportunities

    def identify_shorting_opportunities(self, backtest_results: List[BacktestResult]) -> List[TradingOpportunity]:
        """
        Identify current short-selling opportunities based on backtest results.

        Mirrors identify_buying_opportunities but:
        - Filters results with direction="short"
        - Entry signal: RSI cross-ABOVE rsi_upper (current > rsi_upper AND previous <= rsi_upper)
        - stop_loss_price = entry * (1 + STOP_LOSS_PCT) (inverted)
        - take_profit_price = _compute_rsi_cover_price(...) (cover below entry)
        - Excludes symbols that already have an open SHORT position

        Args:
            backtest_results: List of backtest results (may include both directions)

        Returns:
            List of short trading opportunities
        """
        opportunities = []

        for result in backtest_results:
            try:
                # Only consider short-direction results
                if result.direction != "short":
                    continue

                # Get current and previous RSI for cross-detection.
                current_rsi, previous_rsi = self._get_rsi_with_previous(
                    result.symbol, result.rsi_period)
                if current_rsi is None:
                    continue

                # Cross-above check: current RSI must be above rsi_upper AND
                # previous RSI must have been at or below rsi_upper.
                is_cross_above = current_rsi > result.rsi_upper and (
                    previous_rsi is None or previous_rsi <= result.rsi_upper
                )
                if previous_rsi is None:
                    logger.debug(
                        "Previous RSI unavailable for %s; using level check as fallback", result.symbol)

                if is_cross_above:
                    current_price = self._get_current_price(result.symbol)
                    if current_price is None:
                        continue

                    entry_price = round(current_price, 2)
                    # Short stop-loss: price rises above entry * (1 + STOP_LOSS_PCT)
                    stop_loss_price = round(
                        entry_price * (1 + globalConfig.STOP_LOSS_PCT), 2)
                    # Cover price: RSI-implied target at rsi_lower (must be below entry)
                    take_profit_price = self._compute_rsi_cover_price(
                        result.symbol, result.rsi_lower, result.rsi_period, entry_price
                    )

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
                        direction="short"
                    )

                    opportunities.append(opportunity)

            except Exception as e:
                logger.error(
                    "Error evaluating short opportunity for %s: %s", result.symbol, e)
                continue

        # Sort by alpha (best opportunities first)
        opportunities.sort(key=lambda x: x.alpha, reverse=True)
        # Filter out opportunities with negative alpha
        opportunities = [op for op in opportunities if op.alpha > 0]
        # Filter out opportunities with low win rate
        opportunities = [
            op for op in opportunities if op.win_rate >= globalConfig.MIN_WIN_RATE]
        # Filter opportunities with less than minimum trades
        opportunities = [
            op for op in opportunities if op.num_trades >= globalConfig.MIN_NUM_TRADES]
        # Remove symbols that already have an open SHORT position
        current_short_symbols = {
            pos.symbol for pos in self._positions_manager.positions
            if not pos.closed and getattr(pos, 'side', 'long') == 'short'
        }
        opportunities = [
            op for op in opportunities if op.symbol not in current_short_symbols]

        return opportunities

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
            logger.info("Cash available: $%.2f, Equity: $%.2f", cash, equity)

            # Check if we have enough cash to trade
            cash_pct = cash / equity if equity > 0 else 0
            if cash_pct < globalConfig.MIN_CASH_PCT:
                logger.info("Insufficient cash percentage: %.2f%%",
                            cash_pct * 100)
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

            # Calculate position size for each opportunity
            position_allocations = []

            for opportunity in selected_opportunities:
                # Equal weight allocation
                position_value = equity * globalConfig.POSITION_SIZE_PCT
                shares = int(position_value / opportunity.entry_price)

                if shares > 0:
                    position_allocations.append((opportunity, shares))

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

            # Calculate total notional value of existing short positions
            current_short_notional = sum(
                pos.entry_price * pos.quantity
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
            position_allocations = []

            # Distribute remaining short capacity evenly
            per_position_notional = available_short_notional / \
                len(selected_opportunities)

            for opportunity in selected_opportunities:
                # Cap each to the per_position_notional or a percentage of equity, whichever is smaller
                position_value = min(
                    per_position_notional,
                    equity * globalConfig.POSITION_SIZE_PCT
                )
                shares = int(position_value / opportunity.entry_price)

                if shares > 0:
                    position_allocations.append((opportunity, shares))
                    logger.info(
                        "Short alloc for %s: %d shares @ $%.2f = $%.2f notional",
                        opportunity.symbol, shares, opportunity.entry_price,
                        shares * opportunity.entry_price
                    )

            return position_allocations

        except Exception as e:
            logger.error("Error calculating short position sizes: %s", e)
            return []

    def place_buy_order(self, opportunity: TradingOpportunity, shares: int) -> bool:
        """
        Place a buy order for a trading opportunity.

        Args:
            opportunity: Trading opportunity
            shares: Number of shares to buy

        Returns:
            True if order was placed successfully
        """
        order_success = False
        try:
            if self.dry_run:
                # Dry run mode - simulate order placement
                logger.info("🔍 DRY RUN: Would place buy order for %d shares of %s at $%.2f",
                            shares, opportunity.symbol, opportunity.entry_price)
                logger.info("🔍 DRY RUN: Stop loss: $%.2f, Take profit: $%.2f",
                            opportunity.stop_loss_price, opportunity.take_profit_price)
                logger.info("🔍 DRY RUN: Position value: $%.2f",
                            shares * opportunity.entry_price)
                order_success = False
            else:
                # Check if trading client is available
                if self.trading_client is None:
                    logger.error(
                        "Trading client not available - cannot place order")
                    return False

                # Create bracket order (buy with stop loss and take profit)
                order_request = MarketOrderRequest(
                    symbol=opportunity.symbol,
                    qty=shares,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(
                        stop_price=opportunity.stop_loss_price),
                    take_profit=TakeProfitRequest(
                        limit_price=opportunity.take_profit_price)
                )

                order = self.trading_client.submit_order(order_request)
                order_id = getattr(order, 'id', 'Unknown')
                logger.info("Order placed successfully: %s", order_id)

                logger.info("Buy order placed for %d shares of %s at $%.2f",
                            shares, opportunity.symbol, opportunity.entry_price)
                logger.info("Stop loss: $%.2f, Take profit: $%.2f",
                            opportunity.stop_loss_price, opportunity.take_profit_price)

                order_success = True

        except Exception as e:
            error_msg = "Error placing buy order for %s: %s" % (
                opportunity.symbol, e)
            if self.dry_run:
                error_msg = "🔍 DRY RUN: " + error_msg
            logger.error(error_msg)

        # If order was successful, add position to positions manager
        if order_success:
            try:
                new_position = Position(
                    symbol=opportunity.symbol,
                    quantity=float(shares),
                    entry_price=opportunity.entry_price,
                    current_price=opportunity.entry_price,  # Initial current price
                    current_rsi=opportunity.current_rsi,
                    entry_date=datetime.now(),
                    alpha=opportunity.alpha,
                    rsi_period=opportunity.rsi_period,
                    rsi_lower=opportunity.target_rsi_lower,
                    rsi_upper=opportunity.target_rsi_upper,
                    stop_loss_price=opportunity.stop_loss_price,
                    take_profit_price=opportunity.take_profit_price,
                    closed=False,
                    exit_date=None
                )

                self._positions_manager.open_position(new_position)
            except Exception as e:
                logger.error(
                    "Error adding position to positions manager for %s: %s",
                    opportunity.symbol, e)

        return order_success

    def place_short_order(self, opportunity: TradingOpportunity, shares: int) -> bool:
        """
        Place a short-sell order for a trading opportunity.

        Args:
            opportunity: Short trading opportunity
            shares: Number of shares to short

        Returns:
            True if order was placed successfully
        """
        order_success = False
        try:
            if self.dry_run:
                logger.info("🔍 DRY RUN: Would place SHORT order for %d shares of %s at $%.2f",
                            shares, opportunity.symbol, opportunity.entry_price)
                logger.info("🔍 DRY RUN: Short stop loss: $%.2f, Cover target: $%.2f",
                            opportunity.stop_loss_price, opportunity.take_profit_price)
                logger.info("🔍 DRY RUN: Position notional: $%.2f",
                            shares * opportunity.entry_price)
                order_success = False
            else:
                if self.trading_client is None:
                    logger.error(
                        "Trading client not available - cannot place short order")
                    return False

                # Create bracket order (sell-short with stop loss and cover target).
                # For shorts: take_profit is a buy-to-cover BELOW entry,
                # stop_loss is a buy-to-cover ABOVE entry.
                order_request = MarketOrderRequest(
                    symbol=opportunity.symbol,
                    qty=shares,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                    order_class=OrderClass.BRACKET,
                    stop_loss=StopLossRequest(
                        stop_price=opportunity.stop_loss_price),
                    take_profit=TakeProfitRequest(
                        limit_price=opportunity.take_profit_price)
                )

                order = self.trading_client.submit_order(order_request)
                order_id = getattr(order, 'id', 'Unknown')
                logger.info("Short order placed successfully: %s", order_id)

                logger.info("Short order for %d shares of %s at $%.2f",
                            shares, opportunity.symbol, opportunity.entry_price)
                logger.info("Stop loss: $%.2f, Cover target: $%.2f",
                            opportunity.stop_loss_price, opportunity.take_profit_price)

                order_success = True

        except Exception as e:
            error_msg = "Error placing short order for %s: %s" % (
                opportunity.symbol, e)
            if self.dry_run:
                error_msg = "🔍 DRY RUN: " + error_msg
            logger.error(error_msg)

        if order_success:
            try:
                new_position = Position(
                    symbol=opportunity.symbol,
                    quantity=float(shares),
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
                    side="short"
                )

                self._positions_manager.open_position(new_position)
            except Exception as e:
                logger.error(
                    "Error adding short position to positions manager for %s: %s",
                    opportunity.symbol, e)

        return order_success

    def _close_conflicting_position(self, symbol: str, new_direction: str) -> bool:
        """
        Close an existing position in the opposite direction (flip logic).

        When a new signal fires in the opposite direction for the same symbol,
        we close the existing position first before opening the new one.

        Args:
            symbol: Stock symbol
            new_direction: The direction we want to open ("long" or "short")

        Returns:
            True if a conflicting position was closed, False if none existed
        """
        existing = next(
            (pos for pos in self._positions_manager.positions
             if pos.symbol == symbol and not pos.closed),
            None
        )
        if existing is None:
            return False

        existing_side = getattr(existing, 'side', 'long')
        if existing_side == new_direction:
            return False  # Same direction, no conflict

        logger.info(
            "🔄 Flipping %s: closing existing %s position before opening %s",
            symbol, existing_side, new_direction
        )

        try:
            if self.dry_run:
                logger.info(
                    "🔍 DRY RUN: Would market-sell %d shares of %s to close %s position",
                    existing.quantity, symbol, existing_side
                )
            else:
                if self.trading_client is not None:
                    self.trading_client.close_position(
                        symbol_or_asset_id=symbol
                    )
                    logger.info(
                        "Market-sold %d shares of %s to close %s position (flip)",
                        existing.quantity, symbol, existing_side
                    )

            # Mark position as closed in positions manager
            self._positions_manager.close_position(symbol)
            return True

        except Exception as e:
            logger.error(
                "Error closing conflicting position for %s: %s", symbol, e)
            return False

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

            # Get historical data for RSI calculation
            end_date = datetime.now() - timedelta(minutes=20)
            start_date = end_date - timedelta(days=position.rsi_period * 3)

            data = data_provider.get_single_stock_bars(
                position.symbol, start_date, end_date)

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

                if target_price <= current_price or target_price <= position.entry_price:
                    take_profit_price = round(current_price * (1.0005), 2)
                else:
                    take_profit_price = round(target_price, 2)

                stop_loss_price = round(
                    position.entry_price * (1 - globalConfig.STOP_LOSS_PCT), 2)

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

    def place_oco_sell_order(self, symbol: str, shares: int, stop_loss_price: float, take_profit_price: float) -> bool:
        """
        Place an OCO (One Cancels Other) sell order for a symbol.

        Args:
            symbol: Stock symbol to sell
            shares: Number of shares to sell
            stop_loss_price: Stop loss price (sell below this price)
            take_profit_price: Take profit price (sell above this price)

        Returns:
            True if order was placed successfully
        """
        try:
            if self.dry_run:
                # Dry run mode - simulate order placement
                logger.info(
                    "🔍 DRY RUN: Would place OCO sell order for %d shares of %s",
                    shares, symbol)
                logger.info("🔍 DRY RUN: Stop loss at $%.2f, Take profit at $%.2f",
                            stop_loss_price, take_profit_price)
                return True

            # Get current price for validation
            current_price = self._get_current_price(symbol)
            if current_price is None:
                logger.error("Could not get current price for %s", symbol)
                return False

            # For OCO orders, we need to cancel any existing orders for this symbol first
            try:
                # Check if trading client is available
                if self.trading_client is None:
                    logger.error(
                        "Trading client not available - cannot get orders")
                    return False

                order_filter = GetOrdersRequest(status=QueryOrderStatus.OPEN)
                open_orders = self.trading_client.get_orders(
                    filter=order_filter)

                # Find and cancel any orders for this symbol
                if open_orders:
                    for order in open_orders:
                        # Handle the case where order might be a dict or object
                        order_symbol = getattr(order, 'symbol', None) or (
                            order.get('symbol') if isinstance(order, dict) else None)
                        order_id = getattr(order, 'id', None) or (
                            order.get('id') if isinstance(order, dict) else None)

                        if order_symbol == symbol and order_id:
                            logger.info(
                                "Cancelling existing order %s for %s", order_id, symbol)
                            self.trading_client.cancel_order_by_id(order_id)

                # Small delay to ensure orders are cancelled
                time.sleep(2)
            except Exception as e:
                logger.warning(
                    "Error cancelling existing orders for %s: %s", symbol, e)

            # Create OCO order according to Alpaca documentation
            # OCO orders must be limit orders with take_profit and stop_loss parameters
            oco_order = LimitOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.SELL,
                type=OrderType.LIMIT,  # Must be limit for OCO
                time_in_force=TimeInForce.GTC,
                # Main order limit price (take profit)
                limit_price=take_profit_price,
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(limit_price=take_profit_price),
                stop_loss=StopLossRequest(
                    stop_price=stop_loss_price,
                    # Stop-limit order with small buffer
                    limit_price=round(stop_loss_price * 0.995, 2)
                )
            )

            # Submit the order
            if self.trading_client is None:
                logger.error(
                    "Trading client not available - cannot submit order")
                return False

            order = self.trading_client.submit_order(oco_order)
            order_id = getattr(order, 'id', 'Unknown')
            logger.info("Order placed successfully: %s", order_id)

            logger.info(
                "OCO sell order placed for %d shares of %s", shares, symbol)
            logger.info("Take profit limit: $%.2f, Stop loss: $%.2f",
                        take_profit_price, stop_loss_price)

            return True

        except Exception as e:
            error_msg = "Error placing OCO sell order for %s: %s" % (symbol, e)
            if self.dry_run:
                error_msg = "🔍 DRY RUN: " + error_msg
            logger.error(error_msg)
            return False

    def place_market_sell_order(self, symbol: str, shares: int, reason: str = "manual") -> bool:
        """
        Place a simple market sell order (used for max-hold-day forced exits).

        Args:
            symbol: Stock symbol to sell
            shares: Number of shares to sell
            reason: Human-readable reason for the exit (for logging)

        Returns:
            True if order was placed successfully
        """
        try:
            if self.dry_run:
                logger.info(
                    "🔍 DRY RUN: Would place market sell for %d shares of %s (reason: %s)",
                    shares, symbol, reason)
                return True

            if self.trading_client is None:
                logger.error(
                    "Trading client not available — cannot place sell order for %s", symbol)
                return False

            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_request)
            order_id = getattr(order, 'id', 'Unknown')
            logger.info("Market sell order placed for %d shares of %s (reason: %s) — order %s",
                        shares, symbol, reason, order_id)
            return True

        except Exception as e:
            logger.error(
                "Error placing market sell order for %s: %s", symbol, e)
            return False

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
        now = datetime.now()
        positions_to_close = []
        for position in current_positions:
            days_held = (now - position.entry_date).days
            if days_held >= globalConfig.MAX_HOLD_DAYS:
                logger.info(
                    "⏰ Position %s held for %d days (max: %d) — force closing",
                    position.symbol, days_held, globalConfig.MAX_HOLD_DAYS)
                if self.place_market_sell_order(
                    position.symbol, int(
                        abs(position.quantity)), "max_hold_days"
                ):
                    position.exit_reason = "max_hold_days"
                    self._positions_manager.close_position(position.symbol)
                    session_summary['positions_exited'] += 1
                    positions_to_close.append(position.symbol)
                else:
                    logger.error(
                        "Failed to force-close expired position: %s", position.symbol)

        # Second pass: update remaining open positions with new OCO orders.
        active_positions = [
            p for p in current_positions if p.symbol not in positions_to_close
        ]
        for position in active_positions:
            # Calculate today's stop loss and take profit based on current price
            position.stop_loss_price, position.take_profit_price = self.calculate_todays_stop_loss_and_take_profit(
                position)
            if self.dry_run:
                logger.info("🔍 DRY RUN: Would update stop loss for %s to $%.2f and take profit to $%.2f",
                            position.symbol, position.stop_loss_price, position.take_profit_price)
            else:
                # Place OCO sell order with updated stop loss and take profit
                if self.place_oco_sell_order(position.symbol, int(abs(position.quantity)), position.stop_loss_price, position.take_profit_price):
                    session_summary['orders_placed'] += 1
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

        # Calculate position sizes
        position_allocations = self.calculate_position_sizes(opportunities)

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

            # Execute buy orders (with flip check for conflicting shorts)
            for opportunity, shares in position_allocations:
                # Close conflicting short position if one exists (flip to long)
                self._close_conflicting_position(opportunity.symbol, "long")
                if self.place_buy_order(opportunity, shares):
                    session_summary['orders_placed'] += 1
                    session_summary['new_positions'] += 1
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

        # Calculate position sizes (respects leverage cap)
        position_allocations = self.calculate_short_position_sizes(
            short_opportunities)

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

            # Execute short orders with flip check
            for opportunity, shares in position_allocations:
                # Close conflicting long position if one exists (flip to short)
                self._close_conflicting_position(opportunity.symbol, "short")
                if self.place_short_order(opportunity, shares):
                    session_summary['orders_placed'] += 1
                    session_summary['new_positions'] += 1
        return session_summary

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
            'dry_run': self.dry_run
        }

        try:
            logger.info("Starting trading session...")
            # Refresh positions once at the beginning of the session
            positions = self.refresh_positions()

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
                return session_summary

            self.identify_purchases(session_summary, backtest_results)

            # Identify short-selling opportunities (when enabled)
            if globalConfig.ENABLE_SHORT_SELLING:
                logger.info(
                    "📉 Short selling enabled — checking for short opportunities...")
                self.identify_and_execute_shorts(
                    session_summary, backtest_results)

            # save updated positions to cloud storage
            if not self.dry_run:
                cloud_storage.save_positions(self._positions_manager.positions)
            else:
                logger.info(
                    "Dry run mode: Skipping positions save to cloud storage")

            logger.info("Trading session complete: %s", session_summary)

        except Exception as e:
            error_msg = "Error in trading session (Partial execution to positions): %s" % e
            logger.error(error_msg)
            # save updated positions to cloud storage
            if not self.dry_run:
                cloud_storage.save_positions(self._positions_manager.positions)
            else:
                logger.info(
                    "Dry run mode: Skipping positions save to cloud storage")
            session_summary['errors'].append(error_msg)

        return session_summary

    def get_current_positions(self) -> List[Position]:
        """
        Get current positions from the positions manager.
        Note: This returns the cached positions list. Call refresh_positions()
        first if you need the latest data from broker/cloud storage.

        Returns:
            List of current Position objects
        """
        return self._positions_manager.positions

    def refresh_positions(self) -> List[Position]:
        """
        Refresh positions from broker and cloud storage, then return the updated list.

        Returns:
            Updated list of Position objects
        """
        return self._positions_manager.get_and_reconcile_positions()

    def _get_current_rsi(self, symbol: str, period: int) -> Optional[float]:
        """Get current RSI value for a symbol."""
        try:
            end_date = datetime.now() - timedelta(minutes=20)
            # Buffer for weekends/holidays
            start_date = end_date - timedelta(days=period * 3)

            data = data_provider.get_single_stock_bars(
                symbol, start_date, end_date)

            if data.empty or len(data) < period:
                return None

            rsi = TechnicalIndicators.calculate_rsi(data, period)
            return rsi.iloc[-1] if not rsi.empty else None

        except Exception as e:
            logger.error("Error getting RSI for %s: %s", symbol, e)
            return None

    def _get_rsi_with_previous(self, symbol: str, period: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current and previous RSI values for cross-detection.

        Returns:
            Tuple of (current_rsi, previous_rsi). Either may be None if unavailable.
        """
        try:
            end_date = datetime.now() - timedelta(minutes=20)
            start_date = end_date - timedelta(days=period * 3)

            data = data_provider.get_single_stock_bars(
                symbol, start_date, end_date)

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
            end_date = datetime.now() - timedelta(minutes=20)
            # Use a longer lookback period to account for weekends and holidays
            start_date = end_date - timedelta(days=7)

            data = data_provider.get_single_stock_bars(
                symbol, start_date, end_date)

            if data.empty:
                return None

            # Get the most recent close price available
            return data['close'].iloc[-1]

        except Exception as e:
            logger.error("Error getting current price for %s: %s", symbol, e)
            return None

    def _compute_rsi_take_profit(self, symbol: str, rsi_upper: int, rsi_period: int, entry_price: float) -> float:
        """
        Compute RSI-implied take-profit price for backtest/live parity.

        Uses the same calculate_price_for_target_rsi() as the backtest's
        _generate_signals(). Falls back to the fixed TAKE_PROFIT_PCT percentage
        if the RSI calculation cannot produce a valid target.

        Args:
            symbol: Stock symbol
            rsi_upper: Target RSI upper threshold
            rsi_period: RSI calculation period
            entry_price: Current entry price (used as fallback basis)

        Returns:
            Take-profit price (rounded to 2 decimal places)
        """
        try:
            end_date = datetime.now() - timedelta(minutes=20)
            # Buffer for RSI calculation — need at least rsi_period + 1 bars
            start_date = end_date - timedelta(days=rsi_period * 3)

            data = data_provider.get_single_stock_bars(
                symbol, start_date, end_date)

            if data.empty or len(data) < rsi_period + 1:
                logger.warning(
                    "Insufficient data for RSI take-profit calculation for %s. "
                    "Falling back to fixed %.1f%% take-profit.",
                    symbol, globalConfig.TAKE_PROFIT_PCT * 100
                )
                return round(entry_price * (1 + globalConfig.TAKE_PROFIT_PCT), 2)

            target_price = RSIStrategy.calculate_price_for_target_rsi(
                data, rsi_upper, rsi_period
            )

            if target_price is None or target_price <= entry_price:
                logger.info(
                    "RSI-implied take-profit for %s (target RSI=%d) is at or below entry. "
                    "Falling back to fixed %.1f%% take-profit.",
                    symbol, rsi_upper, globalConfig.TAKE_PROFIT_PCT * 100
                )
                return round(entry_price * (1 + globalConfig.TAKE_PROFIT_PCT), 2)

            logger.info(
                "RSI-implied take-profit for %s: $%.2f (RSI target: %d, entry: $%.2f)",
                symbol, target_price, rsi_upper, entry_price
            )
            return round(target_price, 2)

        except Exception as e:
            logger.error(
                "Error computing RSI take-profit for %s: %s. Falling back to fixed percentage.",
                symbol, e
            )
            return round(entry_price * (1 + globalConfig.TAKE_PROFIT_PCT), 2)

    def _compute_rsi_cover_price(self, symbol: str, rsi_lower: int, rsi_period: int, entry_price: float) -> float:
        """
        Compute RSI-implied cover price for short positions (backtest/live parity).

        Uses calculate_price_for_target_rsi() targeting rsi_lower (the oversold
        threshold at which RSI would trigger a cover). The cover price must be
        BELOW entry_price for a profitable short. Falls back to fixed percentage.

        Args:
            symbol: Stock symbol
            rsi_lower: Target RSI lower threshold (cover trigger)
            rsi_period: RSI calculation period
            entry_price: Short entry price (used as fallback basis)

        Returns:
            Cover price (rounded to 2 decimal places)
        """
        try:
            end_date = datetime.now() - timedelta(minutes=20)
            start_date = end_date - timedelta(days=rsi_period * 3)

            data = data_provider.get_single_stock_bars(
                symbol, start_date, end_date)

            if data.empty or len(data) < rsi_period + 1:
                logger.warning(
                    "Insufficient data for RSI cover-price calculation for %s. "
                    "Falling back to fixed %.1f%% below entry.",
                    symbol, globalConfig.TAKE_PROFIT_PCT * 100
                )
                return round(entry_price * (1 - globalConfig.TAKE_PROFIT_PCT), 2)

            target_price = RSIStrategy.calculate_price_for_target_rsi(
                data, rsi_lower, rsi_period
            )

            if target_price is None or target_price >= entry_price:
                logger.info(
                    "RSI-implied cover for %s (target RSI=%d) is at or above entry. "
                    "Falling back to fixed %.1f%% below entry.",
                    symbol, rsi_lower, globalConfig.TAKE_PROFIT_PCT * 100
                )
                return round(entry_price * (1 - globalConfig.TAKE_PROFIT_PCT), 2)

            logger.info(
                "RSI-implied cover for %s: $%.2f (RSI target: %d, entry: $%.2f)",
                symbol, target_price, rsi_lower, entry_price
            )
            return round(target_price, 2)

        except Exception as e:
            logger.error(
                "Error computing RSI cover for %s: %s. Falling back to fixed percentage.",
                symbol, e
            )
            return round(entry_price * (1 - globalConfig.TAKE_PROFIT_PCT), 2)
