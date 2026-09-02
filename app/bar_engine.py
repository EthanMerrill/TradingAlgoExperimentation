"""
Bar-loop engine for intraday strategies (Phase D of MULTI_STRATEGY_PLAN.md).

Strategies with ``execution_style == "bar_loop"`` (e.g. RVOL-ORB) are not
traded by the daily session engine. Instead, this engine evaluates them on a
schedule during regular trading hours (RTH) using fresh intraday bars, places
day-TIF bracket entries, and force-closes intraday positions at session end.

Reuses TradingEngine for order placement/sizing and PositionsManager for state,
so dry-run and position tagging behave identically to the daily path.
"""
import logging
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional

import pytz

from config import globalConfig  # type: ignore
from data_provider import data_provider
from strategies.base import Strategy, StrategyContext
from strategies.registry import get_strategy

logger = logging.getLogger(__name__)

US_EASTERN = pytz.timezone("US/Eastern")
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)


def _as_et(dt: datetime) -> datetime:
    """Normalize a datetime to US/Eastern.

    TZ-aware datetimes are converted; naive datetimes are treated as UTC
    (the container default, matching strategy.py's EST conversion) then
    converted.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(US_EASTERN)
    return dt.replace(tzinfo=pytz.UTC).astimezone(US_EASTERN)


class BarLoopEngine:
    """Evaluates bar_loop strategies during RTH and manages intraday exits."""

    def __init__(self, trading_engine, positions_manager):
        self.trading_engine = trading_engine
        self.positions_manager = positions_manager
        self.dry_run = False

    def set_dry_run_mode(self, dry_run: bool) -> None:
        """Enable/disable dry run (no real orders, no persistence)."""
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Strategy discovery
    # ------------------------------------------------------------------

    def enabled_bar_loop_strategies(self) -> List[Strategy]:
        """Registered bar_loop strategies enabled in config, instantiated."""
        strategies: List[Strategy] = []
        for name in (getattr(globalConfig, "STRATEGIES_ENABLED", None) or []):
            try:
                cls = get_strategy(name)
            except ValueError:
                logger.warning(
                    "Bar loop: unknown strategy '%s' in config — skipping", name)
                continue
            if getattr(cls, "execution_style", "session") != "bar_loop":
                continue
            try:
                strategies.append(cls.create())
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "Bar loop: failed to instantiate '%s': %s", name, e)
        return strategies

    def has_open_intraday_positions(self) -> bool:
        """True if any open position is flagged intraday."""
        for pos in (self.positions_manager.positions or []):
            if not getattr(pos, "closed", False) and getattr(pos, "intraday", False):
                return True
        return False

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def is_rth(self, as_of: Optional[datetime] = None) -> bool:
        """True during regular trading hours (Mon–Fri, 9:30–16:00 ET)."""
        now = _as_et(as_of or datetime.now())
        if now.weekday() >= 5:
            return False
        return RTH_OPEN <= now.time() <= RTH_CLOSE

    def session_ended(self, as_of: Optional[datetime] = None) -> bool:
        """True when RTH has ended for the day (weekday after 16:00 ET)."""
        now = _as_et(as_of or datetime.now())
        if now.weekday() >= 5:
            return False
        return now.time() > RTH_CLOSE

    # ------------------------------------------------------------------
    # Entry cycle
    # ------------------------------------------------------------------

    def run_intraday_cycle(self, backtest_results: List[Any], as_of: Optional[datetime] = None) -> Dict[str, Any]:
        """Evaluate each enabled bar_loop strategy and place entry orders.

        Args:
            backtest_results: Backtest results for the current cycle (grouped by
                strategy_name; each bar_loop strategy receives its own subset).
            as_of: Evaluation timestamp (defaults to now).

        Returns:
            Summary dict mirroring the daily session summary shape.
        """
        summary: Dict[str, Any] = {
            'timestamp': as_of or datetime.now(),
            'signals': 0,
            'new_positions': 0,
            'orders_placed': 0,
            'positions_exited': 0,
            'errors': [],
            'dry_run': self.dry_run,
        }
        if not backtest_results:
            logger.info("📈 Intraday cycle: no backtest results — skipping")
            return summary

        strategies = self.enabled_bar_loop_strategies()
        if not strategies:
            logger.debug("📈 Intraday cycle: no enabled bar_loop strategies")
            return summary

        for strategy in strategies:
            results = [
                r for r in backtest_results
                if getattr(r, "strategy_name", None) == strategy.name
            ]
            if not results:
                logger.debug(
                    "📈 Intraday cycle: no results for '%s'", strategy.name)
                continue

            ctx = StrategyContext(
                data_provider=data_provider,
                positions_manager=self.positions_manager,
                config=globalConfig,
                as_of=as_of or datetime.now(),
                ohlcv_cache=self.trading_engine._ohlcv_cache,
                strategy_results=results,
            )
            try:
                signals = strategy.evaluate_live_signals(ctx) or []
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "📈 Bar-loop signal error for '%s': %s", strategy.name, e)
                summary['errors'].append(
                    f"{strategy.name}: signal error: {e}")
                continue

            long_opps = self.trading_engine._signals_to_opportunities(
                signals, "long", strategy.name)
            short_opps = (
                self.trading_engine._signals_to_opportunities(
                    signals, "short", strategy.name)
                if globalConfig.ENABLE_SHORT_SELLING else []
            )
            summary['signals'] += len(long_opps) + len(short_opps)

            if long_opps:
                self.trading_engine._execute_purchases(summary, long_opps)
            if short_opps:
                self.trading_engine._execute_shorts(summary, short_opps)

            logger.info(
                "📈 %s: %d long + %d short signals",
                strategy.name, len(long_opps), len(short_opps))

        return summary

    # ------------------------------------------------------------------
    # Session-close exits
    # ------------------------------------------------------------------

    def close_intraday_positions(self, as_of: Optional[datetime] = None) -> Dict[str, Any]:
        """Force-close all open intraday positions (called at session end).

        Returns:
            Summary dict with positions_exited / errors.
        """
        summary: Dict[str, Any] = {'positions_exited': 0, 'errors': []}
        positions = list(self.positions_manager.positions or [])
        for pos in positions:
            if getattr(pos, "closed", False) or not getattr(pos, "intraday", False):
                continue
            side = getattr(pos, "side", "long")
            try:
                placed = self.trading_engine.place_market_sell_order(
                    pos.symbol, abs(pos.quantity), "intraday_session_close",
                    side=side)
            except Exception as e:  # pylint: disable=broad-exception-caught
                placed = False
                logger.error(
                    "🕓 Failed to close intraday %s: %s", pos.symbol, e)
                summary['errors'].append(f"{pos.symbol}: {e}")
            if placed:
                summary['positions_exited'] += 1
                if not self.dry_run:
                    pos.exit_reason = "intraday_session_close"
                    self.positions_manager.close_position(pos.symbol)
            else:
                logger.error(
                    "🕓 Failed to close intraday position %s", pos.symbol)

        if not self.dry_run and summary['positions_exited']:
            try:
                self.positions_manager.persist_positions()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.error(
                    "🕓 Failed to persist positions after intraday close: %s", e)
                summary['errors'].append(f"persist: {e}")

        logger.info("🕓 Intraday session close complete: %s", summary)
        return summary
