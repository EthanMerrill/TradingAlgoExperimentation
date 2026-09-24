"""
Main application entry point for the trading algorithm.
Orchestrates the entire trading workflow.
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from data_provider import data_provider
from storage import storage
from positions import PositionsManager
from optimizer import StrategyOptimizer
from walk_forward import WalkForwardValidator
from trading_engine import TradingEngine
from strategies.registry import get_strategy
from utils import TradingCalendar, setup_logging, utc_now

from config import globalConfig  # type: ignore
from health_server import start_health_server

logger = logging.getLogger(__name__)
TEST_MODE_UNIVERSE_LIMIT = 50


class TradingAlgorithm:
    """Main trading algorithm orchestrator."""

    def __init__(self):
        self.optimizer = StrategyOptimizer()
        self.trading_calendar = TradingCalendar()
        self.positions_manager = PositionsManager(
            storage, data_provider
        )
        self.trading_engine = TradingEngine()
        # Inject shared PositionsManager so TradingAlgorithm and
        # TradingEngine operate on a single source of position state.
        self.trading_engine.set_positions_manager(self.positions_manager)
        self.session_metadata = {
            'start_time': None,
            'end_time': None,
            'config': globalConfig.to_dict(),
            'portfolio_value': 0,
            'results_summary': {}
        }
        # Backtest results from the most recent cycle — consumed by the
        # bar-loop worker (Phase D) to evaluate intraday strategies during RTH.
        self._last_backtest_results: List = []

    @property
    def last_backtest_results(self) -> List:
        """Public accessor for the bar-loop worker (no private-state reach-in)."""
        return self._last_backtest_results

    async def run_full_cycle(self, force_backtest: bool = False, dry_run: bool = False, test_mode: bool = False, run_session_only: bool = False, progress_cb=None) -> dict:
        """
        Run the complete trading algorithm cycle.

        Args:
            force_backtest: Force running backtest even if recent results exist
            dry_run: Run in dry run mode without placing actual orders
            test_mode: Run backtest on a limited stock universe for fast end-to-end validation
            run_session_only: Reuse the latest cached backtest data (ignoring the
                24h freshness window) and run ONLY the trading session — skip the
                slow optimization/backtest pass entirely.
            progress_cb: Optional callback(percent: int, message: str) invoked at
                stage boundaries for background-job progress reporting (Phase 4).

        Returns:
            Dictionary with session results
        """
        def _report(percent, message):
            if progress_cb is not None:
                try:
                    progress_cb(percent, message)
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

        self.session_metadata['start_time'] = utc_now()
        _report(1, "Starting cycle")

        # Clear per-cycle caches to avoid stale data
        self.trading_engine._clear_ohlcv_cache()

        # Startup banner
        logger.info("🚀" * 20)
        logger.info("🚀 TRADING ALGORITHM STARTING")
        logger.info("🚀" * 20)
        logger.info("📅 Session Date: %s",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("💼 Paper Trading: %s", globalConfig.PAPER_TRADE)
        logger.info("🔄 Force Backtest: %s", force_backtest)
        logger.info("🔍 Dry Run Mode: %s", dry_run)
        logger.info("🧪 Test Mode: %s", test_mode)
        logger.info("⚡ Session Only (reuse latest backtest): %s",
                    run_session_only)
        logger.info("🪟 Walk-Forward: %s", globalConfig.WF_ENABLED)
        logger.info("🔬 RSI Fine Tuning: %s",
                    globalConfig.RSI_FINE_TUNING_ENABLED)
        logger.info("=" * 60)

        # Set dry run mode on trading engine
        self.trading_engine.set_dry_run_mode(dry_run)

        try:
            # Check if it's a trading day
            if not self.trading_calendar.is_trading_day():
                if force_backtest:
                    logger.info(
                        "Market is closed today, but force backtest is enabled - continuing execution")
                else:
                    logger.info("Market is closed today - skipping execution")
                    if (dry_run is False):
                        return {'status': 'market_closed'}
                    else:
                        # In dry run mode, we can still simulate the trading day
                        logger.info(
                            "Simulating trading day in dry run mode...")

            # Step 1: Check current positions and account status
            logger.info("🔍 Checking account status and current positions...")
            _report(3, "Checking account & positions")
            account_info = data_provider.get_account_info()
            current_positions = self.positions_manager.get_and_reconcile_positions()
            if current_positions is None:
                logger.warning(
                    "Positions manager returned None during reconciliation; defaulting to empty positions list")
                current_positions = []
            if self.positions_manager.positions is None:
                logger.warning(
                    "Positions manager in-memory positions is None; defaulting to empty list")
                self.positions_manager.positions = []

            logger.info("💰 Account Summary:")
            logger.info("   • Equity: $%.2f", account_info.get('equity', 0))
            logger.info("   • Cash Available: $%.2f",
                        account_info.get('cash', 0))
            logger.info("   • Current Open Positions: %d",
                        len(current_positions))
            logger.info("   • In-Memory Position Records (open + closed): %d",
                        len(self.positions_manager.positions))
            logger.info("─" * 40)

            # Check if we have enough buying power to potentially trade
            buying_power = account_info.get('buying_power', 0)

            # Initialize backtest_results to avoid UnboundLocalError
            backtest_results = []

            if run_session_only:
                # Session-only run: reuse the latest cached backtest data
                # WITHOUT re-running the (slow) optimization pass. The normal
                # 24h freshness window is intentionally ignored so the user can
                # trade on the most recent analysis on demand.
                _report(5, "Loading cached backtest results")
                backtest_results = self._load_latest_backtest_results()
                if backtest_results:
                    logger.info(
                        "⚡ Session-only run: loaded %d cached strategies",
                        len(backtest_results))
                else:
                    logger.warning(
                        "No cached backtest data found — session will manage existing positions only")
            elif buying_power > 0 or force_backtest:
                # Step 2: Get or run backtests
                _report(5, "Running backtests (may take 30-90 minutes)")
                backtest_results = await self._get_backtest_results(
                    force_backtest, test_mode, progress_cb=_report)
            else:
                logger.warning(
                    "Insufficient buying power available for purchases")
                logger.info(
                    "Skipping backtest due to insufficient buying power - will only process existing positions")

            if not backtest_results:
                logger.warning(
                    "No backtest results available - processing existing positions only")

            # Keep the latest results for the bar-loop worker (Phase D).
            self._last_backtest_results = backtest_results

            # Step 3: Execute trading session
            logger.info(
                "🎯 Analyzing trading opportunities and executing orders...")
            _report(90, "Executing trading session")
            trading_summary = self.trading_engine.execute_trading_session(
                backtest_results)

            # Step 4: Save results and metadata
            self.session_metadata['end_time'] = utc_now()
            self.session_metadata['results_summary'] = trading_summary
            logger.info("💾 Saving session results and metadata...")
            _report(97, "Saving session results")
            await self._save_session_results(dry_run, account_info, backtest_results, trading_summary)
            _report(100, "Cycle complete")

            # Success banner
            session_duration = (
                self.session_metadata['end_time'] - self.session_metadata['start_time']).total_seconds()
            logger.info("🎉" * 20)
            logger.info("🎉 TRADING ALGORITHM COMPLETE!")
            logger.info("🎉" * 20)
            logger.info("⏱️  Session Duration: %.1f minutes",
                        session_duration/60)
            logger.info("📊 Backtest Results: %d strategies",
                        len(backtest_results))
            logger.info("💼 Trading Summary: %s", trading_summary)
            logger.info("=" * 60)
            return {
                'status': 'success',
                'trading_summary': trading_summary,
                'backtest_count': len(backtest_results),
                'duration': (self.session_metadata['end_time'] - self.session_metadata['start_time']).total_seconds()
            }

        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error in trading algorithm: %s", e)
            return {'status': 'error', 'error': str(e)}

    async def _get_backtest_results(self, force_backtest: bool, test_mode: bool = False, progress_cb=None) -> List:
        """Get backtest results, either from cache or by running new backtests."""

        # Check for recent backtest results
        if not force_backtest:
            logger.info("🔍 Checking for recent cached backtest results...")
            recent_results = self._load_recent_backtest_results()
            if recent_results:
                logger.info(
                    "✅ Found cached results: %d profitable strategies", len(recent_results))
                logger.info("⚡ Skipping backtest - using cached data")
                return recent_results
            else:
                logger.info("❌ No recent cached results found")
                if test_mode:
                    logger.info(
                        "🧪 Test mode enabled - running limited universe since no cache is available")
        else:
            logger.info("🔄 Force backtest enabled - ignoring cached results")

        logger.info("Running new backtests...")

        # Step 1: Get stock universe
        universe_df = data_provider.get_stock_universe()

        if universe_df.empty:
            logger.error("Failed to get stock universe")
            return []

        symbols = universe_df['symbol'].tolist()

        if test_mode:
            symbols = symbols[:TEST_MODE_UNIVERSE_LIMIT]
            logger.info(
                "🧪 Test mode universe limit applied: first %d symbols", len(symbols))

        logger.info("📋 Stock universe loaded: %d symbols", len(symbols))

        # Step 2: Set backtest date range
        end_date = datetime.now() - timedelta(minutes=20)
        start_date = globalConfig.BACKTEST_START_DATE

        logger.info("📊 Starting comprehensive backtest analysis...")
        logger.info(
            "🕐 This may take 30-90 minutes depending on market conditions")

        # Step 3: Run optimization per enabled strategy (Phase C multi-strategy).
        # Each strategy gets its own optimizer (own grid + z-score pool), then
        # results are merged and filtered together.
        enabled = globalConfig.STRATEGIES_ENABLED or ["rsi_mean_reversion"]
        raw_results: List = []
        filtered_results: List = []
        for strategy_name in enabled:
            logger.info("=" * 60)
            logger.info("🚀 Running backtests for strategy: %s", strategy_name)
            try:
                strategy_cls = get_strategy(strategy_name)
            except ValueError as e:
                logger.error("Unknown strategy '%s' in config — skipping. %s",
                             strategy_name, e)
                continue
            strategy = strategy_cls.create()
            optimizer = StrategyOptimizer(strategy=strategy)

            # Strategies may declare their own symbol universe (e.g. the
            # underlying stocks of leveraged ETFs). Fall back to the global
            # universe when they don't.
            strategy_symbols = symbols
            try:
                override = strategy.symbol_universe()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Strategy '%s' symbol_universe() failed: %s",
                    strategy_name, e)
                override = None
            if override:
                strategy_symbols = list(override)
                logger.info(
                    "📌 Strategy '%s' uses its own universe: %d symbols",
                    strategy_name, len(strategy_symbols))

            if globalConfig.WF_ENABLED:
                logger.info(
                    "🪟 Walk-forward validation enabled — splitting into IS/OOS windows")
                wf_validator = WalkForwardValidator(optimizer)
                wf_results = await wf_validator.validate_universe(
                    strategy_symbols, start_date, end_date,
                    progress_cb=progress_cb)

                # Convert WalkForwardResult → BacktestResult for downstream compatibility
                raw = [r.to_backtest_result() for r in wf_results]
            else:
                raw = await optimizer.optimize_universe(
                    strategy_symbols, start_date, end_date,
                    progress_cb=progress_cb)

            raw_results.extend(raw)
            # Per-strategy filtering (alpha > 0, profitable, trades, win rate)
            filtered_results.extend(optimizer.filter_results(raw))

        # Step 4: Filter results
        logger.info("🔍 Filtering and analyzing results...")
        logger.info("📈 Backtest analysis complete!")
        logger.info("   • Total strategies tested: %d", len(raw_results))
        logger.info("   • Profitable strategies: %d", len(filtered_results))
        logger.info("   • Success rate: %.1f%%",
                    (len(filtered_results)/len(raw_results)*100) if raw_results else 0)

        # Step 5: Save results to storage
        logger.info("💾 Saving results to storage...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        storage.save_backtest_results(
            filtered_results, timestamp)

        # Step 6: Enforce retention so the results store cannot grow without
        # bound. Runs once per backtest pass (after the new results are saved).
        self._prune_old_backtest_results()

        return filtered_results

    def _prune_old_backtest_results(self) -> None:
        """Delete backtest results older than the configured retention window.

        Retention is best-effort: a failure here is logged and swallowed so it
        can never fail an otherwise-successful trading cycle.
        """
        retention_days = getattr(
            globalConfig, 'BACKTEST_RESULTS_RETENTION_DAYS', 0)

        # Coerce defensively. A missing or malformed setting must never crash
        # the cycle, and must fail *closed* (keep the data) rather than guess
        # a window. Note that a thrown-away type (e.g. a test double) or an
        # unexpected JSON type lands here as invalid.
        parsed_days: Optional[int] = None
        if isinstance(retention_days, bool):
            parsed_days = None                      # bool is not a day count
        elif isinstance(retention_days, (int, float)):
            parsed_days = int(retention_days)
        elif isinstance(retention_days, str):
            try:
                parsed_days = int(retention_days.strip())
            except ValueError:
                parsed_days = None

        if parsed_days is None:
            logger.warning(
                "🗑️  Invalid backtest retention setting %r — skipping prune",
                retention_days)
            return

        if parsed_days <= 0:
            logger.info(
                "🗑️  Backtest retention disabled — keeping all results")
            return

        try:
            deleted = storage.prune_backtest_results(parsed_days)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Backtest retention prune failed: %s", e)
            return

        if deleted:
            logger.info(
                "🗑️  Pruned %d backtest result rows older than %d days",
                deleted, parsed_days)
        else:
            logger.info(
                "🗑️  No backtest results older than %d days to prune",
                parsed_days)

    def _load_recent_backtest_results(self, max_age_seconds: Optional[float] = 24 * 3600) -> List:
        """Load the most recent backtest results from storage.

        Args:
            max_age_seconds: Maximum allowed age of the cached file. Pass
                ``None`` to skip the freshness check entirely (used by the
                "run session" action to trade on the latest analysis without
                waiting for a fresh optimization pass).
        """
        try:
            backtest_files = storage.list_backtest_files()

            if not backtest_files:
                return []

            # Sort by filename (which contains timestamp) and get most recent
            backtest_files.sort(reverse=True)
            most_recent = backtest_files[0]
            logger.info("Most recent backtest file: %s", most_recent)
            # Check if file is recent enough (within max_age_seconds)
            try:
                # For filenames like backtest_results_20250610_170343.csv
                date_part = most_recent.split(
                    '_')[2]  # Extract date (20250610)
                time_part = most_recent.split('_')[3].split(
                    '.')[0]  # Extract time (170343)

                # Parse as date+time
                file_datetime = datetime.strptime(
                    f"{date_part}_{time_part}", '%Y%m%d_%H%M%S')

                if (max_age_seconds is not None and
                        (datetime.now() - file_datetime).total_seconds() >= max_age_seconds):
                    logger.info(
                        "Most recent backtest file is older than %.1f hours — ignoring",
                        max_age_seconds / 3600)
                    return []

                cached = storage.load_backtest_results(most_recent)
                # Cache-key guard: never reuse results produced by a
                # different strategy set than the one configured now
                # (matters once multiple strategies are enabled).
                if cached and not all(
                    getattr(r, "strategy_name", "rsi_mean_reversion")
                    in globalConfig.STRATEGIES_ENABLED
                    for r in cached
                ):
                    logger.info(
                        "Ignoring cached results: strategy set differs from configured strategies")
                    return []
                return cached
            except (IndexError, ValueError):
                pass

            return []

        except (ValueError, IndexError, TypeError) as e:
            logger.error("Error loading recent backtest results: %s", e)
            return []

    def _load_latest_backtest_results(self) -> List:
        """Load the most recent backtest results regardless of age.

        Used by the "run session" action to trade on the latest available
        analysis without re-running the (slow) optimization/backtest pass.
        """
        return self._load_recent_backtest_results(max_age_seconds=None)

    async def _save_session_results(self, dryRun: bool, account_info: Dict[str, Any], backtest_results: List, trading_summary: dict):
        """Save session results and metadata."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save session metadata
            self.session_metadata['backtest_count'] = len(backtest_results)
            self.session_metadata['portfolio_value'] = account_info.get(
                'equity', 0)
            self.session_metadata['long_market_value'] = account_info.get(
                'long_market_value', 0)
            self.session_metadata['short_market_value'] = account_info.get(
                'short_market_value', 0)
            self.session_metadata['dry_run'] = dryRun
            # Flatten trading_summary into individual columns
            for key, value in trading_summary.items():
                self.session_metadata[f'trading_{key}'] = value

            storage.save_metadata(
                self.session_metadata, timestamp)

            # Per-strategy end-of-day performance snapshot (Phase: multi-
            # strategy attribution). One row per strategy per day, upserted.
            self._save_strategy_performance_snapshot(account_info)

        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error saving session results: %s", e)

    # ------------------------------------------------------------------
    # Per-strategy daily performance snapshots
    # ------------------------------------------------------------------

    def _build_strategy_performance_records(
        self, account_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build one performance record per enabled strategy for today.

        Combines the account equity with per-strategy capital budgets
        (allocation) and per-strategy position attribution from the in-memory
        position book:

          * open_market_value — Σ current_price × |quantity| over open positions
          * unrealized_pnl    — Σ (current − entry) × quantity (negative qty =
                                short, so the sign works out naturally)
          * realized_pnl      — Σ realized_return × entry_price × |quantity|
                                over closed positions in the book

        Strategies with no positions still get a row (zero P&L) so the daily
        series never has gaps for an enabled strategy.
        """
        equity = float(account_info.get('equity', 0.0) or 0.0)
        enabled = list(getattr(globalConfig, 'STRATEGIES_ENABLED', None)
                       or ['rsi_mean_reversion'])
        budgets = self.trading_engine._strategy_budgets(equity)
        alloc = getattr(globalConfig, 'STRATEGY_ALLOCATION', None) or {}

        open_by_strategy: Dict[str, List] = {}
        for pos in (self.positions_manager.positions or []):
            if getattr(pos, 'closed', False):
                continue
            open_by_strategy.setdefault(
                getattr(pos, 'strategy_name', 'rsi_mean_reversion'), []).append(pos)

        realized_by_strategy: Dict[str, float] = {}
        for pos in (self.positions_manager.positions or []):
            if not getattr(pos, 'closed', False):
                continue
            ret = getattr(pos, 'realized_return', None)
            if ret is None:
                continue
            name = getattr(pos, 'strategy_name', 'rsi_mean_reversion')
            try:
                realized_by_strategy[name] = (
                    realized_by_strategy.get(name, 0.0)
                    + float(ret) * float(pos.entry_price)
                    * abs(float(pos.quantity)))
            except (TypeError, ValueError):
                continue

        records: List[Dict[str, Any]] = []
        for name in enabled:
            open_positions = open_by_strategy.get(name, [])
            market_value = 0.0
            unrealized = 0.0
            for pos in open_positions:
                try:
                    qty = float(pos.quantity)
                    market_value += float(pos.current_price) * abs(qty)
                    unrealized += (float(pos.current_price)
                                   - float(pos.entry_price)) * qty
                except (TypeError, ValueError, AttributeError):
                    continue
            weight = alloc.get(name)
            try:
                weight = float(weight) if weight is not None else None
            except (TypeError, ValueError):
                weight = None
            records.append({
                'snapshot_date': datetime.now().strftime('%Y-%m-%d'),
                'strategy_name': name,
                'equity': equity,
                'allocation_weight': weight,
                'budget_notional': budgets.get(name),
                'open_positions': len(open_positions),
                'open_market_value': market_value,
                'unrealized_pnl': unrealized,
                'realized_pnl': realized_by_strategy.get(name, 0.0),
            })
        return records

    def _save_strategy_performance_snapshot(
        self, account_info: Dict[str, Any]) -> None:
        """Persist today's per-strategy performance snapshot (best-effort)."""
        try:
            records = self._build_strategy_performance_records(account_info)
            if not records:
                return
            snapshot_date = datetime.now().strftime('%Y-%m-%d')
            saved = storage.save_strategy_performance(
                records, snapshot_date)
            if saved:
                logger.info(
                    "📊 Saved strategy performance snapshot (%d strategies, %s)",
                    len(records), snapshot_date)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Strategy performance snapshot failed (non-fatal): %s", e)


def _daily_scheduler(schedule_time: str, shared_state: dict, algorithm: 'TradingAlgorithm'):
    """Background thread that triggers a trading cycle at a fixed time each day.

    Waits until the next occurrence of *schedule_time* (HH:MM in Eastern),
    then sets the trigger_event.  Repeats every 24 hours.
    """
    import time as _time

    logger.info(
        "⏰ Daily scheduler started — will trigger at %s ET each day", schedule_time)

    while True:
        try:
            now = datetime.now()
            hour, minute = map(int, schedule_time.split(':'))
            target = now.replace(hour=hour, minute=minute,
                                 second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            logger.info(
                "⏰ Next scheduled run: %s ET (in %.1f hours)",
                target.strftime('%Y-%m-%d %H:%M'), wait_seconds / 3600,
            )

            # Sleep in 60-second chunks so we remain responsive to shutdown
            while wait_seconds > 0:
                chunk = min(wait_seconds, 60)
                _time.sleep(chunk)
                wait_seconds -= chunk
                # If a manual trigger is already in progress, skip this tick
                if shared_state.get('cycle_running', False):
                    logger.info(
                        "⏰ Skipping scheduled trigger — a cycle is already running")
                    break
            else:
                # Timer expired cleanly — fire the trigger
                if not shared_state.get('cycle_running', False):
                    logger.info(
                        "⏰ Scheduled time reached — triggering daily cycle")
                    shared_state['cycle_flags'] = {}
                    trigger = shared_state.get('trigger_event')
                    if trigger:
                        trigger.set()
        except Exception as e:
            logger.error("Scheduler error: %s", e)
            _time.sleep(60)  # back off on error


def _make_job_progress_cb(job_id):
    """Build a progress callback for a background job (None if no job)."""
    if not job_id:
        return None
    # pylint: disable=import-outside-toplevel
    from jobs import job_manager
    return job_manager.make_progress_callback(job_id)


def _mark_job_running(job_id: Optional[str]) -> None:
    if not job_id:
        return
    # pylint: disable=import-outside-toplevel
    from jobs import job_manager
    job_manager.mark_running(job_id)


def _finish_job(job_id: Optional[str], success: bool,
                result_summary: Optional[dict] = None,
                error: Optional[str] = None) -> None:
    if not job_id:
        return
    # pylint: disable=import-outside-toplevel
    from jobs import job_manager
    job_manager.finish_job(job_id, success,
                           result_summary=result_summary, error=error)


def _finish_initial_job_if_any(shared_state: dict, session_result: dict) -> None:
    """Complete the startup job (if the first cycle was API-triggered)."""
    job_id = (shared_state.get('cycle_flags') or {}).get('job_id')
    if job_id:
        _finish_job(
            job_id,
            success=session_result.get('status') == 'success',
            result_summary={'status': session_result.get('status')},
            error=session_result.get('error'),
        )
        shared_state['cycle_flags'] = {}


def _bar_loop_worker(algorithm: 'TradingAlgorithm', shared_state: dict):
    """Poll bar-loop strategies during RTH; close intraday positions at session end.

    Runs in keep-alive mode alongside the daily scheduler. Evaluates intraday
    strategies every 60s while the market is open (using the latest cycle's
    backtest results), and force-closes intraday positions once per day after
    the 16:00 ET close.
    """
    import time as _time
    from bar_engine import BarLoopEngine  # pylint: disable=import-outside-toplevel

    bar_engine = BarLoopEngine(
        algorithm.trading_engine, algorithm.positions_manager)
    bar_engine.set_dry_run_mode(bool(shared_state.get('dry_run', False)))
    last_close_date: Optional[str] = None
    logger.info("📈 Bar-loop worker started for intraday strategies")

    while True:
        try:
            _time.sleep(60)
            # Don't interfere with a running daily cycle.
            if shared_state.get('cycle_running', False):
                continue

            now = datetime.now()
            results = algorithm.last_backtest_results

            if bar_engine.is_rth(now):
                summary = bar_engine.run_intraday_cycle(results, as_of=now)
                if summary.get('signals') or summary.get('orders_placed'):
                    logger.info("📈 Intraday cycle: %s", summary)
            elif bar_engine.session_ended(now) and bar_engine.has_open_intraday_positions():
                today = now.strftime('%Y-%m-%d')
                if last_close_date != today:
                    summary = bar_engine.close_intraday_positions(as_of=now)
                    last_close_date = today
                    logger.info("🕓 Intraday session close: %s", summary)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Bar-loop worker error: %s", e)
            _time.sleep(60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading Algorithm')
    parser.add_argument('--force-backtest', action='store_true',
                        help='Force running new backtests')
    parser.add_argument('--paper-trading', action='store_true',
                        help='Enable paper trading mode')
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set logging level')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run analysis without placing orders')
    parser.add_argument('--test-mode', action='store_true',
                        help=f'Run backtest on first {TEST_MODE_UNIVERSE_LIMIT} symbols to validate full flow quickly')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Override globalConfig if needed
    if args.paper_trading:
        globalConfig.PAPER_TRADE = True

    logger.info("=" * 50)
    logger.info("Trading Algorithm Starting")
    logger.info("Paper Trading: %s", globalConfig.PAPER_TRADE)
    logger.info("Dry Run: %s", args.dry_run)
    logger.info("Test Mode: %s", args.test_mode)
    logger.info("=" * 50)

    try:
        # Initialize and run the trading algorithm
        algorithm = TradingAlgorithm()

        # Always start the health server so the Docker HEALTHCHECK passes
        # (the HEALTHCHECK runs curl against :8080/health regardless of KEEP_ALIVE).
        # When KEEP_ALIVE is false, the process exits after run_full_cycle completes
        # and the daemon thread is torn down automatically.
        import threading
        shared_state: dict = {
            'last_result': None,
            'cycle_running': False,
            'cycle_flags': {},
            'trigger_event': threading.Event(),
        }
        health_thread = threading.Thread(
            target=start_health_server,
            args=(globalConfig.HEALTH_PORT,
                  shared_state, storage, data_provider),
            daemon=True,
        )
        health_thread.start()
        logger.info(
            "🛟 Health server started on port %d",
            globalConfig.HEALTH_PORT,
        )
        if globalConfig.KEEP_ALIVE:
            logger.info(
                "🛟 Dashboard available at http://localhost:%d/",
                globalConfig.HEALTH_PORT,
            )
            # Start the daily scheduler if SCHEDULE_TIME is configured
            if globalConfig.SCHEDULE_TIME:
                scheduler_thread = threading.Thread(
                    target=_daily_scheduler,
                    args=(globalConfig.SCHEDULE_TIME, shared_state, algorithm),
                    daemon=True,
                )
                scheduler_thread.start()

            # Start the bar-loop worker when intraday strategies are enabled
            from bar_engine import BarLoopEngine  # pylint: disable=import-outside-toplevel
            if BarLoopEngine(
                algorithm.trading_engine,
                algorithm.positions_manager,
            ).enabled_bar_loop_strategies():
                bar_loop_thread = threading.Thread(
                    target=_bar_loop_worker,
                    args=(algorithm, shared_state),
                    daemon=True,
                )
                bar_loop_thread.start()

        session_result = await algorithm.run_full_cycle(
            force_backtest=args.force_backtest,
            dry_run=args.dry_run,
            test_mode=args.test_mode,
        )
        # Expose the completed result to the health server
        shared_state['last_result'] = session_result
        shared_state['cycle_running'] = False
        _finish_initial_job_if_any(shared_state, session_result)

        logger.info("=" * 50)
        logger.info("Trading Algorithm Complete")
        logger.info("Result: %s", session_result)
        logger.info("=" * 50)

        # ---- KEEP_ALIVE idle loop (event-driven) ----
        if globalConfig.KEEP_ALIVE:
            logger.info(
                "🛟 KEEP_ALIVE — container idling. Dashboard at http://localhost:%d/. "
                "Trigger cycles via POST /api/run-cycle or press Ctrl+C to exit.",
                globalConfig.HEALTH_PORT)
            trigger = shared_state['trigger_event']
            try:
                while True:
                    # Wait until a cycle is triggered (or wake every 60s to
                    # keep the thread responsive to signals).
                    trigger.wait(timeout=60)
                    if not trigger.is_set():
                        continue

                    trigger.clear()
                    shared_state['cycle_running'] = True
                    flags = shared_state.get('cycle_flags', {})

                    job_id = flags.get('job_id')
                    progress_cb = _make_job_progress_cb(job_id)

                    logger.info(
                        "🔄 Cycle triggered via API (flags=%s) — starting new run...",
                        flags)

                    try:
                        _mark_job_running(job_id)
                        session_result = await algorithm.run_full_cycle(
                            force_backtest=flags.get('force_backtest', False),
                            dry_run=flags.get('dry_run', False),
                            test_mode=flags.get('test_mode', False),
                            run_session_only=flags.get(
                                'run_session_only', False),
                            progress_cb=progress_cb,
                        )
                        shared_state['last_result'] = session_result
                        _finish_job(
                            job_id,
                            success=session_result.get('status') == 'success',
                            result_summary={
                                'status': session_result.get('status'),
                                'backtest_count': session_result.get(
                                    'backtest_count', 0),
                            },
                            error=session_result.get('error'),
                        )
                        logger.info("✅ Triggered cycle complete: %s",
                                    session_result.get('status'))
                    except Exception as e:
                        logger.error("Triggered cycle failed: %s", e)
                        _finish_job(job_id, success=False, error=str(e))
                        shared_state['last_result'] = {
                            'status': 'error', 'error': str(e)}
                    finally:
                        shared_state['cycle_running'] = False
                        shared_state['cycle_flags'] = {}
            except KeyboardInterrupt:
                logger.info("Idle loop interrupted — shutting down.")

        return session_result

    except (KeyboardInterrupt, SystemExit):
        logger.info("Algorithm interrupted by user")
        return {'status': 'interrupted'}
    except (ValueError, TypeError, KeyError) as e:
        logger.error("Unexpected error: %s", e)
        return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    result = asyncio.run(main())

    # Exit with appropriate code
    if result.get('status') == 'success':
        sys.exit(0)
    else:
        sys.exit(1)
