"""
Main application entry point for the trading algorithm.
Orchestrates the entire trading workflow.
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

from data_provider import data_provider
from storage import storage
from positions import PositionsManager
from optimizer import StrategyOptimizer
from walk_forward import WalkForwardValidator
from trading_engine import TradingEngine
from utils import TradingCalendar, setup_logging

from config import globalConfig  # type: ignore
from health_server import start_health_server

logger = logging.getLogger(__name__)
TEST_MODE_UNIVERSE_LIMIT = 50


class TradingAlgorithm:
    """Main trading algorithm orchestrator."""

    def __init__(self):
        self.optimizer = StrategyOptimizer()
        self.trading_engine = TradingEngine()
        self.trading_calendar = TradingCalendar()
        self.positions_manager = PositionsManager(
            storage, data_provider
        )
        self.session_metadata = {
            'start_time': None,
            'end_time': None,
            'config': globalConfig.to_dict(),
            'portfolio_value': 0,
            'results_summary': {}
        }

    async def run_full_cycle(self, force_backtest: bool = False, dry_run: bool = False, test_mode: bool = False) -> dict:
        """
        Run the complete trading algorithm cycle.

        Args:
            force_backtest: Force running backtest even if recent results exist
            dry_run: Run in dry run mode without placing actual orders
            test_mode: Run backtest on a limited stock universe for fast end-to-end validation

        Returns:
            Dictionary with session results
        """
        self.session_metadata['start_time'] = datetime.now()

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

            if buying_power > 0 or force_backtest:
                # Step 2: Get or run backtests
                backtest_results = await self._get_backtest_results(force_backtest, test_mode)
            else:
                logger.warning(
                    "Insufficient buying power available for purchases")
                logger.info(
                    "Skipping backtest due to insufficient buying power - will only process existing positions")

            if not backtest_results:
                logger.warning(
                    "No backtest results available - processing existing positions only")

            # Step 3: Execute trading session
            logger.info(
                "🎯 Analyzing trading opportunities and executing orders...")
            trading_summary = self.trading_engine.execute_trading_session(
                backtest_results)

            # Step 4: Save results and metadata
            self.session_metadata['end_time'] = datetime.now()
            self.session_metadata['results_summary'] = trading_summary
            logger.info("💾 Saving session results and metadata...")
            await self._save_session_results(dry_run, account_info, backtest_results, trading_summary)

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

    async def _get_backtest_results(self, force_backtest: bool, test_mode: bool = False) -> List:
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

        # Step 3: Run optimization for all symbols
        if globalConfig.WF_ENABLED:
            logger.info(
                "🪟 Walk-forward validation enabled — splitting into IS/OOS windows")
            wf_validator = WalkForwardValidator(self.optimizer)
            wf_results = await wf_validator.validate_universe(symbols, start_date, end_date)

            # Convert WalkForwardResult → BacktestResult for downstream compatibility
            raw_results = [r.to_backtest_result() for r in wf_results]
        else:
            raw_results = await self.optimizer.optimize_universe(symbols, start_date, end_date)

        # Step 4: Filter results
        logger.info("🔍 Filtering and analyzing results...")
        filtered_results = self.optimizer.filter_results(raw_results)

        logger.info("📈 Backtest analysis complete!")
        logger.info("   • Total strategies tested: %d", len(raw_results))
        logger.info("   • Profitable strategies: %d", len(filtered_results))
        logger.info("   • Success rate: %.1f%%",
                    (len(filtered_results)/len(raw_results)*100) if raw_results else 0)

        # Step 5: Save results to cloud storage
        logger.info("💾 Saving results to cloud storage...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        storage.save_backtest_results(
            filtered_results, timestamp)

        return filtered_results

    def _load_recent_backtest_results(self) -> List:
        """Load recent backtest results from cloud storage."""
        try:
            backtest_files = storage.list_backtest_files()

            if not backtest_files:
                return []

            # Sort by filename (which contains timestamp) and get most recent
            backtest_files.sort(reverse=True)
            most_recent = backtest_files[0]
            logger.info("Most recent backtest file: %s", most_recent)
            # Check if file is recent enough (within last 24 hours)
            try:
                # For filenames like backtest_results_20250610_170343.csv
                date_part = most_recent.split(
                    '_')[2]  # Extract date (20250610)
                time_part = most_recent.split('_')[3].split(
                    '.')[0]  # Extract time (170343)

                # Parse as date+time
                file_datetime = datetime.strptime(
                    f"{date_part}_{time_part}", '%Y%m%d_%H%M%S')

                if (datetime.now() - file_datetime).total_seconds() < 24 * 3600:
                    return storage.load_backtest_results(most_recent)
            except (IndexError, ValueError):
                pass

            return []

        except (ValueError, IndexError, TypeError) as e:
            logger.error("Error loading recent backtest results: %s", e)
            return []

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

        except (ValueError, TypeError, KeyError) as e:
            logger.error("Error saving session results: %s", e)


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

        # If KEEP_ALIVE is set, start the dashboard server immediately
        # so you can watch progress while the backtest runs.
        shared_state = None
        if globalConfig.KEEP_ALIVE:
            import threading
            shared_state: dict = {'last_result': None}
            health_thread = threading.Thread(
                target=start_health_server,
                args=(globalConfig.HEALTH_PORT, shared_state, storage, data_provider),
                daemon=True,
            )
            health_thread.start()
            logger.info(
                "🛟 Dashboard server started on port %d — visit http://localhost:%d/",
                globalConfig.HEALTH_PORT, globalConfig.HEALTH_PORT,
            )

        session_result = await algorithm.run_full_cycle(
            force_backtest=args.force_backtest,
            dry_run=args.dry_run,
            test_mode=args.test_mode,
        )

        # Expose the completed result to the health server
        if shared_state is not None:
            shared_state['last_result'] = session_result

        logger.info("=" * 50)
        logger.info("Trading Algorithm Complete")
        logger.info("Result: %s", session_result)
        logger.info("=" * 50)

        return session_result

    except (KeyboardInterrupt, SystemExit):
        logger.info("Algorithm interrupted by user")
        return {'status': 'interrupted'}
    except (ValueError, TypeError, KeyError) as e:
        logger.error("Unexpected error: %s", e)
        return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    result = asyncio.run(main())

    # If KEEP_ALIVE is set, the server was already started inside main()
    # before the backtest ran.  Just idle so the container stays alive.
    if globalConfig.KEEP_ALIVE:
        logger.info("🛟 KEEP_ALIVE — container idling. Press Ctrl+C to exit.")
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Idle loop interrupted — shutting down.")
            sys.exit(0)

    # Exit with appropriate code
    if result.get('status') == 'success':
        sys.exit(0)
    else:
        sys.exit(1)
