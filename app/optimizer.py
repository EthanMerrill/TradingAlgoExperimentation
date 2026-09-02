"""
Strategy optimization module.
Grid search over RSI parameters with two-stage (coarse + fine) optimization.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from data_provider import data_provider
from strategies.base import Strategy
from strategies.rsi import RSIStrategy
from utils import ProgressIndicator, resolve_worker_counts

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Grid-search optimizer over a Strategy's parameter space.

    Owns data fetching + universe orchestration; the actual grid search is
    delegated to ``strategy.optimize(...)`` so strategies can specialize
    (RSI keeps its two-stage search and per-period RSI cache).
    """

    def __init__(self, strategy: Optional[Strategy] = None):
        # Backward-compatible default: RSI strategy with config-derived ranges.
        self.strategy = strategy if strategy is not None else RSIStrategy.create()
        # RSI ranges kept for backward compatibility (legacy attribute surface).
        self.rsi_periods = list(range(*globalConfig.RSI_PERIOD_RANGE))
        self.rsi_lowers = list(range(*globalConfig.RSI_LOWER_RANGE))
        self.rsi_uppers = list(range(*globalConfig.RSI_UPPER_RANGE))
        self.last_consolidated_trades_df = pd.DataFrame()

    def optimize_symbol(self, symbol: str, start_date: datetime, end_date: datetime, direction: str = "long", prefetched_data: Optional[pd.DataFrame] = None) -> Optional["BacktestResult"]:  # noqa: F821
        """
        Optimize the strategy's parameters for a single symbol.

        Args:
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            direction: "long" or "short"
            prefetched_data: Optional pre-fetched OHLCV DataFrame (skips API call when provided).

        Returns:
            Best BacktestResult or None if optimization fails
        """
        try:
            # Shift start_date back to warm up indicator history.  The number of
            # calendar days needed is strategy-specific (e.g. RSI periods).
            warmup_days = self.strategy.warmup_days()
            warmup_start = start_date - timedelta(days=warmup_days)

            # Use pre-fetched data when available (avoids duplicate API calls
            # when testing both long and short for the same symbol).
            if prefetched_data is not None:
                data = prefetched_data
            else:
                data = data_provider.get_single_stock_bars(
                    symbol, warmup_start, end_date)

            if data.empty or len(data) < 50:
                logger.debug(
                    f"⚠️  {symbol}: Insufficient data ({len(data)} rows)")
                return None

            # Delegate the grid search to the strategy (RSI keeps its
            # two-stage search + per-period RSI cache; new strategies get the
            # generic grid-search default from Strategy.optimize).
            return self.strategy.optimize(
                data, symbol, direction, globalConfig.BACKTEST_INIT_CASH)

        except Exception as e:
            logger.error(f"💥 Error optimizing {symbol}: {e}")
            return None

    async def optimize_universe(self, symbols: List[str], start_date: datetime, end_date: datetime) -> List["BacktestResult"]:  # noqa: F821
        """
        Optimize RSI parameters for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            List of BacktestResult objects
        """
        # Lazy import to avoid circular dependency at module level.
        from strategy import BacktestResult  # pylint: disable=import-outside-toplevel

        results = []
        processed_count = 0
        successful_count = 0
        total_symbols = len(symbols)

        # Track timing for progress estimates
        start_time = time.time()

        logger.info(
            f"🚀 Starting backtest optimization for {total_symbols} symbols")
        logger.info(
            f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(
            f"⚙️  Strategy: {self.strategy.name} — "
            f"grid: {len(self.strategy.get_param_grid('long'))} combos")
        logger.info(
            f"📉 Short selling: {'ENABLED' if globalConfig.ENABLE_SHORT_SELLING else 'DISABLED'}")
        logger.info("=" * 60)

        # Initialize progress indicator
        progress = ProgressIndicator(total_symbols, "🔍 Optimizing strategies")

        # Use ThreadPoolExecutor for I/O bound operations
        loop = asyncio.get_event_loop()

        # Process symbols in batches to avoid overwhelming the API.
        # When grid-level parallelism is active (n_jobs != 1), process
        # symbols one at a time since each symbol already saturates cores.
        (
            os_detected_cpus,
            joblib_detected_cpus,
            detected_cpus,
            effective_workers,
        ) = resolve_worker_counts(globalConfig.N_JOBS)
        batch_size = 1 if effective_workers > 1 else 10
        logger.info(
            "⚡ CPU detection: os.cpu_count=%s, joblib.cpu_count=%s, selected=%d",
            os_detected_cpus, joblib_detected_cpus, detected_cpus
        )
        logger.info(
            "⚡ Parallelism: configured_n_jobs=%d, effective_workers=%d, batch_size=%d",
            globalConfig.N_JOBS, effective_workers, batch_size
        )
        total_batches = (total_symbols + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(symbols), batch_size), 1):
            batch = symbols[i:i + batch_size]
            batch_start_time = time.time()

            logger.info(
                f"📊 Processing batch {batch_num}/{total_batches} ({len(batch)} symbols): {', '.join(batch)}")

            # Compute warmup days once for the batch (strategy-specific)
            warmup_days = self.strategy.warmup_days()
            warmup_start = start_date - timedelta(days=warmup_days)

            # Pre-fetch OHLCV data per symbol so long and short share the same fetch
            symbol_data_map: Dict[str, pd.DataFrame] = {}
            for symbol in batch:
                fetched = data_provider.get_single_stock_bars(
                    symbol, warmup_start, end_date)
                if not fetched.empty and len(fetched) >= 50:
                    symbol_data_map[symbol] = fetched
                else:
                    logger.debug(
                        f"⚠️  {symbol}: Insufficient data ({len(fetched)} rows), skipping")

            tasks = []
            task_symbols: List[Tuple[str, str]] = []  # (symbol, direction)
            for symbol in batch:
                prefetched = symbol_data_map.get(symbol)
                if prefetched is None:
                    continue  # skip symbols with no data

                task = loop.run_in_executor(
                    None,
                    self.optimize_symbol,
                    symbol,
                    start_date,
                    end_date,
                    "long",
                    prefetched,
                )
                tasks.append(task)
                task_symbols.append((symbol, "long"))
                if globalConfig.ENABLE_SHORT_SELLING:
                    short_task = loop.run_in_executor(
                        None,
                        self.optimize_symbol,
                        symbol,
                        start_date,
                        end_date,
                        "short",
                        prefetched,
                    )
                    tasks.append(short_task)
                    task_symbols.append((symbol, "short"))

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results — one BacktestResult per task
            batch_successful = 0
            for (sym, _dir), result in zip(task_symbols, batch_results):
                if isinstance(result, BacktestResult):
                    results.append(result)
                    successful_count += 1
                    batch_successful += 1
                elif result is not None:
                    logger.error(f"Error in batch processing: {result}")

            processed_count += len(batch)
            for _ in batch:
                progress.update(1, f"Batch {batch_num}/{total_batches}")

            # Calculate progress and time estimates
            batch_time = time.time() - batch_start_time
            total_elapsed = time.time() - start_time
            completion_pct = (processed_count / total_symbols) * 100

            if processed_count > 0:
                avg_time_per_symbol = total_elapsed / processed_count
                remaining_symbols = total_symbols - processed_count
                estimated_remaining_time = avg_time_per_symbol * remaining_symbols

                # Clear progress line and show batch summary
                print()  # New line after progress bar
                logger.info(
                    f"✅ Batch {batch_num} complete: {batch_successful}/{len(batch)} successful (took {batch_time:.1f}s)")
                logger.info(f"📈 Progress: {processed_count}/{total_symbols} ({completion_pct:.1f}%) | "
                            f"Successful: {successful_count} | "
                            f"ETA: {estimated_remaining_time/60:.1f} min")
                logger.info("─" * 60)

            # Small delay between batches
            await asyncio.sleep(1)

        # Finish progress indicator
        progress.finish("All symbols processed!")

        # Final summary
        total_time = time.time() - start_time
        success_rate = (successful_count / total_symbols) * \
            100 if total_symbols > 0 else 0

        logger.info("=" * 60)
        logger.info("🎯 BACKTEST OPTIMIZATION COMPLETE!")
        logger.info("📊 Results Summary:")
        logger.info(
            f"   • Total symbols processed: {processed_count}/{total_symbols}")
        logger.info(
            f"   • Successful optimizations: {successful_count} ({success_rate:.1f}%)")
        logger.info(f"   • Total time: {total_time/60:.1f} minutes")
        logger.info(
            f"   • Average time per symbol: {total_time/total_symbols:.1f}s")
        if successful_count > 0:
            profitable_count = len([r for r in results if r.profitable])
            logger.info(
                f"   • Profitable strategies: {profitable_count}/{successful_count}")
        logger.info("=" * 60)

        # Build consolidated trades DataFrame for optional runtime use.
        if results:
            logger.info(
                "📊 Building consolidated trades DataFrame for runtime use...")
            self.last_consolidated_trades_df = self.build_consolidated_trades(
                results)

        # Compute cross-symbol Z-scores so results are comparable across symbols.
        if results:
            logger.info("📊 Computing cross-symbol Z-scores...")
            import zscore  # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported
            zscore.compute_cross_symbol_zscores(results)

        return [r for r in results if r is not None]

    def build_consolidated_trades(self, results: List["BacktestResult"]) -> pd.DataFrame:  # noqa: F821
        """
        Build consolidated trades DataFrame from optimization results.

        Args:
            results: List of BacktestResult objects

        Returns:
            Consolidated trades DataFrame
        """
        # Lazy import to avoid circular dependency at module level.
        return self.strategy.build_consolidated_trades(results)

    def filter_results(self, results: List["BacktestResult"]) -> List["BacktestResult"]:  # noqa: F821
        """
        Filter backtest results for trading opportunities.

        Args:
            results: List of BacktestResult objects

        Returns:
            Filtered list of profitable strategies with positive alpha
        """
        filtered = []

        for result in results:
            # Filter criteria from legacy get_entries function
            if (result.alpha > 0 and
                result.profitable and
                result.num_trades > 0 and
                    result.win_rate > 0.3):  # At least 30% win rate
                filtered.append(result)

        # Sort by composite score (cross-symbol Z-score, descending)
        filtered.sort(key=lambda x: x.composite_score, reverse=True)

        return filtered
