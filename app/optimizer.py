"""
Strategy optimization module.
Grid search over RSI parameters with two-stage (coarse + fine) optimization.
"""
import asyncio
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import pytz
from data_provider import TechnicalIndicators, data_provider
from joblib import Parallel, delayed
from utils import PerformanceMetrics, ProgressIndicator

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Optimize RSI strategy parameters for multiple symbols."""

    def __init__(self):
        self.rsi_periods = list(range(*globalConfig.RSI_PERIOD_RANGE))
        self.rsi_lowers = list(range(*globalConfig.RSI_LOWER_RANGE))
        self.rsi_uppers = list(range(*globalConfig.RSI_UPPER_RANGE))
        self.last_consolidated_trades_df = pd.DataFrame()

    @staticmethod
    def _composite_score(result: "BacktestResult") -> float:  # noqa: F821
        """Deprecated: use zscore.compute_stage_zscores / compute_cross_symbol_zscores.

        Returns result.composite_score if already set (cross-symbol), otherwise
        falls back to the legacy formula.  Still used by positions.py for display.
        """
        if result.composite_score != 0.0:
            return result.composite_score
        return StrategyOptimizer._composite_score_from_parts(
            result.alpha, result.sharpe_ratio, result.calmar_ratio
        )

    @staticmethod
    def _composite_score_from_parts(alpha: float, sharpe: float, calmar: float) -> float:
        """Legacy fallback: (alpha*100) + sharpe + calmar.

        Only used when no Z-score pool is available.
        """
        return (alpha * 100) + sharpe + calmar

    @staticmethod
    def _test_single_combo(
        data: pd.DataFrame,
        symbol: str,
        rsi_series: pd.Series,
        rsi_period: int,
        rsi_lower: int,
        rsi_upper: int,
        direction: str,
        initial_cash: float,
    ) -> Tuple["BacktestResult", int, int, int]:  # noqa: F821
        """Run a single backtest for one parameter combination.

        Extracted as a static method so it's trivially shareable across
        threads/processes — no mutable instance state, all inputs are
        read-only primitives or numpy/pandas objects.

        Returns:
            Tuple of (BacktestResult, rsi_period, rsi_lower, rsi_upper)
        """
        # Lazy import to avoid circular dependency at module level.
        from strategy import RSIStrategy  # pylint: disable=import-outside-toplevel

        strategy = RSIStrategy(
            rsi_period, rsi_lower, rsi_upper, direction=direction
        )
        result = strategy.backtest_with_rsi(
            data, symbol, rsi_series, initial_cash
        )
        return result, rsi_period, rsi_lower, rsi_upper

    def optimize_symbol(self, symbol: str, start_date: datetime, end_date: datetime, direction: str = "long") -> Optional["BacktestResult"]:  # noqa: F821
        """
        Optimize RSI parameters for a single symbol.

        Args:
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            direction: "long" or "short"

        Returns:
            Best BacktestResult or None if optimization fails
        """
        # Lazy imports to avoid circular dependency at module level.
        from strategy import RSIStrategy, BacktestResult  # pylint: disable=import-outside-toplevel

        try:
            # Shift start_date back to warm up RSI calculation.
            # RSI needs rsi_period + 1 bars; the optimizer tests periods up to
            # max(self.rsi_periods).  Multiply by 2 to convert trading days to
            # calendar days (covers weekends + holidays with margin).
            max_rsi_period = max(self.rsi_periods) if self.rsi_periods else 14
            warmup_days = max_rsi_period * 2
            warmup_start = start_date - timedelta(days=warmup_days)

            # Get historical data
            data = data_provider.get_single_stock_bars(
                symbol, warmup_start, end_date)

            if data.empty or len(data) < 50:
                logger.debug(
                    f"⚠️  {symbol}: Insufficient data ({len(data)} rows)")
                return None

            # --- Tier 1: Precompute RSI for every unique period once ---
            # RSI depends only on rsi_period, not lower/upper, so this eliminates
            # ~96% of redundant RSI calculations (e.g., 28 recomputations → 1 per period).
            price_col = RSIStrategy._get_price_column(data)
            rsi_cache: Dict[int, pd.Series] = {}
            for period in self.rsi_periods:
                rsi_cache[period] = TechnicalIndicators.calculate_rsi(
                    data, period, price_col
                )

            best_result: Optional[BacktestResult] = None
            best_score = -float('inf')
            tested_combinations = 0
            tested_set: set = set()  # dedup between stages
            n_jobs = globalConfig.N_JOBS

            # Lazy import to avoid circular dependency at module level.
            import zscore  # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported

            # Determine whether two-stage optimization is worthwhile.
            # Requires at least 3 values in both lower and upper ranges.
            fine_step_lower = (
                self.rsi_lowers[1] - self.rsi_lowers[0]
                if len(self.rsi_lowers) >= 2 else 1
            )
            fine_step_upper = (
                self.rsi_uppers[1] - self.rsi_uppers[0]
                if len(self.rsi_uppers) >= 2 else 1
            )
            use_two_stage = (
                len(self.rsi_lowers) >= 4
                and len(self.rsi_uppers) >= 4
                and fine_step_lower > 0
                and fine_step_upper > 0
            )

            if use_two_stage:
                # --- Tier 2: Stage 1 — Coarse grid (double step) ---
                coarse_step_lower = fine_step_lower * 2
                coarse_step_upper = fine_step_upper * 2
                coarse_lowers = list(range(
                    self.rsi_lowers[0], self.rsi_lowers[-1] +
                    1, coarse_step_lower
                ))
                coarse_uppers = list(range(
                    self.rsi_uppers[0], self.rsi_uppers[-1] +
                    1, coarse_step_upper
                ))

                coarse_candidates: List[Tuple[float, int, int, int]] = []

                # Build flat list of combos for parallel dispatch
                coarse_combos: List[Tuple[int, int, int, pd.Series]] = []
                for rsi_period in self.rsi_periods:
                    rsi = rsi_cache[rsi_period]
                    for rsi_lower in coarse_lowers:
                        for rsi_upper in coarse_uppers:
                            if rsi_lower >= rsi_upper:
                                continue
                            tested_set.add((rsi_period, rsi_lower, rsi_upper))
                            coarse_combos.append(
                                (rsi_period, rsi_lower, rsi_upper, rsi))

                coarse_count = len(coarse_combos)
                tested_combinations = coarse_count
                logger.debug(
                    "🔍 %s: Stage 1/2 — coarse grid (%d lowers × %d uppers × %d periods = %d combos)",
                    symbol, len(coarse_lowers), len(coarse_uppers),
                    len(self.rsi_periods), coarse_count
                )

                # Run coarse grid in parallel
                parallel_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                    delayed(StrategyOptimizer._test_single_combo)(
                        data, symbol, rsi_series, rsi_period, rsi_lower, rsi_upper,
                        direction, globalConfig.BACKTEST_INIT_CASH
                    )
                    for rsi_period, rsi_lower, rsi_upper, rsi_series in coarse_combos
                )
                assert isinstance(
                    parallel_results, list), "Parallel() returned non-list for coarse grid"
                parallel_results = cast(
                    List[Tuple[BacktestResult, int, int, int]], parallel_results)

                # Compute Z-scores within the coarse pool
                coarse_zscores = zscore.compute_stage_zscores(parallel_results)

                for idx, (result, rp, rl, ru) in enumerate(parallel_results):
                    score = coarse_zscores[idx]
                    if result.profitable:
                        coarse_candidates.append((score, rp, rl, ru))
                    if score > best_score and result.profitable:
                        best_score = score
                        best_result = result

                # Keep top-3 for fine refinement
                coarse_candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = coarse_candidates[:3]

                # --- Tier 2: Stage 2 — Fine grid around top candidates (parallel) ---
                if coarse_candidates:

                    fine_combos: List[Tuple[int, int, int, pd.Series]] = []
                    for _, c_period, c_lower, c_upper in top_candidates:
                        fine_lowers = [
                            x for x in range(
                                c_lower - fine_step_lower,
                                c_lower + fine_step_lower + 1,
                                fine_step_lower
                            )
                            if x >= self.rsi_lowers[0] and x < self.rsi_uppers[-1]
                        ]
                        fine_uppers = [
                            x for x in range(
                                c_upper - fine_step_upper,
                                c_upper + fine_step_upper + 1,
                                fine_step_upper
                            )
                            if x > self.rsi_lowers[0] and x <= self.rsi_uppers[-1]
                        ]

                        rsi = rsi_cache[c_period]
                        for rsi_lower in fine_lowers:
                            for rsi_upper in fine_uppers:
                                if rsi_lower >= rsi_upper:
                                    continue
                                key = (c_period, rsi_lower, rsi_upper)
                                if key in tested_set:
                                    continue
                                tested_set.add(key)
                                fine_combos.append(
                                    (c_period, rsi_lower, rsi_upper, rsi))

                    if fine_combos:
                        fine_count = len(fine_combos)
                        tested_combinations += fine_count

                        fine_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                            delayed(StrategyOptimizer._test_single_combo)(
                                data, symbol, rsi_series, rsi_period, rsi_lower, rsi_upper,
                                direction, globalConfig.BACKTEST_INIT_CASH
                            )
                            for rsi_period, rsi_lower, rsi_upper, rsi_series in fine_combos
                        )
                        assert isinstance(
                            fine_results, list), "Parallel() returned non-list for fine grid"
                        fine_results = cast(
                            List[Tuple[BacktestResult, int, int, int]], fine_results)

                        # Compute Z-scores within the fine pool
                        fine_zscores = zscore.compute_stage_zscores(
                            fine_results)

                        for idx, (result, rp, rl, ru) in enumerate(fine_results):
                            score = fine_zscores[idx]
                            if score > best_score and result.profitable:
                                best_score = score
                                best_result = result
                                logger.debug(
                                    "🎯 %s: New best — RSI(%d, %d, %d) Z-Score: %.2f",
                                    symbol, rp, rl, ru, score
                                )

                        logger.debug(
                            "🔍 %s: Stage 2/2 — %d fine combos tested around top-3 candidates",
                            symbol, fine_count
                        )
                    else:
                        logger.debug(
                            "🔍 %s: Stage 2/2 — no new fine combos to test",
                            symbol
                        )
            else:
                # Fallback: single-stage fine grid with cached RSI (Tier 1 only).
                # Used when parameter ranges are too narrow for two-stage to help.
                fallback_combos: List[Tuple[int, int, int, pd.Series]] = []
                for rsi_period in self.rsi_periods:
                    rsi = rsi_cache[rsi_period]
                    for rsi_lower in self.rsi_lowers:
                        for rsi_upper in self.rsi_uppers:
                            if rsi_lower >= rsi_upper:
                                continue
                            fallback_combos.append(
                                (rsi_period, rsi_lower, rsi_upper, rsi))

                tested_combinations = len(fallback_combos)

                if fallback_combos:
                    fallback_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                        delayed(StrategyOptimizer._test_single_combo)(
                            data, symbol, rsi_series, rsi_period, rsi_lower, rsi_upper,
                            direction, globalConfig.BACKTEST_INIT_CASH
                        )
                        for rsi_period, rsi_lower, rsi_upper, rsi_series in fallback_combos
                    )
                    assert isinstance(
                        fallback_results, list), "Parallel() returned non-list for fallback grid"
                    fallback_results = cast(
                        List[Tuple[BacktestResult, int, int, int]], fallback_results)

                    # Compute Z-scores within the fallback pool
                    fallback_zscores = zscore.compute_stage_zscores(
                        fallback_results)

                    for idx, (result, rp, rl, ru) in enumerate(fallback_results):
                        score = fallback_zscores[idx]
                        if score > best_score and result.profitable:
                            best_score = score
                            best_result = result
                            logger.debug(
                                "🎯 %s: New best — RSI(%d, %d, %d) Z-Score: %.2f",
                                symbol, rp, rl, ru, score
                            )

            if best_result:
                # Store the within-symbol Z-score on the winning result.
                # (This will be overwritten by cross-symbol Z-scores later.)
                best_result.composite_score = best_score
                logger.debug(
                    "✅ %s: Optimization complete — "
                    "Best: RSI(%d, %d, %d) Z-Score: %.2f "
                    "(α=%.2f%%, Sharpe=%.2f, Calmar=%.2f), Trades: %d (tested %d combos)",
                    symbol, best_result.rsi_period, best_result.rsi_lower,
                    best_result.rsi_upper, best_score,
                    best_result.alpha * 100, best_result.sharpe_ratio,
                    best_result.calmar_ratio,
                    best_result.num_trades, tested_combinations
                )
            else:
                logger.debug(
                    "❌ %s: No profitable strategies found from %d combinations",
                    symbol, tested_combinations
                )

            return best_result

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
            f"⚙️  RSI Parameters - Periods: {self.rsi_periods}, Lower: {self.rsi_lowers}, Upper: {self.rsi_uppers}")
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
        effective_workers = (
            os.cpu_count() or 4
        ) if globalConfig.N_JOBS == -1 else globalConfig.N_JOBS
        batch_size = 1 if effective_workers > 1 else 10
        logger.info(
            "⚡ Parallelism: %d joblib workers per symbol, batch_size=%d",
            effective_workers, batch_size
        )
        total_batches = (total_symbols + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(symbols), batch_size), 1):
            batch = symbols[i:i + batch_size]
            batch_start_time = time.time()

            logger.info(
                f"📊 Processing batch {batch_num}/{total_batches} ({len(batch)} symbols): {', '.join(batch)}")

            tasks = []
            for symbol in batch:
                task = loop.run_in_executor(
                    None,
                    self.optimize_symbol,
                    symbol,
                    start_date,
                    end_date,
                    "long"
                )
                tasks.append(task)
                if globalConfig.ENABLE_SHORT_SELLING:
                    short_task = loop.run_in_executor(
                        None,
                        self.optimize_symbol,
                        symbol,
                        start_date,
                        end_date,
                        "short"
                    )
                    tasks.append(short_task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Group results per symbol to count progress correctly.
            # Each symbol may have 1 (long-only) or 2 (long+short) tasks.
            results_per_symbol = len(tasks) // len(batch)
            batch_successful = 0
            for sym_idx, symbol in enumerate(batch):
                processed_count += 1
                progress.update(1, f"Batch {batch_num}/{total_batches}")

                # Collect all task results for this symbol
                sym_start = sym_idx * results_per_symbol
                sym_end = sym_start + results_per_symbol
                sym_results = batch_results[sym_start:sym_end]

                for result in sym_results:
                    if isinstance(result, BacktestResult):
                        results.append(result)
                        successful_count += 1
                        batch_successful += 1
                    elif result is not None:
                        logger.error(f"Error in batch processing: {result}")

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
        from strategy import RSIStrategy  # pylint: disable=import-outside-toplevel

        strategy = RSIStrategy(
            14, 30, 70)  # Dummy strategy instance for the method
        return strategy.build_consolidated_trades_df(results)

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
