"""
Walk-forward validation module.

Splits the backtest window into rolling in-sample (IS) and out-of-sample (OOS)
periods. For each window, optimizes RSI parameters on IS data then validates
on the subsequent OOS period. Aggregate OOS performance provides a less-biased
estimate of strategy quality than pure in-sample grid search.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Result from a single walk-forward window."""
    window_index: int
    is_start: datetime
    is_end: datetime
    oos_start: datetime
    oos_end: datetime
    # Best IS parameters
    best_period: int
    best_lower: int
    best_upper: int
    # IS metrics
    is_total_return: float
    is_sharpe_ratio: float
    is_num_trades: int
    # OOS metrics
    oos_total_return: float = 0.0
    oos_sharpe_ratio: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0
    oos_num_trades: int = 0
    oos_calmar_ratio: float = 0.0
    oos_profitable: bool = False
    # Status
    is_optimized: bool = False
    oos_validated: bool = False
    error: Optional[str] = None


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward validation result for one symbol × direction."""
    symbol: str
    direction: str
    windows: List[WalkForwardWindow] = field(default_factory=list)

    # Aggregate OOS metrics (across all windows with OOS validation)
    oos_total_return: float = 0.0
    oos_sharpe_ratio: float = 0.0
    oos_max_drawdown: float = 0.0
    oos_win_rate: float = 0.0
    oos_num_trades: int = 0
    oos_calmar_ratio: float = 0.0

    # Best parameters from most recent profitable OOS window (or IS fallback)
    best_rsi_period: Optional[int] = None
    best_rsi_lower: Optional[int] = None
    best_rsi_upper: Optional[int] = None

    # Parameter stability: fraction of windows with same (period, lower, upper)
    param_stability: float = 0.0

    # Cross-symbol Z-score (set later by zscore.compute_cross_symbol_zscores)
    composite_score: float = 0.0
    profitable: bool = False
    alpha: float = 0.0  # For backward compatibility with filter_results

    @property
    def num_windows(self) -> int:
        """Total number of windows (IS-optimized + OOS-validated)."""
        return len(self.windows)

    @property
    def num_profitable_oos_windows(self) -> int:
        """Number of windows with profitable OOS performance."""
        return sum(1 for w in self.windows if w.oos_profitable)

    def to_backtest_result(self):
        """Convert walk-forward result to BacktestResult for downstream consumers.

        Maps aggregate OOS metrics into the BacktestResult schema so that
        trading_engine.py and positions.py need no changes.
        """
        # Lazy import to avoid circular dependency at module level.
        from strategy import BacktestResult  # pylint: disable=import-outside-toplevel

        return BacktestResult(
            symbol=self.symbol,
            rsi_period=self.best_rsi_period or 14,
            rsi_lower=self.best_rsi_lower or 30,
            rsi_upper=self.best_rsi_upper or 70,
            total_return=self.oos_total_return,
            buy_and_hold_return=0.0,  # Not computed in walk-forward
            alpha=self.alpha,
            num_trades=self.oos_num_trades,
            win_rate=self.oos_win_rate,
            avg_trade_duration=0.0,  # Aggregate across windows
            max_drawdown=self.oos_max_drawdown,
            sharpe_ratio=self.oos_sharpe_ratio,
            calmar_ratio=self.oos_calmar_ratio,
            composite_score=self.composite_score,
            profitable=self.profitable,
            current_rsi=None,
            trade_details=None,
            direction=self.direction,
        )


class WalkForwardValidator:
    """Walk-forward validation orchestrator.

    Splits the full backtest window into rolling IS/OOS slices,
    optimizes parameters on each IS window, and evaluates on
    the subsequent OOS window. Aggregate OOS performance provides
    a less-biased estimate of strategy robustness.
    """

    def __init__(self, optimizer):
        """Initialize with a StrategyOptimizer instance for per-window IS grid search.

        Args:
            optimizer: StrategyOptimizer instance
        """
        self.optimizer = optimizer
        # One-shot sentinel: only log the detailed "why min_windows?" explanation
        # once per universe run, even if many symbols fall short.
        self._insufficient_windows_explained = False

    # ------------------------------------------------------------------
    # Window boundary computation
    # ------------------------------------------------------------------

    def _compute_window_boundaries(
        self, start_date: datetime, end_date: datetime
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Compute rolling (IS_start, IS_end, OOS_start, OOS_end) windows.

        Windows slide forward by step_months. Each IS window is is_months long.
        The OOS window immediately follows and is oos_months long.

        Returns empty list if not enough data for at least one full IS+OOS window.
        """
        is_delta = timedelta(days=globalConfig.WF_IS_MONTHS * 30)
        oos_delta = timedelta(days=globalConfig.WF_OOS_MONTHS * 30)
        step_delta = timedelta(days=globalConfig.WF_STEP_MONTHS * 30)

        windows: List[Tuple[datetime, datetime, datetime, datetime]] = []
        current_start = start_date

        while True:
            is_end = current_start + is_delta
            oos_start = is_end
            oos_end = oos_start + oos_delta

            # Stop if OOS end exceeds the available data end date
            if oos_end > end_date:
                break

            windows.append((current_start, is_end, oos_start, oos_end))
            current_start += step_delta

        return windows

    # ------------------------------------------------------------------
    # Single-symbol validation
    # ------------------------------------------------------------------

    def validate_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        direction: str = "long",
    ) -> Optional[WalkForwardResult]:
        """Run walk-forward validation for a single symbol and direction.

        Args:
            symbol: Stock symbol
            start_date: Start of the full backtest window
            end_date: End of the full backtest window
            direction: "long" or "short"

        Returns:
            WalkForwardResult with aggregate OOS metrics, or None if insufficient
            windows or all windows fail IS optimization.
        """
        try:
            # Compute window boundaries
            windows = self._compute_window_boundaries(start_date, end_date)

            if len(windows) < globalConfig.WF_MIN_WINDOWS:
                available_days = (end_date - start_date).days
                needed_days = (
                    globalConfig.WF_IS_MONTHS
                    + globalConfig.WF_OOS_MONTHS
                    + globalConfig.WF_STEP_MONTHS * (globalConfig.WF_MIN_WINDOWS - 1)
                ) * 30

                # Emit a detailed explanation once per universe run.
                if not self._insufficient_windows_explained:
                    self._insufficient_windows_explained = True
                    logger.warning(
                        "🪟 Walk-forward skipped: only %d window(s) computed "
                        "but min_windows=%d is required. "
                        "Available date span: %s → %s (%d days). "
                        "Each window needs IS=%dm + OOS=%dm; with step=%dm, "
                        "%d windows need roughly %d days of data. "
                        "To fix: either (a) increase backtesting.months in config, "
                        "(b) decrease walk_forward.is_months / oos_months / step_months, "
                        "or (c) lower walk_forward.min_windows (≥2 recommended).",
                        len(windows), globalConfig.WF_MIN_WINDOWS,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        available_days,
                        globalConfig.WF_IS_MONTHS, globalConfig.WF_OOS_MONTHS,
                        globalConfig.WF_STEP_MONTHS,
                        globalConfig.WF_MIN_WINDOWS, needed_days,
                    )
                    if windows:
                        for idx, (is_s, is_e, oos_s, oos_e) in enumerate(windows):
                            logger.warning(
                                "  Window %d: IS=[%s → %s]  OOS=[%s → %s]",
                                idx,
                                is_s.strftime('%Y-%m-%d'),
                                is_e.strftime('%Y-%m-%d'),
                                oos_s.strftime('%Y-%m-%d'),
                                oos_e.strftime('%Y-%m-%d'),
                            )
                        logger.warning(
                            "  Next window would need OOS ending %s, "
                            "but data ends %s.",
                            (
                                windows[-1][0]
                                + timedelta(days=globalConfig.WF_STEP_MONTHS * 30)
                                + timedelta(days=globalConfig.WF_IS_MONTHS * 30)
                                + timedelta(days=globalConfig.WF_OOS_MONTHS * 30)
                            ).strftime('%Y-%m-%d'),
                            end_date.strftime('%Y-%m-%d'),
                        )
                else:
                    logger.debug(
                        "⚠️  %s (%s): Only %d walk-forward window(s) available "
                        "(need ≥%d). Skipping walk-forward.",
                        symbol, direction, len(windows), globalConfig.WF_MIN_WINDOWS,
                    )
                return None

            logger.debug(
                "🔍 %s (%s): Walk-forward with %d windows "
                "(IS=%dm, OOS=%dm, step=%dm)",
                symbol, direction, len(windows),
                globalConfig.WF_IS_MONTHS, globalConfig.WF_OOS_MONTHS,
                globalConfig.WF_STEP_MONTHS,
            )

            wf_windows: List[WalkForwardWindow] = []

            for idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
                wf_win = WalkForwardWindow(
                    window_index=idx,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    best_period=14,
                    best_lower=30,
                    best_upper=70,
                    is_total_return=0.0,
                    is_sharpe_ratio=0.0,
                    is_num_trades=0,
                )

                # --- Step A: Optimize on IS window ---
                try:
                    is_result = self.optimizer.optimize_symbol(
                        symbol, is_start, is_end, direction
                    )
                except Exception as e:
                    wf_win.error = f"IS optimization failed: {e}"
                    wf_windows.append(wf_win)
                    logger.warning(
                        "⚠️  %s (%s) window %d: IS optimization error: %s",
                        symbol, direction, idx, e,
                    )
                    continue

                if is_result is None:
                    wf_win.error = "IS optimization returned None"
                    wf_windows.append(wf_win)
                    logger.debug(
                        "⚠️  %s (%s) window %d: No profitable IS strategy found",
                        symbol, direction, idx,
                    )
                    continue

                wf_win.is_optimized = True
                wf_win.best_period = is_result.rsi_period
                wf_win.best_lower = is_result.rsi_lower
                wf_win.best_upper = is_result.rsi_upper
                wf_win.is_total_return = is_result.total_return
                wf_win.is_sharpe_ratio = is_result.sharpe_ratio
                wf_win.is_num_trades = is_result.num_trades

                # --- Step B: Validate on OOS window ---
                try:
                    oos_result = self._run_oos_backtest(
                        symbol, oos_start, oos_end, is_result, direction
                    )
                except Exception as e:
                    wf_win.error = f"OOS validation failed: {e}"
                    wf_windows.append(wf_win)
                    logger.warning(
                        "⚠️  %s (%s) window %d: OOS validation error: %s",
                        symbol, direction, idx, e,
                    )
                    continue

                if oos_result is not None:
                    wf_win.oos_validated = True
                    wf_win.oos_total_return = oos_result.total_return
                    wf_win.oos_sharpe_ratio = oos_result.sharpe_ratio
                    wf_win.oos_max_drawdown = oos_result.max_drawdown
                    wf_win.oos_win_rate = oos_result.win_rate
                    wf_win.oos_num_trades = oos_result.num_trades
                    wf_win.oos_calmar_ratio = oos_result.calmar_ratio
                    wf_win.oos_profitable = oos_result.profitable

                wf_windows.append(wf_win)

                logger.debug(
                    "✅ %s (%s) window %d: IS=%s→%s OOS=%s→%s "
                    "params=(%d,%d,%d) OOS_ret=%.2f%% OOS_sharpe=%.2f",
                    symbol, direction, idx,
                    is_start.strftime('%Y-%m-%d'), is_end.strftime('%Y-%m-%d'),
                    oos_start.strftime(
                        '%Y-%m-%d'), oos_end.strftime('%Y-%m-%d'),
                    wf_win.best_period, wf_win.best_lower, wf_win.best_upper,
                    wf_win.oos_total_return * 100, wf_win.oos_sharpe_ratio,
                )

            if not wf_windows:
                return None

            # --- Aggregate results ---
            result = self._aggregate_windows(symbol, direction, wf_windows)
            return result

        except Exception as e:
            logger.error("💥 Error in walk-forward for %s (%s): %s",
                         symbol, direction, e)
            return None

    def _run_oos_backtest(
        self,
        symbol: str,
        oos_start: datetime,
        oos_end: datetime,
        is_result,
        direction: str,
    ):
        """Run a backtest on OOS data using IS-optimized parameters.

        Args:
            symbol: Stock symbol
            oos_start: OOS window start date
            oos_end: OOS window end date
            is_result: BacktestResult from IS optimization (carries best params)
            direction: "long" or "short"

        Returns:
            BacktestResult from OOS evaluation, or None if insufficient data
        """
        from strategy import RSIStrategy  # pylint: disable=import-outside-toplevel
        from data_provider import data_provider  # pylint: disable=import-outside-toplevel,reimported

        # Fetch OOS data with warmup for RSI calculation
        warmup_days = is_result.rsi_period * 2
        warmup_start = oos_start - timedelta(days=warmup_days)
        oos_data = data_provider.get_single_stock_bars(
            symbol, warmup_start, oos_end)

        if oos_data.empty or len(oos_data) < is_result.rsi_period + 10:
            logger.debug(
                "⚠️  %s: Insufficient OOS data (%d rows) for RSI(%d)",
                symbol, len(oos_data), is_result.rsi_period,
            )
            return None

        strategy = RSIStrategy(
            rsi_period=is_result.rsi_period,
            rsi_lower=is_result.rsi_lower,
            rsi_upper=is_result.rsi_upper,
            direction=direction,
        )
        return strategy.backtest(oos_data, symbol, globalConfig.BACKTEST_INIT_CASH)

    def _aggregate_windows(
        self,
        symbol: str,
        direction: str,
        wf_windows: List[WalkForwardWindow],
    ) -> WalkForwardResult:
        """Aggregate per-window OOS metrics into a single WalkForwardResult.

        Selects best parameters from the most recent profitable OOS window.
        Falls back to most recent IS-optimized window if no OOS-profitable windows.
        Computes aggregate OOS metrics across all validated windows.
        """
        result = WalkForwardResult(
            symbol=symbol, direction=direction, windows=wf_windows)

        # Validated windows = those that succeeded in both IS and OOS
        validated = [w for w in wf_windows if w.oos_validated]

        if validated:
            # Aggregate OOS metrics: equal-weight average of per-window returns
            result.oos_total_return = float(
                np.mean([w.oos_total_return for w in validated]))
            result.oos_sharpe_ratio = float(
                np.mean([w.oos_sharpe_ratio for w in validated]))
            result.oos_max_drawdown = float(
                np.max([w.oos_max_drawdown for w in validated]))
            result.oos_win_rate = float(
                np.mean([w.oos_win_rate for w in validated]))
            result.oos_num_trades = sum(w.oos_num_trades for w in validated)
            result.oos_calmar_ratio = float(
                np.mean([w.oos_calmar_ratio for w in validated]))

            # Alpha: OOS return minus zero (we don't have buy-and-hold per-window)
            result.alpha = result.oos_total_return

            # Select best parameters: most recent OOS-profitable window
            profitable_validated = [w for w in validated if w.oos_profitable]
            if profitable_validated:
                chosen = profitable_validated[-1]  # Most recent profitable
            else:
                chosen = validated[-1]  # Most recent validated (fallback)

            result.best_rsi_period = chosen.best_period
            result.best_rsi_lower = chosen.best_lower
            result.best_rsi_upper = chosen.best_upper
            result.profitable = result.oos_total_return > 0

        else:
            # No OOS-validated windows — fall back to most recent IS optimization
            optimized = [w for w in wf_windows if w.is_optimized]
            if optimized:
                chosen = optimized[-1]
                result.best_rsi_period = chosen.best_period
                result.best_rsi_lower = chosen.best_lower
                result.best_rsi_upper = chosen.best_upper
                result.oos_total_return = chosen.is_total_return
                result.oos_sharpe_ratio = chosen.is_sharpe_ratio
                result.oos_num_trades = chosen.is_num_trades
                result.alpha = chosen.is_total_return
                result.profitable = chosen.is_total_return > 0
            else:
                # No successful windows at all
                result.profitable = False

        # Parameter stability: fraction of IS-optimized windows with same params
        optimized = [w for w in wf_windows if w.is_optimized]
        if optimized:
            param_counts: Dict[Tuple[int, int, int], int] = {}
            for w in optimized:
                key = (w.best_period, w.best_lower, w.best_upper)
                param_counts[key] = param_counts.get(key, 0) + 1
            max_count = max(param_counts.values())
            result.param_stability = max_count / len(optimized)

        logger.debug(
            "📊 %s (%s) Walk-Forward Aggregate: "
            "windows=%d validated=%d profitable_oos=%d "
            "OOS_ret=%.2f%% OOS_sharpe=%.2f stability=%.0f%%",
            symbol, direction,
            len(wf_windows), len(validated), result.num_profitable_oos_windows,
            result.oos_total_return * 100, result.oos_sharpe_ratio,
            result.param_stability * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Universe-level orchestration
    # ------------------------------------------------------------------

    async def validate_universe(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> List[WalkForwardResult]:
        """Run walk-forward validation for all symbols concurrently.

        Mirrors StrategyOptimizer.optimize_universe() in structure:
        same batching, same progress reporting, same direction support.

        Args:
            symbols: List of stock symbols
            start_date: Start of the full backtest window
            end_date: End of the full backtest window

        Returns:
            List of WalkForwardResult objects (one per symbol × direction)
        """
        results: List[WalkForwardResult] = []
        total_symbols = len(symbols)
        processed_count = 0
        successful_count = 0

        # Reset per-run sentinel so the detailed explanation fires once
        # for this universe run (even if a prior run already emitted it).
        self._insufficient_windows_explained = False

        start_time = time.time()

        logger.info(
            "🚀 Starting walk-forward validation for %d symbols", total_symbols,
        )
        logger.info(
            "📅 Date range: %s to %s",
            start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'),
        )
        logger.info(
            "🪟 Windows: IS=%dm OOS=%dm step=%dm (≥%d required)",
            globalConfig.WF_IS_MONTHS, globalConfig.WF_OOS_MONTHS,
            globalConfig.WF_STEP_MONTHS, globalConfig.WF_MIN_WINDOWS,
        )
        logger.info("=" * 60)

        # Import here to avoid circular dependency at module level.
        from utils import ProgressIndicator  # pylint: disable=import-outside-toplevel

        progress = ProgressIndicator(
            total_symbols, "🔍 Walk-forward validation")

        loop = asyncio.get_event_loop()

        # Same batching as StrategyOptimizer.optimize_universe
        effective_workers = (
            os.cpu_count() or 4
        ) if globalConfig.N_JOBS == -1 else globalConfig.N_JOBS
        batch_size = 1 if effective_workers > 1 else 10
        total_batches = (total_symbols + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(symbols), batch_size), 1):
            batch = symbols[i:i + batch_size]
            batch_start_time = time.time()

            logger.info(
                "📊 Processing batch %d/%d (%d symbols): %s",
                batch_num, total_batches, len(batch), ', '.join(batch),
            )

            tasks = []
            for symbol in batch:
                task = loop.run_in_executor(
                    None,
                    self.validate_symbol,
                    symbol,
                    start_date,
                    end_date,
                    "long",
                )
                tasks.append(task)
                if globalConfig.ENABLE_SHORT_SELLING:
                    short_task = loop.run_in_executor(
                        None,
                        self.validate_symbol,
                        symbol,
                        start_date,
                        end_date,
                        "short",
                    )
                    tasks.append(short_task)

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            results_per_symbol = len(tasks) // len(batch)
            batch_successful = 0
            for sym_idx, symbol in enumerate(batch):
                processed_count += 1
                progress.update(1, f"Batch {batch_num}/{total_batches}")

                sym_start = sym_idx * results_per_symbol
                sym_end = sym_start + results_per_symbol
                sym_results = batch_results[sym_start:sym_end]

                for result in sym_results:
                    if isinstance(result, WalkForwardResult):
                        results.append(result)
                        successful_count += 1
                        batch_successful += 1
                    elif result is not None:
                        logger.error("Error in batch processing: %s", result)

            batch_time = time.time() - batch_start_time
            total_elapsed = time.time() - start_time

            if processed_count > 0:
                avg_time = total_elapsed / processed_count
                remaining = total_symbols - processed_count
                eta = avg_time * remaining

                logger.info(
                    "✅ Batch %d complete: %d/%d successful (%.1fs) | "
                    "Progress: %d/%d | ETA: %.1f min",
                    batch_num, batch_successful, len(batch), batch_time,
                    processed_count, total_symbols, eta / 60,
                )

            await asyncio.sleep(1)

        progress.finish("Walk-forward validation complete!")

        total_time = time.time() - start_time
        success_rate = (successful_count / total_symbols) * \
            100 if total_symbols > 0 else 0

        logger.info("=" * 60)
        logger.info("🎯 WALK-FORWARD VALIDATION COMPLETE!")
        logger.info("📊 Results Summary:")
        logger.info("   • Symbols processed: %d/%d",
                    processed_count, total_symbols)
        logger.info("   • Successful: %d (%.1f%%)",
                    successful_count, success_rate)
        logger.info("   • Total time: %.1f min", total_time / 60)

        if results:
            profitable_count = len([r for r in results if r.profitable])
            logger.info("   • Profitable (OOS): %d/%d",
                        profitable_count, len(results))

            # Compute cross-symbol Z-scores on aggregate OOS metrics
            logger.info("📊 Computing cross-symbol Z-scores...")
            self._compute_wf_cross_symbol_zscores(results)

        logger.info("=" * 60)

        return results

    # ------------------------------------------------------------------
    # Cross-symbol Z-score computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_wf_cross_symbol_zscores(results: List[WalkForwardResult]) -> None:
        """Compute cross-symbol Z-scores on walk-forward aggregate OOS metrics.

        Mutates each result's composite_score in place. Uses the same approach
        as zscore.compute_cross_symbol_zscores but operates on WalkForwardResult
        fields (oos_total_return, oos_sharpe_ratio, oos_calmar_ratio).
        """
        if len(results) < 2:
            # Single result: set neutral score
            for r in results:
                r.composite_score = 0.0
            return

        alphas = np.array([r.oos_total_return for r in results])
        sharpes = np.array([r.oos_sharpe_ratio for r in results])
        calmars = np.array([min(r.oos_calmar_ratio, 10.0) for r in results])

        alpha_mean, alpha_std = alphas.mean(), alphas.std()
        sharpe_mean, sharpe_std = sharpes.mean(), sharpes.std()
        calmar_mean, calmar_std = calmars.mean(), calmars.std()

        for i, r in enumerate(results):
            alpha_z = (alphas[i] - alpha_mean) / \
                alpha_std if alpha_std > 0 else 0
            sharpe_z = (sharpes[i] - sharpe_mean) / \
                sharpe_std if sharpe_std > 0 else 0
            calmar_z = (calmars[i] - calmar_mean) / \
                calmar_std if calmar_std > 0 else 0
            r.composite_score = float(alpha_z + sharpe_z + calmar_z)

        logger.debug(
            "WF Z-score pool stats: α(μ=%.4f,σ=%.4f) sharpe(μ=%.4f,σ=%.4f) calmar(μ=%.4f,σ=%.4f)",
            alpha_mean, alpha_std, sharpe_mean, sharpe_std, calmar_mean, calmar_std,
        )
