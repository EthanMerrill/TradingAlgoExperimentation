"""
RSI mean-reversion strategy.

Moved from ``strategy.py`` in Phase B (multi-strategy framework). Implements
the ``Strategy`` interface; the legacy module path
``from strategies.rsi import RSIStrategy``.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import pytz
from indicators import TechnicalIndicators
from joblib import Parallel, delayed

from config import globalConfig  # type: ignore
from strategies.base import BacktestResult, Strategy
from utils import PerformanceMetrics

logger = logging.getLogger(__name__)


class RSIStrategy(Strategy):
    """Vectorized RSI trading strategy."""

    # Registry key + interface metadata.
    name = "rsi_mean_reversion"
    execution_style = "session"
    bar_size = None

    @classmethod
    def create(cls) -> "Strategy":
        """Default RSI instance (config-derived ranges come from the optimizer grid)."""
        return cls(14, 30, 70)

    def __init__(self, rsi_period: int, rsi_lower: int, rsi_upper: int, max_hold_days: Optional[int] = None, direction: str = "long"):
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.max_hold_days = max_hold_days or globalConfig.MAX_HOLD_DAYS
        self.direction = direction  # "long" or "short"

    @staticmethod
    def calculate_price_for_target_rsi(data: pd.DataFrame, target_rsi: float, rsi_period: int = 14) -> Optional[float]:
        """
        Calculate the price needed to achieve a target RSI value.

        Args:
            data: DataFrame with OHLCV data (needs at least rsi_period + 1 rows)
            target_rsi: Desired RSI value (0-100)
            rsi_period: RSI calculation period

        Returns:
            Price needed to achieve target RSI, or None if calculation fails
        """
        try:
            if len(data) < rsi_period + 1:
                logger.warning(
                    f"Insufficient data for RSI calculation. Need {rsi_period + 1} rows, got {len(data)}")
                return None

            if not (0 <= target_rsi <= 100):
                logger.warning(
                    f"Invalid RSI target: {target_rsi}. Must be between 0 and 100")
                return None

            # Get the most recent prices (excluding the current/future price we're solving for)
            # Last rsi_period closing prices - handle both 'c' and 'close' column names
            price_col = RSIStrategy._get_price_column(data)
            prices = data[price_col].iloc[-(rsi_period):].values

            if len(prices) < rsi_period:
                return None

            # Calculate price changes for the historical period
            price_changes = np.diff(np.array(prices))

            # Separate gains and losses
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            # Calculate current average gain and loss
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0

            # Avoid division by zero
            if avg_loss == 0:
                if target_rsi < 100:
                    # If no losses and target RSI < 100, we need a loss
                    # Return a price lower than current to create a loss
                    return prices[-1] * 0.99
                else:
                    # If no losses and target RSI = 100, any gain works
                    return prices[-1] * 1.01

            # RSI formula: RSI = 100 - (100 / (1 + RS))
            # Where RS = Average Gain / Average Loss
            # Solving for RS: RS = (100 - target_rsi) / target_rsi * 100
            # We need: new_avg_gain / new_avg_loss = rs_target
            # With Wilder's smoothing: new_avg_gain = (old_avg_gain * (period-1) + new_gain) / period
            # Similar for losses

            current_price = prices[-1]

            # Binary search for the target price
            min_price = current_price * 0.5  # 50% down
            max_price = current_price * 1.5  # 50% up
            tolerance = 0.01  # RSI tolerance
            max_iterations = 100

            for _ in range(max_iterations):
                test_price = (min_price + max_price) / 2
                price_change = test_price - current_price

                # Calculate new gains and losses with the test price
                if price_change > 0:
                    new_gain = price_change
                    new_loss = 0
                else:
                    new_gain = 0
                    new_loss = -price_change

                # Calculate new averages using Wilder's smoothing
                new_avg_gain = (avg_gain * (rsi_period - 1) +
                                new_gain) / rsi_period
                new_avg_loss = (avg_loss * (rsi_period - 1) +
                                new_loss) / rsi_period

                if new_avg_loss == 0:
                    calculated_rsi = 100
                else:
                    rs = new_avg_gain / new_avg_loss
                    calculated_rsi = 100 - (100 / (1 + rs))

                # Check if we're close enough
                if abs(calculated_rsi - target_rsi) < tolerance:
                    return test_price

                # Adjust search range
                if calculated_rsi < target_rsi:
                    min_price = test_price
                else:
                    max_price = test_price

            # Return best approximation
            return (min_price + max_price) / 2

        except Exception as e:
            logger.error("Error calculating price for target RSI: %s", e)
            return None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def backtest(self, data: pd.DataFrame, symbol: str, initial_cash: float = 10000,
                 prepared: Any = None, close_open_at_end: bool = False,
                 **params: Any) -> BacktestResult:
        """Run a vectorized RSI backtest.

        Backward-compatible with the legacy 3-arg call. When ``params`` are
        provided (optimizer grid search), a strategy is built from them and the
        backtest is delegated. ``prepared`` is an RSI cache keyed by period
        (see :meth:`prepare`), used to avoid recomputing RSI per grid combo.
        ``close_open_at_end`` (opt-in, default False) force-closes a position
        still open on the final bar at that bar's close (see ``_simulate_trades``).
        """
        if params:
            merged = {
                "rsi_period": self.rsi_period,
                "rsi_lower": self.rsi_lower,
                "rsi_upper": self.rsi_upper,
                "max_hold_days": self.max_hold_days,
                "direction": self.direction,
            }
            merged.update(params)
            strat = RSIStrategy(
                merged["rsi_period"], merged["rsi_lower"], merged["rsi_upper"],
                max_hold_days=merged["max_hold_days"], direction=merged["direction"],
            )
            return strat.backtest(data, symbol, initial_cash, prepared=prepared,
                                  close_open_at_end=close_open_at_end)
        if prepared is not None:
            rsi = prepared.get(self.rsi_period)
            if rsi is not None and len(rsi) == len(data):
                return self.backtest_with_rsi(data, symbol, rsi, initial_cash,
                                              close_open_at_end=close_open_at_end)
        return self._backtest_impl(data, symbol, initial_cash,
                                   close_open_at_end=close_open_at_end)

    def _backtest_impl(self, data: pd.DataFrame, symbol: str, initial_cash: float = 10000,
                       close_open_at_end: bool = False) -> BacktestResult:
        """Core backtest entry point (legacy behavior, no params/prepared)."""
        try:
            if len(data) < self.rsi_period + 10:
                return self._create_null_result(symbol or "UNKNOWN")

            # Determine symbol name - prioritize passed parameter, then try to extract from data
            if symbol is None:
                if 'symbol' in data.columns:
                    symbol = str(data['symbol'].iloc[0])
                elif hasattr(data, 'symbol'):
                    symbol = str(data.symbol)
                else:
                    symbol = "UNKNOWN"

            # Clean symbol name to ensure it's safe for filenames
            symbol = str(symbol).strip().replace('\n', '').replace('\r', '')
            if not symbol or symbol.isspace():
                symbol = "UNKNOWN"

            # Calculate RSI
            price_col = self._get_price_column(data)
            rsi = TechnicalIndicators.calculate_rsi(
                data, self.rsi_period, price_col)

            return self._run_backtest_core(data, symbol, rsi, initial_cash,
                                           close_open_at_end=close_open_at_end)

        except Exception as e:
            logger.error("Error in backtest: %s", e)
            return self._create_null_result("ERROR")

    def backtest_with_rsi(self, data: pd.DataFrame, symbol: str, rsi: pd.Series, initial_cash: float = 10000,
                          close_open_at_end: bool = False) -> BacktestResult:
        """
        Run backtest using a precomputed RSI series (avoids redundant recalculation).

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol (required — caller must provide it)
            rsi: Precomputed RSI series matching the strategy's rsi_period
            initial_cash: Starting cash amount
            close_open_at_end: force-close a position open on the final bar at its close

        Returns:
            BacktestResult object with performance metrics
        """
        try:
            if len(data) < self.rsi_period + 10:
                return self._create_null_result(symbol)
            return self._run_backtest_core(data, symbol, rsi, initial_cash,
                                           close_open_at_end=close_open_at_end)
        except Exception as e:
            logger.error("Error in backtest_with_rsi: %s", e)
            return self._create_null_result("ERROR")

    def get_param_grid(self, direction: str = "long") -> List[Dict[str, Any]]:
        """Full (period, lower, upper) grid from the configured RSI ranges."""
        grid: List[Dict[str, Any]] = []
        for period in list(range(*globalConfig.RSI_PERIOD_RANGE)):
            for lower in list(range(*globalConfig.RSI_LOWER_RANGE)):
                for upper in list(range(*globalConfig.RSI_UPPER_RANGE)):
                    if lower >= upper:
                        continue
                    grid.append({
                        "rsi_period": period,
                        "rsi_lower": lower,
                        "rsi_upper": upper,
                        "direction": direction,
                    })
        return grid

    def prepare(self, data: pd.DataFrame) -> Dict[int, pd.Series]:
        """Precompute the RSI series once per period (Tier-1 grid cache)."""
        price_col = self._get_price_column(data)
        cache: Dict[int, pd.Series] = {}
        for period in list(range(*globalConfig.RSI_PERIOD_RANGE)):
            cache[period] = TechnicalIndicators.calculate_rsi(
                data, period, price_col
            )
        return cache

    def warmup_days(self) -> int:
        """Calendar-day warmup for the largest RSI period in the grid."""
        periods = list(range(*globalConfig.RSI_PERIOD_RANGE))
        max_period = max(periods) if periods else 14
        return max_period * 2

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
    ) -> Tuple[BacktestResult, int, int, int]:
        """Run a single backtest for one parameter combination.

        Extracted as a static method so it's trivially shareable across
        threads/processes — no mutable instance state, all inputs are
        read-only primitives or numpy/pandas objects.

        Returns:
            Tuple of (BacktestResult, rsi_period, rsi_lower, rsi_upper)
        """
        strategy = RSIStrategy(
            rsi_period, rsi_lower, rsi_upper, direction=direction
        )
        result = strategy.backtest_with_rsi(
            data, symbol, rsi_series, initial_cash
        )
        return result, rsi_period, rsi_lower, rsi_upper

    def optimize(self, data: pd.DataFrame, symbol: str, direction: str = "long",
                 initial_cash: float = 10000) -> Optional[BacktestResult]:
        """Grid-search RSI params on ``data`` (two-stage coarse+fine when enabled).

        Ported from the former ``StrategyOptimizer.optimize_symbol`` body so the
        search behavior is identical to the pre-framework implementation:
        Tier-1 precomputes the RSI series once per period; the grid is then run
        in parallel and z-scored per stage.
        """
        # Lazy import to avoid circular dependency at module level.
        import zscore  # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported

        rsi_periods = list(range(*globalConfig.RSI_PERIOD_RANGE))
        rsi_lowers = list(range(*globalConfig.RSI_LOWER_RANGE))
        rsi_uppers = list(range(*globalConfig.RSI_UPPER_RANGE))

        # --- Tier 1: Precompute RSI for every unique period once ---
        # RSI depends only on rsi_period, not lower/upper, so this eliminates
        # ~96% of redundant RSI calculations (e.g., 28 recomputations → 1 per period).
        price_col = self._get_price_column(data)
        rsi_cache: Dict[int, pd.Series] = {}
        for period in rsi_periods:
            rsi_cache[period] = TechnicalIndicators.calculate_rsi(
                data, period, price_col
            )

        best_result: Optional[BacktestResult] = None
        best_score = -float('inf')
        tested_combinations = 0
        tested_set: set = set()  # dedup between stages
        n_jobs = globalConfig.N_JOBS

        # Determine whether two-stage (coarse + fine) optimization is enabled.
        # Gated behind the fine_tuning_enabled feature flag and requires at least
        # 4 values in both lower and upper ranges for the coarse grid to be meaningful.
        fine_step_lower = (
            rsi_lowers[1] - rsi_lowers[0]
            if len(rsi_lowers) >= 2 else 1
        )
        fine_step_upper = (
            rsi_uppers[1] - rsi_uppers[0]
            if len(rsi_uppers) >= 2 else 1
        )
        use_two_stage = (
            globalConfig.RSI_FINE_TUNING_ENABLED
            and len(rsi_lowers) >= 4
            and len(rsi_uppers) >= 4
            and fine_step_lower > 0
            and fine_step_upper > 0
        )

        if use_two_stage:
            # --- Tier 2: Stage 1 — Coarse grid (double step) ---
            coarse_step_lower = fine_step_lower * 2
            coarse_step_upper = fine_step_upper * 2
            coarse_lowers = list(range(
                rsi_lowers[0], rsi_lowers[-1] +
                1, coarse_step_lower
            ))
            coarse_uppers = list(range(
                rsi_uppers[0], rsi_uppers[-1] +
                1, coarse_step_upper
            ))

            coarse_candidates: List[Tuple[float, int, int, int]] = []

            # Build flat list of combos for parallel dispatch
            coarse_combos: List[Tuple[int, int, int, pd.Series]] = []
            for rsi_period in rsi_periods:
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
                len(rsi_periods), coarse_count
            )

            # Run coarse grid in parallel
            parallel_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                delayed(RSIStrategy._test_single_combo)(
                    data, symbol, rsi_series, rsi_period, rsi_lower, rsi_upper,
                    direction, initial_cash
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
                        if x >= rsi_lowers[0] and x < rsi_uppers[-1]
                    ]
                    fine_uppers = [
                        x for x in range(
                            c_upper - fine_step_upper,
                            c_upper + fine_step_upper + 1,
                            fine_step_upper
                        )
                        if x > rsi_lowers[0] and x <= rsi_uppers[-1]
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
                        delayed(RSIStrategy._test_single_combo)(
                            data, symbol, rsi_series, rsi_period, rsi_lower, rsi_upper,
                            direction, initial_cash
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
            for rsi_period in rsi_periods:
                rsi = rsi_cache[rsi_period]
                for rsi_lower in rsi_lowers:
                    for rsi_upper in rsi_uppers:
                        if rsi_lower >= rsi_upper:
                            continue
                        fallback_combos.append(
                            (rsi_period, rsi_lower, rsi_upper, rsi))

            tested_combinations = len(fallback_combos)

            if fallback_combos:
                fallback_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
                    delayed(RSIStrategy._test_single_combo)(
                        data, symbol, rsi_series, rsi_period, rsi_lower, rsi_upper,
                        direction, initial_cash
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
                symbol, best_result.params.get("rsi_period", 0),
                best_result.params.get("rsi_lower", 0),
                best_result.params.get("rsi_upper", 0), best_score,
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

    def _run_backtest_core(self, data: pd.DataFrame, symbol: str, rsi: pd.Series, initial_cash: float,
                           close_open_at_end: bool = False) -> BacktestResult:
        """Core backtest logic shared by backtest() and backtest_with_rsi().

        One pass (``_simulate_trades``) produces the trade list and the equity
        curve from the same fills, so every metric below is consistent.
        """
        current_rsi = rsi.iloc[-1] if not rsi.empty else None

        trades, portfolio_values = self._simulate_trades(
            data, rsi, initial_cash, close_open_at_end=close_open_at_end)
        portfolio = pd.Series(portfolio_values, index=data.index)
        daily_returns = portfolio.pct_change().fillna(0)

        price_col = self._get_price_column(data)
        buy_and_hold_return = (
            data[price_col].iloc[-1] / data[price_col].iloc[0]) - 1

        total_return = portfolio_values[-1] / initial_cash - 1
        alpha = total_return - buy_and_hold_return

        trades_summary = self._summarize_trades(trades)
        sharpe = self._calculate_sharpe_ratio(daily_returns)
        calmar = PerformanceMetrics.calculate_calmar_ratio(
            daily_returns, portfolio)

        return BacktestResult(
            symbol=symbol,
            total_return=total_return,
            buy_and_hold_return=buy_and_hold_return,
            alpha=alpha,
            num_trades=trades_summary['num_trades'],
            win_rate=trades_summary['win_rate'],
            avg_trade_duration=trades_summary['avg_duration'],
            max_drawdown=self._calculate_max_drawdown(portfolio),
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            profitable=total_return > 0,
            trade_details=trades,
            direction=self.direction,
            strategy_name=self.name,
            params={
                "rsi_period": self.rsi_period,
                "rsi_lower": self.rsi_lower,
                "rsi_upper": self.rsi_upper,
                "current_rsi": current_rsi,
                "direction": self.direction,
                "max_hold_days": self.max_hold_days,
            }
        )

    @staticmethod
    def _ohlc_arrays(data: pd.DataFrame, price_col: str):
        """(open, high, low, close) as float lists; missing columns fall back to close."""
        close = data[price_col].values.astype(np.float64)

        def col(*names):
            for nm in names:
                if nm in data.columns:
                    return data[nm].values.astype(np.float64)
            return close
        return (col('open', 'o').tolist(), col('high', 'h').tolist(),
                col('low', 'l').tolist(), close.tolist())

    def _simulate_trades(self, data: pd.DataFrame, rsi: pd.Series,
                         initial_cash: float,
                         close_open_at_end: bool = False) -> Tuple[List[Dict], np.ndarray]:
        """Single O(n) pass producing the trade list and the equity curve.

        Execution model (mirrors live trading):
          * Entry: RSI cross on bar i close -> market fill at bar i+1 OPEN.
          * Protective exits (stop = fill*(1-STOP_LOSS_PCT) long / (1+) short,
            take-profit = RSI-implied price, else fill*(1+/-TAKE_PROFIT_PCT))
            are evaluated intraday with bar high/low, including the entry bar
            itself (protection is armed right after the open fill, so the
            whole entry bar's range after the open counts):
              long  stop: open<=stop -> fill open (gap); elif low<=stop -> fill stop
              long  tp  : open>=tp   -> fill open (gap); elif high>=tp -> fill tp
              short mirrored. If stop and TP are both touched in one bar and
              neither gapped, the stop is assumed to fill first (conservative).
          * RSI-cross and max-hold exits are decided on a bar close and fill
            at the next bar's OPEN.
          * Every fill is moved adversely by BACKTEST_SLIPPAGE_BPS; per-share
            commission applies to each fill and the SEC fee to each sell.
          * Returns are per unit of entry notional with 100% of equity per
            trade, so equity compounds exactly the net trade returns. Equity
            is marked on closes while in a position. By default a position still
            open on the final bar is marked to market but not reported as a
            trade. With ``close_open_at_end=True`` it is instead closed at the
            final bar's CLOSE (same adverse slippage/costs, exit_reason
            'window_end'), reported as a trade, and the final equity value
            reflects the realised return.
        """
        n = len(data)
        price_col = self._get_price_column(data)
        opens, highs, lows, closes = self._ohlc_arrays(data, price_col)

        r = rsi.values.astype(np.float64)
        prev = np.concatenate(([np.nan], r[:-1]))
        with np.errstate(invalid='ignore'):
            if self.direction == "long":
                buy = ((r < self.rsi_lower) & (prev >= self.rsi_lower)).tolist()
                sell = ((r > self.rsi_upper) & (prev <= self.rsi_upper)).tolist()
            else:
                buy = ((r > self.rsi_upper) & (prev <= self.rsi_upper)).tolist()
                sell = ((r < self.rsi_lower) & (prev >= self.rsi_lower)).tolist()

        idx = data.index
        if isinstance(idx, pd.DatetimeIndex):
            days = ((idx - idx[0]) / pd.Timedelta(days=1)).tolist()
        else:
            days = list(range(n))

        is_long = self.direction == "long"
        stop_pct = globalConfig.STOP_LOSS_PCT
        tp_pct = globalConfig.TAKE_PROFIT_PCT
        max_hold = self.max_hold_days
        slip = getattr(globalConfig, 'BACKTEST_SLIPPAGE_BPS', 5) / 1e4
        comm = getattr(globalConfig, 'BACKTEST_COMMISSION_PER_SHARE', 0.0)
        sec = getattr(globalConfig, 'BACKTEST_SEC_FEE_BPS', 0.0278) / 1e4
        rsi_target = self.rsi_upper if is_long else self.rsi_lower
        floor = math.floor

        equity = [float(initial_cash)] * n
        cash = float(initial_cash)
        trades: List[Dict] = []

        pos = False
        pend_entry = -1          # bar index where a pending entry fills
        pend_exit = None         # reason for an exit pending at next open
        e_j = 0
        e_px = stop = tp = 0.0   # entry fill (post-slip), protective levels
        tp_reason = 'take_profit'
        base = cash

        def net_return(entry_px, exit_px):
            if is_long:
                return ((exit_px * (1 - sec) - comm) - (entry_px + comm)) / entry_px
            return ((entry_px * (1 - sec) - comm) - (exit_px + comm)) / entry_px

        def record(j, reason, exit_raw):
            nonlocal cash, pos
            x_px = exit_raw * (1 - slip) if is_long else exit_raw * (1 + slip)
            ret = net_return(e_px, x_px)
            cash = base * (1 + ret)
            equity[j] = cash
            if isinstance(idx, pd.DatetimeIndex):
                duration = (idx[j] - idx[e_j]).days
            else:
                duration = j - e_j
            trades.append({
                'entry_date': idx[e_j],
                'exit_date': idx[j],
                'entry_price': e_px,
                'exit_price': x_px,
                'return': ret,
                'return_pct': ret * 100.0,
                'duration': duration,
                'exit_reason': reason,
                'direction': self.direction,
                'side': self.direction,
            })
            pos = False

        for j in range(n):
            exited = False
            exit_raw = 0.0
            reason = None

            if pend_exit is not None:
                exit_raw, reason, pend_exit = opens[j], pend_exit, None
                exited = True

            if pend_entry == j and not pos:
                pend_entry = -1
                o = opens[j]
                e_px = o * (1 + slip) if is_long else o * (1 - slip)
                e_j = j
                base = cash
                pos = True
                if is_long:
                    stop = e_px * (1 - stop_pct)
                else:
                    stop = e_px * (1 + stop_pct)
                tp = self._target_price(data, signal_i=j - 1, target_rsi=rsi_target)
                if tp is None:
                    tp = e_px * (1 + tp_pct) if is_long else e_px * (1 - tp_pct)
                    tp_reason = 'take_profit_fallback'
                else:
                    tp_reason = 'take_profit'

            if pos and not exited:
                o, h, l = opens[j], highs[j], lows[j]
                if is_long:
                    if o <= stop:
                        exit_raw, reason = o, 'stop_loss'
                    elif o >= tp:
                        exit_raw, reason = o, tp_reason
                    elif l <= stop:
                        exit_raw, reason = stop, 'stop_loss'
                    elif h >= tp:
                        exit_raw, reason = tp, tp_reason
                else:
                    if o >= stop:
                        exit_raw, reason = o, 'stop_loss'
                    elif o <= tp:
                        exit_raw, reason = o, tp_reason
                    elif h >= stop:
                        exit_raw, reason = stop, 'stop_loss'
                    elif l <= tp:
                        exit_raw, reason = tp, tp_reason
                exited = reason is not None

            if exited:
                record(j, reason, exit_raw)
            elif pos:
                c = closes[j]
                if is_long:
                    mtm = (c - e_px - comm) / e_px
                else:
                    mtm = (e_px * (1 - sec) - comm - c) / e_px
                equity[j] = base * (1 + mtm)
                # Decisions on this close; fills happen at the next open.
                if sell[j]:
                    pend_exit = 'rsi_cross'
                elif floor(days[j] - days[e_j]) >= max_hold:
                    pend_exit = 'max_hold_days'
            else:
                equity[j] = cash

            if not pos and pend_exit is None and buy[j] and j + 1 < n:
                pend_entry = j + 1

        if close_open_at_end and pos and n:
            record(n - 1, 'window_end', closes[n - 1])

        return trades, np.asarray(equity, dtype=np.float64)

    def _target_price(self, data: pd.DataFrame, signal_i: int,
                      target_rsi: float) -> Optional[float]:
        """RSI-implied take-profit price using only data up to the signal bar."""
        if signal_i < self.rsi_period:
            return None
        return RSIStrategy.calculate_price_for_target_rsi(
            data.iloc[:signal_i + 1], target_rsi, self.rsi_period)

    @staticmethod
    def _summarize_trades(trades: List[Dict]) -> Dict:
        if not trades:
            return {'num_trades': 0, 'win_rate': 0, 'avg_duration': 0}
        num = len(trades)
        wins = sum(1 for t in trades if t['return'] > 0)
        return {
            'num_trades': num,
            'win_rate': wins / num,
            'avg_duration': float(np.mean([t['duration'] for t in trades])),
        }

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min())

    def _calculate_sharpe_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0

        excess_returns = daily_returns - \
            (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / daily_returns.std()

    def _create_null_result(self, symbol: str) -> BacktestResult:
        """Create null result for failed backtests."""
        return BacktestResult(
            symbol=symbol,
            total_return=0.0,
            buy_and_hold_return=0.0,
            alpha=0.0,
            num_trades=0,
            win_rate=0.0,
            avg_trade_duration=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            composite_score=0.0,
            profitable=False,
            direction=self.direction,
            strategy_name=self.name,
            params={
                "rsi_period": self.rsi_period,
                "rsi_lower": self.rsi_lower,
                "rsi_upper": self.rsi_upper,
                "direction": self.direction,
                "max_hold_days": self.max_hold_days,
            }
        )

    def build_consolidated_trades(self, results: List[BacktestResult]) -> pd.DataFrame:
        """
        Build a consolidated trade DataFrame from multiple backtest results.

        Args:
            results: List of BacktestResult objects containing trade details

        Returns:
            Consolidated trades DataFrame. Empty DataFrame if no trades are available.
        """
        try:
            all_trades = []

            # Collect all trades from all results
            for result in results:
                if result.trade_details:
                    for trade in result.trade_details:
                        # Add strategy and symbol info to each trade
                        trade_record = {
                            'symbol': result.symbol,
                            'rsi_period': (result.params or {}).get('rsi_period'),
                            'rsi_lower': (result.params or {}).get('rsi_lower'),
                            'rsi_upper': (result.params or {}).get('rsi_upper'),
                            'entry_date': trade['entry_date'],
                            'entry_price': trade['entry_price'],
                            'exit_date': trade['exit_date'],
                            'exit_price': trade['exit_price'],
                            'return': trade['return'],
                            'duration': trade['duration'],
                            'exit_reason': trade.get('exit_reason', 'unknown'),
                            'direction': trade.get('direction', result.direction)
                        }
                        all_trades.append(trade_record)

            if not all_trades:
                logger.info("No trades to consolidate")
                return pd.DataFrame()

            logger.info(
                f"Consolidated {len(all_trades)} trades from {len(results)} strategies")

            # Convert to DataFrame
            trades_df = pd.DataFrame(all_trades)

            # Convert datetime to EST timezone and format for readability
            est = pytz.timezone('US/Eastern')

            def convert_to_est(timestamp):
                """Convert timestamp to EST, handling both tz-aware and tz-naive timestamps."""
                if timestamp.tz is None:
                    # If timezone-naive, assume UTC
                    return timestamp.tz_localize('UTC').tz_convert(est).strftime('%Y-%m-%d %H:%M:%S EST')
                else:
                    # If already timezone-aware, just convert
                    return timestamp.tz_convert(est).strftime('%Y-%m-%d %H:%M:%S EST')

            trades_df['entry_date_est'] = trades_df['entry_date'].apply(
                convert_to_est)
            trades_df['exit_date_est'] = trades_df['exit_date'].apply(
                convert_to_est)

            # Round numeric values for readability
            trades_df['entry_price'] = trades_df['entry_price'].round(4)
            trades_df['exit_price'] = trades_df['exit_price'].round(4)
            trades_df['return'] = trades_df['return'].round(6)

            # Reorder columns for better readability
            final_columns = [
                'symbol', 'rsi_period', 'rsi_lower', 'rsi_upper', 'direction',
                'entry_date_est', 'entry_price', 'exit_date_est', 'exit_price',
                'return', 'duration', 'exit_reason'
            ]
            trades_df = trades_df[final_columns]

            # Sort by entry date for chronological order
            trades_df = trades_df.sort_values('entry_date_est')

            return trades_df

        except Exception as e:
            logger.error(f"Error building consolidated trade DataFrame: {e}")
            return pd.DataFrame()

    @staticmethod
    def _get_price_column(data: pd.DataFrame) -> str:
        """Get the correct price column name from the DataFrame."""
        if 'close' in data.columns:
            return 'close'
        elif 'c' in data.columns:
            return 'c'
        else:
            raise ValueError(
                f"No price column found. Expected 'close' or 'c' in data columns: {list(data.columns)}")
