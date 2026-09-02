"""
Strategy interface and shared types for the multi-strategy framework.

Phase B of MULTI_STRATEGY_PLAN.md: defines the ``Strategy`` base class every
strategy implements, plus the shared data types (``BacktestResult``, which
moved here from ``strategy.py``, ``LiveSignal``, and ``StrategyContext``).

``strategy.py`` remains a backward-compatible re-export shim so existing
``from strategy import BacktestResult`` imports keep working.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    symbol: str
    rsi_period: int
    rsi_lower: int
    rsi_upper: int
    total_return: float
    buy_and_hold_return: float
    alpha: float
    num_trades: int
    win_rate: float
    avg_trade_duration: float
    max_drawdown: float
    sharpe_ratio: float
    profitable: bool
    calmar_ratio: float = 0.0
    composite_score: float = 0.0
    # Current RSI value at time of backtest (RSI strategies only)
    current_rsi: Optional[float] = None
    # Add trade details to the result
    trade_details: Optional[List[Dict]] = None
    direction: str = "long"
    # Owning strategy (registry key) and exact params used for this run.
    # Defaults to the legacy RSI strategy so existing persisted rows and
    # callers stay backward compatible (see MULTI_STRATEGY_PLAN.md Phase A).
    strategy_name: str = "rsi_mean_reversion"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LiveSignal:
    """A live trading signal emitted by a strategy (consumed by the engine).

    Generalizes ``TradingOpportunity``: strategy-specific fields live in
    ``extra``; the engine only interprets the common fields below.
    """
    symbol: str
    direction: str = "long"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    backtest_return: float = 0.0
    alpha: float = 0.0
    win_rate: float = 0.0
    composite_score: float = 0.0
    num_trades: int = 0
    strategy_name: str = "rsi_mean_reversion"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyContext:
    """Read-only per-cycle context handed to strategies at signal time.

    Injected by the engine (Phase C). Strategies must not mutate it.
    """
    data_provider: Any = None
    positions_manager: Any = None
    config: Any = None
    as_of: Any = None
    ohlcv_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Backtest results for THIS strategy (the engine groups by strategy_name).
    strategy_results: List["BacktestResult"] = field(default_factory=list)


class Strategy(ABC):
    """Base class for all trading strategies.

    A strategy owns its signal math, its parameter search space, and (via
    :meth:`optimize`) optionally its own grid-search procedure. The optimizer,
    engine, storage, and UI are strategy-agnostic and operate through this
    interface.
    """

    # Registry key (must be unique, stable, and config-friendly).
    name: str = "base"
    # "session" (daily) | "bar_loop" (intraday, Phase D).
    execution_style: str = "session"
    # Bar timeframe for bar_loop strategies, e.g. "5m".
    bar_size: Optional[str] = None

    @classmethod
    def create(cls) -> "Strategy":
        """Instantiate the strategy with defaults (registry convenience)."""
        return cls()

    @abstractmethod
    def backtest(
        self,
        data: pd.DataFrame,
        symbol: str,
        initial_cash: float = 10000,
        prepared: Any = None,
        **params: Any,
    ) -> BacktestResult:
        """Run a vectorized backtest.

        ``params`` override constructor defaults (used by grid search);
        ``prepared`` is an optional strategy-specific precomputation produced
        by :meth:`prepare` (e.g. an indicator cache) to avoid recomputing
        shared work per grid combo.
        """

    def get_param_grid(self, direction: str = "long") -> List[Dict[str, Any]]:
        """Parameter search space for optimization.

        Default: a single point (no search) using constructor defaults.
        """
        return [{"direction": direction}]

    def prepare(self, data: pd.DataFrame) -> Optional[Any]:
        """Optional shared precomputation across grid combos. Default: none."""
        return None

    def optimize(
        self,
        data: pd.DataFrame,
        symbol: str,
        direction: str = "long",
        initial_cash: float = 10000,
    ) -> Optional[BacktestResult]:
        """Default generic grid-search over :meth:`get_param_grid`.

        Runs every combo, z-scores the stage pool, and returns the best
        profitable result (composite_score set). Strategies with specialized
        search procedures (e.g. RSI's two-stage search + per-period RSI cache)
        override this.
        """
        import zscore  # pylint: disable=import-outside-toplevel,redefined-outer-name,reimported

        grid = self.get_param_grid(direction)
        if not grid:
            return None

        prepared = self.prepare(data)
        results = [
            self.backtest(data, symbol, initial_cash, prepared=prepared, **p)
            for p in grid
        ]
        # compute_stage_zscores expects (result, period, lower, upper) tuples;
        # it only reads item[0], so dummy ints are fine for the generic path.
        zscores = zscore.compute_stage_zscores(
            [(r, 0, 0, 0) for r in results])

        best: Optional[BacktestResult] = None
        best_score = -float("inf")
        for result, score in zip(results, zscores):
            if result.profitable and score > best_score:
                best, best_score = result, score
        if best is not None:
            best.composite_score = best_score
        return best

    def warmup_days(self) -> int:
        """Calendar days of extra history to fetch before the backtest window.

        Override to match the strategy's longest indicator lookback.
        """
        return 30

    def build_consolidated_trades(
        self, results: List[BacktestResult]
    ) -> pd.DataFrame:
        """Build a consolidated trade DataFrame from result.trade_details.

        Default implementation; strategies may override for extra columns.
        """
        all_trades = []
        for result in results:
            for trade in (result.trade_details or []):
                all_trades.append({
                    "symbol": result.symbol,
                    "strategy_name": result.strategy_name,
                    "params": result.params,
                    "entry_date": trade.get("entry_date"),
                    "entry_price": trade.get("entry_price"),
                    "exit_date": trade.get("exit_date"),
                    "exit_price": trade.get("exit_price"),
                    "return": trade.get("return"),
                    "duration": trade.get("duration"),
                    "exit_reason": trade.get("exit_reason", "unknown"),
                    "direction": trade.get("direction", result.direction),
                })
        return pd.DataFrame(all_trades)

    def evaluate_live_signals(self, ctx: StrategyContext) -> List[LiveSignal]:
        """Evaluate live signals (Phase C engine hook). Default: no signals."""
        return []
