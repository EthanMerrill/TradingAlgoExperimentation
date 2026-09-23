"""
Leveraged single-stock ETF rebalance front-run strategy.

Trades the **underlying stocks** of single-stock leveraged/inverse ETFs near
the close, attempting to front-run the daily delta re-hedge those funds (via
their swap counterparties) execute in the underlying.

See ``strategies/leveraged_single_stock_etfs.py`` for the ETF → underlying map
and ``strategies/leveraged_rebalance_signal.py`` for the detection math.

Design notes
------------
* ``execution_style = "session"``: v1 is a daily-entry strategy. The default
  exit is flat-at-close, which is also what a future live bar-loop wiring would
  do (enter in the late window, flat by the bell).
* ``data_timeframe = "5m"`` with ``Adjustment.SPLIT``: intraday bars are needed
  to see the late-session volume/momentum fingerprint. SPLIT-only adjustment
  avoids dividend gap-fills that would corrupt intraday signals.
* ``symbol_universe()`` returns the map's underlyings, so the backtest loop
  runs this strategy on that set instead of the global universe.
* ``prepare()`` computes the expensive RVOL/momentum features **once**; each
  grid combo then only re-thresholds the compact feature frame.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment

from strategies.base import BacktestResult, Strategy
from strategies.leveraged_rebalance_signal import (
    DEFAULT_ENTRY_END,
    DEFAULT_ENTRY_START,
    DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_RVOL_LOOKBACK_DAYS,
    DEFAULT_RVOL_THRESHOLD,
    detect_events,
    placebo_alpha_test,
    prepare_features,
    simulate_events,
)
from strategies.leveraged_single_stock_etfs import underlyings
from utils import PerformanceMetrics

logger = logging.getLogger(__name__)


class LeveragedRebalanceStrategy(Strategy):
    """Front-run the daily rebalance flow of single-stock leveraged ETFs."""

    name = "leveraged_etf_rebalance"
    execution_style = "session"
    bar_size = None
    data_timeframe = "5m"
    data_adjustment = Adjustment.SPLIT

    @classmethod
    def create(cls) -> "Strategy":
        """Default instance (single param point; grid comes from get_param_grid)."""
        return cls()

    def __init__(
        self,
        entry_start: str = DEFAULT_ENTRY_START,
        entry_end: str = DEFAULT_ENTRY_END,
        rvol_threshold: float = DEFAULT_RVOL_THRESHOLD,
        momentum_threshold: float = DEFAULT_MOMENTUM_THRESHOLD,
        rvol_lookback_days: int = DEFAULT_RVOL_LOOKBACK_DAYS,
        exit_mode: str = "close",
        hold_days: int = 0,
        cost_bps: float = 2.0,
        direction_mode: str = "both",
        enter_at: str = "window_start",
    ):
        self.entry_start = entry_start
        self.entry_end = entry_end
        self.rvol_threshold = float(rvol_threshold)
        self.momentum_threshold = float(momentum_threshold)
        self.rvol_lookback_days = int(rvol_lookback_days)
        self.exit_mode = exit_mode
        self.hold_days = int(hold_days)
        self.cost_bps = float(cost_bps)
        self.direction_mode = direction_mode
        self.enter_at = enter_at

    # ------------------------------------------------------------------
    # Universe / data
    # ------------------------------------------------------------------

    def symbol_universe(self) -> Optional[List[str]]:
        """The underlying stocks of the mapped leveraged ETFs."""
        return underlyings()

    def warmup_days(self) -> int:
        """Enough calendar days for the RVOL baseline plus a buffer."""
        return max(20, self.rvol_lookback_days * 3)

    # ------------------------------------------------------------------
    # Parameter search
    # ------------------------------------------------------------------

    def get_param_grid(self, direction: str = "long") -> List[Dict[str, Any]]:
        """Grid over the entry window, day-move magnitude and RVOL gate.

        ``direction`` (the framework's long/short pass) maps onto
        ``direction_mode`` so each direction is optimized separately.

        Design is driven by the rebalance *mechanism*: a fund's required hedge
        is ``leverage × AUM × day_return``, so the flow is proportional to the
        **magnitude of the session's move** and aligned with its **sign**. That
        magnitude is fully known well before the close, which is what makes
        front-running possible at all.

        Consequently:

        * ``momentum_threshold`` (minimum ``|day return|``) **is** searched — it
          is the primary mechanism variable, not an arbitrary knob.
        * ``rvol_threshold`` is optional. RVOL is the *symptom* of the flow
          (it only fires once the rebalance is already under way), so a value of
          ``0.0`` disables it and tests the pure "day-move predicts the flow"
          hypothesis.
        * ``entry_start`` spans the afternoon: if the signal is the day's move,
          entering earlier captures more of the drift and pays less of the
          late-window spread.
        * ``entry_end`` is fixed at the close, since the flow is a close event.
        """
        mode = direction if direction in ("long", "short") else "both"
        grid: List[Dict[str, Any]] = []
        for entry_start in ("12:30", "13:30", "14:30", "15:20", "15:45"):
            for rvol_threshold in (0.0, 2.5):
                for momentum_threshold in (0.0, 0.015, 0.03):
                    for exit_mode, hold_days in (("close", 0), ("next_open", 0)):
                        grid.append({
                            "entry_start": entry_start,
                            "entry_end": "15:55",
                            "rvol_threshold": rvol_threshold,
                            "momentum_threshold": momentum_threshold,
                            "exit_mode": exit_mode,
                            "hold_days": hold_days,
                            "direction_mode": mode,
                            "enter_at": "window_start",
                            "rvol_lookback_days": self.rvol_lookback_days,
                            "cost_bps": self.cost_bps,
                        })
        return grid

    def _benchmark_trades(
        self, features: pd.DataFrame, resolved: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Unfiltered "always positioned" baseline over **every** session.

        This is the honest counterfactual for measuring the *detection's* value:
        be positioned in the window on every session (entering at the window
        open, same exit mode, same cost) regardless of whether any spike fired.
        Comparing the strategy's per-session average against this baseline
        answers the only question that matters — *does picking these days beat
        simply holding the same window every day?* — with market drift, intraday
        timing convention, exit and costs all held constant.

        Note the baseline deliberately spans all sessions rather than only the
        ones the strategy traded. Restricting it would make the comparison
        degenerate whenever the strategy's entry timing equals the baseline's
        (a "trade every day" baseline restricted to the same days becomes the
        strategy itself, forcing alpha to 0 by construction).
        """
        events = detect_events(
            features,
            entry_start=resolved["entry_start"],
            entry_end=resolved["entry_end"],
            rvol_threshold=0.0,
            momentum_threshold=0.0,
            direction_mode=resolved["direction_mode"],
            # Always enter at the window open: a single, well-defined timing
            # convention for the "hold the window every day" baseline.
            enter_at="window_start",
        )
        if events.empty:
            return []
        return simulate_events(
            features,
            events,
            exit_mode=resolved["exit_mode"],
            hold_days=resolved["hold_days"],
            cost_bps=resolved["cost_bps"],
        )

    def prepare(self, data: pd.DataFrame) -> Optional[Any]:
        """Precompute the RVOL/momentum feature frame once per symbol."""
        if data is None or data.empty:
            return None
        features = prepare_features(data, self.rvol_lookback_days)
        features.attrs["rvol_lookback_days"] = self.rvol_lookback_days
        return features

    def placebo_p_value(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        alpha: float,
        num_trades: int,
        n_draws: int = 2000,
    ) -> Optional[Dict[str, Any]]:
        """Permutation test: did the detected days beat random day-picking?

        Called by walk-forward on the OUT-OF-SAMPLE slice with the parameters
        chosen in-sample, so the p-value is not contaminated by selecting the
        test statistic.
        """
        if data is None or data.empty or num_trades <= 0:
            return None
        resolved = self._resolve_params(params or {})
        features = prepare_features(data, resolved["rvol_lookback_days"])
        if features is None or features.empty:
            return None
        return placebo_alpha_test(
            features,
            entry_start=resolved["entry_start"],
            entry_end=resolved["entry_end"],
            exit_mode=resolved["exit_mode"],
            hold_days=resolved["hold_days"],
            cost_bps=resolved["cost_bps"],
            direction_mode=resolved["direction_mode"],
            n_selected=int(num_trades),
            observed_alpha=float(alpha),
            n_draws=int(n_draws),
        )

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def _resolve_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge constructor defaults with per-combo overrides."""
        resolved = {
            "entry_start": self.entry_start,
            "entry_end": self.entry_end,
            "rvol_threshold": self.rvol_threshold,
            "momentum_threshold": self.momentum_threshold,
            "rvol_lookback_days": self.rvol_lookback_days,
            "exit_mode": self.exit_mode,
            "hold_days": self.hold_days,
            "cost_bps": self.cost_bps,
            "direction_mode": self.direction_mode,
            "enter_at": self.enter_at,
        }
        for key, value in params.items():
            if key in resolved and value is not None:
                resolved[key] = value
        return resolved

    def backtest(
        self,
        data: pd.DataFrame,
        symbol: str,
        initial_cash: float = 10000,
        prepared: Any = None,
        **params: Any,
    ) -> BacktestResult:
        resolved = self._resolve_params(params)

        features = None
        if isinstance(prepared, pd.DataFrame) and "rvol" in prepared.columns:
            if prepared.attrs.get("rvol_lookback_days") == resolved["rvol_lookback_days"]:
                features = prepared
        if features is None:
            features = prepare_features(data, resolved["rvol_lookback_days"])

        if features is None or features.empty:
            return self._null_result(symbol, resolved)

        events = detect_events(
            features,
            entry_start=resolved["entry_start"],
            entry_end=resolved["entry_end"],
            rvol_threshold=resolved["rvol_threshold"],
            momentum_threshold=resolved["momentum_threshold"],
            direction_mode=resolved["direction_mode"],
            enter_at=resolved.get("enter_at", "trigger"),
        )
        if events.empty:
            return self._null_result(symbol, resolved)

        trades = simulate_events(
            features,
            events,
            exit_mode=resolved["exit_mode"],
            hold_days=resolved["hold_days"],
            cost_bps=resolved["cost_bps"],
        )
        if not trades:
            return self._null_result(symbol, resolved)

        return self._build_result(
            symbol, symbol_direction=resolved["direction_mode"],
            features=features, trades=trades, resolved=resolved)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_sharpe_ratio(daily_returns: pd.Series,
                                risk_free_rate: float = 0.02) -> float:
        """Annualized Sharpe (matches the RSI strategy's convention).

        Returns 0.0 when there is not enough data to estimate a dispersion:
        a single observation has an undefined sample std (NaN under ddof=1),
        which would otherwise poison the Z-score pool.
        """
        if len(daily_returns) < 2:
            return 0.0
        std = daily_returns.std()
        if not np.isfinite(std) or std == 0:
            return 0.0
        excess = daily_returns - (risk_free_rate / 252)
        return float(np.sqrt(252) * excess.mean() / std)

    def _build_result(
        self,
        symbol: str,
        symbol_direction: str,
        features: pd.DataFrame,
        trades: List[Dict[str, Any]],
        resolved: Dict[str, Any],
    ) -> BacktestResult:
        trade_returns = pd.Series([t["return"]
                                  for t in trades], dtype=np.float64)
        equity = (1.0 + trade_returns).cumprod()

        num_trades = len(trades)
        win_rate = float(
            sum(1 for t in trades if t["return"] > 0) / num_trades) if num_trades else 0.0
        avg_duration = float(np.mean([t["duration"]
                             for t in trades])) if trades else 0.0
        total_return = float(equity.iloc[-1] - 1.0) if num_trades else 0.0

        first_close = float(features["close"].iloc[0])
        last_close = float(features["close"].iloc[-1])
        window_buy_and_hold = (last_close / first_close -
                               1.0) if first_close else 0.0

        # Real alpha, measured PER SESSION so differing trade counts cannot
        # flatter the result:
        #   alpha = (strategy mean return per trade)
        #         − (baseline mean return per session, all sessions)
        # The baseline is "be positioned in the window every day", which is what
        # a naive trader would get without any detection. Anything the RVOL /
        # momentum filter adds over that is genuine edge.
        benchmark_trades = self._benchmark_trades(features, resolved)
        strategy_mean_return = float(
            trade_returns.mean()) if num_trades else 0.0
        if benchmark_trades:
            bench_rets = pd.Series(
                [t["return"] for t in benchmark_trades], dtype=np.float64)
            benchmark_mean_return = float(bench_rets.mean())
            benchmark_total_return = float(
                (1.0 + bench_rets).cumprod().iloc[-1] - 1.0)
        else:
            # No comparable baseline — fall back to the whole-window buy-and-hold
            # so alpha stays conservative rather than flattering.
            benchmark_mean_return = window_buy_and_hold
            benchmark_total_return = window_buy_and_hold
        alpha = strategy_mean_return - benchmark_mean_return

        max_drawdown = float(
            PerformanceMetrics.calculate_max_drawdown(equity)) if num_trades else 0.0

        # Daily attribution over ALL sessions (0 on flat days). Using every
        # session rather than only trade days keeps the Sharpe estimate from
        # exploding on a 1-2 trade sample — this strategy is flat most days,
        # so the flat days are genuinely part of its return distribution.
        session_index = pd.Index(
            sorted(features["date"].unique()), name="date")
        daily = (
            pd.Series(
                [t["return"] for t in trades],
                index=pd.Index([t["entry_date"] for t in trades], name="date"),
                dtype=np.float64,
            )
            .groupby(level=0).sum()
            .reindex(session_index, fill_value=0.0)
        )
        sharpe = self._calculate_sharpe_ratio(daily)

        calmar = (
            float(total_return / max_drawdown) if max_drawdown > 0 else 0.0
        )

        result_params = dict(resolved)
        result_params["direction"] = symbol_direction
        result_params["strategy_mean_return"] = strategy_mean_return
        result_params["benchmark_mean_return"] = benchmark_mean_return
        result_params["benchmark_return"] = benchmark_total_return
        result_params["benchmark_trades"] = len(benchmark_trades)
        result_params["window_buy_and_hold"] = window_buy_and_hold

        return BacktestResult(
            symbol=symbol,
            total_return=total_return,
            # Per-session baseline expectation (documented in params) — used as
            # the reference return rather than a whole-window buy-and-hold.
            buy_and_hold_return=benchmark_mean_return,
            alpha=alpha,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_trade_duration=avg_duration,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            profitable=total_return > 0,
            calmar_ratio=calmar,
            trade_details=trades,
            direction=symbol_direction,
            strategy_name=self.name,
            params=result_params,
        )

    def _null_result(self, symbol: str, resolved: Dict[str, Any]) -> BacktestResult:
        result_params = dict(resolved)
        result_params["direction"] = resolved.get("direction_mode", "both")
        return BacktestResult(
            symbol=symbol,
            profitable=False,
            direction=result_params["direction"],
            strategy_name=self.name,
            params=result_params,
        )
