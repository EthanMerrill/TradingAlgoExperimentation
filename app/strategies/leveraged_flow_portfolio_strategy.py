"""
Live trading strategy: long-only leveraged-ETF rebalance-flow portfolio.

This operationalizes the cross-sectional flow portfolio researched in
``leveraged_flow_portfolio.py`` (see that module and the research runner
``leveraged_flow_backtest.py`` for the full thesis and falsification history).

Research result being operationalized
-------------------------------------
Rank the mapped underlyings each session by the estimated dealer hedge flow
(Σ leverage×AUM× the funds' return-so-far), buy the top-K **positive**-flow
names at the decision bar, exit at the close. Long-only: the short side was
shown to be structurally unprofitable (no dose-response in |flow|, negative
on crash days' context — negative-flow days mean-revert), and bucket analysis
showed the entire effect lives in the largest positive flows.

Best validated configuration (proxy flows, 4 bps round trip, 14mo):
15:20 ET decision, top-3, +0.253%/session, t=3.70, placebo p=0.001,
split-half consistent. Treat as a *starting point for paper trading*, not a
proven edge — the momentum control is also significant and costs were
optimistic.

Live mechanics (bar_loop)
-------------------------
``execution_style = "bar_loop"`` so the BarLoopEngine evaluates during RTH:

* at/after the decision time each session, emit one LiveSignal per selected
  underlying (deduplicated per session date),
* entries are day-TIF bracket orders placed by the engine,
* the engine force-closes all ``intraday`` positions at the session close,
  which reproduces the backtest's flat-at-close exit exactly.

Lookahead safety: the flow is computed from the funds' bars **up to the
decision bar only** (``etf_flow_estimates`` filters ``tod == asof``), and the
entry price is the underlying's close of that same bar.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pytz

from strategies.base import BacktestResult, LiveSignal, Strategy, StrategyContext
from strategies.leveraged_flow_portfolio import (
    DEFAULT_AUM_USD,
    etf_flow_estimates,
)
from strategies.leveraged_rebalance_signal import (
    parse_hhmm,
    prepare_features,
)
from strategies.leveraged_single_stock_etfs import (
    LEVERAGED_SINGLE_STOCK_ETFS,
    underlyings,
)
from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)

US_EASTERN = pytz.timezone("US/Eastern")

# Lookback for intraday bar fetches: enough sessions for the flow's cum_ret
# (needs only the prior close) plus a buffer for half-days/holidays.
_LIVE_FETCH_SESSIONS = 4


class LeveragedFlowPortfolioStrategy(Strategy):
    """Long-only top-K flow portfolio, traded intraday, flat at close."""

    name = "leveraged_flow_portfolio"
    execution_style = "bar_loop"
    bar_size = "5m"
    data_timeframe = "5m"
    # SPLIT only: dividend gap-fills would corrupt intraday flows.
    from alpaca.data.enums import Adjustment as _Adjustment
    data_adjustment = _Adjustment.SPLIT

    def __init__(
        self,
        entry_time: str = "15:20",
        top_k: int = 3,
        min_abs_notional: float = 10_000_000.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ):
        self.entry_time = entry_time
        self.entry_tod = parse_hhmm(entry_time)
        self.top_k = int(top_k)
        self.min_abs_notional = float(min_abs_notional)
        # Bracket orders require stop/TP levels; intraday positions are closed
        # at the session close regardless, so these are catastrophic-risk
        # bounds rather than the exit mechanism. Default to config values.
        self.stop_loss_pct = (
            float(stop_loss_pct)
            if stop_loss_pct is not None
            else getattr(globalConfig, "STOP_LOSS_PCT", 0.05))
        self.take_profit_pct = (
            float(take_profit_pct)
            if take_profit_pct is not None
            else max(getattr(globalConfig, "TAKE_PROFIT_PCT", 0.15), 0.02))

        # Per-session dedupe: emit at most once per ET trading date.
        self._last_signal_date: Optional[Any] = None

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def symbol_universe(self) -> List[str]:
        """The underlyings of the mapped single-stock leveraged ETFs."""
        return underlyings()

    def warmup_days(self) -> int:
        return 10

    def get_param_grid(self, direction: str = "long") -> List[Dict[str, Any]]:
        """Single-point grid: the researched configuration (long-only)."""
        return [{
            "entry_time": self.entry_time,
            "top_k": self.top_k,
            "min_abs_notional": self.min_abs_notional,
            "direction": "long",
        }]

    def backtest(
        self,
        data,
        symbol: str,
        initial_cash: float = 10000,
        prepared: Any = None,
        **params: Any,
    ) -> BacktestResult:
        """Per-symbol approximation of the cross-sectional portfolio.

        The full portfolio ranks ALL underlyings cross-sectionally, which the
        per-symbol backtest framework cannot do (it sees one symbol's bars).
        This approximation trades the symbol whenever its own flow is positive
        at the decision bar — the same mechanism, without the ranking.

        Use ``leveraged_flow_backtest.py`` for the true cross-sectional result;
        this exists so the engine's backtest cycle produces the per-symbol
        result rows the bar loop needs to activate the strategy.
        """
        entry_tod = parse_hhmm(
            params.get("entry_time", self.entry_time))
        top_k = int(params.get("top_k", self.top_k))
        min_abs = float(params.get("min_abs_notional", self.min_abs_notional))

        features = prepared if (
            isinstance(prepared, dict) and "cum_ret" in getattr(
                prepared, "columns", [])) else prepare_features(data)

        # Aggregate leverage×AUM for the funds tracking this symbol.
        lev_aum = 0.0
        for fund in LEVERAGED_SINGLE_STOCK_ETFS.values():
            if fund.underlying == symbol:
                lev_aum += fund.leverage * (fund.aum_usd or DEFAULT_AUM_USD)

        if features.empty or lev_aum <= 0:
            return self._null_result(symbol, entry_tod, top_k, min_abs)

        decision = features[features["tod"] == entry_tod]
        closes = features.groupby("date")["close"].last()
        records = []
        for row in decision.itertuples(index=False):
            if row.cum_ret is None or not np.isfinite(row.cum_ret):
                continue
            flow = lev_aum * float(row.cum_ret)
            if abs(flow) < min_abs or flow <= 0:
                continue  # long-only: positive flows above the floor only
            entry_px = float(row.close)
            exit_px = closes.get(row.date)
            if not exit_px or exit_px <= 0:
                continue
            records.append(float(exit_px) / entry_px - 1.0)

        if not records:
            return self._null_result(symbol, entry_tod, top_k, min_abs)

        import numpy as _np
        rets = _np.array(records, dtype=float)
        total = float((1.0 + rets).prod() - 1.0)
        win_rate = float((rets > 0).mean())
        params_out = {
            "entry_time": params.get("entry_time", self.entry_time),
            "top_k": top_k,
            "min_abs_notional": min_abs,
            "direction": "long",
            "note": "per-symbol approximation; ranking needs the runner",
        }
        return BacktestResult(
            symbol=symbol,
            total_return=total,
            buy_and_hold_return=0.0,
            alpha=total,
            num_trades=len(rets),
            win_rate=win_rate,
            profitable=total > 0,
            direction="long",
            strategy_name=self.name,
            params=params_out,
        )

    @staticmethod
    def _null_result(symbol, entry_tod, top_k, min_abs) -> BacktestResult:
        return BacktestResult(
            symbol=symbol,
            profitable=False,
            direction="long",
            strategy_name="leveraged_flow_portfolio",
            params={
                "entry_time": "unavailable",
                "top_k": top_k,
                "min_abs_notional": min_abs,
                "direction": "long",
            },
        )

    # ------------------------------------------------------------------
    # Live signals (bar loop)
    # ------------------------------------------------------------------

    def _et_now(self, as_of: Optional[datetime]) -> datetime:
        if as_of is None:
            as_of = datetime.now()
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=pytz.UTC)
        return as_of.astimezone(US_EASTERN)

    def _fetch_features(self, ctx: StrategyContext, symbol: str):
        """Features for ``symbol``'s recent intraday bars (provider-cached)."""
        provider = ctx.data_provider
        cache = ctx.ohlcv_cache if ctx.ohlcv_cache is not None else {}
        key = f"{symbol}@{self.entry_tod}"
        cached = cache.get(key)
        if cached is not None:
            return cached
        end = self._et_now(ctx.as_of)
        start = end - timedelta(days=_LIVE_FETCH_SESSIONS + 2)
        try:
            bars = provider.get_single_stock_bars(
                symbol, start, end, timeframe=self.data_timeframe,
                adjustment=self.data_adjustment)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Flow portfolio: bar fetch failed for %s: %s",
                           symbol, e)
            return None
        if bars is None or bars.empty:
            return None
        features = prepare_features(bars)
        cache[key] = features
        return features

    def evaluate_live_signals(
            self, ctx: StrategyContext) -> List[LiveSignal]:
        """Emit the day's top-K positive-flow long signals, once per session."""
        now_et = self._et_now(ctx.as_of)
        if now_et.weekday() >= 5:
            return []
        if now_et.hour * 60 + now_et.minute < self.entry_tod:
            return []  # decision time not reached yet
        if self._last_signal_date == now_et.date():
            return []  # already emitted today

        # Underlying features (entry prices + date alignment).
        underlying_features: Dict[str, Any] = {}
        for symbol in self.symbol_universe():
            features = self._fetch_features(ctx, symbol)
            if features is not None and not features.empty:
                underlying_features[symbol] = features
        if not underlying_features:
            logger.warning(
                "Flow portfolio: no underlying data at %s", now_et)
            return []

        # Real flows: fetch each mapped fund's own bars (dedup via cache).
        etf_features: Dict[str, Any] = {}
        for etf in LEVERAGED_SINGLE_STOCK_ETFS:
            features = self._fetch_features(ctx, etf)
            if features is not None and not features.empty:
                etf_features[etf] = features
        if not etf_features:
            logger.warning("Flow portfolio: no ETF data at %s", now_et)
            return []

        flows = etf_flow_estimates(etf_features, asof_tod=self.entry_tod)
        if flows.empty:
            self._last_signal_date = now_et.date()
            return []

        # Decision bar for entry prices: the underlying's close at entry_tod.
        entry_px: Dict[Any, Dict[str, float]] = {}
        for symbol, features in underlying_features.items():
            bars = features[features["tod"] == self.entry_tod]
            for row in bars.itertuples(index=False):
                entry_px.setdefault(row.date, {})[symbol] = float(row.close)

        today = now_et.date()
        day_flows = flows[flows["date"] == today]
        if day_flows.empty:
            # Partial day / holiday: fall back to the latest available date.
            day_flows = flows.tail(len(flows))
            if not day_flows.empty:
                today = day_flows["date"].iloc[-1]
        prices_today = entry_px.get(today, {})

        candidates = day_flows[
            (day_flows["abs_notional"] >= self.min_abs_notional)
            & (day_flows["notional"] > 0.0)                      # long-only
        ].sort_values("abs_notional", ascending=False)

        signals: List[LiveSignal] = []
        for row in candidates.itertuples(index=False):
            if len(signals) >= self.top_k:
                break
            symbol = row.underlying
            entry = prices_today.get(symbol)
            if entry is None or entry <= 0:
                continue
            signals.append(LiveSignal(
                symbol=symbol,
                direction="long",
                entry_price=round(entry, 2),
                stop_loss=round(entry * (1.0 - self.stop_loss_pct), 2),
                take_profit=round(
                    entry * (1.0 + self.take_profit_pct), 2),
                strategy_name=self.name,
                extra={
                    "flow_notional_usd": float(row.notional),
                    "n_funds": int(row.n_funds),
                    "decision_time": self.entry_time,
                    "exit": "intraday_session_close",
                },
            ))

        self._last_signal_date = now_et.date()
        logger.info(
            "Flow portfolio: %d long signal(s) at %s (%s)",
            len(signals), now_et.strftime("%H:%M"), self.entry_time)
        return signals
