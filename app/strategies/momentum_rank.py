"""
Live strategy: backtest-ranked momentum, weekly rank-buffer rebalance.

Research (see app/research/momentum_rank_portfolio.py and the 2026-11 runs):
rank every common stock in the universe by the Sharpe of ONE fixed rule
(``long while close > SMA50``) over the trailing 12-1 window, hold the top
``slots`` names, and replace a holding only once it drops out of the top
``buffer`` ranks. On a point-in-time universe (delisted names included,
2017-03 -> 2026-11, 5 bps/side) the 5-slot weekly variant returned ~28%
CAGR vs SPY ~14%, 99-100th percentile vs random rankings in every period;
max drawdown ~-48%, beta ~1.2. Timing luck across rebalance days was
+/-5pp CAGR. Treat paper trading as the real out-of-sample test.

Live mechanics (``execution_style = "portfolio"``)
--------------------------------------------------
* Rebalance session: the first trading session of each ISO week. The plan
  uses daily bars through the PREVIOUS session's close (the last close of
  the prior week) and executes at the open -- same timing as the backtest
  (decide at close, trade next open). Today's partial bar is never used.
* Exits: held names (of this strategy) ranked worse than ``buffer``, without a
  score, or outside the price band. A holding is not sold just because it
  dropped out of the daily-rebuilt universe (one-day volume snapshot).
* Entries: best-ranked names whose rule is long (close > SMA50), not already
  held by ANY strategy, until ``slots`` positions are held.
* No stop-loss / take-profit and no max-hold: the backtest had none.
* Universe: the live trading universe (``data_provider.get_stock_universe``,
  liquidity/price/options filters) restricted to common stocks, plus this
  strategy's current holdings so they can be ranked. The backtest used a
  point-in-time top-700-by-dollar-volume universe -- similar size and
  liquidity, not identical.
* Idempotent: re-running a rebalance on the same session reproduces the same
  ranking from the same closes; sold names are gone and held names are not
  re-bought, so a restart mid-session does not double-trade.
"""
import logging
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import (
    BacktestResult, LiveSignal, RebalancePlan, Strategy, StrategyContext)
from strategies.momentum_rank_signal import (
    backtest_scores, is_common_stock, rule_positions)
from utils.calendar import to_eastern  # type: ignore
from utils.datetime_ import get_session  # type: ignore

logger = logging.getLogger(__name__)


def _previous_session(d: date, max_back: int = 10) -> Optional[date]:
    for k in range(1, max_back + 1):
        prev = d - timedelta(days=k)
        if get_session(prev) is not None:
            return prev
    return None


class MomentumRankStrategy(Strategy):
    """Top-``slots`` stocks by rule Sharpe, weekly rebalance with a rank buffer."""

    name = "momentum_rank"
    execution_style = "portfolio"

    def __init__(self, slots: int = 5, buffer: int = 20, sma_days: int = 50,
                 lookback: int = 252, skip: int = 21, cost_bps: float = 5.0,
                 min_days_long: int = 40, min_cash_pct: float = 0.10,
                 fetch_chunk: int = 200):
        self.slots = int(slots)
        self.buffer = int(buffer)
        self.sma_days = int(sma_days)
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.cost_bps = float(cost_bps)
        self.min_days_long = int(min_days_long)
        self.min_cash_pct = float(min_cash_pct)
        self.fetch_chunk = int(fetch_chunk)

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------

    def warmup_days(self) -> int:
        # SMA + scoring window in trading days, as calendar days with buffer.
        return int((self.lookback + self.sma_days) * 1.5) + 30

    def backtest(self, data: pd.DataFrame, symbol: str, initial_cash: float = 10000,
                 prepared: Any = None, **params: Any) -> BacktestResult:
        """Single-symbol view of the rule (informational; ranking is cross-sectional)."""
        close = data["close"].astype(float).to_frame(symbol)
        pos = rule_positions(close, self.sma_days)
        ret = close[symbol].pct_change(fill_method=None)
        held = pos[symbol].shift(1)
        strat = (held * ret - held.diff().abs() * self.cost_bps / 1e4).dropna()
        sharpe = float(strat.mean() / strat.std() * np.sqrt(252)) if strat.std() > 0 else 0.0
        total = float((1 + strat).prod() - 1) if len(strat) else 0.0
        bh = float(close[symbol].iloc[-1] / close[symbol].iloc[0] - 1) if len(close) > 1 else 0.0
        return BacktestResult(symbol=symbol, total_return=total, buy_and_hold_return=bh,
                              alpha=total - bh, sharpe_ratio=sharpe, profitable=total > 0,
                              strategy_name=self.name,
                              params={"sma_days": self.sma_days})

    def is_rebalance_due(self, as_of: Any) -> bool:
        """True on the first trading session of the ISO week."""
        today = to_eastern(as_of).date()
        if get_session(today) is None:
            return False
        prev = _previous_session(today)
        return prev is None or prev.isocalendar()[:2] != today.isocalendar()[:2]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _fetch_closes(self, data_provider: Any, symbols: List[str],
                      start: datetime, end: datetime) -> pd.DataFrame:
        """Wide all-adjusted daily closes (date x symbol), batched."""
        from alpaca.data.enums import Adjustment  # pylint: disable=import-outside-toplevel
        from alpaca.data.requests import StockBarsRequest  # pylint: disable=import-outside-toplevel
        from alpaca.data.timeframe import TimeFrame  # pylint: disable=import-outside-toplevel

        frames = []
        for i in range(0, len(symbols), self.fetch_chunk):
            batch = list(symbols[i:i + self.fetch_chunk])
            attempt = 0
            while batch and attempt < 3:
                try:
                    df = data_provider.historical_client.get_stock_bars(StockBarsRequest(
                        symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                        start=start, end=end, adjustment=Adjustment.ALL)).df
                    if df is not None and not df.empty:
                        frames.append(df.reset_index())
                    break
                except Exception as e:  # pylint: disable=broad-exception-caught
                    bad = re.search(r"invalid symbol: ([^\"']+)", str(e))
                    if bad and bad.group(1) in batch:
                        batch.remove(bad.group(1))
                        continue
                    attempt += 1
                    logger.warning("momentum_rank: bar batch %d attempt %d failed: %s",
                                   i // self.fetch_chunk, attempt, e)
                    time.sleep(2 * attempt)
            if batch and attempt >= 3:
                raise RuntimeError(f"momentum_rank: could not fetch bars for batch {i // self.fetch_chunk}")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        df["date"] = (pd.to_datetime(df["timestamp"], utc=True)
                      .dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize())
        return df.pivot_table(index="date", columns="symbol", values="close").sort_index()

    # ------------------------------------------------------------------
    # Rebalance
    # ------------------------------------------------------------------

    def rank_table(self, close: pd.DataFrame, universe: List[str]) -> pd.DataFrame:
        """Score, rank and trend state at the last row of ``close``.

        Rank is among ``universe`` members with a finite score (1 = best).
        """
        pos = rule_positions(close, self.sma_days)
        score = backtest_scores(close, pos, self.lookback, self.skip, self.cost_bps,
                                self.min_days_long, metric="sharpe")
        last = pd.DataFrame({"score": score.iloc[-1], "trend": pos.iloc[-1],
                             "close": close.iloc[-1]})
        last["member"] = last.index.isin(universe)
        ranked = last[last["member"] & np.isfinite(last["score"])]
        last["rank"] = ranked["score"].rank(ascending=False, method="first")
        return last.sort_values("rank")

    def rebalance_plan(self, ctx: StrategyContext) -> Optional[RebalancePlan]:
        as_of = to_eastern(ctx.as_of)
        today = as_of.date()
        decision_day = _previous_session(today)
        if decision_day is None:
            logger.error("momentum_rank: no previous session before %s", today)
            return None

        dp = ctx.data_provider
        universe_df = dp.get_stock_universe()
        if universe_df is None or universe_df.empty:
            logger.error("momentum_rank: empty universe -- no rebalance")
            return None
        names = dict(zip(universe_df["symbol"],
                         universe_df.get("name", pd.Series([""] * len(universe_df)))))
        universe = sorted(s for s, n in names.items() if is_common_stock(s, n or ""))

        positions = [p for p in (ctx.positions_manager.positions or []) if not p.closed]
        mine = sorted(p.symbol for p in positions if p.strategy_name == self.name)
        held_any = {p.symbol for p in positions}

        start = datetime.combine(decision_day, datetime.min.time()) - timedelta(
            days=self.warmup_days())
        end = datetime.combine(decision_day, datetime.min.time()) + timedelta(hours=23)
        close = self._fetch_closes(dp, sorted(set(universe) | set(mine)), start, end)
        if close.empty:
            logger.error("momentum_rank: no bars -- no rebalance")
            return None
        close = close[close.index.date <= decision_day]  # never today's partial bar
        if close.index[-1].date() != decision_day:
            logger.error("momentum_rank: latest bar %s != decision session %s -- "
                         "data not ready, no rebalance", close.index[-1].date(), decision_day)
            return None

        # Holdings stay members while inside the price band: the live universe
        # is rebuilt daily from ONE day's volume, so a quiet session must not
        # force a sale (the backtest used a 63-day median dollar volume).
        lo = float(getattr(ctx.config, "MIN_PRICE", 0.0) or 0.0)
        hi = float(getattr(ctx.config, "MAX_PRICE", float("inf")) or float("inf"))
        last_px = close.iloc[-1]
        members = set(universe) | {s for s in mine
                                   if s in last_px.index and lo <= last_px[s] <= hi}
        table = self.rank_table(close, sorted(members))

        exits = []
        for sym in mine:
            row = table.loc[sym] if sym in table.index else None
            if row is None or not row["member"]:
                exits.append((sym, "rank_exit_universe"))
            elif not np.isfinite(row["rank"]):
                exits.append((sym, "rank_exit_no_score"))
            elif row["rank"] > self.buffer:
                exits.append((sym, f"rank_exit_{int(row['rank'])}"))

        keep = len(mine) - len(exits)
        free = max(0, self.slots - keep)
        cands = table[(table["trend"] == 1.0) & np.isfinite(table["rank"])
                      & ~table.index.isin(held_any)]
        entries = [
            LiveSignal(symbol=sym, direction="long", entry_price=float(row["close"]),
                       composite_score=float(row["score"]), strategy_name=self.name,
                       extra={"rank": int(row["rank"]), "score": float(row["score"])})
            for sym, row in cands.head(free + 3).iterrows()  # spares if a buy fails
        ]
        notes = {"decision_close": str(decision_day), "universe": len(universe),
                 "ranked": int(np.isfinite(table["rank"]).sum()),
                 "held": mine, "free_slots": free,
                 "top10": [(s, round(float(r["score"]), 2)) for s, r in table.head(10).iterrows()]}
        logger.info("momentum_rank plan (close %s): exits=%s entries(first %d of)=%s notes=%s",
                    decision_day, exits, free, [e.symbol for e in entries], notes)
        return RebalancePlan(strategy_name=self.name, exits=exits, entries=entries,
                             slots=self.slots, min_cash_pct=self.min_cash_pct, notes=notes)
