"""Portfolio-level, walk-forward-honest backtest of the live selection pipeline.

Pipeline (mirrors what is traded live, minus look-ahead):
  1. For each rolling window, optimise params per symbol on the IN-SAMPLE slice
     only, apply the live entry filters (profitable, alpha>0, win_rate>=MIN_WIN_RATE,
     num_trades>=MIN_NUM_TRADES) to the IS result, and rank survivors by a
     cross-symbol composite z-score computed from IS results only.
  2. Backtest the chosen params on the OOS slice (with IS bars as indicator
     warmup) and keep only trades ENTERED inside the OOS window; trades still
     open at the window's last bar are closed there (``close_open_at_end``).
  3. Replay all OOS trades through portfolio rules (MAX_POSITIONS,
     MAX_NEW_POSITIONS_PER_DAY, POSITION_SIZE_PCT of equity, MIN_CASH_PCT).
  4. Placebo: same pipeline with params drawn uniformly from the grid; the real
     pipeline's Sharpe/CAGR is located in the placebo distribution.

Limitations (read before trusting numbers):
  * SURVIVORSHIP BIAS: the live universe is today's active Alpaca assets filtered
    by today's price / market cap. Delisted names are absent unless
    ``--include-delisted`` is used, and even then historical market-cap and price
    filters are NOT reproduced.
  * Without per-symbol daily closes, positions are marked at entry price until
    exit (equity is flat while held, so drawdown/Sharpe are understated).
  * Window-end exits: a position still open on an OOS window's last bar is
    force-closed at that bar's close (exit_reason 'window_end') with the usual
    slippage/costs. Conservative: adds an exit cost and may cut winners short.
  * A candidate skipped for capacity is lost, not deferred to a later day.
"""
import argparse
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

SURVIVORSHIP_WARNING = (
    "WARNING: survivorship bias - symbols are drawn from today's survivors; "
    "historical price/market-cap universe filters are not reproduced.")
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD",
                   "NFLX", "JPM", "XOM", "BA", "INTC"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research", "data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _day(ts: Any) -> pd.Timestamp:
    """Normalise to a tz-naive midnight Timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def _default_config() -> Any:
    from config import globalConfig  # type: ignore  # pylint: disable=import-outside-toplevel
    return globalConfig


def _norm_trade(t: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a trade dict (old or new rsi.py key sets) to the sim schema."""
    side = str(t.get("side") or t.get("direction") or "long").lower()
    return {
        "symbol": t["symbol"], "entry_date": _day(t["entry_date"]),
        "exit_date": _day(t["exit_date"]), "entry_price": float(t["entry_price"]),
        "exit_price": float(t["exit_price"]),
        "side": "short" if side in ("short", "sell") else "long",
        "score": float(t.get("score", 0.0) or 0.0),
        "exit_reason": t.get("exit_reason", ""),
    }


# ---------------------------------------------------------------------------
# 1. Portfolio simulator
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    equity: pd.Series
    trades_taken: pd.DataFrame
    trades_skipped: pd.DataFrame
    invested: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: Dict[str, float] = field(default_factory=dict)

    def __getattr__(self, name: str) -> float:  # cagr, sharpe, max_drawdown, ...
        m = self.__dict__.get("metrics", {})
        if name in m:
            return m[name]
        raise AttributeError(name)


def _metrics(equity: pd.Series, invested: pd.Series, taken: pd.DataFrame,
             traded_notional: float, rf: float) -> Dict[str, float]:
    out = dict(cagr=0.0, sharpe=0.0, max_drawdown=0.0, exposure=0.0, turnover=0.0,
               win_rate=0.0, total_return=0.0, num_trades=float(len(taken)))
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return out
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    out["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1)
    ratio = equity.iloc[-1] / equity.iloc[0]
    out["cagr"] = float(ratio ** (1 / years) - 1) if ratio > 0 else -1.0
    rets = equity.pct_change().dropna()
    if len(rets) > 1 and rets.std() > 0:
        out["sharpe"] = float(math.sqrt(252) * (rets.mean() - rf / 252) / rets.std())
    out["max_drawdown"] = float(abs((equity / equity.cummax() - 1).min()))
    out["exposure"] = float((invested / equity).mean())
    out["turnover"] = float(traded_notional / 2 / equity.mean() / years)
    if len(taken):
        out["win_rate"] = float((taken["pnl"] > 0).mean())
    return out


def simulate_portfolio(candidate_trades: Sequence[Dict[str, Any]], initial_cash: float,
                       config: Any = None, prices: Optional[Dict[str, pd.Series]] = None,
                       cost_bps_per_side: float = 0.0, risk_free: float = 0.02) -> PortfolioResult:
    """Replay candidate trades under live portfolio rules.

    Each trade: symbol, entry_date, exit_date, entry_price, exit_price, side,
    score. Each day: (1) exits, cash returns; (2) entries in descending score,
    one position per symbol, MAX_POSITIONS / MAX_NEW_POSITIONS_PER_DAY caps,
    size = POSITION_SIZE_PCT * current equity in whole shares, reduced so cash
    stays >= MIN_CASH_PCT * equity; (3) same-day round trips close at day end.
    Shorts reserve notional as cash and mark as notional + shares*(entry-px).

    Marking: with ``prices`` (symbol -> close Series) open positions are marked
    at the last close <= date; otherwise at entry price until exit.
    Skip reasons: duplicate_symbol, max_positions, max_new_per_day,
    insufficient_cash, zero_shares.
    """
    cfg = config if config is not None else _default_config()
    trades = [_norm_trade(t) for t in candidate_trades]
    cost = cost_bps_per_side / 1e4
    cols = ["symbol", "side", "entry_date", "exit_date", "entry_price", "exit_price",
            "shares", "score", "pnl", "return_pct", "exit_reason"]
    if not trades:
        return PortfolioResult(pd.Series(dtype=float), pd.DataFrame(columns=cols),
                               pd.DataFrame(columns=["symbol", "entry_date", "score", "reason"]))

    px = {s: pd.Series(v).copy() for s, v in (prices or {}).items()}
    for s, v in px.items():
        v.index = pd.DatetimeIndex([_day(i) for i in v.index])
        px[s] = v[~v.index.duplicated(keep="last")].sort_index()

    dates = {t["entry_date"] for t in trades} | {t["exit_date"] for t in trades}
    for v in px.values():
        dates |= set(v.index)
    lo = min(t["entry_date"] for t in trades)
    timeline = sorted(d for d in dates if d >= lo)
    by_entry: Dict[pd.Timestamp, List[dict]] = {}
    for t in trades:
        by_entry.setdefault(t["entry_date"], []).append(t)

    cash = float(initial_cash)
    open_pos: Dict[str, dict] = {}
    taken: List[dict] = []
    skipped: List[dict] = []
    eq_vals: List[float] = []
    inv_vals: List[float] = []
    traded = 0.0

    def mark(sym: str, p: dict, d: pd.Timestamp) -> float:
        last = p["entry_price"]
        if sym in px:
            s = px[sym].loc[:d]
            if len(s):
                last = float(s.iloc[-1])
        sign = 1 if p["side"] == "long" else -1
        return p["shares"] * p["entry_price"] + sign * p["shares"] * (last - p["entry_price"])

    def close(sym: str) -> None:
        nonlocal cash, traded
        p = open_pos.pop(sym)
        sign = 1 if p["side"] == "long" else -1
        notional_in = p["shares"] * p["entry_price"]
        gross = sign * p["shares"] * (p["exit_price"] - p["entry_price"])
        fees = cost * (notional_in + p["shares"] * p["exit_price"])
        pnl = gross - fees
        cash += notional_in + pnl
        traded += p["shares"] * p["exit_price"]
        taken.append({**{k: p[k] for k in ("symbol", "side", "entry_date", "exit_date",
                                           "entry_price", "exit_price", "shares", "score",
                                           "exit_reason")},
                      "pnl": pnl, "return_pct": pnl / notional_in})

    for d in timeline:
        for sym in [s for s, p in open_pos.items() if p["exit_date"] <= d]:
            close(sym)
        equity = cash + sum(mark(s, p, d) for s, p in open_pos.items())
        new_today = 0
        for t in sorted(by_entry.get(d, []), key=lambda x: -x["score"]):
            reason = None
            if t["symbol"] in open_pos:
                reason = "duplicate_symbol"
            elif len(open_pos) >= cfg.MAX_POSITIONS:
                reason = "max_positions"
            elif new_today >= cfg.MAX_NEW_POSITIONS_PER_DAY:
                reason = "max_new_per_day"
            else:
                shares = int(cfg.POSITION_SIZE_PCT * equity // t["entry_price"])
                if shares < 1:
                    reason = "zero_shares"
                else:
                    spendable = cash - cfg.MIN_CASH_PCT * equity
                    shares = min(shares, int(spendable // (t["entry_price"] * (1 + cost))))
                    if shares < 1:
                        reason = "insufficient_cash"
            if reason:
                skipped.append({"symbol": t["symbol"], "entry_date": d,
                                "score": t["score"], "reason": reason})
                continue
            cash -= shares * t["entry_price"]
            traded += shares * t["entry_price"]
            open_pos[t["symbol"]] = {**t, "shares": shares}
            new_today += 1
        for sym in [s for s, p in open_pos.items() if p["exit_date"] <= d]:  # same-day round trips
            close(sym)
        inv = sum(mark(s, p, d) for s, p in open_pos.items())
        eq_vals.append(cash + inv)
        inv_vals.append(inv)

    idx = pd.DatetimeIndex(timeline)
    equity_s = pd.Series(eq_vals, index=idx, name="equity")
    inv_s = pd.Series(inv_vals, index=idx, name="invested")
    taken_df = pd.DataFrame(taken, columns=cols)
    skipped_df = pd.DataFrame(skipped, columns=["symbol", "entry_date", "score", "reason"])
    return PortfolioResult(equity_s, taken_df, skipped_df, inv_s,
                           _metrics(equity_s, inv_s, taken_df, traded, risk_free))


# ---------------------------------------------------------------------------
# 2. Walk-forward candidate generation
# ---------------------------------------------------------------------------

def _windows(start: datetime, end: datetime, is_m: int, oos_m: int, step_m: int
             ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """(IS_start, IS_end, OOS_start, OOS_end); same 30-day-month convention as
    WalkForwardValidator._compute_window_boundaries (which reads fixed config
    values, so it cannot be called with explicit month arguments)."""
    out, cur = [], start
    while True:
        is_end = cur + timedelta(days=is_m * 30)
        oos_end = is_end + timedelta(days=oos_m * 30)
        if oos_end > end:
            return out
        out.append((cur, is_end, is_end, oos_end))
        cur += timedelta(days=step_m * 30)


def _passes_filters(res: Any, cfg: Any) -> bool:
    return bool(res is not None and getattr(res, "profitable", True) and res.alpha > 0
                and res.win_rate >= cfg.MIN_WIN_RATE and res.num_trades >= cfg.MIN_NUM_TRADES)


def _slice(data: pd.DataFrame, a: datetime, b: datetime) -> pd.DataFrame:
    from walk_forward import WalkForwardValidator  # pylint: disable=import-outside-toplevel
    return WalkForwardValidator._slice_bars(data, a, b)  # pylint: disable=protected-access


def _generate(symbols: Sequence[str], data_by_symbol: Dict[str, pd.DataFrame],
              strategy_factory: Callable[[], Any], is_months: int, oos_months: int,
              step_months: int, chooser: str, rng: Optional[random.Random],
              config: Any = None, direction: str = "long",
              close_open_at_end: bool = True) -> List[Dict[str, Any]]:
    cfg = config if config is not None else _default_config()
    strat = strategy_factory()
    warm = timedelta(days=int(strat.warmup_days() * 1.5) + 5) if hasattr(strat, "warmup_days") else timedelta(days=60)
    starts = [d.index.min() for d in data_by_symbol.values() if d is not None and len(d)]
    ends = [d.index.max() for d in data_by_symbol.values() if d is not None and len(d)]
    if not starts:
        return []
    wins = _windows(_day(min(starts)).to_pydatetime(), _day(max(ends)).to_pydatetime(),
                    is_months, oos_months, step_months)
    grid = strat.get_param_grid(direction) if chooser == "random" else []
    out: List[Dict[str, Any]] = []
    prev_oos_end: Optional[datetime] = None
    for is_s, is_e, oos_s, oos_e in wins:
        picks: List[Tuple[str, Any]] = []
        for sym in symbols:
            data = data_by_symbol.get(sym)
            if data is None or data.empty:
                continue
            is_data = _slice(data, is_s, is_e - timedelta(days=1))  # IS strictly before OOS
            if len(is_data) < 30:
                continue
            if chooser == "best":
                res = strat.optimize(is_data, sym, direction)
            else:
                assert rng is not None
                res = strat.backtest(is_data, sym, **dict(rng.choice(grid)))
            if _passes_filters(res, cfg):
                picks.append((sym, res))
        if not picks:
            continue
        # Cross-symbol composite score from IS results only (as live).
        import zscore  # pylint: disable=import-outside-toplevel
        zscore.compute_cross_symbol_zscores([r for _, r in picks])
        eff_start = oos_s if prev_oos_end is None else max(oos_s, prev_oos_end)
        prev_oos_end = oos_e
        for sym, res in picks:
            data = data_by_symbol[sym]
            # IS bars = warmup only. With forced window-end exits the slice stops the
            # day BEFORE oos_e so the forced close is strictly before the next
            # window's first allowed entry (>= oos_e): no same-symbol overlap.
            oos_last = oos_e - timedelta(days=1) if close_open_at_end else oos_e
            oos_data = _slice(data, oos_s - warm, oos_last)
            params = {k: v for k, v in res.params.items()}
            if close_open_at_end:
                bt = strat.backtest(oos_data, sym, close_open_at_end=True, **params)
            else:
                bt = strat.backtest(oos_data, sym, **params)
            for t in (bt.trade_details or []):
                ed = _day(t["entry_date"])
                if eff_start <= ed < _day(oos_e):
                    out.append(_norm_trade({**t, "symbol": sym, "score": res.composite_score,
                                            "side": t.get("side") or t.get("direction") or direction}))
    return out


def generate_candidates_walk_forward(symbols: Sequence[str], data_by_symbol: Dict[str, pd.DataFrame],
                                     strategy_factory: Callable[[], Any], is_months: int,
                                     oos_months: int, step_months: int, config: Any = None,
                                     direction: str = "long",
                                     close_open_at_end: bool = True) -> List[Dict[str, Any]]:
    """OOS-only candidate trades; params and score come from IS data alone.

    Overlapping OOS windows (step < oos) are de-duplicated: a later window's
    params apply only from the previous window's OOS end. Positions open at the
    end of a window are closed there at the last close (exit_reason
    'window_end'); ``close_open_at_end=False`` restores the old drop behaviour.
    """
    return _generate(symbols, data_by_symbol, strategy_factory, is_months, oos_months,
                     step_months, "best", None, config, direction, close_open_at_end)


# ---------------------------------------------------------------------------
# 3. Placebo
# ---------------------------------------------------------------------------

def _percentile(value: float, dist: Sequence[float]) -> float:
    """Mid-rank percentile (0-100) of value within dist."""
    if not len(dist):
        return float("nan")
    d = np.asarray(dist, dtype=float)
    return float(100.0 * ((d < value).sum() + 0.5 * (d == value).sum()) / len(d))


def placebo_test(symbols: Sequence[str], data_by_symbol: Dict[str, pd.DataFrame],
                 strategy_factory: Callable[[], Any], is_months: int, oos_months: int,
                 step_months: int, initial_cash: float = 100_000.0, n_runs: int = 20,
                 seed: Optional[int] = None, config: Any = None, direction: str = "long",
                 prices: Optional[Dict[str, pd.Series]] = None,
                 real: Optional[PortfolioResult] = None) -> Dict[str, Any]:
    """Compare real (best-IS) selection to random-param selection.

    Placebo runs use the same filters, scoring, windows and portfolio rules but
    draw each (symbol, window) param set uniformly from the grid. Returns the
    real and placebo Sharpe/CAGR plus the real value's percentile in each
    distribution. A percentile near or below 50-ish means optimisation adds
    no demonstrable value over random parameters.
    """
    cfg = config if config is not None else _default_config()
    if real is None:
        cands = generate_candidates_walk_forward(symbols, data_by_symbol, strategy_factory,
                                                 is_months, oos_months, step_months, cfg, direction)
        real = simulate_portfolio(cands, initial_cash, cfg, prices)
    rng = random.Random(seed)
    sharpes: List[float] = []
    cagrs: List[float] = []
    for _ in range(n_runs):
        c = _generate(symbols, data_by_symbol, strategy_factory, is_months, oos_months,
                      step_months, "random", rng, cfg, direction)
        r = simulate_portfolio(c, initial_cash, cfg, prices)
        sharpes.append(r.metrics.get("sharpe", 0.0))
        cagrs.append(r.metrics.get("cagr", 0.0))
    return {
        "real_sharpe": real.metrics.get("sharpe", 0.0), "real_cagr": real.metrics.get("cagr", 0.0),
        "placebo_sharpes": sharpes, "placebo_cagrs": cagrs,
        "sharpe_percentile": _percentile(real.metrics.get("sharpe", 0.0), sharpes),
        "cagr_percentile": _percentile(real.metrics.get("cagr", 0.0), cagrs),
        "n_runs": n_runs, "seed": seed, "real": real,
    }


# ---------------------------------------------------------------------------
# 4/5. CLI
# ---------------------------------------------------------------------------

def _delisted_symbols() -> List[str]:
    """Inactive US-equity tickers from Alpaca (raw JSON; tolerant of unknown enums)."""
    from data_provider import data_provider  # pylint: disable=import-outside-toplevel
    client = data_provider.trading_client
    if client is None:
        return []
    try:
        rows = client.get("/assets", {"status": "inactive", "asset_class": "us_equity"})
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Inactive asset fetch failed: %s", e)
        return []
    return [r["symbol"] for r in rows if isinstance(r, dict) and r.get("symbol")]


def _fetch(symbols: List[str], start: datetime, end: datetime) -> Dict[str, pd.DataFrame]:
    from data_provider import data_provider  # pylint: disable=import-outside-toplevel
    warm_start = start - timedelta(days=120)
    out = {}
    for s in symbols:
        df = data_provider.get_single_stock_bars(s, warm_start, end)
        if df is not None and not df.empty:
            out[s] = df
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward portfolio backtest + placebo test")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--placebo", type=int, default=0, help="placebo runs (0 = skip)")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--include-delisted", action="store_true")
    ap.add_argument("--is-months", type=int, default=None)
    ap.add_argument("--oos-months", type=int, default=None)
    ap.add_argument("--step-months", type=int, default=None)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output path prefix (default app/research/data/portfolio_bt_<date>)")
    a = ap.parse_args(argv)

    from config import globalConfig  # type: ignore  # pylint: disable=import-outside-toplevel
    from strategies.rsi import RSIStrategy  # pylint: disable=import-outside-toplevel
    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = list(DEFAULT_SYMBOLS)
        print(f"No --symbols given; using default list of {len(symbols)} large caps.")
    print(SURVIVORSHIP_WARNING)
    if a.include_delisted:
        extra = _delisted_symbols()
        print(f"--include-delisted: {len(extra)} inactive assets; kept only those with bar data. "
              "Historical market-cap/price filters are NOT applied.")
        symbols += [s for s in extra if s not in symbols]
    if a.max_symbols:
        symbols = symbols[:a.max_symbols]
    start, end = datetime.fromisoformat(a.start), datetime.fromisoformat(a.end)
    data = _fetch(symbols, start, end)
    print(f"Fetched bars for {len(data)}/{len(symbols)} symbols.")
    is_m = a.is_months or globalConfig.WF_IS_MONTHS
    oos_m = a.oos_months or globalConfig.WF_OOS_MONTHS
    step_m = a.step_months or globalConfig.WF_STEP_MONTHS
    factory = RSIStrategy.create
    prices = {s: d[RSIStrategy._get_price_column(d)] for s, d in data.items()}  # pylint: disable=protected-access

    cands = generate_candidates_walk_forward(list(data), data, factory, is_m, oos_m, step_m)
    real = simulate_portfolio(cands, a.cash, globalConfig, prices)
    m = real.metrics
    print(f"\n{'metric':<14}{'value':>12}")
    for k in ("total_return", "cagr", "sharpe", "max_drawdown", "exposure", "turnover", "win_rate"):
        print(f"{k:<14}{m.get(k, 0.0):>12.3f}")
    print(f"{'trades taken':<14}{len(real.trades_taken):>12}\n{'trades skipped':<14}{len(real.trades_skipped):>12}")
    if len(real.trades_skipped):
        print(real.trades_skipped["reason"].value_counts().to_string())

    if a.placebo > 0:
        p = placebo_test(list(data), data, factory, is_m, oos_m, step_m, a.cash, a.placebo,
                         a.seed, globalConfig, prices, real=real)
        print(f"\nPlacebo ({a.placebo} runs, seed={a.seed}): real Sharpe {p['real_sharpe']:.2f} is at "
              f"percentile {p['sharpe_percentile']:.0f}; real CAGR {p['real_cagr']:.1%} at "
              f"percentile {p['cagr_percentile']:.0f}. Placebo Sharpe mean "
              f"{np.mean(p['placebo_sharpes']):.2f}. Not clearly above ~90 => optimisation adds no proven value.")

    prefix = a.out or os.path.join(OUT_DIR, f"portfolio_bt_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    real.trades_taken.to_csv(prefix + "_trades.csv", index=False)
    real.trades_skipped.to_csv(prefix + "_skipped.csv", index=False)
    pd.DataFrame({"equity": real.equity, "invested": real.invested}).to_csv(prefix + "_equity.csv")
    print(f"\nWrote {prefix}_trades.csv, _skipped.csv, _equity.csv")
    print(SURVIVORSHIP_WARNING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
