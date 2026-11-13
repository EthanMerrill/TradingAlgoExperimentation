#!/usr/bin/env python3
"""
Backtest-ranked momentum: pick stocks by how well ONE fixed trend rule
performed on each of them over the trailing 12-1 window.

Idea
----
RSI's live pipeline backtests every stock and enters the best results. This
does the same with a momentum rule, but with FIXED settings (no per-stock
optimization):

* Per-stock rule (fixed): long while ``close > SMA(sma_days)``. The signal is
  read at the close; trades happen at the next open.
* Score on decision day d: backtest that rule on each stock over the trailing
  window ``[d-lookback, d-skip]`` trading days (default 252 -> 21, i.e. 12-1)
  and take the Sharpe of its daily returns (net of switching costs). Only
  data up to d-skip is used, so there is no lookahead.
* Portfolio: each day, the stocks whose rule is currently long and that pass
  the price filter are ranked by score; free slots are filled (best first)
  at the next open under the LIVE portfolio rules (MAX_POSITIONS,
  MAX_NEW_POSITIONS_PER_DAY, POSITION_SIZE_PCT, MIN_CASH_PCT) via
  ``portfolio_backtest.simulate_portfolio``.
* Exit: follow the per-stock rule -- sell at the next open after the close
  drops below the SMA. Positions open at the end are closed at the last close.

Controls (identical eligibility, entries, exits, costs; only the RANKING
differs):

* ``momentum_12_1`` -- rank by the plain 12-1 return. If the backtest ranking
  does not beat this, the per-stock backtest adds nothing over momentum.
* ``random``        -- random ranking, many runs; the real result's
  percentile in that distribution.
* SPY buy-and-hold, and an equal-weight buy-and-hold of the universe.

Survivorship bias: the universe is TODAY's live RSI universe (active Alpaca
assets passing today's filters). Short windows are dominated by noise -- read
the random-ranking percentile before the headline return.

Read-only: fetches data, places no orders, writes CSVs under research/logs.

    python app/research/momentum_rank_backtest.py --test-months 3
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from portfolio_backtest import simulate_portfolio  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("momentum_rank_backtest")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_daily_panel(symbols: List[str], start: datetime, end: datetime,
                      chunk: int = 200) -> Dict[str, pd.DataFrame]:
    """Wide OHLCV panels (date x symbol), all-adjusted daily bars, batched."""
    from alpaca.data.enums import Adjustment  # pylint: disable=import-outside-toplevel
    from alpaca.data.requests import StockBarsRequest  # pylint: disable=import-outside-toplevel
    from alpaca.data.timeframe import TimeFrame  # pylint: disable=import-outside-toplevel
    from data_provider import data_provider  # pylint: disable=import-outside-toplevel

    frames = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            bars = data_provider.historical_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                start=start, end=end, adjustment=Adjustment.ALL))
            df = bars.df
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Batch %d failed: %s", i // chunk, e)
            continue
        if df is not None and not df.empty:
            frames.append(df)
        logger.info("Fetched %d/%d symbols", min(i + chunk, len(symbols)),
                    len(symbols))
    if not frames:
        return {}
    df = pd.concat(frames).reset_index()
    df["date"] = (pd.to_datetime(df["timestamp"], utc=True)
                  .dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize())
    return {col: df.pivot_table(index="date", columns="symbol", values=col)
            .sort_index() for col in ("open", "close", "volume")}


# ---------------------------------------------------------------------------
# Signal + scores (vectorized over the panel)
# ---------------------------------------------------------------------------

def rule_positions(close: pd.DataFrame, sma_days: int) -> pd.DataFrame:
    """1.0 where the per-stock rule is long at that day's close, else 0.0."""
    sma = close.rolling(sma_days, min_periods=sma_days).mean()
    return (close > sma).astype(float).where(sma.notna())


def backtest_scores(close: pd.DataFrame, pos: pd.DataFrame, lookback: int,
                    skip: int, cost_bps: float, min_days_long: int) -> pd.DataFrame:
    """Sharpe of the fixed rule per stock over the trailing [d-lookback, d-skip].

    Daily rule return on day t = pos[t-1] * r[t] - cost * |pos[t-1] - pos[t-2]|
    (decided at the prior close, cost charged when the position changes).
    """
    ret = close.pct_change(fill_method=None)
    held = pos.shift(1)
    switch = held.diff().abs()
    strat = held * ret - switch * (cost_bps / 1e4)
    window = lookback - skip
    mean = strat.rolling(window, min_periods=window).mean()
    std = strat.rolling(window, min_periods=window).std()
    days_long = held.rolling(window, min_periods=window).sum()
    sharpe = (mean / std) * np.sqrt(252.0)
    sharpe = sharpe.where((std > 0) & (days_long >= min_days_long))
    return sharpe.shift(skip)  # window ends `skip` days before d


def momentum_scores(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Plain 12-1 style return: close[d-skip] / close[d-lookback] - 1."""
    return close.shift(skip) / close.shift(lookback) - 1.0


# ---------------------------------------------------------------------------
# Candidate trades
# ---------------------------------------------------------------------------

def build_candidates(panel: Dict[str, pd.DataFrame], pos: pd.DataFrame,
                     test_start: pd.Timestamp, min_price: float,
                     max_price: float) -> pd.DataFrame:
    """Every eligible (symbol, decision day) -> trade with rule-based exit.

    Decision at the close of d (rule long, price filter), entry at the open
    of d+1, exit at the open after the first close below the SMA (x >= d+1).
    Trades not exited by the last bar close at the last close.
    """
    close, open_ = panel["close"], panel["open"]
    dates = close.index
    rows = []
    last = len(dates) - 1
    for sym in close.columns:
        c = close[sym].to_numpy()
        o = open_[sym].to_numpy()
        p = pos[sym].to_numpy()
        below = (p == 0.0)
        # next_below[i] = smallest x >= i with below[x], else -1
        next_below = np.full(len(dates), -1, dtype=int)
        nxt = -1
        for i in range(last, -1, -1):
            if below[i]:
                nxt = i
            next_below[i] = nxt
        for d in range(len(dates) - 1):
            if dates[d] < test_start or p[d] != 1.0:
                continue
            if not (min_price <= c[d] <= max_price):
                continue
            e = d + 1
            if not np.isfinite(o[e]) or o[e] <= 0:
                continue
            x = next_below[e]
            if x == -1 or x + 1 > last:
                exit_i, exit_px, reason = last, c[last], "window_end"
            else:
                exit_i, exit_px, reason = x + 1, o[x + 1], "rule_exit"
            if not np.isfinite(exit_px):
                continue
            rows.append((sym, dates[d], dates[e], float(o[e]),
                         dates[exit_i], float(exit_px), reason))
    return pd.DataFrame(rows, columns=["symbol", "decision_date", "entry_date",
                                       "entry_price", "exit_date", "exit_price",
                                       "exit_reason"])


def score_candidates(cands: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Attach the decision-day score; drop candidates without one."""
    stacked = scores.stack()
    keys = pd.MultiIndex.from_arrays([cands["decision_date"], cands["symbol"]])
    out = cands.assign(score=stacked.reindex(keys).to_numpy())
    return out[np.isfinite(out["score"])].reset_index(drop=True)


def prune(cands: pd.DataFrame, keep_per_day: int) -> List[dict]:
    """Keep only the top ``keep_per_day`` candidates per entry day.

    The simulator enters in descending score and at most
    MAX_NEW_POSITIONS_PER_DAY per day, skipping names already held (at most
    MAX_POSITIONS). So no candidate ranked below MAX_POSITIONS + MAX_NEW (+
    a small buffer for cash/zero-share skips) can ever be entered: dropping
    them leaves the result unchanged and makes long runs tractable.
    """
    top = (cands.sort_values(["entry_date", "score"], ascending=[True, False])
           .groupby("entry_date", sort=False).head(keep_per_day))
    return top.assign(side="long").to_dict("records")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _metrics(equity: pd.Series) -> dict:
    if len(equity) < 2:
        return {"total_return": np.nan, "cagr": np.nan, "max_drawdown": np.nan,
                "sharpe": np.nan}
    daily = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    return {
        "total_return": total,
        "cagr": float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0.5 else np.nan,
        "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
        "sharpe": (float(daily.mean() / daily.std() * np.sqrt(252))
                   if daily.std() > 0 else np.nan),
    }


def _equity(cands: List[dict], cash, cfg, prices, cost_bps, anchor, dates):
    """Daily equity on the full test calendar (flat cash before 1st entry)."""
    res = simulate_portfolio(cands, cash, cfg, prices, cost_bps_per_side=cost_bps)
    eq = res.equity
    eq = eq[~eq.index.duplicated(keep="last")]
    curve = eq.reindex(dates).ffill().fillna(cash)
    curve = pd.concat([pd.Series([cash], index=[anchor]), curve])
    return curve, res


def _periods(curve: pd.Series, splits) -> Dict[str, pd.Series]:
    """Sub-curves: each calendar year plus the named splits."""
    out = {}
    for year in sorted(set(curve.index.year)):
        prev = curve[curve.index.year < year]
        seg = curve[curve.index.year == year]
        if len(prev):
            seg = pd.concat([prev.iloc[-1:], seg])
        out[str(year)] = seg
    for name, (a, b) in splits.items():
        prev = curve[curve.index < a]
        seg = curve[(curve.index >= a) & (curve.index <= b)]
        if len(prev):
            seg = pd.concat([prev.iloc[-1:], seg])
        out[name] = seg
    return out


def _beta(curve: pd.Series, spy: pd.Series) -> float:
    j = pd.concat([curve.pct_change(), spy.pct_change()], axis=1).dropna()
    if len(j) < 20 or j.iloc[:, 1].var() == 0:
        return np.nan
    return float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--test-start", default=None,
                    help="YYYY-MM-DD (default: --test-months before --end)")
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--holdout-start", default=None,
                    help="YYYY-MM-DD: report selection (before) vs holdout "
                         "(from) periods separately.")
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--skip", type=int, default=21)
    ap.add_argument("--sma-days", type=int, nargs="+", default=[50])
    ap.add_argument("--min-days-long", type=int, default=40,
                    help="Min in-position days in the scoring window.")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Per side.")
    ap.add_argument("--random-runs", type=int, default=200)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--min-cash-pct", type=float, default=0.10,
                    help="Cash floor (default 0.10 = prod).")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from config import globalConfig  # type: ignore  # pylint: disable=import-outside-toplevel
    from data_provider import data_provider  # pylint: disable=import-outside-toplevel

    end = datetime.strptime(a.end, "%Y-%m-%d") if a.end else datetime.now()
    test_start = pd.Timestamp(
        datetime.strptime(a.test_start, "%Y-%m-%d") if a.test_start
        else end - timedelta(days=int(a.test_months * 30.4))).normalize()
    warm = a.lookback + max(a.sma_days)
    fetch_start = test_start - timedelta(days=int(warm * 1.5) + 30)

    universe = data_provider.get_stock_universe()
    symbols = sorted(universe["symbol"].tolist()) if not universe.empty else []
    if a.max_symbols:
        symbols = symbols[:a.max_symbols]
    logger.info("Universe: %d symbols (today's live RSI universe)", len(symbols))
    panel = fetch_daily_panel(symbols + ["SPY"], fetch_start, end)
    if not panel:
        print("No data.")
        return 1
    spy_all = panel["close"].pop("SPY") if "SPY" in panel["close"] else None
    for k in ("open", "volume"):
        panel[k] = panel[k].drop(columns=["SPY"], errors="ignore")
    close = panel["close"]
    all_dates = close.index
    test_start = all_dates[all_dates >= test_start][0]
    anchor = all_dates[all_dates < test_start][-1]
    test_dates = all_dates[all_dates >= test_start]

    cfg = SimpleNamespace(
        MAX_POSITIONS=globalConfig.MAX_POSITIONS,
        MAX_NEW_POSITIONS_PER_DAY=globalConfig.MAX_NEW_POSITIONS_PER_DAY,
        POSITION_SIZE_PCT=globalConfig.POSITION_SIZE_PCT,
        MIN_CASH_PCT=a.min_cash_pct)
    keep = cfg.MAX_POSITIONS + cfg.MAX_NEW_POSITIONS_PER_DAY + 3
    prices = {s: close[s].dropna() for s in close.columns}

    splits = {}
    if a.holdout_start:
        h = pd.Timestamp(a.holdout_start)
        splits = {"SELECTION": (test_start, h - pd.Timedelta(days=1)),
                  "HOLDOUT": (h, test_dates[-1])}
    splits["FULL"] = (test_start, test_dates[-1])

    spy = spy_all.loc[anchor:] if spy_all is not None else None
    ew = close.loc[anchor:].pct_change().mean(axis=1).fillna(0.0)
    ew_curve = a.cash * (1.0 + ew).cumprod()
    spy_curve = a.cash * spy / spy.iloc[0] if spy is not None else None

    rng = np.random.default_rng(a.seed)
    rows, period_rows, all_trades = [], [], []
    for sma in a.sma_days:
        logger.info("SMA%d: building candidates…", sma)
        pos = rule_positions(close, sma)
        cands = build_candidates(panel, pos, test_start, globalConfig.MIN_PRICE,
                                 globalConfig.MAX_PRICE)
        bt = score_candidates(cands, backtest_scores(
            close, pos, a.lookback, a.skip, a.cost_bps, a.min_days_long))
        mom = score_candidates(cands, momentum_scores(close, a.lookback, a.skip))

        curves = {}
        curves["backtest score"], res = _equity(
            prune(bt, keep), a.cash, cfg, prices, a.cost_bps, anchor, test_dates)
        curves["plain 12-1"], _ = _equity(
            prune(mom, keep), a.cash, cfg, prices, a.cost_bps, anchor, test_dates)
        trades = res.trades_taken.assign(sma=sma)
        all_trades.append(trades)

        logger.info("SMA%d: %d random rankings…", sma, a.random_runs)
        rand_curves = []
        for _ in range(a.random_runs):
            shuffled = bt.assign(score=rng.random(len(bt)))
            curve, _ = _equity(prune(shuffled, keep), a.cash, cfg, prices,
                               a.cost_bps, anchor, test_dates)
            rand_curves.append(curve)

        def summarize(label, curve, rand=None):
            for period, seg in _periods(curve, splits).items():
                m = _metrics(seg)
                row = {"sma": sma, "ranking": label, "period": period, **m}
                if rand is not None:
                    dist = np.array([_metrics(_periods(rc, splits)[period])
                                     ["total_return"] for rc in rand])
                    row["rand_median"] = float(np.nanmedian(dist))
                    row["pctile_vs_random"] = float(
                        (dist < m["total_return"]).mean() * 100)
                period_rows.append(row)

        summarize("backtest score", curves["backtest score"], rand_curves)
        summarize("plain 12-1", curves["plain 12-1"], rand_curves)
        full_bt = _metrics(curves["backtest score"])
        rows.append({
            "sma": sma, "trades": len(trades),
            "hit": float((trades["pnl"] > 0).mean()) if len(trades) else np.nan,
            "med_hold_days": float((trades["exit_date"] - trades["entry_date"])
                                   .dt.days.median()) if len(trades) else np.nan,
            "beta": _beta(curves["backtest score"], spy) if spy is not None else np.nan,
            **{k: full_bt[k] for k in ("cagr", "sharpe", "max_drawdown")},
        })

    for label, curve in (("SPY buy&hold", spy_curve), ("EW universe", ew_curve)):
        if curve is None:
            continue
        for period, seg in _periods(curve, splits).items():
            period_rows.append({"sma": "-", "ranking": label, "period": period,
                                **_metrics(seg)})

    table = pd.DataFrame(period_rows)
    print("=" * 96)
    print("BACKTEST-RANKED MOMENTUM (fixed rule, per-stock backtest score)")
    print("=" * 96)
    print(f"  Test window : {test_start.date()} -> {test_dates[-1].date()} "
          f"({len(test_dates)} sessions)" + (
              f"; holdout from {a.holdout_start}" if a.holdout_start else ""))
    print(f"  Universe    : {close.shape[1]} symbols (today's live universe; "
          f"SURVIVORSHIP-BIASED - absolute returns overstated)")
    print(f"  Rule        : long while close > SMA(n), n in {a.sma_days}; "
          f"score = rule Sharpe over [d-{a.lookback}, d-{a.skip}]")
    print(f"  Portfolio   : max {cfg.MAX_POSITIONS} positions, "
          f"{cfg.MAX_NEW_POSITIONS_PER_DAY} new/day, "
          f"{cfg.POSITION_SIZE_PCT:.0%} each, {cfg.MIN_CASH_PCT:.0%} min cash, "
          f"{a.cost_bps:.0f} bps/side; {a.random_runs} random rankings per SMA")

    print("\n--- Full-period summary (backtest-score ranking) ---")
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "hit": "{:.0%}".format, "cagr": "{:+.1%}".format,
        "sharpe": "{:.2f}".format, "max_drawdown": "{:+.1%}".format,
        "beta": "{:.2f}".format, "med_hold_days": "{:.0f}".format}))

    fmt = lambda v: "" if pd.isna(v) else f"{v:+.1%}"  # noqa: E731
    pivot = table.pivot_table(index=["ranking", "sma"], columns="period",
                              values="total_return", aggfunc="first")
    order = [c for c in pivot.columns if c.isdigit()] + [
        c for c in ("SELECTION", "HOLDOUT", "FULL") if c in pivot.columns]
    print("\n--- Total return by period ---")
    print(pivot[order].to_string(formatters={c: fmt for c in order}))

    sharpe = table.pivot_table(index=["ranking", "sma"], columns="period",
                               values="sharpe", aggfunc="first")
    cols = [c for c in ("SELECTION", "HOLDOUT", "FULL") if c in sharpe.columns]
    print("\n--- Sharpe by split ---")
    print(sharpe[cols].to_string(float_format=lambda v: f"{v:.2f}"))

    rel = table[table["pctile_vs_random"].notna()].pivot_table(
        index=["ranking", "sma"], columns="period", values="pctile_vs_random",
        aggfunc="first")
    print("\n--- Percentile vs random rankings (same rule; >90 = ranking adds value) ---")
    print(rel[order].to_string(float_format=lambda v: f"{v:.0f}"))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(out_dir, f"momentum_rank_{stamp}")
    pd.concat(all_trades).to_csv(prefix + "_trades.csv", index=False)
    table.to_csv(prefix + "_periods.csv", index=False)
    print(f"\nWrote: {prefix}_trades.csv, _periods.csv")
    print("WARNING: survivorship bias - universe is today's survivors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
