#!/usr/bin/env python3
"""
Backtest-ranked momentum, portfolio-construction variants (pre-declared grid).

Score (fixed, the only ranking that beat random on the point-in-time
universe): Sharpe of the per-stock rule ``long while close > SMA(50)`` over
the trailing [d-252, d-21] window (see momentum_rank_backtest).

Grid (12 variants, declared before looking at results):

* exit   -- ``sma``:  sell at the next open after the close drops below SMA50;
                      entries any day, at most MAX_NEW_POSITIONS_PER_DAY.
            ``rank``: monthly rebalance; sell holdings that left the top
                      ``--buffer`` ranks (or the universe); fill free slots
                      with the best-ranked names whose rule is long.
* slots  -- 3, 5, 10 positions, each (1 - min cash) / slots of equity.
* crash  -- off, or ``spy200``: no entries and everything sold while SPY
            closes below its 200-day SMA.

Protocol: pick the variant with the highest SELECTION (2019-2023) CAGR among
those at >= 90th percentile vs random rankings in SELECTION; report its
VALIDATION (first scorable day .. 2018-12-31, never examined before) once.
2024+ ("SEEN") was looked at in earlier runs and cannot validate anything.

Universe: point-in-time common stocks (active + delisted listings), price band
+ top-N 63d median dollar volume. A holding whose data stops for more than
``GAP_DAYS`` sessions (delisted / acquired / bankrupt, or a ticker later
reused) is sold at its last close.

Decisions at the close of day i execute at the open of i+1; marks use the last
close. Costs per side on both legs. Read-only research: places no orders.

    python app/research/momentum_rank_portfolio.py --cache <pit_panel.pkl>
"""
import argparse
import itertools
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from momentum_rank_backtest import (  # noqa: E402
    asset_names, backtest_scores, is_common_stock, load_panel, pit_symbols,
    pit_universe, rule_positions)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("momentum_rank_portfolio")

GAP_DAYS = 5


def simulate(close: np.ndarray, open_: np.ndarray, member: np.ndarray,
             trend: np.ndarray, score: np.ndarray, regime_ok: np.ndarray,
             rebalance: np.ndarray, start_i: int, slots: int, exit_mode: str,
             buffer: int, cash0: float = 100_000.0, min_cash: float = 0.10,
             cost_bps: float = 5.0, max_new: int = 2, symbols: Optional[List[str]] = None,
             dates: Optional[pd.DatetimeIndex] = None, pre: Optional[dict] = None):
    """Daily portfolio simulation on (date x symbol) arrays.

    Returns (equity array over [start_i, n), list of trade dicts).
    """
    n, m = close.shape
    cost = cost_bps / 1e4
    size = (1.0 - min_cash) / slots
    pre = pre or precompute(close)
    finite, nxt, closef = pre["finite"], pre["nxt"], pre["closef"]

    cash = cash0
    hold: Dict[int, list] = {}          # j -> [shares, entry_px, entry_i]
    sells: Dict[int, str] = {}          # pending sells at next open: j -> reason
    buys: List[int] = []
    eq = np.empty(n - start_i)
    trades = []

    def close_pos(j, px, i, reason):
        nonlocal cash
        sh, epx, ei = hold.pop(j)
        cash += sh * px * (1 - cost)
        trades.append({"symbol": symbols[j] if symbols else j,
                       "entry_date": dates[ei] if dates is not None else ei,
                       "exit_date": dates[i] if dates is not None else i,
                       "entry_price": epx, "exit_price": px, "shares": sh,
                       "return_pct": px * (1 - cost) / (epx * (1 + cost)) - 1.0,
                       "exit_reason": reason})

    prev_eq = cash0
    for i in range(start_i, n):
        # 1) execute yesterday's decisions at today's open
        for j in list(sells):
            px = open_[i, j]
            if np.isfinite(px) and px > 0:
                close_pos(j, px, i, sells.pop(j))
        for j in buys:
            if len(hold) >= slots or j in hold:
                continue
            px = open_[i, j]
            if not (np.isfinite(px) and px > 0):
                continue
            amount = min(size * prev_eq, cash - min_cash * prev_eq)
            sh = int(amount // (px * (1 + cost)))
            if sh >= 1:
                cash -= sh * px * (1 + cost)
                hold[j] = [sh, px, i]
        buys = []
        # 2) data stops (delisting / ticker reuse gap): sell at the last close
        for j in list(hold):
            if finite[i, j] and nxt[i + 1, j] > i + GAP_DAYS and i + GAP_DAYS < n - 1:
                sells.pop(j, None)
                close_pos(j, close[i, j], i, "data_end")
        # 3) mark
        value = cash + sum(h[0] * closef[i, j] for j, h in hold.items())
        eq[i - start_i] = value
        prev_eq = value
        if i == n - 1:
            break
        # 4) decisions at today's close
        ok = bool(regime_ok[i])
        for j in hold:
            if j in sells:
                continue
            if not ok:
                sells[j] = "regime"
            elif exit_mode == "sma" and trend[i, j] == 0.0:
                sells[j] = "rule_exit"
        if exit_mode == "rank" and rebalance[i] and ok:
            s = np.where(member[i] & np.isfinite(score[i]), score[i], -np.inf)
            order = np.argsort(-s)
            rank = np.empty(m, dtype=np.int64)
            rank[order] = np.arange(1, m + 1)
            for j in hold:
                if j not in sells and (not np.isfinite(s[j]) or rank[j] > buffer):
                    sells[j] = "rank_exit"
        if ok and (exit_mode == "sma" or rebalance[i]):
            free = slots - (len(hold) - len(sells))
            cap = min(free, max_new) if exit_mode == "sma" else free
            if cap > 0:
                elig = member[i] & (trend[i] == 1.0) & np.isfinite(score[i])
                idx = np.flatnonzero(elig)
                idx = idx[np.argsort(-score[i, idx])]
                buys = [j for j in idx if j not in hold][:cap]
    return eq, trades


def precompute(close: np.ndarray) -> dict:
    """Arrays shared by every simulation on the same panel."""
    n, m = close.shape
    finite = np.isfinite(close)
    nxt = np.full((n + 1, m), n + 10**6, dtype=np.int64)  # next finite close >= i
    for i in range(n - 1, -1, -1):
        nxt[i] = np.where(finite[i], i, nxt[i + 1])
    return {"finite": finite, "nxt": nxt,
            "closef": pd.DataFrame(close).ffill().to_numpy()}


def _metrics(curve: pd.Series) -> dict:
    if len(curve) < 2:
        return {"total": np.nan, "cagr": np.nan, "sharpe": np.nan, "mdd": np.nan}
    r = curve.pct_change().dropna()
    yrs = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    tot = curve.iloc[-1] / curve.iloc[0] - 1.0
    return {"total": tot, "cagr": (1 + tot) ** (1 / yrs) - 1,
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "mdd": (curve / curve.cummax() - 1).min()}


def _seg(curve: pd.Series, a, b) -> pd.Series:
    prev = curve[curve.index < a]
    seg = curve[(curve.index >= a) & (curve.index <= b)]
    return pd.concat([prev.iloc[-1:], seg]) if len(prev) else seg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache", required=True, help="PIT panel pickle (fetched if missing).")
    ap.add_argument("--start", default="2016-01-01", help="Data start (fetch only).")
    ap.add_argument("--end", default=None)
    ap.add_argument("--pit-top", type=int, default=700)
    ap.add_argument("--slots", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--exits", nargs="+", default=["sma", "rank"])
    ap.add_argument("--crash", nargs="+", default=["off", "spy200"])
    ap.add_argument("--buffer", type=int, default=20)
    ap.add_argument("--rebalance", nargs="+", default=["M"],
                    help="Rank-exit schedule(s): M = last session of each month; "
                         "W = last session of each week; K or K+O = every K trading "
                         "days starting O sessions after the simulation start.")
    ap.add_argument("--random-runs", type=int, default=100)
    ap.add_argument("--min-cash-pct", type=float, default=0.10)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from config import globalConfig  # type: ignore  # pylint: disable=import-outside-toplevel

    end = datetime.strptime(a.end, "%Y-%m-%d") if a.end else datetime.now()
    panel = load_panel(sorted(set(pit_symbols()) | {"SPY"}) if not os.path.exists(a.cache) else [],
                       datetime.strptime(a.start, "%Y-%m-%d"), end, a.cache, 250_000.0)
    panel = {k: v.astype("float64") for k, v in panel.items()}
    spy = panel["close"]["SPY"].copy()
    names = asset_names()
    keep = [c for c in panel["close"].columns if c != "SPY" and is_common_stock(c, names.get(c, ""))]
    panel = {k: v[keep] for k, v in panel.items()}
    close, open_ = panel["close"], panel["open"]
    dates = close.index
    member = pit_universe(close, panel["volume"], a.pit_top,
                          globalConfig.MIN_PRICE, globalConfig.MAX_PRICE)
    pos = rule_positions(close, 50)
    score = backtest_scores(close, pos, 252, 21, a.cost_bps, 40, metric="sharpe")
    ever = (member & score.notna()).any()
    cols = ever[ever].index
    close, open_, member, pos, score = (x[cols] for x in (close, open_, member, pos, score))
    logger.info("%d symbols ever eligible", len(cols))

    scorable = (member & score.notna()).sum(axis=1)
    start_i = int(np.flatnonzero(scorable.to_numpy() >= 200)[0])
    start = dates[start_i]
    spy_sma = spy.rolling(200, min_periods=200).mean()
    regimes = {"off": np.ones(len(dates), bool),
               "spy200": (spy > spy_sma).reindex(dates).fillna(False).to_numpy()}
    month = dates.to_period("M")

    def schedule(spec: str) -> np.ndarray:
        if spec == "M":
            return np.r_[month[1:] != month[:-1], True]
        if spec == "W":
            week = dates.to_period("W")
            return np.r_[week[1:] != week[:-1], True]
        k, _, off = spec.partition("+")
        mask = np.zeros(len(dates), bool)
        mask[start_i + int(off or 0)::int(k)] = True
        return mask

    arrs = dict(close=close.to_numpy(), open_=open_.to_numpy(),
                member=member.to_numpy(dtype=bool), trend=pos.to_numpy())
    sc = score.to_numpy()
    syms = list(cols)
    periods = {"VALIDATION": (start, pd.Timestamp("2018-12-31")),
               "SELECTION": (pd.Timestamp("2019-01-01"), pd.Timestamp("2023-12-31")),
               "SEEN 2024+": (pd.Timestamp("2024-01-01"), dates[-1]),
               "FULL": (start, dates[-1])}
    test_dates = dates[start_i:]
    spy_curve = 100_000 * spy.loc[test_dates] / spy.loc[test_dates[0]]
    rets = close.pct_change(fill_method=None)
    ew = rets.where(member.shift(1, fill_value=False)).mean(axis=1).loc[test_dates].fillna(0.0)
    ew.iloc[0] = 0.0
    ew_curve = 100_000 * (1 + ew).cumprod()

    arrs["pre"] = precompute(arrs["close"])
    scorable_mask = np.isfinite(sc)

    def rand_score(k):  # same k-th random ranking for every variant
        return np.where(scorable_mask, np.random.default_rng(a.seed + k).random(sc.shape), np.nan)
    rows, curves, all_trades = [], {}, []
    grid = [(e, s, c, r) for e, s, c in itertools.product(a.exits, a.slots, a.crash)
            for r in (a.rebalance if e == "rank" else ["-"])]
    for exit_mode, slots, crash, reb in grid:
        tag = f"{exit_mode}/{slots}/{crash}" + (f"/{reb}" if exit_mode == "rank" else "")
        kw = dict(regime_ok=regimes[crash], rebalance=schedule(reb) if reb != "-" else
                  np.zeros(len(dates), bool), start_i=start_i,
                  slots=slots, exit_mode=exit_mode, buffer=a.buffer,
                  min_cash=a.min_cash_pct, cost_bps=a.cost_bps,
                  max_new=globalConfig.MAX_NEW_POSITIONS_PER_DAY)
        eq, trades = simulate(score=sc, symbols=syms, dates=dates, **arrs, **kw)
        curve = pd.Series(eq, index=test_dates)
        curves[tag] = curve
        all_trades.append(pd.DataFrame(trades).assign(variant=tag))
        rand = [pd.Series(simulate(score=rand_score(k), **arrs, **kw)[0], index=test_dates)
                for k in range(a.random_runs)]
        tr = pd.DataFrame(trades)
        for pname, (pa, pb) in periods.items():
            m_ = _metrics(_seg(curve, pa, pb))
            dist = np.array([_metrics(_seg(rc, pa, pb))["total"] for rc in rand])
            spy_m = _metrics(_seg(spy_curve, pa, pb))
            rows.append({"variant": tag, "period": pname, **m_,
                         "excess_cagr_vs_spy": m_["cagr"] - spy_m["cagr"],
                         "rand_median": float(np.median(dist)) if len(dist) else np.nan,
                         "pctile": float((dist < m_["total"]).mean() * 100) if len(dist) else np.nan,
                         "trades": len(tr), "med_hold": float(
                             (tr["exit_date"] - tr["entry_date"]).dt.days.median()) if len(tr) else np.nan})
        logger.info("%s done: FULL CAGR %.1f%%", tag, rows[-1]["cagr"] * 100)

    for label, curve in (("SPY", spy_curve), ("EW universe", ew_curve)):
        curves[label] = curve
        for pname, (pa, pb) in periods.items():
            rows.append({"variant": label, "period": pname, **_metrics(_seg(curve, pa, pb))})
    t = pd.DataFrame(rows)

    print("=" * 100)
    print("BACKTEST-RANKED MOMENTUM - PORTFOLIO VARIANTS (point-in-time common stocks)")
    print("=" * 100)
    print(f"  Data {dates[0].date()} -> {dates[-1].date()}; simulation from {start.date()}; "
          f"{len(syms)} symbols; top {a.pit_top} by $vol; buffer {a.buffer}; "
          f"{a.random_runs} random rankings/variant; {a.cost_bps:.0f} bps/side")
    pct = lambda v: "" if pd.isna(v) else f"{v:+.1%}"  # noqa: E731
    for metric, f in (("cagr", pct), ("sharpe", lambda v: "" if pd.isna(v) else f"{v:.2f}"),
                      ("mdd", pct), ("pctile", lambda v: "" if pd.isna(v) else f"{v:.0f}")):
        pv = t.pivot_table(index="variant", columns="period", values=metric, aggfunc="first")
        pv = pv[[p for p in periods if p in pv.columns]]
        print(f"\n--- {metric} by period ---")
        print(pv.to_string(formatters={c: f for c in pv.columns}))

    sel = t[(t.period == "SELECTION") & (t.pctile >= 90)]
    print("\n--- Pre-declared selection: best SELECTION CAGR with pctile >= 90 ---")
    if sel.empty:
        print("  No variant passes the >= 90th-percentile bar in SELECTION.")
        winner = None
    else:
        winner = sel.sort_values("cagr", ascending=False).iloc[0]["variant"]
        w = t[t.variant == winner].set_index("period")
        s_ = t[t.variant == "SPY"].set_index("period")
        print(f"  Winner: {winner}")
        for p in periods:
            print(f"  {p:<11} CAGR {pct(w.loc[p, 'cagr']):>7} (SPY {pct(s_.loc[p, 'cagr']):>7})  "
                  f"Sharpe {w.loc[p, 'sharpe']:.2f} (SPY {s_.loc[p, 'sharpe']:.2f})  "
                  f"MDD {pct(w.loc[p, 'mdd'])} (SPY {pct(s_.loc[p, 'mdd'])})  "
                  f"pctile vs random {w.loc[p, 'pctile']:.0f}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"momentum_rank_portfolio_{datetime.now():%Y%m%d_%H%M%S}")
    t.to_csv(prefix + "_periods.csv", index=False)
    pd.DataFrame(curves).to_csv(prefix + "_equity.csv")
    pd.concat(all_trades).to_csv(prefix + "_trades.csv", index=False)
    print(f"\nWrote: {prefix}_periods.csv, _equity.csv, _trades.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
