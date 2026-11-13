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

Survivorship bias: ``--universe live`` uses TODAY's RSI universe (survivors).
``--universe pit`` rebuilds it point-in-time from active + delisted listings
(price band + top-N trailing dollar volume each day); delisted holdings exit
at their last close. Short windows are dominated by noise -- read the
random-ranking percentile before the headline return.

Read-only: fetches data, places no orders, writes CSVs under research/logs.

    python app/research/momentum_rank_backtest.py --test-months 3
"""
import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from portfolio_backtest import simulate_portfolio  # noqa: E402
from strategies.momentum_rank_signal import (  # noqa: E402,F401  (re-exported for research)
    backtest_scores, is_common_stock, rule_positions)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("momentum_rank_backtest")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_daily_panel(symbols: List[str], start: datetime, end: datetime,
                      chunk: int = 200, min_peak_dv: float = 0.0
                      ) -> Dict[str, pd.DataFrame]:
    """Wide OHLCV panels (date x symbol), all-adjusted daily bars, batched.

    Each batch is pivoted on arrival (float32) so a ~16k-symbol point-in-time
    universe fits in memory. ``min_peak_dv`` drops symbols whose 63-day median
    dollar volume never reaches that level -- a cheap prefilter that must sit
    well below the top-N cutoff (checked by ``pit_universe``).
    """
    from alpaca.data.enums import Adjustment  # pylint: disable=import-outside-toplevel
    from alpaca.data.requests import StockBarsRequest  # pylint: disable=import-outside-toplevel
    from alpaca.data.timeframe import TimeFrame  # pylint: disable=import-outside-toplevel
    from data_provider import data_provider  # pylint: disable=import-outside-toplevel

    parts: Dict[str, list] = {"open": [], "close": [], "volume": []}
    failed: List[str] = []
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        df = None
        attempt = 0
        while batch and attempt < 3:
            try:
                df = data_provider.historical_client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=batch, timeframe=TimeFrame.Day,
                    start=start, end=end, adjustment=Adjustment.ALL)).df
                break
            except Exception as e:  # pylint: disable=broad-exception-caught
                # One unknown ticker rejects the whole request: drop it and retry.
                bad = re.search(r"invalid symbol: ([^\"']+)", str(e))
                if bad and bad.group(1) in batch:
                    logger.info("Dropping invalid symbol %s", bad.group(1))
                    batch = [b for b in batch if b != bad.group(1)]
                    continue
                attempt += 1
                logger.warning("Batch %d attempt %d failed: %s", i // chunk, attempt, e)
                time.sleep(5 * attempt)
        if df is None and batch:
            failed.extend(batch)
        if df is not None and not df.empty:
            df = df.reset_index()
            df["date"] = (pd.to_datetime(df["timestamp"], utc=True)
                          .dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize())
            wide = {c: df.pivot_table(index="date", columns="symbol", values=c)
                    .astype("float32") for c in ("open", "close", "volume")}
            if min_peak_dv > 0:
                dv = (wide["close"] * wide["volume"]).rolling(63, min_periods=50).median()
                keep = dv.max() >= min_peak_dv
                keep = keep[keep].index.union(
                    pd.Index([s for s in ("SPY",) if s in wide["close"].columns]))
                wide = {c: w[keep] for c, w in wide.items()}
            for c in parts:
                parts[c].append(wide[c])
        logger.info("Fetched %d/%d symbols (%d kept so far)", min(i + chunk, len(symbols)),
                    len(symbols), sum(w.shape[1] for w in parts["close"]))
    if failed:
        logger.error("%d symbols could not be fetched (e.g. %s)", len(failed), failed[:10])
    if not parts["close"]:
        return {}
    out = {c: pd.concat(ws, axis=1).sort_index() for c, ws in parts.items()}
    return {c: w.loc[:, ~w.columns.duplicated()] for c, w in out.items()}


def asset_names() -> Dict[str, str]:
    """symbol -> name for active and inactive US-equity assets."""
    from data_provider import data_provider  # pylint: disable=import-outside-toplevel
    out: Dict[str, str] = {}
    for status in ("active", "inactive"):
        for r in data_provider.trading_client.get(
                "/assets", {"status": status, "asset_class": "us_equity"}):
            if isinstance(r, dict) and r.get("symbol"):
                out.setdefault(r["symbol"], r.get("name") or "")
    return out


def pit_symbols() -> List[str]:
    """Every listing that could have been tradable: active + delisted US equities.

    Active: major exchanges plus OTC (bankrupt names often move to OTC with a
    ``Q`` suffix). Inactive: major exchanges only (inactive OTC is ~16k
    shells). Leveraged/inverse funds dropped by name, as in the live filter.
    """
    from data_provider import data_provider, is_leveraged_or_inverse_fund  # pylint: disable=import-outside-toplevel
    major = {"NASDAQ", "NYSE", "ARCA", "BATS", "AMEX"}
    out = set()
    for status in ("active", "inactive"):
        rows = data_provider.trading_client.get(
            "/assets", {"status": status, "asset_class": "us_equity"})
        for r in rows:
            if not isinstance(r, dict) or not r.get("symbol"):
                continue
            ex = r.get("exchange")
            if ex in major or (status == "active" and ex == "OTC"):
                if not is_leveraged_or_inverse_fund(r.get("name")):
                    # Delisted assets are renamed X_DELISTED; bars live under X.
                    sym = r["symbol"].replace("_DELISTED", "")
                    if re.fullmatch(r"[A-Z]+([./][A-Z]+)?", sym):  # drops CUSIP-like codes
                        out.add(sym)
    return sorted(out)


def pit_universe(close: pd.DataFrame, volume: pd.DataFrame, top_n: int,
                 min_price: float, max_price: float, dv_days: int = 63) -> pd.DataFrame:
    """Point-in-time membership at each close: price band + top-N liquidity.

    Rank by trailing ``dv_days`` median dollar volume (data up to and incl. d,
    known at the decision close). Stands in for the live filters that cannot
    be reconstructed historically (options listed, margin requirement).
    """
    dv = (close * volume).rolling(dv_days, min_periods=int(dv_days * 0.8)).median()
    in_band = (close >= min_price) & (close <= max_price)
    rank = dv.where(in_band).rank(axis=1, ascending=False, method="first")
    member = rank <= top_n
    cutoff = dv.where(member).min(axis=1)
    logger.info("PIT universe: %d members/day median; dollar-volume cutoff "
                "min $%.1fm, median $%.1fm", int(member.sum(axis=1).median()),
                cutoff.min() / 1e6, cutoff.median() / 1e6)
    return member


def load_panel(symbols: List[str], start: datetime, end: datetime, cache: str = None,
               min_peak_dv: float = 0.0) -> Dict[str, pd.DataFrame]:
    """fetch_daily_panel with an optional pickle cache (keyed by path only)."""
    if cache and os.path.exists(cache):
        logger.info("Loading cached panel %s", cache)
        return pd.read_pickle(cache)
    panel = fetch_daily_panel(symbols, start, end, min_peak_dv=min_peak_dv)
    if cache and panel:
        pd.to_pickle(panel, cache)
        logger.info("Cached panel -> %s", cache)
    return panel


# ---------------------------------------------------------------------------
# Signal + scores (vectorized over the panel)
# ---------------------------------------------------------------------------

def momentum_scores(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """Plain 12-1 style return: close[d-skip] / close[d-lookback] - 1."""
    return close.shift(skip) / close.shift(lookback) - 1.0


# ---------------------------------------------------------------------------
# Candidate trades
# ---------------------------------------------------------------------------

def build_candidates(panel: Dict[str, pd.DataFrame], pos: pd.DataFrame,
                     test_start: pd.Timestamp, min_price: float,
                     max_price: float, universe: pd.DataFrame = None) -> pd.DataFrame:
    """Every eligible (symbol, decision day) -> trade with rule-based exit.

    Decision at the close of d (rule long, in universe / price band), entry
    at the next valid open, exit at the next valid open after the first close
    below the SMA. If the symbol's data ends first (delisted, acquired,
    bankrupt) the trade exits at its LAST close ("data_end") -- dropping such
    trades would hide exactly the losers survivorship bias hides. Trades
    still open on the last bar close at the last close ("window_end").
    """
    close, open_ = panel["close"], panel["open"]
    dates = close.index
    n = len(dates)
    last = n - 1
    t0 = int(np.searchsorted(dates.values, np.datetime64(test_start)))
    rows = []
    for sym in close.columns:
        c = close[sym].to_numpy(dtype=float)
        o = open_[sym].to_numpy(dtype=float)
        p = pos[sym].to_numpy(dtype=float)
        valid_c = np.flatnonzero(np.isfinite(c))
        if not len(valid_c):
            continue
        lv = int(valid_c[-1])
        elig = (p == 1.0)
        if universe is not None:
            elig &= universe[sym].to_numpy(dtype=bool)
        else:
            elig &= (c >= min_price) & (c <= max_price)
        elig[:t0] = False
        ds = np.flatnonzero(elig[:last])
        if not len(ds):
            continue
        # next_open[i]: smallest j >= i with a finite open (else -1)
        next_open = np.full(n + 1, -1, dtype=int)
        next_below = np.full(n + 1, -1, dtype=int)
        no = nb = -1
        for i in range(last, -1, -1):
            if np.isfinite(o[i]) and o[i] > 0:
                no = i
            if p[i] == 0.0:
                nb = i
            next_open[i], next_below[i] = no, nb
        for d in ds:
            e = next_open[d + 1]
            if e == -1 or e > d + 5:  # no tradable open soon after the signal
                continue
            x = next_below[e]
            xe = next_open[x + 1] if x != -1 and x + 1 <= last else -1
            if xe != -1:
                exit_i, exit_px, reason = xe, o[xe], "rule_exit"
            elif lv < last:
                exit_i, exit_px, reason = lv, c[lv], "data_end"
            else:
                exit_i, exit_px, reason = last, c[last], "window_end"
            if exit_i < e:
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


def _alpha_beta(curve: pd.Series, bench: pd.Series):
    """Annualized OLS intercept and slope of daily returns on a benchmark."""
    j = pd.concat([curve.pct_change(), bench.pct_change()], axis=1).dropna()
    if len(j) < 20 or j.iloc[:, 1].var() == 0:
        return np.nan, np.nan
    b = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var())
    a = float(j.iloc[:, 0].mean() - b * j.iloc[:, 1].mean()) * 252
    return a, b


def rank_buckets(cands: pd.DataFrame, edges=(1, 3, 5, 10, 20, 50, 10**9)) -> pd.DataFrame:
    """Does a higher score predict a better trade? Candidate return by daily rank.

    Every eligible candidate (not just those the portfolio could take) is
    ranked within its entry day; the table shows the mean / median trade
    return and hold per rank bucket. Monotone => concentrating in the top
    ranks should add return; flat => the ranking only filters.
    """
    c = cands.copy()
    c["rank"] = c.groupby("entry_date")["score"].rank(ascending=False, method="first")
    c["ret"] = c["exit_price"] / c["entry_price"] - 1.0
    c["hold"] = (c["exit_date"] - c["entry_date"]).dt.days
    labels = [f"{lo}-{hi - 1}" if hi - 1 < 10**8 else f"{lo}+"
              for lo, hi in zip(edges[:-1], edges[1:])]
    c["bucket"] = pd.cut(c["rank"], bins=[e - 0.5 for e in edges], labels=labels)
    g = c.groupby("bucket", observed=True)
    return pd.DataFrame({"n": g.size(), "mean_ret": g["ret"].mean(),
                         "median_ret": g["ret"].median(), "hit": g["ret"].apply(lambda r: (r > 0).mean()),
                         "mean_hold_d": g["hold"].mean(),
                         "ret_per_yr_held": g["ret"].mean() / g["hold"].mean() * 365})


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
    ap.add_argument("--score", nargs="+", default=["sharpe"], choices=["sharpe", "return"],
                    help="Rule-backtest metric used to rank.")
    ap.add_argument("--max-positions", type=int, nargs="+", default=[0],
                    help="Portfolio slots; position size = (1-min cash)/slots. "
                         "0 = live config (MAX_POSITIONS, POSITION_SIZE_PCT).")
    ap.add_argument("--min-days-long", type=int, default=40,
                    help="Min in-position days in the scoring window.")
    ap.add_argument("--cost-bps", type=float, default=5.0, help="Per side.")
    ap.add_argument("--random-runs", type=int, default=200)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--min-cash-pct", type=float, default=0.10,
                    help="Cash floor (default 0.10 = prod).")
    ap.add_argument("--universe", choices=["live", "pit"], default="live",
                    help="live = today's RSI universe (survivorship-biased); "
                         "pit = point-in-time: active+delisted listings, "
                         "price band + top-N trailing dollar volume each day.")
    ap.add_argument("--pit-top", type=int, default=700)
    ap.add_argument("--min-peak-dv", type=float, default=250_000.0,
                    help="pit prefilter: drop symbols whose 63d median $vol never "
                         "reaches this (feed terms).")
    ap.add_argument("--include-funds", action="store_true",
                    help="Keep ETFs/CEFs/preferreds (default: common stocks only).")
    ap.add_argument("--cache", default=None, help="Pickle path for the price panel.")
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

    if a.universe == "pit":
        symbols = pit_symbols()
        desc = (f"POINT-IN-TIME: active+delisted listings, ${globalConfig.MIN_PRICE:.0f}-"
                f"${globalConfig.MAX_PRICE:.0f}, top {a.pit_top} by 63d median $vol")
    else:
        universe = data_provider.get_stock_universe()
        symbols = sorted(universe["symbol"].tolist()) if not universe.empty else []
        desc = "today's live universe; SURVIVORSHIP-BIASED - absolute returns overstated"
    if a.max_symbols:
        symbols = symbols[:a.max_symbols]
    logger.info("Universe source: %d symbols (%s)", len(symbols), a.universe)
    panel = load_panel(sorted(set(symbols) | {"SPY"}), fetch_start, end, a.cache,
                       a.min_peak_dv if a.universe == "pit" else 0.0)
    if not panel:
        print("No data.")
        return 1
    panel = {k: v.astype("float64") for k, v in panel.items()}
    spy_all = panel["close"]["SPY"].copy() if "SPY" in panel["close"] else None
    if a.universe == "live":
        panel = {k: v.drop(columns=["SPY"], errors="ignore") for k, v in panel.items()}
    if not a.include_funds:
        names = asset_names()
        drop = [c for c in panel["close"].columns
                if c != "SPY" and not is_common_stock(c, names.get(c, ""))]
        panel = {k: v.drop(columns=drop) for k, v in panel.items()}
        logger.info("Stocks only: dropped %d funds/CEFs/preferreds", len(drop))
        desc += "; common stocks only"
    close = panel["close"]
    all_dates = close.index
    test_start = all_dates[all_dates >= test_start][0]
    anchor = all_dates[all_dates < test_start][-1]
    test_dates = all_dates[all_dates >= test_start]

    member = None
    if a.universe == "pit":
        member = pit_universe(close, panel["volume"], a.pit_top,
                              globalConfig.MIN_PRICE, globalConfig.MAX_PRICE)
        ever = member.loc[test_start:].any()
        keep_cols = ever[ever].index
        panel = {k: v[keep_cols] for k, v in panel.items()}
        member = member[keep_cols]
        close = panel["close"]
        logger.info("PIT: %d symbols ever in the universe during the test", len(keep_cols))

    prices = {s: close[s].dropna() for s in close.columns}
    splits = {}
    if a.holdout_start:
        h = pd.Timestamp(a.holdout_start)
        splits = {"SELECTION": (test_start, h - pd.Timedelta(days=1)),
                  "HOLDOUT": (h, test_dates[-1])}
    splits["FULL"] = (test_start, test_dates[-1])

    spy = spy_all.loc[anchor:] if spy_all is not None else None
    rets = close.loc[anchor:].pct_change(fill_method=None)
    if member is not None:   # EW of yesterday's members, rebalanced daily
        ew = rets.where(member.loc[anchor:].shift(1).fillna(False).astype(bool)).mean(axis=1)
    else:
        ew = rets.mean(axis=1)
    ew_curve = a.cash * (1.0 + ew.fillna(0.0)).cumprod()
    spy_curve = a.cash * spy / spy.iloc[0] if spy is not None else None

    def make_cfg(slots):
        if not slots:
            return SimpleNamespace(
                MAX_POSITIONS=globalConfig.MAX_POSITIONS,
                MAX_NEW_POSITIONS_PER_DAY=globalConfig.MAX_NEW_POSITIONS_PER_DAY,
                POSITION_SIZE_PCT=globalConfig.POSITION_SIZE_PCT,
                MIN_CASH_PCT=a.min_cash_pct)
        return SimpleNamespace(
            MAX_POSITIONS=slots,
            MAX_NEW_POSITIONS_PER_DAY=globalConfig.MAX_NEW_POSITIONS_PER_DAY,
            POSITION_SIZE_PCT=(1.0 - a.min_cash_pct) / slots,
            MIN_CASH_PCT=a.min_cash_pct)

    rng = np.random.default_rng(a.seed)
    rows, period_rows, all_trades, buckets = [], [], [], []
    equity_out = {"SPY buy&hold": spy_curve, "EW universe": ew_curve}
    for sma in a.sma_days:
        logger.info("SMA%d: building candidates…", sma)
        pos = rule_positions(close, sma)
        cands = build_candidates(panel, pos, test_start, globalConfig.MIN_PRICE,
                                 globalConfig.MAX_PRICE, universe=member)
        logger.info("SMA%d: %d candidates (%s)", sma, len(cands),
                    cands["exit_reason"].value_counts().to_dict())
        scored = {m: score_candidates(cands, backtest_scores(
            close, pos, a.lookback, a.skip, a.cost_bps, a.min_days_long, metric=m))
            for m in a.score}
        scored["12-1"] = score_candidates(cands, momentum_scores(close, a.lookback, a.skip))
        for m, sc in scored.items():
            bk = rank_buckets(sc).reset_index().assign(sma=sma, ranking=m)
            buckets.append(bk)

        for slots in a.max_positions:
            cfg = make_cfg(slots)
            keep = cfg.MAX_POSITIONS + cfg.MAX_NEW_POSITIONS_PER_DAY + 3
            tag = f"{sma}/{cfg.MAX_POSITIONS}"
            curves = {}
            for m, sc in scored.items():
                label = m if m == "12-1" else f"bt:{m}"
                curves[label], res = _equity(prune(sc, keep), a.cash, cfg, prices,
                                             a.cost_bps, anchor, test_dates)
                equity_out[f"{label} {tag}"] = curves[label]
                tr = res.trades_taken.assign(sma=sma, slots=cfg.MAX_POSITIONS, ranking=label)
                all_trades.append(tr)
                full = _metrics(curves[label])
                a_spy, b_spy = _alpha_beta(curves[label], spy) if spy is not None else (np.nan, np.nan)
                a_ew, b_ew = _alpha_beta(curves[label], ew_curve)
                rows.append({
                    "sma/slots": tag, "ranking": label, "trades": len(tr),
                    "hit": float((tr["pnl"] > 0).mean()) if len(tr) else np.nan,
                    "med_hold": float((tr["exit_date"] - tr["entry_date"]).dt.days.median())
                    if len(tr) else np.nan,
                    "data_end_exits": int((tr["exit_reason"] == "data_end").sum()),
                    **{k: full[k] for k in ("cagr", "sharpe", "max_drawdown")},
                    "beta_spy": b_spy, "alpha_spy": a_spy, "beta_ew": b_ew, "alpha_ew": a_ew})

            logger.info("%s: %d random rankings…", tag, a.random_runs)
            base = next(iter(scored.values()))
            rand_curves = []
            for _ in range(a.random_runs):
                shuffled = base.assign(score=rng.random(len(base)))
                curve, _ = _equity(prune(shuffled, keep), a.cash, cfg, prices,
                                   a.cost_bps, anchor, test_dates)
                rand_curves.append(curve)
            rand_periods = [_periods(rc, splits) for rc in rand_curves]

            for label, curve in curves.items():
                for period, seg in _periods(curve, splits).items():
                    m_ = _metrics(seg)
                    row = {"sma/slots": tag, "ranking": label, "period": period, **m_}
                    if rand_curves:
                        dist = np.array([_metrics(rp[period])["total_return"]
                                         for rp in rand_periods])
                        row["rand_median"] = float(np.nanmedian(dist))
                        row["pctile_vs_random"] = float(
                            (dist < m_["total_return"]).mean() * 100)
                    period_rows.append(row)
            if rand_curves:
                for period in _periods(rand_curves[0], splits):
                    tot = [_metrics(rp[period])["total_return"] for rp in rand_periods]
                    period_rows.append({"sma/slots": tag, "ranking": "random (median)",
                                        "period": period, "total_return": float(np.nanmedian(tot))})

    for label, curve in (("SPY buy&hold", spy_curve), ("EW universe", ew_curve)):
        if curve is None:
            continue
        for period, seg in _periods(curve, splits).items():
            period_rows.append({"sma/slots": "-", "ranking": label, "period": period,
                                **_metrics(seg)})

    table = pd.DataFrame(period_rows)
    print("=" * 110)
    print("BACKTEST-RANKED MOMENTUM (fixed rule, per-stock backtest score)")
    print("=" * 110)
    print(f"  Test window : {test_start.date()} -> {test_dates[-1].date()} "
          f"({len(test_dates)} sessions)" + (
              f"; holdout from {a.holdout_start}" if a.holdout_start else ""))
    print(f"  Universe    : {close.shape[1]} symbols ({desc})")
    print(f"  Rule        : long while close > SMA(n), n in {a.sma_days}; "
          f"score in {a.score} over [d-{a.lookback}, d-{a.skip}]")
    print(f"  Portfolio   : slots {a.max_positions} (0 = live config), "
          f"{globalConfig.MAX_NEW_POSITIONS_PER_DAY} new/day, {a.min_cash_pct:.0%} min cash, "
          f"{a.cost_bps:.0f} bps/side; {a.random_runs} random rankings per sma/slots")

    pct = "{:+.1%}".format
    print("\n--- Full-period summary (alpha = annualized regression intercept) ---")
    print(pd.DataFrame(rows).to_string(index=False, formatters={
        "hit": "{:.0%}".format, "cagr": pct, "sharpe": "{:.2f}".format,
        "max_drawdown": pct, "beta_spy": "{:.2f}".format, "alpha_spy": pct,
        "beta_ew": "{:.2f}".format, "alpha_ew": pct, "med_hold": "{:.0f}".format}))

    fmt = lambda v: "" if pd.isna(v) else f"{v:+.1%}"  # noqa: E731
    pivot = table.pivot_table(index=["ranking", "sma/slots"], columns="period",
                              values="total_return", aggfunc="first")
    order = [c for c in pivot.columns if c.isdigit()] + [
        c for c in ("SELECTION", "HOLDOUT", "FULL") if c in pivot.columns]
    print("\n--- Total return by period ---")
    print(pivot[order].to_string(formatters={c: fmt for c in order}))

    sharpe = table.pivot_table(index=["ranking", "sma/slots"], columns="period",
                               values="sharpe", aggfunc="first")
    cols = [c for c in ("SELECTION", "HOLDOUT", "FULL") if c in sharpe.columns]
    print("\n--- Sharpe by split ---")
    print(sharpe[cols].to_string(float_format=lambda v: f"{v:.2f}"))

    if "pctile_vs_random" in table:
        rel = table[table["pctile_vs_random"].notna()].pivot_table(
            index=["ranking", "sma/slots"], columns="period", values="pctile_vs_random",
            aggfunc="first")
        print("\n--- Percentile vs random rankings (same rule; >90 = ranking adds value) ---")
        print(rel[order].to_string(float_format=lambda v: f"{v:.0f}"))

    bk = pd.concat(buckets)
    print("\n--- Candidate trade return by daily score rank (all eligible candidates) ---")
    print(bk.to_string(index=False, formatters={
        "mean_ret": pct, "median_ret": pct, "hit": "{:.0%}".format,
        "mean_hold_d": "{:.0f}".format, "ret_per_yr_held": pct}))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(out_dir, f"momentum_rank_{a.universe}_{stamp}")
    pd.concat(all_trades).to_csv(prefix + "_trades.csv", index=False)
    table.to_csv(prefix + "_periods.csv", index=False)
    pd.DataFrame(rows).to_csv(prefix + "_summary.csv", index=False)
    bk.to_csv(prefix + "_buckets.csv", index=False)
    pd.DataFrame({k: v for k, v in equity_out.items() if v is not None}).to_csv(
        prefix + "_equity.csv")
    print(f"\nWrote: {prefix}_trades.csv, _periods.csv, _summary.csv, _buckets.csv, _equity.csv")
    if a.universe == "live":
        print("WARNING: survivorship bias - universe is today's survivors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
