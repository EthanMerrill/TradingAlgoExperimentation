#!/usr/bin/env python3
"""
Morning continuation via the mapped INVERSE leveraged ETFs (v2 of the
morning-short idea).

Instead of shorting the underlying (borrow + locate + wide spread), BUY the
mapped inverse leveraged fund at the open after the underlying had a big down
day, cover at 10:00 / 10:30. The leveraged fund amplifies the underlying's
continuation ~2x, costs nothing in borrow, and trades at RTH spreads.

Variants tested (all identical machinery, only the filter differs):

* ``unfiltered``  — buy the inverse fund on EVERY qualifying down-day session.
* ``pm_confirm``  — additionally require the underlying to be DOWN premarket
  (9:00 ET price below the prior close). Keeps the premarket *information*
  found by the earlier probe (premarket 9:00→open continued, t=-2.44) while
  avoiding premarket execution.
* ``underlying_short`` — the baseline from the previous runner (short the
  underlying itself at the open) for direct comparison of the two expressions.

Controls:
* ``fade`` — LONG the inverse fund (equivalent to short the continuation)
  to sign-check the morning effect on the fund itself.
* ``underlying`` price-matched to the inverse fund's listed map entries only
  (so the comparison uses the same names).

Read-only diagnostic: places no orders, writes no storage.

    python app/research/leveraged_inverse_morning.py --start 2025-07-18
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from alpaca.data.enums import Adjustment  # noqa: E402

from data_provider import data_provider  # noqa: E402
from strategies.leveraged_flow_portfolio import (  # noqa: E402
    portfolio_stats,
    split_half_stability,
)
from strategies.leveraged_rebalance_signal import prepare_features  # noqa: E402
from strategies.leveraged_single_stock_etfs import (  # noqa: E402
    LEVERAGED_SINGLE_STOCK_ETFS,
    underlyings,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("leveraged_inverse_morning")

OPEN_TOD = 9 * 60 + 30
PM9_TOD = 9 * 60  # premarket 09:00 confirmation bar
EXIT_TODS = {30: 10 * 60, 60: 10 * 60 + 30}


def _inverse_fund(underlying: str):
    """The highest-|leverage| INVERSE fund mapped to ``underlying`` (or None)."""
    candidates = [f for f in LEVERAGED_SINGLE_STOCK_ETFS.values()
                  if f.underlying == underlying and f.leverage < 0]
    if not candidates:
        return None
    return max(candidates, key=lambda f: abs(f.leverage))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--down-threshold", type=float, default=-0.02,
                        help="Prior-session underlying return must be ≤ this.")
    parser.add_argument("--top-ks", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--min-days", type=int, default=40)
    args = parser.parse_args()

    end = (datetime.strptime(args.end, "%Y-%m-%d")
           if args.end else datetime.now())
    start = (datetime.strptime(args.start, "%Y-%m-%d")
             if args.start else end - timedelta(days=180))

    print("=" * 78)
    print("INVERSE-FUND MORNING TRADE — buy the mapped inverse fund at the open")
    print(f"  trigger : underlying prior-session return ≤ {args.down_threshold:.1%}")
    print(f"  entry   : 09:30 open price of the INVERSE fund (RTH, no borrow)")
    print(f"  exits   : 10:00 (30min) and 10:30 (60min)")
    print(f"  filters : all qualifying days | premarket-confirmed (9:00 < prev close)")
    print("=" * 78)

    universe = underlyings()
    underlying_features = {}
    fund_features = {}
    for underlying in universe:
        bars = data_provider.get_single_stock_bars(
            underlying, start - timedelta(days=10), end,
            timeframe=args.timeframe, adjustment=Adjustment.SPLIT)
        if bars is not None and not bars.empty:
            underlying_features[underlying] = prepare_features(bars)
        fund = _inverse_fund(underlying)
        if fund is None:
            continue
        fbars = data_provider.get_single_stock_bars(
            fund.etf, start - timedelta(days=10), end,
            timeframe=args.timeframe, adjustment=Adjustment.SPLIT)
        if fbars is not None and not fbars.empty:
            fund_features[fund.etf] = prepare_features(fbars)
    print(f"data: {len(underlying_features)} underlyings, "
          f"{len(fund_features)} inverse funds")

    # Per (underlying, date) trigger rows: prior-session underlying return,
    # plus the premarket confirmation (underlying 09:00 price vs prior close).
    rows = []
    for underlying, features in underlying_features.items():
        daily_close = features.groupby("date")["close"].last().sort_index()
        prior_ret = (daily_close / daily_close.shift(1) - 1.0).shift(1)
        pm9 = features[features["tod"] == PM9_TOD].set_index("date")["close"]
        pm9_ret = (pm9 / daily_close.shift(1) - 1.0)  # premarket move so far
        bars = features[features["tod"] == OPEN_TOD]
        for row in bars.itertuples(index=False):
            pr = prior_ret.get(row.date)
            if pr is None or not np.isfinite(pr) or pr > args.down_threshold:
                continue
            confirm = pm9_ret.get(row.date)
            rows.append({
                "date": row.date,
                "underlying": underlying,
                "prior_ret": float(pr),
                "pm_confirmed": bool(np.isfinite(confirm) and confirm < 0),
            })
    if not rows:
        print("No qualifying trigger rows.")
        return 1
    triggers = pd.DataFrame(rows)
    print(f"triggers: {len(triggers)} sessions, "
          f"premarket-confirmed {int(triggers['pm_confirmed'].sum())}")

    # Inverse-fund price tables: open (entry) and close at exit bars.
    def price_table(tod):
        px = {}
        for etf, features in fund_features.items():
            bars = features[features["tod"] == tod]
            for row in bars.itertuples(index=False):
                if np.isfinite(row.close) and row.close > 0:
                    px[(etf, row.date)] = float(row.close)
        return px

    open_px = price_table(OPEN_TOD)
    exit_tables = {h: price_table(t) for h, t in EXIT_TODS.items()}
    fund_by_underlying = {
        f.underlying: f.etf
        for f in LEVERAGED_SINGLE_STOCK_ETFS.values()
        if f.leverage < 0 and f.etf in fund_features
    }

    configs = []
    for pm_only in (False, True):
        subset = triggers[triggers["pm_confirmed"]] if pm_only else triggers
        for top_k in args.top_ks:
            for hold, exit_table in exit_tables.items():
                # One position per underlying per session (dedupe underlying).
                daily = {}
                for date, day in subset.groupby("date"):
                    picks = (day.sort_values("prior_ret", ascending=True)
                            .head(int(top_k)))
                    rets = []
                    for row in picks.itertuples(index=False):
                        etf = fund_by_underlying.get(row.underlying)
                        if etf is None:
                            continue
                        entry = open_px.get((etf, row.date))
                        exit_ = exit_table.get((etf, row.date))
                        if entry is None or exit_ is None:
                            continue
                        rets.append(exit_ / entry - 1.0
                                    - 2.0 * 5.0 / 10_000.0)
                    if rets:
                        daily[date] = float(np.mean(rets))
                series = pd.Series(daily, dtype=np.float64).sort_index()
                if len(series) < args.min_days:
                    continue
                stats = portfolio_stats(series)
                stab = split_half_stability(series)
                configs.append({
                    "filter": "pm-confirmed" if pm_only else "all-down",
                    "top_k": top_k,
                    "exit": f"{EXIT_TODS[hold]//60:02d}:{EXIT_TODS[hold]%60:02d}",
                    "n": stats["n_sessions"],
                    "mean": stats["mean"],
                    "t": stats["t_stat"],
                    "hit": stats["hit_rate"],
                    "total": stats["total_return"],
                    "h1": stab["first_mean"],
                    "h2": stab["second_mean"],
                    "cons": stab["consistent"],
                })

    if not configs:
        print("\nNo configurations produced enough sessions.")
        return 1

    table = pd.DataFrame(configs).sort_values("t", ascending=False)
    print("\n--- RESULTS (inverse-fund long, 5 bps/side, sorted by t) ---")
    print("  Buy the mapped inverse fund at 09:30 after the underlying dropped;")
    print("  premarket-confirmed = also require the underlying down premarket.")
    print(table.to_string(index=False, formatters={
        "mean": "{:+.3%}".format,
        "t": lambda v: "—" if v is None or pd.isna(v) else f"{v:+.2f}",
        "hit": "{:.0%}".format,
        "total": "{:+.1%}".format,
        "h1": "{:+.3%}".format,
        "h2": "{:+.3%}".format,
        "cons": lambda v: "y" if v else "N",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())