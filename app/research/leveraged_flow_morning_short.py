#!/usr/bin/env python3
"""
Morning-short research runner for the leveraged-ETF flow universe.

Hypothesis (user's): after a big DOWN day, the dealer/ETF hedge overhang keeps
pressuring the name — short the worst-flow names at the open and cover 30-60
minutes later. Empirical grounding: a 6-month probe showed prior-day-down
names CONTINUE into the open (premarket 9:00→open: −0.161%, t=−2.44 on
prior-day ≤ −2% days) — the opposite regime of the close window, where
down-days bounce.

Configuration (user decisions):
* selection = most-negative estimated hedge flow (leverage×AUM×prior-day
  return) from the ETF map, top-K
* timing = compare open→30min vs open→60min (premarket entry excluded: the
  premarket edge is inside the premarket bar-range noise, ~0.14%)
* entry price = the open bar's OPEN price (market order at the bell)
* costs = 5 bps/side base, 10 bps/side stress

Controls (all identical machinery, only the ranking differs):
* ``momentum`` — top-K by prior-day |return| (biggest losers), no ETF layer
* ``fade``     — LONG the biggest prior-day losers at the open: tests whether
  the morning continues or reverses the down move. If fade wins, the sign of
  the hypothesis is wrong.

Read-only diagnostic: places no orders, writes no storage.

    python app/leveraged_flow_morning_short.py --start 2026-03-22 --end 2026-09-22
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
from strategies.leveraged_rebalance_signal import (  # noqa: E402
    parse_hhmm,
    prepare_features,
)
from strategies.leveraged_single_stock_etfs import (  # noqa: E402
    LEVERAGED_SINGLE_STOCK_ETFS,
    underlyings,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("leveraged_flow_morning_short")

OPEN_TOD = 9 * 60 + 30          # 09:30 ET first RTH bar
EXIT_TODS = {30: 10 * 60,       # open→10:00
             60: 10 * 60 + 30}  # open→10:30


def _price_table(underlying_features: dict, exit_tod: int):
    """Exit prices at ``exit_tod`` per (symbol, date)."""
    px = {}
    for symbol, features in underlying_features.items():
        bars = features[features["tod"] == exit_tod]
        for row in bars.itertuples(index=False):
            if np.isfinite(row.close) and row.close > 0:
                px[(symbol, row.date)] = float(row.close)
    return px


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--top-ks", nargs="+", type=int, default=[3])
    parser.add_argument("--down-threshold", type=float, default=-0.02,
                        help="Prior-session return must be ≤ this (default -0.02).")
    parser.add_argument("--cost-ladder", nargs="+", type=float,
                        default=[5.0, 10.0])
    parser.add_argument("--min-days", type=int, default=40)
    args = parser.parse_args()

    end = (datetime.strptime(args.end, "%Y-%m-%d")
           if args.end else datetime.now())
    start = (datetime.strptime(args.start, "%Y-%m-%d")
             if args.start else end - timedelta(days=180))

    print("=" * 78)
    print("MORNING SHORT ��� most-negative-flow names after a down day")
    print(f"  window   : prior-day ret ≤ {args.down_threshold:.1%}, "
          f"short at 09:30 open, cover at 10:00 / 10:30")
    print(f"  ranking  : flow (lev×AUM×prior-ret) vs momentum control vs fade control")
    print(f"  costs    : 5 bps/side base, 10 bps/side stress")
    print("=" * 78)

    universe = underlyings()
    underlying_features = {}
    lev_aum = {}
    for fund in LEVERAGED_SINGLE_STOCK_ETFS.values():
        aum = fund.aum_usd or 500_000_000.0
        prev = lev_aum.get(fund.underlying, 0.0)
        lev_aum[fund.underlying] = prev + fund.leverage * aum

    for symbol in universe:
        bars = data_provider.get_single_stock_bars(
            symbol, start - timedelta(days=10), end,
            timeframe=args.timeframe, adjustment=Adjustment.SPLIT)
        if bars is None or bars.empty:
            continue
        underlying_features[symbol] = prepare_features(bars)
    print(f"Data: {len(underlying_features)}/{len(universe)} underlyings")

    # NOTE: ranking frames are built per exit horizon only for prices; the
    # scores frame is timing-independent.
    base_scores = None
    results = []
    for ranking in ("flow", "momentum", "fade"):
        scores = None
        for symbol, features in underlying_features.items():
            daily_close = features.groupby("date")["close"].last().sort_index()
            prior_ret = (daily_close / daily_close.shift(1) - 1.0).shift(1)
            bars = features[features["tod"] == OPEN_TOD]
            for row in bars.itertuples(index=False):
                pr = prior_ret.get(row.date)
                if pr is None or not np.isfinite(pr):
                    continue
                if pr > args.down_threshold:
                    continue  # shorts require a qualifying down prior session
                weight = lev_aum.get(symbol, 0.0) if ranking == "flow" else 1.0
                results_frame_row = {
                    "date": row.date,
                    "underlying": symbol,
                    "prior_ret": float(pr),
                    "open_price": float(row.open),
                    "score": float(pr) * weight,
                }
                if scores is None:
                    scores = []
                scores.append(results_frame_row)
        frame = pd.DataFrame(scores) if scores else pd.DataFrame()
        if frame.empty:
            continue
        frame.attrs["ranking"] = ranking
        for top_k in args.top_ks:
            for exit_tod in EXIT_TODS.values():
                exit_px = _price_table(underlying_features, exit_tod)
                round_trip = 2.0 * 5.0 / 10_000.0
                daily = {}
                for date, day in frame.groupby("date"):
                    if ranking == "flow":
                        sel = day.sort_values("score", ascending=True)
                    else:
                        sel = day.sort_values("prior_ret", ascending=True)
                    sel = sel.head(int(top_k))
                    if sel.empty:
                        continue
                    rets = []
                    for row in sel.itertuples(index=False):
                        key = (row.underlying, row.date)
                        if key not in exit_px:
                            continue
                        gross = exit_px[key] / row.open_price - 1.0
                        if ranking != "fade":
                            gross *= -1.0
                        rets.append(gross - round_trip)
                    if rets:
                        daily[date] = float(np.mean(rets))
                series = pd.Series(daily, dtype=np.float64).sort_index()
                if len(series) < args.min_days:
                    continue
                stats = portfolio_stats(series)
                stab = split_half_stability(series)
                results.append({
                    "ranking": ranking,
                    "top_k": top_k,
                    "exit": f"{exit_tod // 60:02d}:{exit_tod % 60:02d}",
                    "n": stats["n_sessions"],
                    "mean": stats["mean"],
                    "t": stats["t_stat"],
                    "hit": stats["hit_rate"],
                    "total": stats["total_return"],
                    "h1": stab["first_mean"],
                    "h2": stab["second_mean"],
                    "cons": stab["consistent"],
                })

    if not results:
        print("\nNo configurations produced enough sessions.")
        return 1

    table = pd.DataFrame(results).sort_values("t", ascending=False)
    print("\n--- RESULTS (10 bps round trip shown; sorted by t) ---")
    print("  ranking: flow = most-negative hedge flow; momentum = biggest")
    print("  prior-day losers (control); fade = LONG the losers (sign check).")
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