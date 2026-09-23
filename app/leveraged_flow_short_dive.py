#!/usr/bin/env python3
"""
Short-side deep dive for the leveraged-ETF flow portfolio.

Question: does the short side fail only because the sample lacks EXTREME
negative-flow days (the Samsung/SK Hynix leveraged-ETF unwind scenario)?

Three probes, all long/short symmetric in machinery:

1. BUCKETS — every qualifying (symbol, session) pair bucketed into |flow|
   quartiles. If the short edge is real but rare, mean return should improve
   monotonically in the largest-|flow| buckets.
2. THRESHOLDS — short-only portfolio restricted to |flow| >= T for escalating
   T (25m/50m/100m/250m). Also reports the MAX |flow| ever seen, so we can
   compare against Korean-meltdown scale.
3. DRAWDOWN DAYS — the worst market days (bottom decile of the equal-weight
   market proxy) versus the short book's return on those days: does the short
   side work when it matters?

    python app/leveraged_flow_short_dive.py --start 2025-07-18 --end 2026-09-18
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

from alpaca.data.enums import Adjustment  # noqa: E402

from config import globalConfig  # noqa: E402
from data_provider import data_provider  # noqa: E402
from strategies.leveraged_flow_portfolio import (  # noqa: E402
    flow_magnitude_buckets,
    portfolio_session_returns,
    portfolio_stats,
)
from strategies.leveraged_rebalance_signal import (  # noqa: E402
    parse_hhmm,
    prepare_features,
)
from strategies.leveraged_single_stock_etfs import (  # noqa: E402
    LEVERAGED_SINGLE_STOCK_ETFS,
    underlyings,
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-07-18")
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--entry-starts", nargs="+",
                        default=["14:30", "15:20"])
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[25e6, 50e6, 100e6, 250e6],
                        help="Short-only |flow| thresholds in USD.")
    parser.add_argument("--buckets", type=int, default=4)
    return parser.parse_args()


def _proxy_flows(underlying_features, entry_tod, leverage_aum):
    """Proxy flows: Σ(leverage×AUM) × underlying return-so-far."""
    rows = []
    for symbol, feats in underlying_features.items():
        lev_aum, _ = leverage_aum.get(symbol, (0.0, 0))
        if not lev_aum:
            continue
        bars = feats[feats["tod"] == entry_tod]
        for row in bars.itertuples(index=False):
            if row.cum_ret is None or not pd.notna(row.cum_ret):
                continue
            notional = lev_aum * float(row.cum_ret)
            rows.append({"date": row.date, "underlying": symbol,
                         "notional": notional, "abs_notional": abs(notional),
                         "n_funds": 1})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["date", "abs_notional"], ascending=[True, False])


def main() -> int:
    args = _parse_args()
    end = (datetime.strptime(args.end, "%Y-%m-%d")
           if args.end else datetime.now())
    start = datetime.strptime(args.start, "%Y-%m-%d")

    print("=" * 78)
    print("SHORT-SIDE DEEP DIVE — is the short edge real but rare?")
    print("=" * 78)

    universe = underlyings()
    underlying_features = {}
    for symbol in universe:
        bars = data_provider.get_single_stock_bars(
            symbol, start, end, timeframe=args.timeframe,
            adjustment=Adjustment.SPLIT)
        if bars is not None and not bars.empty:
            underlying_features[symbol] = prepare_features(bars)
    print(
        f"Underlying data: {len(underlying_features)}/{len(universe)} symbols")

    leverage_aum = {}
    for fund in LEVERAGED_SINGLE_STOCK_ETFS.values():
        if fund.underlying in underlying_features:
            aum = fund.aum_usd or 500_000_000.0
            prev = leverage_aum.get(fund.underlying, (0.0, 0))
            leverage_aum[fund.underlying] = (prev[0] + fund.leverage * aum,
                                             prev[1] + 1)

    # Market proxy for the drawdown-day probe (first entry time).
    base_tod = parse_hhmm(args.entry_starts[0])
    market_by_date = {}
    for feats in underlying_features.values():
        bars = feats[feats["tod"] == base_tod]
        for row in bars.itertuples(index=False):
            market_by_date.setdefault(row.date, []).append(float(row.cum_ret))
    market_returns = pd.Series(
        {d: float(np.nanmean(v)) for d, v in sorted(market_by_date.items())}
    ).dropna()

    for entry in args.entry_starts:
        entry_tod = parse_hhmm(entry)
        flows = _proxy_flows(underlying_features, entry_tod, leverage_aum)
        if flows.empty:
            continue
        print(f"\n{'=' * 78}\nENTRY {entry} (costs {args.cost_bps:.0f} bps/side)"
              f"\n{'=' * 78}")

        # 1. |flow| quartile buckets, short side only.
        buckets = flow_magnitude_buckets(
            flows, underlying_features, entry_tod,
            cost_bps=args.cost_bps, direction_filter="short",
            buckets=args.buckets)
        print("\n--- Short returns by |flow| bucket (1=smallest … N=largest) ---")
        if buckets.empty:
            print("  (no short-side data)")
        else:
            print(buckets.to_string(index=False, formatters={
                "mean": "{:+.3%}".format,
                "t": lambda v: "—" if v is None or pd.isna(v) else f"{v:+.2f}",
                "hit_rate": "{:.0%}".format,
                "total": "{:+.1%}".format,
                "median_abs_notional": "${:,.0f}".format,
            }))
            improving = buckets["mean"].is_monotonic_increasing
            print(f"  monotonic improvement toward large flows: "
                  f"{'YES' if improving else 'no'}")

        # Long buckets too, for contrast.
        lbuckets = flow_magnitude_buckets(
            flows, underlying_features, entry_tod,
            cost_bps=args.cost_bps, direction_filter="long",
            buckets=args.buckets)
        print("\n--- Long returns by |flow| bucket (contrast) ---")
        if not lbuckets.empty:
            print(lbuckets.to_string(index=False, formatters={
                "mean": "{:+.3%}".format,
                "t": lambda v: "—" if v is None or pd.isna(v) else f"{v:+.2f}",
                "hit_rate": "{:.0%}".format,
                "total": "{:+.1%}".format,
                "median_abs_notional": "${:,.0f}".format,
            }))

        # 2. Threshold sweep, short-only portfolio.
        print("\n--- Short-only portfolio vs |flow| threshold ---")
        max_flow = flows["abs_notional"].max()
        print(f"  max |flow| observed: ${max_flow/1e6:,.0f}m "
              f"(Samsung/SK scenario would be $Bns)")
        rows = []
        for threshold in args.thresholds:
            returns, _ = portfolio_session_returns(
                flows, underlying_features, entry_tod=entry_tod, top_k=3,
                direction_filter="short", cost_bps=args.cost_bps,
                min_abs_notional=float(threshold))
            stats = portfolio_stats(returns)
            rows.append({
                "min |flow|": f"${threshold/1e6:,.0f}m",
                "n_sess": stats["n_sessions"],
                "mean/sess": stats["mean"],
                "t": stats["t_stat"],
                "hit%": stats["hit_rate"],
                "total": stats["total_return"],
            })
        print(pd.DataFrame(rows).to_string(index=False, formatters={
            "mean/sess": "{:+.3%}".format,
            "t": lambda v: "—" if v is None or pd.isna(v) else f"{v:+.2f}",
            "hit%": "{:.0%}".format,
            "total": "{:+.1%}".format,
        }))

        # 3. Worst market days: does the short book fire when it matters?
        short_returns, _ = portfolio_session_returns(
            flows, underlying_features, entry_tod=entry_tod, top_k=3,
            direction_filter="short", cost_bps=args.cost_bps,
            min_abs_notional=0.0)
        if not short_returns.empty:
            joined = pd.concat(
                {"mkt": market_returns, "short": short_returns},
                axis=1).dropna()
            if not joined.empty:
                cutoff = joined["mkt"].quantile(0.10)
                worst = joined[joined["mkt"] <= cutoff]
                print(f"\n--- Bottom-decile market days "
                      f"(mkt <= {cutoff:+.2%}, n={len(worst)}) ---")
                if not worst.empty:
                    print(f"  market mean : {worst['mkt'].mean():+.3%}")
                    print(f"  short mean  : {worst['short'].mean():+.3%}")
                    print(f"  short hit   : {(worst['short'] > 0).mean():.0%}")
                else:
                    print("  (no overlap)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
