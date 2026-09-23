#!/usr/bin/env python3
"""
Research runner for the ETF-level cross-sectional rebalance-flow portfolio.

Estimates each single-stock leveraged ETF's required daily hedge directly
(leverage x AUM x the fund's own return-so-far), aggregates per underlying,
ranks all underlyings cross-sectionally each session, and trades the top-K.
One global rule — no per-symbol parameter fitting.

For each configuration it reports the portfolio's mean daily return with a
t-statistic, split-half stability, and a permutation placebo p-value (random
K-name portfolios from the same session pool).

Read-only diagnostic: places no orders, writes no storage.

    python app/leveraged_flow_backtest.py --start 2025-07-18 --end 2026-09-18
    python app/leveraged_flow_backtest.py --entry-starts 15:20 14:30 --top-ks 3 5
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
    beta_vs_market,
    direction_breakdown,
    etf_flow_estimates,
    flow_magnitude_buckets,
    momentum_scores,
    placebo_portfolio_test,
    portfolio_session_returns,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("leveraged_flow_backtest")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional leveraged-ETF flow portfolio backtest.")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 14 months ago).")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: now).")
    parser.add_argument("--timeframe", default="5m",
                        help="Bar timeframe (default: 5m).")
    parser.add_argument("--entry-starts", nargs="+", default=["14:30", "15:20"],
                        help="Decision times HH:MM ET (default: 14:30 15:20).")
    parser.add_argument("--top-ks", nargs="+", type=int, default=[3, 5],
                        help="Portfolio sizes (default: 3 5).")
    parser.add_argument("--direction", choices=("both", "long", "short"),
                        default="both")
    parser.add_argument("--cost-bps", type=float, default=2.0,
                        help="Per-side cost in bps (default: 2.0).")
    parser.add_argument("--min-flow-usd", type=float, default=1e7,
                        help="Skip sessions whose best |flow| is below this "
                             "(default: $10m). 0 disables.")
    parser.add_argument("--placebo-draws", type=int, default=1000,
                        help="Placebo draws (default: 1000; 0 disables).")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Restrict the underlying universe.")
    parser.add_argument("--skip-etf-data", action="store_true",
                        help="Re-estimate flows from the UNDERLYING's own move "
                             "using the map's leverage/AUM (no ETF fetches). "
                             "Faster, but a proxy rather than the fund return.")
    return parser.parse_args()


def _resolve_dates(args):
    end = datetime.strptime(
        args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = (datetime.strptime(args.start, "%Y-%m-%d")
             if args.start else end - timedelta(days=14 * 30))
    return start, end


def _fetch(symbol: str, start: datetime, end: datetime, timeframe: str):
    bars = data_provider.get_single_stock_bars(
        symbol, start, end, timeframe=timeframe, adjustment=Adjustment.SPLIT)
    if bars is None or bars.empty:
        logger.warning("%s: no bars returned", symbol)
    return bars


def _flow_inputs(args, symbols, start, end):
    """Fetch bars and build the flow source + underlying features."""
    universe = underlyings() if not args.symbols else [
        s.upper() for s in args.symbols]
    logger.info("Fetching underlying bars for %d symbols…", len(universe))
    underlying_features = {}
    for symbol in universe:
        bars = _fetch(symbol, start, end, args.timeframe)
        if bars is None or bars.empty:
            continue
        underlying_features[symbol] = prepare_features(bars)
    logger.info("Underlying data ready for %d/%d symbols.",
                len(underlying_features), len(universe))

    if args.skip_etf_data:
        # Proxy flows: use the underlying's own return-so-far scaled by the
        # aggregated leverage x AUM of the funds tracking it. Cheaper (no ETF
        # fetches) and close in spirit, but it is NOT the fund return.
        leverage: dict = {}
        for fund in LEVERAGED_SINGLE_STOCK_ETFS.values():
            if fund.underlying in underlyings():
                aum = fund.aum_usd or 500_000_000.0
                prev = leverage.get(fund.underlying, (0.0, 0.0))
                leverage[fund.underlying] = (
                    prev[0] + fund.leverage * aum, prev[1] + 1)
        logger.info("Proxy mode: flows = Σ(leverage×AUM) × underlying move.")
        return underlying_features, None, leverage

    etf_symbols = list(LEVERAGED_SINGLE_STOCK_ETFS)
    logger.info("Fetching ETF bars for %d funds…", len(etf_symbols))
    etf_features = {}
    for etf in etf_symbols:
        bars = _fetch(etf, start, end, args.timeframe)
        if bars is None or bars.empty:
            continue
        features = prepare_features(bars)
        if not features.empty:
            etf_features[etf] = features
    logger.info("ETF data ready for %d/%d funds.",
                len(etf_features), len(etf_symbols))
    return underlying_features, etf_features, None


def _evaluate_config(
    ranking_name, flows, underlying_features, entry_tod, args,
    placebo_draws, market_returns,
) -> dict:
    """Evaluate one (ranking rule, entry time, top_k) configuration."""
    returns, decisions = portfolio_session_returns(
        flows, underlying_features, entry_tod=entry_tod,
        top_k=args.top_k, direction_filter=args.direction,
        cost_bps=args.cost_bps,
        min_abs_notional=float(args.min_flow_usd))
    stats = portfolio_stats(returns)
    stability = split_half_stability(returns)
    breakdown = direction_breakdown(returns, decisions)
    beta = beta_vs_market(returns, market_returns) if (
        market_returns is not None) else None
    placebo = None
    if placebo_draws and stats["mean"] is not None:
        placebo = placebo_portfolio_test(
            flows, underlying_features, entry_tod,
            top_k=args.top_k, direction_filter=args.direction,
            cost_bps=args.cost_bps, min_abs_notional=float(args.min_flow_usd),
            observed_mean=stats["mean"], n_draws=placebo_draws)
    return {
        "ranking": ranking_name, "entry": args.entry, "top_k": args.top_k,
        "stats": stats, "stability": stability,
        "breakdown": breakdown, "beta": beta, "placebo": placebo,
        "decisions": decisions, "returns": returns,
    }


def main() -> int:
    args = _parse_args()
    start, end = _resolve_dates(args)
    print("=" * 78)
    print("ETF-LEVEL CROSS-SECTIONAL REBALANCE-FLOW PORTFOLIO")
    print("=" * 78)
    print(f"  Date range : {start.date()} -> {end.date()}")
    print(f"  Timeframe  : {args.timeframe} (adjustment=SPLIT)")
    print(f"  Entry times: {', '.join(args.entry_starts)}")
    print(f"  Top-K      : {args.top_ks}")
    print(f"  Direction  : {args.direction}")
    print(f"  Costs      : {args.cost_bps:.1f} bps/side "
          f"({2*args.cost_bps:.0f} bps round trip)")
    print(f"  Min |flow| : ${args.min_flow_usd:,.0f}")
    print("=" * 78)

    underlying_features, etf_features, proxy_leverage = _flow_inputs(
        args, None, start, end)
    if not underlying_features:
        print("No underlying data available.")
        return 1

    # Market proxy: equal-weight mean of ALL universe names' session returns
    # (close of the decision bar -> session close). This is what the beta
    # regression compares against; uncorrelated return = positive alpha_daily.
    market_by_date: Dict[object, List[float]] = {}
    for symbol, feats in underlying_features.items():
        bars = feats[feats["tod"] == parse_hhmm(args.entry_starts[0])]
        for row in bars.itertuples(index=False):
            market_by_date.setdefault(row.date, []).append(float(row.cum_ret))
    market_returns = pd.Series(
        {d: float(np.nanmean(v)) for d, v in sorted(market_by_date.items())},
        dtype=np.float64).dropna()
    market_returns.index.name = "date"

    results = []
    for entry in args.entry_starts:
        entry_tod = parse_hhmm(entry)

        # Build BOTH ranking frames for this entry time (A/B).
        ranking_frames: Dict[str, pd.DataFrame] = {}
        if etf_features:
            flows = etf_flow_estimates(etf_features, asof_tod=entry_tod)
            if not flows.empty:
                ranking_frames["flow"] = flows
        else:
            rows = []
            for symbol, feats in underlying_features.items():
                lev_aum, _n = proxy_leverage.get(symbol, (0.0, 0))
                if not lev_aum:
                    continue
                bars = feats[feats["tod"] == entry_tod]
                for row in bars.itertuples(index=False):
                    if row.cum_ret is None or not pd.notna(row.cum_ret):
                        continue
                    notional = lev_aum * float(row.cum_ret)
                    rows.append({
                        "date": row.date,
                        "underlying": symbol,
                        "notional": notional,
                        "abs_notional": abs(notional),
                        "n_funds": 1,
                    })
            if rows:
                ranking_frames["flow"] = pd.DataFrame(rows).sort_values(
                    ["date", "abs_notional"], ascending=[True, False])
        # The momentum control uses the SAME machinery, ranked by raw |day ret|.
        momentum = momentum_scores(underlying_features, asof_tod=entry_tod)
        if not momentum.empty:
            ranking_frames["momentum"] = momentum

        if not ranking_frames:
            print(f"\n[{entry}] no ranking data")
            continue

        for ranking_name, flows in ranking_frames.items():
            for top_k in args.top_ks:
                args.entry, args.top_k = entry, top_k
                results.append(_evaluate_config(
                    ranking_name, flows, underlying_features, entry_tod, args,
                    args.placebo_draws, market_returns))

    if not results:
        print("\nNo configurations evaluated.")
        return 1

    print("\n--- RESULTS (sorted by t-statistic) ---")
    print("  ranking: flow = ETF hedge notional (lev×AUM×ret);")
    print("           momentum = CONTROL, raw |day return| (no ETF layer).")
    print("  If flow ≈ momentum, the edge is intraday momentum, NOT rebalances.")
    print("  mean/sess = equal-weighted portfolio daily return, net of costs.")
    print("  alpha_ann = annualized market-regression intercept (uncorrelated ret).")
    rows = []
    for cfg in results:
        s, stab, pl = cfg["stats"], cfg["stability"], cfg["placebo"]
        bd, beta = cfg["breakdown"], cfg["beta"]
        rows.append({
            "ranking": cfg["ranking"],
            "entry": cfg["entry"],
            "top_k": cfg["top_k"],
            "n_sess": s["n_sessions"],
            "mean/sess": s["mean"],
            "t": s["t_stat"],
            "hit%": s["hit_rate"],
            "total": s["total_return"],
            "h1": stab["first_mean"],
            "h2": stab["second_mean"],
            "cons": "y" if stab["consistent"] else "N",
            "L/sess": bd["long_days"]["mean"],
            "L#": bd["long_days"]["n"],
            "S/sess": bd["short_days"]["mean"],
            "S#": bd["short_days"]["n"],
            "beta": beta["beta"] if beta else None,
            "alpha_ann": beta["alpha_ann"] if beta else None,
            "corr": beta["corr"] if beta else None,
            "p": pl["p_value"] if pl else None,
        })
    table = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(table.to_string(index=False, formatters={
        "mean/sess": "{:+.3%}".format,
        "t": "{:.2f}".format,
        "hit%": "{:.0%}".format,
        "total": "{:+.1%}".format,
        "h1": "{:+.3%}".format,
        "h2": "{:+.3%}".format,
        "L/sess": "{:+.3%}".format,
        "S/sess": "{:+.3%}".format,
        "beta": "{:+.2f}".format,
        "alpha_ann": "{:+.1%}".format,
        "corr": "{:+.2f}".format,
        "p": lambda v: "—" if v is None or pd.isna(v) else f"{v:.3f}",
    }))

    significant = table[(table["p"].notna()) & (table["p"] <= 0.05)]
    print(f"\n  Placebo p<=0.05: {len(significant)}/{int(table['p'].notna().sum())} "
          f"configs (expected by chance ≈ "
          f"{0.05 * int(table['p'].notna().sum()):.1f})")

    best = results[int(table.index[0])]
    print("\n--- Best config sample decisions "
          f"({best['entry']}, top_k={best['top_k']}) ---")
    for dec in best["decisions"][-8:]:
        picks = ", ".join(
            f"{s}{'+' if d == 'long' else '-'}(${abs(n)/1e6:.0f}m)"
            for s, d, n in zip(dec.symbols, dec.directions, dec.notionals))
        print(f"  {dec.date}  {picks:<58} {dec.portfolio_return:+.3%}")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"leveraged_flow_portfolio_{stamp}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nWrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
