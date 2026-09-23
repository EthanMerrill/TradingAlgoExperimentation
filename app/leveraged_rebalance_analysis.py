#!/usr/bin/env python3
"""
Research / calibration CLI for the leveraged single-stock ETF rebalance strategy.

For each requested underlying it fetches intraday bars, computes the RVOL /
momentum features, detects late-session block-trade events, and reports how the
underlying moved from the detected entry to the session close (the edge proxy
the strategy is betting on).

This is a standalone diagnostic — it places no orders and touches no storage.
Run it from the repo root:

    python app/leveraged_rebalance_analysis.py --start 2026-06-01 --end 2026-09-01
    python app/leveraged_rebalance_analysis.py --symbols NVDA TSLA --timeframe 1m

Outputs a per-symbol summary to stdout and writes the raw events to CSV.
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Allow running as a script from the repo root or from app/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

from alpaca.data.enums import Adjustment  # noqa: E402

from data_provider import data_provider  # noqa: E402
from strategies.leveraged_rebalance_signal import (  # noqa: E402
    DEFAULT_ENTRY_END,
    DEFAULT_ENTRY_START,
    DEFAULT_MOMENTUM_THRESHOLD,
    DEFAULT_RVOL_LOOKBACK_DAYS,
    DEFAULT_RVOL_THRESHOLD,
    detect_events,
    prepare_features,
)
from strategies.leveraged_single_stock_etfs import (  # noqa: E402
    etfs_for,
    underlyings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("leveraged_rebalance_analysis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect late-session leveraged-ETF rebalance block trades.")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Underlying symbols to analyze (default: all mapped underlyings).")
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: 90 days ago).")
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: now).")
    parser.add_argument(
        "--timeframe", default="5m",
        help="Bar timeframe (default: 5m).")
    parser.add_argument(
        "--entry-start", default=DEFAULT_ENTRY_START,
        help=f"Entry window start HH:MM ET (default: {DEFAULT_ENTRY_START}).")
    parser.add_argument(
        "--entry-end", default=DEFAULT_ENTRY_END,
        help=f"Entry window end HH:MM ET (default: {DEFAULT_ENTRY_END}).")
    parser.add_argument(
        "--rvol-threshold", type=float, default=DEFAULT_RVOL_THRESHOLD,
        help=f"Minimum RVOL (default: {DEFAULT_RVOL_THRESHOLD}).")
    parser.add_argument(
        "--momentum-threshold", type=float, default=DEFAULT_MOMENTUM_THRESHOLD,
        help=f"Minimum |momentum| as a decimal (default: {DEFAULT_MOMENTUM_THRESHOLD}).")
    parser.add_argument(
        "--rvol-lookback", type=int, default=DEFAULT_RVOL_LOOKBACK_DAYS,
        help=f"RVOL baseline lookback sessions (default: {DEFAULT_RVOL_LOOKBACK_DAYS}).")
    parser.add_argument(
        "--enter-at", choices=("trigger", "window_start"), default="trigger",
        help="'trigger' = enter when the spike fires; 'window_start' = enter at "
             "the window open on qualifying days (default: trigger).")
    parser.add_argument(
        "--output", default=None,
        help="CSV output path (default: logs/leveraged_rebalance_events_<tf>.csv).")
    return parser.parse_args()


def _resolve_dates(args: argparse.Namespace):
    end = (datetime.strptime(args.end, "%Y-%m-%d")
           if args.end else datetime.now())
    start = (datetime.strptime(args.start, "%Y-%m-%d")
             if args.start else end - timedelta(days=90))
    return start, end


def analyze_symbol(
    symbol: str,
    start: datetime,
    end: datetime,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Fetch bars for one underlying and return its detected events (with edge)."""
    bars = data_provider.get_single_stock_bars(
        symbol, start, end, timeframe=args.timeframe,
        adjustment=Adjustment.SPLIT)
    if bars is None or bars.empty:
        logger.warning("%s: no bars returned", symbol)
        return pd.DataFrame()

    features = prepare_features(bars, args.rvol_lookback)
    events = detect_events(
        features,
        entry_start=args.entry_start,
        entry_end=args.entry_end,
        rvol_threshold=args.rvol_threshold,
        momentum_threshold=args.momentum_threshold,
        enter_at=args.enter_at,
    )
    if events.empty:
        logger.info("%s: no events detected", symbol)
        return pd.DataFrame()

    session_close = features.groupby("date")["close"].last()
    etf_tickers = ",".join(e.etf for e in etfs_for(symbol))

    records = []
    for event in events.itertuples(index=False):
        close_px = session_close.get(event.date)
        if close_px is None:
            continue
        sign = 1.0 if event.direction == "long" else -1.0
        move_to_close = (float(close_px) / event.entry_price - 1.0) * sign
        records.append({
            "symbol": symbol,
            "etfs": etf_tickers,
            "date": event.date,
            "entry_ts": event.entry_ts,
            "direction": event.direction,
            "rvol": event.rvol,
            "momentum": event.momentum,
            "entry_price": event.entry_price,
            "session_close": float(close_px),
            "move_to_close": move_to_close,
        })
    return pd.DataFrame.from_records(records)


def main() -> int:
    args = _parse_args()
    start, end = _resolve_dates(args)
    symbols = [s.upper() for s in (args.symbols or underlyings())]

    logger.info("Analyzing %d underlyings from %s to %s (tf=%s, window %s-%s)",
                len(symbols), start.date(), end.date(), args.timeframe,
                args.entry_start, args.entry_end)

    frames = []
    for symbol in symbols:
        try:
            frame = analyze_symbol(symbol, start, end, args)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("%s: analysis failed: %s", symbol, e)
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        logger.warning("No events detected across %d symbols.", len(symbols))
        return 1

    events = pd.concat(frames, ignore_index=True)

    summary = (
        events.groupby("symbol")
        .agg(
            events=("move_to_close", "size"),
            avg_rvol=("rvol", "mean"),
            avg_momentum=("momentum", lambda s: s.abs().mean()),
            hit_rate=("move_to_close", lambda s: (s > 0).mean()),
            avg_move_to_close=("move_to_close", "mean"),
        )
        .sort_values("avg_move_to_close", ascending=False)
    )

    print("\n=== Detected rebalance events ===")
    print(summary.to_string(
        formatters={
            "avg_rvol": "{:.2f}".format,
            "avg_momentum": "{:.4f}".format,
            "hit_rate": "{:.0%}".format,
            "avg_move_to_close": "{:+.3%}".format,
        }))

    overall_hit = (events["move_to_close"] > 0).mean()
    print(
        f"\nTotal events: {len(events)} across {events['symbol'].nunique()} symbols")
    print(f"Overall hit rate (entry → close): {overall_hit:.1%}")
    print(
        f"Overall avg move (entry → close): {events['move_to_close'].mean():+.3%}")
    print(f"Total edge if taken every event (equal weight): "
          f"{events['move_to_close'].sum():+.1%}")

    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "logs", f"leveraged_rebalance_events_{args.timeframe}.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    events.to_csv(output, index=False)
    logger.info("Wrote %d events to %s", len(events), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
