#!/usr/bin/env python3
"""
Standalone runner for the leveraged-ETF rebalance strategy's backtest.

Runs **only** ``leveraged_etf_rebalance`` (not the other enabled strategies)
over its own underlying universe and prints the full decision trail: the
parameters the optimizer picked per symbol/direction, the resulting metrics,
how many would pass the framework's live-trading filter, and every simulated
trade with the exact bar that triggered it.

This is a read-only diagnostic — it places no orders and writes no storage.

Usage (from the repo root):

    python app/leveraged_rebalance_backtest.py
    python app/leveraged_rebalance_backtest.py --start 2026-06-01 --end 2026-09-18
    python app/leveraged_rebalance_backtest.py --symbols NVDA TSLA --long-only
    python app/leveraged_rebalance_backtest.py --verbose --max-trades 50
"""
import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Allow running as a script from the repo root or from app/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402

import zscore  # noqa: E402
from config import globalConfig  # noqa: E402
from data_provider import data_provider  # noqa: E402
from optimizer import StrategyOptimizer  # noqa: E402
from strategies.leveraged_rebalance import LeveragedRebalanceStrategy  # noqa: E402
from strategies.leveraged_rebalance_signal import placebo_alpha_test  # noqa: E402
from strategies.leveraged_single_stock_etfs import underlyings  # noqa: E402
from walk_forward import WalkForwardValidator  # noqa: E402

logger = logging.getLogger("leveraged_rebalance_backtest")

# Mirrors StrategyOptimizer.filter_results (the live-trading gate).
_MIN_WIN_RATE = 0.3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the leveraged-ETF rebalance strategy backtest.")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Underlying symbols (default: all mapped underlyings).")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 90 days ago).")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: now).")
    parser.add_argument("--timeframe", default="5m",
                        help="Bar timeframe (default: 5m).")
    parser.add_argument("--cost-bps", type=float, default=2.0,
                        help="Round-trip cost in basis points (default: 2.0).")
    parser.add_argument("--rvol-lookback", type=int, default=10,
                        help="RVOL baseline lookback sessions (default: 10).")
    parser.add_argument("--long-only", action="store_true",
                        help="Skip the short direction (overrides config).")
    parser.add_argument("--max-trades", type=int, default=25,
                        help="Max trade rows to print (default: 25; 0 = all).")
    parser.add_argument("--output-dir", default=None,
                        help="CSV output directory (default: app/logs).")
    parser.add_argument("--verbose", action="store_true",
                        help="Show the optimizer's INFO logs.")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run IS/OOS walk-forward validation instead of a "
                             "single in-sample grid search.")
    parser.add_argument("--is-months", type=int, default=None,
                        help="Walk-forward in-sample window in months.")
    parser.add_argument("--oos-months", type=int, default=None,
                        help="Walk-forward out-of-sample window in months.")
    parser.add_argument("--step-months", type=int, default=None,
                        help="Walk-forward step between windows in months.")
    parser.add_argument("--placebo-draws", type=int, default=2000,
                        help="Random draws for the placebo permutation test "
                             "(default: 2000; 0 disables).")
    return parser.parse_args()


def _resolve_dates(args: argparse.Namespace):
    end = datetime.strptime(
        args.end, "%Y-%m-%d") if args.end else datetime.now()
    start = (datetime.strptime(args.start, "%Y-%m-%d")
             if args.start else end - timedelta(days=90))
    return start, end


def _passes_filter(result) -> bool:
    """The framework's live-trading gate (see StrategyOptimizer.filter_results)."""
    return (result.alpha > 0
            and result.profitable
            and result.num_trades > 0
            and result.win_rate > _MIN_WIN_RATE)


def _print_run_config(args, strategy, symbols, start, end) -> None:
    print("\n" + "=" * 78)
    print("LEVERAGED SINGLE-STOCK ETF REBALANCE — BACKTEST")
    print("=" * 78)
    print(f"  Strategy      : {strategy.name}")
    print(f"  Universe      : {len(symbols)} underlyings "
          f"({len(strategy.symbol_universe())} mapped)")
    print(f"  Date range    : {start.date()} → {end.date()}")
    print(f"  Timeframe     : {args.timeframe}  (adjustment=SPLIT)")
    print(
        f"  Grid combos   : {len(strategy.get_param_grid('long'))} per direction")
    print(f"  Short selling : "
          f"{'ENABLED' if globalConfig.ENABLE_SHORT_SELLING else 'DISABLED'}")
    print(f"  Cost model    : {args.cost_bps:.1f} bps round trip per trade")
    print(f"  RVOL lookback : {args.rvol_lookback} sessions")
    print(f"  Warmup        : {strategy.warmup_days()} calendar days")
    print("=" * 78)


def _decisions_frame(results, info=None) -> pd.DataFrame:
    """One row per symbol/direction: the chosen params + resulting metrics."""
    info = info or {}
    rows = []
    for r in results:
        p = dict(getattr(r, "params", None) or {})
        direction = p.get("direction_mode", r.direction)
        extra = info.get((r.symbol, direction), {})
        rows.append({
            "symbol": r.symbol,
            "dir": direction,
            "window": f"{p.get('entry_start', '?')}-{p.get('entry_end', '?')}",
            "rvol>=": p.get("rvol_threshold"),
            "mom>=": p.get("momentum_threshold"),
            "enter": p.get("enter_at"),
            "exit": p.get("exit_mode"),
            "trades": r.num_trades,
            "prof": (f"{extra.get('profitable', 0)}/{extra.get('combos', 0)}"
                     if extra else ""),
            "win%": r.win_rate,
            "ret%": r.total_return,
            "bench%": (r.params or {}).get("benchmark_return"),
            "alpha%": r.alpha,
            "sharpe": r.sharpe_ratio,
            "maxDD%": r.max_drawdown,
            "score": r.composite_score,
            "pass": "yes" if _passes_filter(r) else "no",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def _print_decisions(df: pd.DataFrame) -> None:
    print("\n--- Chosen parameters & performance (sorted by composite score) ---")
    print("  NOTE: in-sample grid search — the best combo per symbol is selected on")
    print("  the same data it is scored on, so these returns are optimistic. Use")
    print("  walk-forward validation for an out-of-sample estimate.")
    print("  'bench%' = mean per-session return of holding the window every day")
    print("  with no RVOL filter; 'alpha%' = per-session excess = detection value.")
    print("  'p' = placebo permutation p-value on the OOS slice (p<=0.05 = real).")
    if df.empty:
        print("  (no results)")
        return
    print(df.to_string(index=False, formatters={
        "rvol>=": "{:.1f}".format,
        "mom>=": "{:.3%}".format,
        "win%": "{:.0%}".format,
        "ret%": "{:+.2%}".format,
        "bench%": "{:+.2%}".format,
        "alpha%": "{:+.2%}".format,
        "sharpe": "{:.2f}".format,
        "maxDD%": "{:.2%}".format,
        "score": "{:.2f}".format,
    }))


def _print_consensus(df: pd.DataFrame) -> None:
    """Which parameter values the optimizer converged on across the universe."""
    if df.empty:
        return
    print("\n--- Parameter consensus (what the optimizer converged on) ---")
    for column in ("window", "rvol>=", "mom>=", "exit", "dir"):
        counts = df[column].value_counts(dropna=False)
        summary = ", ".join(f"{value} ×{count}" for value,
                            count in counts.items())
        print(f"  {column:<8}: {summary}")


def _trades_frame(results) -> pd.DataFrame:
    """Every simulated trade, keeping the detection detail from trade_details."""
    rows = []
    for r in results:
        for t in (r.trade_details or []):
            rows.append({
                "symbol": r.symbol,
                "dir": t.get("direction"),
                "trigger_ts": t.get("entry_ts"),
                "entry_date": t.get("entry_date"),
                "entry_px": t.get("entry_price"),
                "rvol": t.get("rvol"),
                "momentum": t.get("momentum"),
                "exit_date": t.get("exit_date"),
                "exit_px": t.get("exit_price"),
                "exit_reason": t.get("exit_reason"),
                "gross%": t.get("gross_return"),
                "net%": t.get("return"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("trigger_ts").reset_index(drop=True)
    return df


def _print_trades(df: pd.DataFrame, max_rows: int) -> None:
    print("\n--- Simulated trades (the decisions it made) ---")
    if df.empty:
        print("  (no trades triggered in this window)")
        return
    shown = df if max_rows <= 0 else df.head(max_rows)
    display = shown.copy()
    if "trigger_ts" in display.columns:
        display["trigger_ts"] = display["trigger_ts"].map(
            lambda v: v.strftime("%Y-%m-%d %H:%M") if hasattr(v, "strftime")
            else str(v))
    print(display.to_string(index=False, formatters={
        "entry_px": "{:.2f}".format,
        "exit_px": "{:.2f}".format,
        "rvol": "{:.2f}".format,
        "momentum": "{:+.2%}".format,
        "gross%": "{:+.2%}".format,
        "net%": "{:+.2%}".format,
    }))
    if max_rows > 0 and len(df) > max_rows:
        print(f"  … {len(df) - max_rows} more trade(s) — see the CSV output.")


def _print_summary(results, df: pd.DataFrame, trades: pd.DataFrame) -> None:
    print("\n--- Aggregate ---")
    if df.empty:
        print("  No symbol produced a usable backtest.")
        return
    passed = int((df["pass"] == "yes").sum())
    print(f"  Symbol/direction runs        : {len(df)}")
    print(f"  Runs with trades             : {int((df['trades'] > 0).sum())}")
    print(f"  Pass live-trading filter     : {passed} "
          f"(alpha>0, profitable, trades>0, win>{_MIN_WIN_RATE:.0%})")
    with_trades = df[df["trades"] > 0]
    if not with_trades.empty:
        print(f"  Mean return (all runs)       : {df['ret%'].mean():+.2%}")
        print(
            f"  Mean return (traded runs)    : {with_trades['ret%'].mean():+.2%}")
        print(
            f"  Mean win rate (traded runs)  : {with_trades['win%'].mean():.0%}")
    if not trades.empty:
        print(f"  Total simulated trades       : {len(trades)}")
        print(
            f"  Trade hit rate               : {(trades['net%'] > 0).mean():.1%}")
        print(f"  Mean net return per trade    : {trades['net%'].mean():+.3%}")
        print(
            f"  Mean gross return per trade  : {trades['gross%'].mean():+.3%}")
        print(f"  Cost drag per trade          : "
              f"{(trades['gross%'] - trades['net%']).mean():.3%}")
        print(f"  Long / short trades          : "
              f"{int((trades['dir'] == 'long').sum())} / "
              f"{int((trades['dir'] == 'short').sum())}")


def _run_grid(strategy, symbols, start, end, directions, placebo_draws=0):
    """Run the full parameter grid for each symbol/direction.

    Unlike ``Strategy.optimize`` (which returns only the best *profitable*
    combo, or None), this keeps every combo so the report can show what the
    strategy decided even when nothing was profitable.

    Returns ``(best_results, info)``: one BacktestResult per symbol/direction,
    plus ``info[(symbol, direction)] = {"combos": n, "profitable": k}``.
    """
    best_results = []
    info = {}
    warmup = strategy.warmup_days()
    total = len(symbols)

    for index, symbol in enumerate(symbols, 1):
        print(f"  [{index}/{total}] {symbol} …", end="", flush=True)
        bars = data_provider.get_single_stock_bars(
            symbol, start - timedelta(days=warmup), end,
            timeframe=strategy.data_timeframe,
            adjustment=strategy.data_adjustment)

        if bars is None or bars.empty or len(bars) < 50:
            print(" skipped (insufficient data)")
            continue

        prepared = strategy.prepare(bars)

        for direction in directions:
            grid = strategy.get_param_grid(direction)
            combos = [
                strategy.backtest(
                    bars, symbol, globalConfig.BACKTEST_INIT_CASH,
                    prepared=prepared, **params)
                for params in grid
            ]
            # Score the pool with the framework's own z-score composite.
            scores = zscore.compute_metric_triple_zscores(
                [(r.alpha, r.sharpe_ratio, r.calmar_ratio) for r in combos])
            for result, score in zip(combos, scores):
                result.composite_score = score

            # Prefer combos that actually traded: a no-trade combo has all-zero
            # metrics and can outrank losing combos purely via pool centering.
            traded = [r for r in combos if r.num_trades > 0]
            pool = traded or combos
            best = max(pool, key=lambda r: r.composite_score)

            # Falsification test: is the alpha the detection produced better
            # than randomly picking the same number of sessions?
            if placebo_draws and best.num_trades > 0 and best.alpha != 0.0:
                bp = best.params
                best.params["placebo"] = placebo_alpha_test(
                    prepared,
                    entry_start=bp["entry_start"],
                    entry_end=bp["entry_end"],
                    exit_mode=bp["exit_mode"],
                    hold_days=bp["hold_days"],
                    cost_bps=bp["cost_bps"],
                    direction_mode=bp["direction_mode"],
                    n_selected=best.num_trades,
                    observed_alpha=best.alpha,
                    n_draws=placebo_draws,
                )

            best_results.append(best)
            info[(symbol, direction)] = {
                "combos": len(combos),
                "profitable": sum(1 for r in combos if r.profitable),
            }

        print(f" {len(bars)} bars")

    return best_results, info


def _print_placebo(results) -> pd.DataFrame:
    """Report the placebo permutation test: is the alpha better than luck?"""
    rows = []
    for r in results:
        test = (r.params or {}).get("placebo")
        if not test or test.get("p_value") is None:
            continue
        rows.append({
            "symbol": r.symbol,
            "dir": (r.params or {}).get("direction_mode"),
            "trades": r.num_trades,
            "sessions": test["n_sessions"],
            "alpha/sess": r.alpha,
            "null_mean": test["null_mean"],
            "null_p95": test["null_p95"],
            "pctile": test["percentile"],
            "p": test["p_value"],
            "sig": "YES" if test["p_value"] <= 0.05 else "",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("p").reset_index(drop=True)
    print("\n--- Placebo permutation test (detection vs. random day-picking) ---")
    print("  Null: pick this many sessions at random. p = P(random alpha >= observed).")
    print("  sig=YES ⇒ p<=0.05, i.e. the RVOL/momentum filter beat random picking.")
    print(df.to_string(index=False, formatters={
        "alpha/sess": "{:+.3%}".format,
        "null_mean": "{:+.3%}".format,
        "null_p95": "{:+.3%}".format,
        "pctile": "{:.0f}".format,
        "p": "{:.3f}".format,
    }))
    return df


def _run_walk_forward(strategy, symbols, start, end, directions):
    """Run IS/OOS walk-forward validation per symbol/direction.

    For each rolling window the strategy's grid is optimized on the in-sample
    slice and the chosen parameters are then evaluated on the out-of-sample
    slice. Returns ``(results, info)`` where ``info`` carries per-window detail.
    """
    validator = WalkForwardValidator(StrategyOptimizer(strategy=strategy))
    info = {}

    print(f"\nWalk-forward: IS={globalConfig.WF_IS_MONTHS}m "
          f"OOS={globalConfig.WF_OOS_MONTHS}m "
          f"step={globalConfig.WF_STEP_MONTHS}m "
          f"(min {globalConfig.WF_MIN_WINDOWS} windows)")
    print(
        f"  warmup={strategy.warmup_days()}d, timeframe={strategy.data_timeframe}")

    total = len(symbols)
    per_symbol = []
    for index, symbol in enumerate(symbols, 1):
        print(f"  [{index}/{total}] {symbol} …", end="", flush=True)
        bars = data_provider.get_single_stock_bars(
            symbol, start - timedelta(days=strategy.warmup_days()), end,
            timeframe=strategy.data_timeframe,
            adjustment=strategy.data_adjustment)
        if bars is None or bars.empty:
            print(" skipped (no data)")
            continue

        symbol_results = []
        for direction in directions:
            wf = validator.validate_symbol(
                symbol, start, end, direction, prefetched_full_data=bars)
            if wf is not None:
                symbol_results.append(wf)
                info[(symbol, direction)] = wf
        per_symbol.append((symbol, symbol_results))
        windows = sum(len(w.windows) for w in symbol_results)
        print(f" {windows} window(s)")

    return per_symbol, info


def _print_wf_decisions(per_symbol, info) -> None:
    """Per symbol/direction: aggregate OOS metrics + chosen params."""
    print("\n--- Aggregate OUT-OF-SAMPLE performance (walk-forward) ---")
    print("  'bench%' = MEAN per-session return of holding the same window every")
    print("  day with no RVOL filter — what you'd get with no detection at all.")
    print("  'oos_alpha' = mean per-session excess over that baseline (the only")
    print("  number that reflects genuine edge).")
    rows = []
    for symbol, wf_list in per_symbol:
        for wf in wf_list:
            p = dict(wf.best_params or {})
            rows.append({
                "symbol": wf.symbol,
                "dir": wf.direction,
                "windows": wf.num_windows,
                "prof_oos": wf.num_profitable_oos_windows,
                "stability": wf.param_stability,
                "window": f"{p.get('entry_start', '?')}-{p.get('entry_end', '?')}",
                "rvol>=": p.get("rvol_threshold"),
                "mom>=": p.get("momentum_threshold"),
                "enter": p.get("enter_at"),
                "exit": p.get("exit_mode"),
                "oos_trades": wf.oos_num_trades,
                "oos_win%": wf.oos_win_rate,
                "oos_ret%": wf.oos_total_return,
                "oos_bench%": wf.oos_benchmark_return,
                "oos_alpha": wf.alpha,
                "oos_sharpe": wf.oos_sharpe_ratio,
                "oos_calmar": wf.oos_calmar_ratio,
                "oos_maxDD%": wf.oos_max_drawdown,
                "p": wf.oos_placebo_p,
                "pass": "yes" if (wf.alpha > 0 and wf.profitable
                                  and wf.oos_num_trades > 0
                                  and wf.oos_win_rate > _MIN_WIN_RATE) else "no",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            "oos_alpha", ascending=False).reset_index(drop=True)
    if df.empty:
        print("  (no symbol produced a walk-forward result)")
        return df
    print(df.to_string(index=False, formatters={
        "stability": "{:.0%}".format,
        "rvol>=": "{:.1f}".format,
        "mom>=": "{:.3%}".format,
        "oos_win%": "{:.0%}".format,
        "oos_ret%": "{:+.2%}".format,
        "oos_bench%": "{:+.2%}".format,
        "oos_alpha": "{:+.2%}".format,
        "oos_sharpe": "{:.2f}".format,
        "oos_calmar": "{:.2f}".format,
        "oos_maxDD%": "{:.2%}".format,
        "p": lambda v: "—" if v is None else f"{v:.3f}",
    }))
    return df


def _print_wf_windows(per_symbol) -> None:
    """Per-window IS → OOS detail so the decision path is inspectable."""
    print("\n--- Per-window detail (IS chosen → OOS result) ---")
    any_window = False
    for symbol, wf_list in per_symbol:
        for wf in wf_list:
            if not wf.windows:
                continue
            any_window = True
            print(f"\n  {wf.symbol} ({wf.direction})")
            rows = []
            for w in wf.windows:
                p = dict(w.best_params or {})
                rows.append({
                    "win": w.window_index,
                    "is_window": f"{w.is_start.date()}→{w.is_end.date()}",
                    "oos_window": f"{w.oos_start.date()}→{w.oos_end.date()}",
                    "chosen": (f"{p.get('entry_start', '?')}-"
                               f"{p.get('entry_end', '?')} "
                               f"rv>={p.get('rvol_threshold', '?')} "
                               f"mo>={p.get('momentum_threshold', '?')} "
                               f"{p.get('enter_at', '?')} "
                               f"{p.get('exit_mode', '?')}") if p else "—",
                    "is_ret%": w.is_total_return,
                    "is_trades": w.is_num_trades,
                    "oos_ret%": w.oos_total_return,
                    "oos_trades": w.oos_num_trades,
                    "oos_win%": w.oos_win_rate,
                    "err": (w.error or "")[:38],
                })
            print(pd.DataFrame(rows).to_string(index=False, formatters={
                "is_ret%": "{:+.2%}".format,
                "oos_ret%": "{:+.2%}".format,
                "oos_win%": "{:.0%}".format,
            }))
    if not any_window:
        print("  (no windows were formed — the date range is too short for the "
              "configured IS/OOS/step/min-windows)")


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s")

    start, end = _resolve_dates(args)
    symbols = [s.upper() for s in (args.symbols or underlyings())]

    if args.long_only:
        globalConfig.ENABLE_SHORT_SELLING = False

    strategy = LeveragedRebalanceStrategy(
        cost_bps=args.cost_bps,
        rvol_lookback_days=args.rvol_lookback,
    )
    # data_timeframe is a class attribute; override per-run when requested.
    strategy.data_timeframe = args.timeframe

    if args.is_months is not None:
        globalConfig.WF_IS_MONTHS = args.is_months
    if args.oos_months is not None:
        globalConfig.WF_OOS_MONTHS = args.oos_months
    if args.step_months is not None:
        globalConfig.WF_STEP_MONTHS = args.step_months

    _print_run_config(args, strategy, symbols, start, end)

    directions = ["long"]
    if globalConfig.ENABLE_SHORT_SELLING:
        directions.append("short")

    if args.walk_forward:
        per_symbol, _info = _run_walk_forward(
            strategy, symbols, start, end, directions)
        df = _print_wf_decisions(per_symbol, _info)
        _print_wf_windows(per_symbol)
        if df.empty:
            return 1
        print("\n--- Walk-forward aggregate ---")
        print(f"  Symbol/direction runs : {len(df)}")
        print(f"  Pass live filter      : {int((df['pass'] == 'yes').sum())}")
        print(f"  Mean OOS return       : {df['oos_ret%'].mean():+.2%}")
        print(f"  Mean OOS baseline/sess: {df['oos_bench%'].mean():+.3%}")
        print(f"  Mean OOS alpha/sess   : {df['oos_alpha'].mean():+.3%}")
        print(f"  Mean OOS win rate     : {df['oos_win%'].mean():.0%}")
        print(f"  Mean OOS Sharpe       : {df['oos_sharpe'].mean():.2f}")
        print(f"  Mean param stability  : {df['stability'].mean():.0%}")
        if df["p"].notna().any():
            significant = df[(df["p"] <= 0.05)]
            print(f"  OOS placebo p<=0.05   : {len(significant)}/{int(df['p'].notna().sum())}"
                  f"  (expected by chance ≈ {int(0.05 * df['p'].notna().sum())})")
            for _, row in significant.sort_values("p").iterrows():
                print(f"    {row['symbol']:<6} {row['dir']:<6} p={row['p']:.3f} "
                      f"alpha={row['oos_alpha']:+.3%}")
        print("\n  NOTE: OOS alpha is the mean PER-SESSION excess over holding the"
              "\n  same window every day, so market drift is removed. If alpha ≈ 0"
              "\n  the detection adds nothing over simply trading that window.")
        output_dir = args.output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            output_dir, f"leveraged_rebalance_walkforward_{stamp}.csv")
        df.to_csv(path, index=False)
        print(f"\nWrote:\n  {path}")
        return 0

    print(f"\nRunning {len(strategy.get_param_grid('long'))} combos × "
          f"{len(directions)} direction(s) per symbol…")
    results, info = _run_grid(
        strategy, symbols, start, end, directions,
        placebo_draws=args.placebo_draws)

    if not results:
        print("\nNo usable data — check the date range and symbol availability.")
        return 1

    decisions = _decisions_frame(results, info)
    trades = _trades_frame(results)

    _print_decisions(decisions)
    _print_consensus(decisions)
    placebo = _print_placebo(results)
    _print_trades(trades, args.max_trades)
    _print_summary(results, decisions, trades)

    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    decisions_path = os.path.join(
        output_dir, f"leveraged_rebalance_decisions_{stamp}.csv")
    trades_path = os.path.join(
        output_dir, f"leveraged_rebalance_trades_{stamp}.csv")
    decisions.to_csv(decisions_path, index=False)
    trades.to_csv(trades_path, index=False)
    print(f"\nWrote:\n  {decisions_path}\n  {trades_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
