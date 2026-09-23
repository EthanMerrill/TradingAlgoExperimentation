"""
ETF-level cross-sectional rebalance-flow portfolio.

This is the v2 of the leveraged-ETF rebalance idea and differs from the
per-symbol strategy (``leveraged_rebalance.py``) in three structural ways.

1. Flow is ESTIMATED DIRECTLY, not inferred from a fingerprint in the
   underlying. For each single-stock leveraged/inverse ETF the dealer's
   required daily hedge is approximately

       hedge_notional = leverage * AUM * day_return_of_the_fund

   where ``day_return_of_the_fund`` is the fund's own move from the prior
   session close. Crucially, as of any intraday time T the return-so-far is
   KNOWN, so the hedge notional the dealer must trade by the close is known
   hours in advance. No volume fingerprint (RVOL) is needed — RVOL was the
   v1 signal and it failed falsification.

2. Selection is CROSS-SECTIONAL. Each session the per-underlying notionals are
   aggregated across all funds tracking that underlying, ranked by absolute
   notional, and only the top-K underlyings are traded. One global rule, no
   per-symbol parameter fitting. The v1 failure mode — 34 independent per-
   symbol grid searches with 48 combos each, and a placebo landing at exactly
   the chance rate (1/34 significant) — is structurally removed.

3. Evaluation is PORTFOLIO-LEVEL. The unit of evidence is the daily return of
   an equal-weighted portfolio of K positions, tested against a permutation
   null that forms random K-name portfolios from the same session pool with
   the same costs. Market drift, timing and costs cancel in the comparison,
   so the p-value measures whether *knowing the flows* adds information.

Lookahead safety
----------------
Nothing here may use information from after the decision time T:

* the fund return used in the flow estimate is measured **up to T**,
* the entry price is the underlying's close of the bar AT T,
* the session close exit price is the ONLY forward-looking value, and it is
  the trade itself (that is the bet), not an input to the decision.

AUM caveats
-----------
AUM values in the map are a point-in-time snapshot and are used only to scale
the relative size of each fund's flow. They are not live NAVs. A stale AUM
changes the ranking only when two underlyings are close in notional, which the
placebo test accounts for statistically. Funds without an AUM figure default
to ``DEFAULT_AUM_USD``.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.leveraged_single_stock_etfs import (
    LEVERAGED_SINGLE_STOCK_ETFS,
)

__all__ = [
    "DEFAULT_AUM_USD",
    "FlowDecision",
    "etf_flow_estimates",
    "momentum_scores",
    "portfolio_session_returns",
    "portfolio_stats",
    "direction_breakdown",
    "beta_vs_market",
    "placebo_portfolio_test",
    "split_half_stability",
]

# Fallback AUM for funds without a snapshot figure (USD).
DEFAULT_AUM_USD = 500_000_000.0


@dataclass
class FlowDecision:
    """One session's cross-sectional trade decision."""
    date: object                      # datetime.date
    symbols: Tuple[str, ...]          # chosen underlyings, ranked by |flow|
    directions: Tuple[str, ...]       # "long"/"short", aligned with symbols
    notionals: Tuple[float, ...]      # estimated hedge notional per symbol
    returns: Tuple[float, ...]        # net per-position returns
    portfolio_return: float           # equal-weighted mean of returns


def _effective_aum(etf: str) -> float:
    """AUM for a fund, falling back to the shared default when unknown."""
    fund = LEVERAGED_SINGLE_STOCK_ETFS.get(etf)
    if fund is None:
        return DEFAULT_AUM_USD
    return float(fund.aum_usd) if fund.aum_usd else DEFAULT_AUM_USD


def etf_flow_estimates(
    etf_features: Dict[str, pd.DataFrame],
    asof_tod: int,
) -> pd.DataFrame:
    """Estimate the dealer hedge notional per underlying, per session.

    Args:
        etf_features: Map of ETF ticker -> features frame (output of
            ``leveraged_rebalance_signal.prepare_features``) for the ETF's OWN
            bars. Must contain ``date``, ``tod``, ``cum_ret``.
        asof_tod: Minutes-since-ET-midnight of the decision time (e.g. 920 for
            15:20). Only data up to this bar is used.

    Returns:
        DataFrame indexed by session date with columns ``underlying``,
        ``notional`` (summed signed hedge notional, USD), ``abs_notional``,
        ``n_funds``. Empty when there is no data.
    """
    rows: List[dict] = []
    for etf_symbol, features in etf_features.items():
        fund = LEVERAGED_SINGLE_STOCK_ETFS.get(etf_symbol)
        if fund is None or features is None or features.empty:
            continue
        bar = features[features["tod"] == asof_tod]
        for row in bar.itertuples(index=False):
            if row.cum_ret is None or not np.isfinite(row.cum_ret):
                continue
            aum = _effective_aum(etf_symbol)
            # Signed hedge notional the dealer must trade by the close:
            # positive => buy the underlying, negative => sell.
            notional = fund.leverage * aum * float(row.cum_ret)
            rows.append({
                "date": row.date,
                "underlying": fund.underlying,
                "notional": notional,
                "n_funds": 1,
            })

    if not rows:
        return pd.DataFrame(
            columns=["underlying", "notional", "abs_notional", "n_funds"])

    frame = pd.DataFrame(rows)
    aggregated = (
        frame.groupby(["date", "underlying"], as_index=False)
        .agg(notional=("notional", "sum"), n_funds=("n_funds", "sum"))
    )
    aggregated["abs_notional"] = aggregated["notional"].abs()
    return aggregated.sort_values(["date", "abs_notional"],
                                  ascending=[True, False]).reset_index(drop=True)


def _price_tables(features_by_symbol: Dict[str, pd.DataFrame], entry_tod: int):
    """Entry (at ``entry_tod``) and session-close price tables per symbol."""
    entry_px: Dict[Tuple[str, object], float] = {}
    close_px: Dict[Tuple[str, object], float] = {}
    for symbol, features in features_by_symbol.items():
        if features is None or features.empty:
            continue
        entries = features[features["tod"] == entry_tod]
        closes = features.groupby("date")["close"].last()
        for row in entries.itertuples(index=False):
            if np.isfinite(row.close) and row.close > 0:
                entry_px[(symbol, row.date)] = float(row.close)
        for date, close in closes.items():
            if np.isfinite(close) and close > 0:
                close_px[(symbol, date)] = float(close)
    return entry_px, close_px


def momentum_scores(
    underlying_features: Dict[str, pd.DataFrame],
    asof_tod: int,
    notional_scale_usd: float = DEFAULT_AUM_USD,
) -> pd.DataFrame:
    """CONTROL ranking variable: the underlying's own |day return| so far.

    Identical output shape to :func:`etf_flow_estimates` (columns
    ``underlying``/``notional``/``abs_notional``/``n_funds``, indexed by
    session date) so the two ranking rules are directly comparable — the ONLY
    difference is what is ranked.

    ``notional_scale_usd`` exists purely so the control's scores share the
    same *scale* as dollar notionals: dollar-based filters
    (``min_abs_notional``) then behave identically for both ranking rules.
    Ranking is by the return itself, so the scale does not affect selection.

    This is the control that separates the two hypotheses the flow portfolio
    could be exploiting:

      * ``flow``      — notional = leverage x AUM x return. Ranking is scaled
        by fund AUM/leverage, so a 1% move in a $3.7B AUM name ranks above a
        2% move in a $400M name.
      * ``momentum``  — score = the return itself. Ranking is PURE day-move
        magnitude, with no ETF/AUM information at all.

    If the flow portfolio beats this control, the ETF layer (AUM/leverage
    weighting, fund-level flow aggregation) adds information beyond simple
    "buy the biggest movers". If they tie, the exploitable effect is plain
    intraday momentum and the rebalance mechanism is decoration.
    """
    rows: List[dict] = []
    for symbol, features in underlying_features.items():
        if features is None or features.empty:
            continue
        bars = features[features["tod"] == asof_tod]
        for row in bars.itertuples(index=False):
            if row.cum_ret is None or not np.isfinite(row.cum_ret):
                continue
            ret = float(row.cum_ret)
            score = ret * float(notional_scale_usd)
            rows.append({
                "date": row.date,
                "underlying": symbol,
                "notional": score,         # scaled return as the score
                "abs_notional": abs(score),
                "n_funds": 1,
            })
    if not rows:
        return pd.DataFrame(
            columns=["underlying", "notional", "abs_notional", "n_funds"])
    frame = pd.DataFrame(rows)
    return frame.sort_values(["date", "abs_notional"],
                             ascending=[True, False]).reset_index(drop=True)


def direction_breakdown(
    daily_returns: pd.Series,
    decisions: List[FlowDecision],
) -> dict:
    """Split the portfolio's sessions by whether it went net long or short.

    Reports, per side: session count, mean daily return, hit rate and total
    compounded return, plus a sign-agreement statistic vs the *next* question
    (is the return uncorrelated with the market or just directional beta?).

    A strategy whose long-day and short-day returns are BOTH positive and of
    similar size is closer to market-neutral / uncorrelated. One that makes
    money only on long days (in a bull sample) is repackaged beta.
    """
    by_side: Dict[str, List[float]] = {"long": [], "short": []}
    decision_by_date = {d.date: d for d in decisions}

    for date, ret in daily_returns.items():
        decision = decision_by_date.get(date)
        if decision is None:
            continue
        longs = sum(1 for direction in decision.directions
                    if direction == "long")
        shorts = len(decision.directions) - longs
        side = "long" if longs >= shorts else "short"
        by_side[side].append(float(ret))

    def _stats(values: List[float]) -> dict:
        if not values:
            return {"n": 0, "mean": None, "hit_rate": None, "total": None}
        arr = np.array(values, dtype=np.float64)
        return {
            "n": len(arr),
            "mean": float(arr.mean()),
            "hit_rate": float((arr > 0).mean()),
            "total": float((1.0 + arr).prod() - 1.0),
        }

    return {
        "long_days": _stats(by_side["long"]),
        "short_days": _stats(by_side["short"]),
    }


def flow_magnitude_buckets(
    flows: pd.DataFrame,
    underlying_features: Dict[str, pd.DataFrame],
    entry_tod: int,
    cost_bps: float = 2.0,
    direction_filter: str = "short",
    buckets: int = 4,
) -> pd.DataFrame:
    """Per-position returns bucketed by the position's |flow| rank/magnitude.

    Directly tests the hypothesis that the short side fails only because the
    sample lacks extreme negative-flow days (e.g. the Samsung/SK Hynix
    leveraged-ETF unwind): if the effect is real but rare, mean return should
    IMPROVE monotonically in the largest-|flow| buckets for shorts.

    Rather than picking a threshold, every qualifying (symbol, session) pair
    in the ranking frame is simulated (entry at ``entry_tod``, exit at close,
    same costs) and assigned to a |notional| quartile bucket. Bucket 1 = the
    SMALLEST |flow|, bucket ``buckets`` = the LARGEST.

    Returns:
        DataFrame with per-bucket stats: ``bucket``, ``n``, ``mean``, ``t``,
        ``hit_rate``, ``total``, ``median_abs_notional``.
    """
    entry_px, close_px = _price_tables(underlying_features, entry_tod)
    round_trip = 2.0 * float(cost_bps) / 10_000.0

    records = []
    for row in flows.itertuples(index=False):
        symbol = row.underlying
        entry = entry_px.get((symbol, row.date))
        exit_ = close_px.get((symbol, row.date))
        if entry is None or exit_ is None or entry <= 0:
            continue
        direction = "long" if row.notional >= 0 else "short"
        if direction_filter != "both" and direction != direction_filter:
            continue
        sign = 1.0 if direction == "long" else -1.0
        net = (exit_ / entry - 1.0) * sign - round_trip
        records.append({
            "date": row.date,
            "symbol": symbol,
            "direction": direction,
            "abs_notional": float(row.abs_notional),
            "notional": float(row.notional),
            "return": net,
        })
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records)
    # Quantile buckets on |flow|, 1 = smallest … N = largest.
    frame["bucket"] = pd.qcut(
        frame["abs_notional"], q=buckets, labels=False, duplicates="drop") + 1

    rows = []
    for bucket, group in frame.groupby("bucket"):
        values = group["return"].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        t_stat = (mean / (std / np.sqrt(len(values)))
                  if std > 0 and len(values) > 1 else None)
        rows.append({
            "bucket": int(bucket),
            "n": len(values),
            "mean": mean,
            "t": None if t_stat is None or not np.isfinite(t_stat) else t_stat,
            "hit_rate": float((values > 0).mean()),
            "total": float((1.0 + values).prod() - 1.0),
            "median_abs_notional": float(group["abs_notional"].median()),
        })
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def beta_vs_market(
    daily_returns: pd.Series,
    market_returns: pd.Series,
) -> dict:
    """Regress the portfolio's daily returns on the market's.

    The user's question is whether this can generate *uncorrelated* return
    (diversifying alpha) or whether it is repackaged beta. Regressing the
    strategy's daily returns on an equal-weight market proxy answers it:

      * ``beta``  — how much market exposure the portfolio carries. A
        long-biased book in a bull sample shows beta >> 0 even with no skill.
      * ``alpha_ann`` — the intercept, annualized. This is the return the
        portfolio would earn with the market's move removed — the uncorrelated
        part. It is the number that matters for portfolio construction.
      * ``corr``  — daily return correlation with the market.

    Args:
        daily_returns: Portfolio daily returns (indexed by date).
        market_returns: Market-proxy daily returns (e.g. equal-weight mean of
            ALL universe names, indexed by date).

    Returns:
        Dict with ``beta``, ``alpha_daily``, ``alpha_ann``, ``corr``,
        ``r_squared`` and ``n`` (overlap count). Values are None when the
        overlap is too small for a regression.
    """
    joined = pd.concat(
        {"strategy": daily_returns, "market": market_returns}, axis=1
    ).dropna()
    result = {
        "beta": None, "alpha_daily": None, "alpha_ann": None,
        "corr": None, "r_squared": None, "n": int(len(joined)),
    }
    if len(joined) < 10:
        return result

    x = joined["market"].to_numpy(dtype=np.float64)
    y = joined["strategy"].to_numpy(dtype=np.float64)
    x_var = float(x.var(ddof=1))
    if x_var <= 0.0:
        return result
    beta = float(np.cov(x, y, ddof=1)[0, 1] / x_var)
    alpha_daily = float(y.mean() - beta * x.mean())

    corr = float(np.corrcoef(x, y)[0, 1]) if x.std(ddof=1) > 0 else None
    r_squared = corr ** 2 if corr is not None else None

    result.update({
        "beta": beta,
        "alpha_daily": alpha_daily,
        "alpha_ann": alpha_daily * 252.0,
        "corr": corr,
        "r_squared": r_squared,
    })
    return result


def portfolio_session_returns(
    flows: pd.DataFrame,
    underlying_features: Dict[str, pd.DataFrame],
    entry_tod: int,
    top_k: int = 3,
    direction_filter: str = "both",
    cost_bps: float = 2.0,
    min_abs_notional: float = 0.0,
) -> Tuple[pd.Series, List[FlowDecision]]:
    """Form the cross-sectional portfolio each session and return its P&L.

    GENERIC over the ranking variable: ``flows`` may come from either
    :func:`etf_flow_estimates` (rebalance-flow ranking) or
    :func:`momentum_scores` (pure day-move ranking, the control). Everything
    else — timing, direction rule, weighting, costs — is identical, so running
    both is a clean A/B of the two hypotheses.

    Each session: rank underlyings by the ``flows`` frame's ``abs_notional``
    (descending), take the top ``top_k`` that have tradable prices, go long
    when ``notional`` is positive and short when negative, equal-weighted.

    Args:
        flows: Ranking frame (see above) indexed by session date.
        underlying_features: Map of underlying -> features frame (own bars).
        entry_tod: Decision/entry bar in minutes since ET midnight. Must match
            the ``asof_tod`` used to build ``flows``.
        top_k: Number of underlyings to hold per session.
        direction_filter: ``"long"``, ``"short"`` or ``"both"``.
        cost_bps: Per-side cost; a round trip (2×) is charged per position.
        min_abs_notional: Skip sessions whose best |notional| is below this.

    Returns:
        ``(daily_returns, decisions)`` — a Series of portfolio returns indexed
        by session date, and the per-session decision detail.
    """
    entry_px, close_px = _price_tables(underlying_features, entry_tod)
    round_trip = 2.0 * float(cost_bps) / 10_000.0

    daily_returns: Dict[object, float] = {}
    decisions: List[FlowDecision] = []

    for date, day_flows in flows.groupby("date"):
        candidates = day_flows[
            (day_flows["abs_notional"] >= float(min_abs_notional))
            & (day_flows["abs_notional"] > 0.0)
        ].sort_values("abs_notional", ascending=False)
        if candidates.empty:
            continue

        chosen_symbols, directions, notionals, rets = [], [], [], []
        for row in candidates.itertuples(index=False):
            if len(chosen_symbols) >= int(top_k):
                break
            symbol = row.underlying
            entry = entry_px.get((symbol, date))
            exit_ = close_px.get((symbol, date))
            if entry is None or exit_ is None:
                continue
            direction = "long" if row.notional >= 0 else "short"
            if direction_filter == "long" and direction != "long":
                continue
            if direction_filter == "short" and direction != "short":
                continue
            sign = 1.0 if direction == "long" else -1.0
            net = (exit_ / entry - 1.0) * sign - round_trip
            chosen_symbols.append(symbol)
            directions.append(direction)
            notionals.append(float(row.notional))
            rets.append(float(net))

        if not chosen_symbols:
            continue
        portfolio_return = float(np.mean(rets))
        daily_returns[date] = portfolio_return
        decisions.append(FlowDecision(
            date=date,
            symbols=tuple(chosen_symbols),
            directions=tuple(directions),
            notionals=tuple(notionals),
            returns=tuple(rets),
            portfolio_return=portfolio_return,
        ))

    series = pd.Series(daily_returns, dtype=np.float64)
    series.index.name = "date"
    return series.sort_index(), decisions


def portfolio_stats(daily_returns: pd.Series) -> dict:
    """Mean, t-statistic and hit rate of a daily return series."""
    result = {
        "n_sessions": int(len(daily_returns)),
        "mean": None, "std": None, "t_stat": None, "hit_rate": None,
        "total_return": None,
    }
    if daily_returns.empty:
        return result
    values = daily_returns.to_numpy()
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    t_stat = (
        mean / (std / np.sqrt(len(values))) if std > 0 and len(values) > 1
        else None)
    result.update({
        "mean": mean,
        "std": std,
        "t_stat": None if t_stat is None or not np.isfinite(t_stat) else t_stat,
        "hit_rate": float((values > 0).mean()),
        "total_return": float((1.0 + values).prod() - 1.0),
    })
    return result


def split_half_stability(daily_returns: pd.Series) -> dict:
    """Mean return of the first half vs the second half of the sample.

    A rule that only works in one half is not a rule; this is the cheapest
    robustness check that does not involve any further fitting.
    """
    if len(daily_returns) < 4:
        return {"first_mean": None, "second_mean": None, "consistent": False}
    midpoint = len(daily_returns) // 2
    first_mean = float(daily_returns.iloc[:midpoint].mean())
    second_mean = float(daily_returns.iloc[midpoint:].mean())
    return {
        "first_mean": first_mean,
        "second_mean": second_mean,
        "consistent": bool((first_mean > 0) == (second_mean > 0)),
    }


def placebo_portfolio_test(
    flows: pd.DataFrame,
    underlying_features: Dict[str, pd.DataFrame],
    entry_tod: int,
    top_k: int,
    direction_filter: str,
    cost_bps: float,
    min_abs_notional: float,
    observed_mean: float,
    n_draws: int = 2000,
    seed: int = 0,
) -> dict:
    """Permutation null: random K-name portfolios from the same session pool.

    Each draw shuffles, per session, which underlyings are held and (for the
    direction-null variant) the trade direction, then computes the same
    equal-weighted portfolio mean. Because the draws use the same sessions,
    entry/exit convention and costs, drift and timing cancel — the p-value
    isolates whether the FLOW RANKING carries information.

    Args:
        observed_mean: The real portfolio's mean daily return.
        n_draws: Number of random portfolios in the null distribution.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with ``p_value``, ``null_mean``, ``null_p95``, ``n_draws``,
        ``n_sessions`` and ``percentile`` of the observed mean.
    """
    entry_px, close_px = _price_tables(underlying_features, entry_tod)
    round_trip = 2.0 * float(cost_bps) / 10_000.0

    # Pool of (date -> list of (symbol, raw long-return)) for every tradable
    # name on that date, regardless of the flow ranking.
    pool: Dict[object, List[Tuple[str, float]]] = {}
    for date, day_flows in flows.groupby("date"):
        options = []
        for symbol in day_flows["underlying"]:
            entry = entry_px.get((symbol, date))
            exit_ = close_px.get((symbol, date))
            if entry is None or exit_ is None:
                continue
            options.append((symbol, exit_ / entry - 1.0))
        if options:
            pool[date] = options

    dates = sorted(pool)
    result = {
        "p_value": None, "null_mean": None, "null_p95": None,
        "n_draws": 0, "n_sessions": len(dates), "percentile": None,
    }
    if len(dates) < 5:
        return result

    if direction_filter == "both":
        signs = [1.0, -1.0]
    else:
        signs = [1.0 if direction_filter == "long" else -1.0]

    rng = np.random.default_rng(seed)
    k = max(1, min(int(top_k), max(len(pool[d]) for d in dates)))
    draws = np.empty(int(n_draws), dtype=np.float64)
    for i in range(int(n_draws)):
        session_means = []
        for date in dates:
            options = pool[date]
            chosen = rng.choice(len(options), size=min(k, len(options)),
                                replace=False)
            rets = []
            for idx in chosen:
                symbol, long_ret = options[int(idx)]
                sign = float(rng.choice(signs))
                rets.append(long_ret * sign - round_trip)
            session_means.append(float(np.mean(rets)))
        draws[i] = float(np.mean(session_means))

    better = int(np.sum(draws >= observed_mean))
    result.update({
        "p_value": (better + 1.0) / (len(draws) + 1.0),
        "null_mean": float(draws.mean()),
        "null_p95": float(np.percentile(draws, 95)),
        "n_draws": int(len(draws)),
        "percentile": float((draws < observed_mean).mean() * 100.0),
    })
    return result
