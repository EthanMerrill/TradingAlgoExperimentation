"""
Detection logic for single-stock leveraged-ETF daily rebalance (block) trades.

Thesis
------
Single-stock leveraged/inverse ETFs reset their daily exposure near the close.
Their swap counterparties hedge that delta by trading the underlying stock, so
a chunk of *predictable* flow lands in the underlying in the last part of the
session. This module detects the fingerprint of that flow — a late-session
cluster of unusually high volume *with* directional price momentum — so the
strategy can (in simulation) enter ahead of it.

Signal definition
-----------------
For each intraday bar we compute:

* **RVOL** — the bar's volume divided by a trailing baseline of the *same
  minute-of-day* volume over the previous ``rvol_lookback_days`` sessions. This
  removes the strong intraday volume seasonality (the U-shape) so a 15:35 spike
  is compared to other 15:35 bars, not to the 09:35 open.
* **Momentum** — the cumulative return from the prior session's close to the
  bar's close.

An **event day** is one where at least one bar inside the entry window has
``RVOL >= rvol_threshold`` **and** ``|momentum| >= momentum_threshold``. The
first qualifying bar is the trigger; its momentum sign gives the trade
direction (dealers must buy into an up-move, sell into a down-move).

All functions here are pure and vectorized over the bars frame so they can be
precomputed once by the strategy's ``prepare()`` and re-thresholded cheaply per
grid combo.
"""
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "parse_hhmm",
    "prepare_features",
    "detect_events",
    "simulate_events",
    "session_returns",
    "placebo_alpha_test",
]

US_EASTERN = "US/Eastern"

# Defaults (kept here so the strategy and the analysis CLI agree).
DEFAULT_RVOL_LOOKBACK_DAYS = 10
DEFAULT_RVOL_THRESHOLD = 2.0
DEFAULT_MOMENTUM_THRESHOLD = 0.005   # 0.5%
DEFAULT_ENTRY_START = "15:20"
DEFAULT_ENTRY_END = "15:50"

_FEATURE_COLUMNS = [
    "date", "tod", "open", "high", "low", "close", "volume",
    "bar_ret", "prior_close", "cum_ret", "rvol_base", "rvol",
]


def parse_hhmm(value: str) -> int:
    """Parse ``"HH:MM"`` into minutes since midnight (US/Eastern wall clock)."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected 'HH:MM', got {value!r}")
    hours, minutes = int(parts[0]), int(parts[1])
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ValueError(f"Invalid time-of-day: {value!r}")
    return hours * 60 + minutes


def _to_eastern_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert a bar index to US/Eastern, treating naive timestamps as UTC."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(US_EASTERN)


def prepare_features(
    bars: pd.DataFrame,
    rvol_lookback_days: int = DEFAULT_RVOL_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Compute per-bar RVOL / momentum features for one symbol.

    Args:
        bars: OHLCV DataFrame indexed by timestamp (tz-aware or naive-UTC).
            Must contain ``open``/``high``/``low``/``close``/``volume``.
        rvol_lookback_days: Number of prior sessions used for the same-time-of-day
            volume baseline.

    Returns:
        A copy of ``bars`` (sorted by time) with added columns:
        ``date``, ``tod`` (minutes since ET midnight), ``bar_ret``,
        ``prior_close``, ``cum_ret``, ``rvol_base``, ``rvol``.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=_FEATURE_COLUMNS)

    df = bars.sort_index().copy()
    idx_et = _to_eastern_index(df.index)

    df["date"] = [ts.date() for ts in idx_et]
    df["tod"] = [ts.hour * 60 + ts.minute for ts in idx_et]

    df["bar_ret"] = df["close"].pct_change()

    # Prior session close (per ET trading date), mapped back onto every bar.
    day_last_close = df.groupby("date")["close"].last()
    prior_close_by_date = day_last_close.shift(1)
    df["prior_close"] = df["date"].map(prior_close_by_date)
    df["cum_ret"] = df["close"] / df["prior_close"] - 1.0

    # Same-time-of-day volume baseline: rolling median over prior sessions,
    # shifted so a session never contributes to its own baseline.
    vol_by_date_tod = df.pivot_table(
        index="date", columns="tod", values="volume", aggfunc="sum")
    window = max(1, int(rvol_lookback_days))
    baseline = (
        vol_by_date_tod
        .rolling(window=window, min_periods=max(1, window // 2))
        .median()
        .shift(1)
    )
    baseline_long = baseline.stack()
    multi_index = pd.MultiIndex.from_arrays([df["date"], df["tod"]])
    df["rvol_base"] = baseline_long.reindex(multi_index).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        df["rvol"] = df["volume"] / df["rvol_base"]
    df["rvol"] = df["rvol"].replace([np.inf, -np.inf], np.nan)

    return df


def detect_events(
    features: pd.DataFrame,
    entry_start: str = DEFAULT_ENTRY_START,
    entry_end: str = DEFAULT_ENTRY_END,
    rvol_threshold: float = DEFAULT_RVOL_THRESHOLD,
    momentum_threshold: float = DEFAULT_MOMENTUM_THRESHOLD,
    direction_mode: str = "both",
    enter_at: str = "trigger",
) -> pd.DataFrame:
    """Reduce a features frame to one detection event per session.

    Args:
        features: Output of :func:`prepare_features`.
        entry_start / entry_end: Entry window (``"HH:MM"`` ET, inclusive).
        rvol_threshold: Minimum RVOL for a bar to qualify.
        momentum_threshold: Minimum ``|cum_ret|`` (decimal) to qualify.
        direction_mode: ``"long"``, ``"short"`` or ``"both"`` — which event
            directions to keep.
        enter_at: Where within the window to enter on a qualifying session.

            * ``"trigger"`` (default) — enter on the first qualifying bar, i.e.
              wait for the volume/momentum fingerprint to appear.
            * ``"window_start"`` — enter on the first bar of the window, using
              the *trigger* bar only to decide direction/qualification.

            ``"window_start"`` matters because the late-session drift is
            positive on average; waiting for the spike forfeits that drift, so
            "trigger" can post negative alpha relative to simply holding the
            final half hour. Qualifying the *day* while entering at the start
            keeps the drift and still conditions on the flow fingerprint.

    Returns:
        DataFrame with one row per detected session and columns
        ``date``, ``entry_ts``, ``entry_price``, ``direction``, ``rvol``,
        ``momentum``, ``prior_close``. Empty when nothing qualifies.
    """
    columns = ["date", "entry_ts", "entry_price", "direction",
               "rvol", "momentum", "prior_close"]
    if features is None or features.empty:
        return pd.DataFrame(columns=columns)

    if enter_at not in ("trigger", "window_start"):
        raise ValueError(
            f"enter_at must be 'trigger' or 'window_start', got {enter_at!r}")

    start_tod = parse_hhmm(entry_start)
    end_tod = parse_hhmm(entry_end)
    if end_tod < start_tod:
        raise ValueError(
            f"entry_end ({entry_end}) must not precede entry_start ({entry_start})")

    window = features[
        (features["tod"] >= start_tod) & (features["tod"] <= end_tod)
    ]
    if window.empty:
        return pd.DataFrame(columns=columns)

    # A bar qualifies on the information available AT that bar: the magnitude of
    # the session's move so far, and (optionally) the relative volume.
    qualified = (
        window["cum_ret"].notna()
        & (window["cum_ret"].abs() >= float(momentum_threshold))
    )
    if float(rvol_threshold) > 0.0:
        qualified &= window["rvol"] >= float(rvol_threshold)
    candidates = window[qualified]
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    # Choose the entry bar. Both modes only ever use data up to the entry bar.
    #
    #   "trigger"      — the first bar in the window that qualifies. Waits for
    #                    the move/spike to actually appear.
    #   "window_start" — the window's opening bar, but ONLY on sessions whose
    #                    move already qualifies at that moment.
    #
    # NOTE: an earlier version paired a window-open entry PRICE with a
    # direction taken from a later qualifying bar. That was lookahead bias —
    # harmless across a 30-minute window, but badly wrong once entry windows
    # span the afternoon. Entry price, direction, rvol and momentum are now all
    # read from the same bar.
    first_per_session = candidates.sort_index().groupby("date", sort=True).head(1)
    if enter_at == "window_start":
        # Only sessions already qualifying at the opening bar of the window.
        entries = candidates[candidates["tod"] == start_tod]
    else:
        entries = first_per_session

    if entries.empty:
        return pd.DataFrame(columns=columns)

    records = []
    for ts, row in entries.sort_index().iterrows():
        direction = "long" if row["cum_ret"] >= 0 else "short"
        if direction_mode == "long" and direction != "long":
            continue
        if direction_mode == "short" and direction != "short":
            continue
        # Report the entry time in US/Eastern (bars arrive tz-aware UTC); the
        # features' date/tod columns are already ET-derived, so stay aligned.
        entry_ts = _to_eastern_index(pd.DatetimeIndex([ts]))[0]
        records.append({
            "date": row["date"],
            "entry_ts": entry_ts,
            "entry_price": float(row["close"]),
            "direction": direction,
            "rvol": float(row["rvol"]) if pd.notna(row["rvol"]) else float("nan"),
            "momentum": float(row["cum_ret"]),
            "prior_close": float(row["prior_close"]),
        })

    result = pd.DataFrame.from_records(records, columns=columns)
    return result.sort_values("date").reset_index(drop=True)


def simulate_events(
    features: pd.DataFrame,
    events: pd.DataFrame,
    exit_mode: str = "close",
    hold_days: int = 0,
    cost_bps: float = 0.0,
) -> List[dict]:
    """Simulate one flat-by-default trade per detected event.

    Args:
        features: Output of :func:`prepare_features` (must contain ``close``,
            ``open`` and ``date`` columns).
        events: Output of :func:`detect_events`.
        exit_mode: ``"close"`` (flat at the session close), ``"next_open"``
            (exit at the next session's first bar), or ``"hold_days"`` (exit at
            the close ``hold_days`` sessions later).
        hold_days: Sessions to hold for ``exit_mode="hold_days"``.
        cost_bps: Round-trip cost in basis points (fees + slippage) subtracted
            from each trade's return.

    Returns:
        List of trade dicts with ``entry_date``, ``entry_price``, ``exit_date``,
        ``exit_price``, ``return``, ``gross_return``, ``duration``,
        ``exit_reason``, ``direction``, ``rvol``, ``momentum``.
    """
    if events is None or events.empty or features is None or features.empty:
        return []

    trades: List[dict] = []
    session_dates = sorted(features["date"].unique())
    date_rank = {d: i for i, d in enumerate(session_dates)}

    session_last = features.groupby("date").tail(1)
    last_close_by_date = dict(zip(session_last["date"], session_last["close"]))
    session_first = features.groupby("date").head(1)
    first_row_by_date = {
        row.date: row for row in session_first.itertuples(index=False)
    }

    round_trip_cost = 2.0 * float(cost_bps) / 10_000.0

    for event in events.itertuples(index=False):
        entry_date = event.date
        entry_price = float(event.entry_price)
        if not entry_price or not np.isfinite(entry_price):
            continue

        rank = date_rank.get(entry_date)
        if rank is None:
            continue

        exit_date = None
        exit_price = None
        exit_reason = exit_mode

        if exit_mode == "close":
            exit_date = entry_date
            exit_price = last_close_by_date.get(entry_date)
        elif exit_mode == "next_open":
            if rank + 1 < len(session_dates):
                exit_date = session_dates[rank + 1]
                row = first_row_by_date.get(exit_date)
                exit_price = float(row.open) if row is not None else None
            else:
                exit_reason = "close_no_next_session"
                exit_date = entry_date
                exit_price = last_close_by_date.get(entry_date)
        elif exit_mode == "hold_days":
            target_rank = min(rank + max(0, int(hold_days)),
                              len(session_dates) - 1)
            exit_date = session_dates[target_rank]
            exit_price = last_close_by_date.get(exit_date)
        else:
            raise ValueError(f"Unknown exit_mode: {exit_mode!r}")

        if exit_price is None or not np.isfinite(exit_price):
            continue

        sign = 1.0 if event.direction == "long" else -1.0
        gross = (float(exit_price) / entry_price - 1.0) * sign
        net = gross - round_trip_cost

        trades.append({
            "entry_date": entry_date,
            "entry_ts": getattr(event, "entry_ts", None),
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": float(exit_price),
            "return": net,
            "gross_return": gross,
            "duration": (exit_date - entry_date).days if exit_date else 0,
            "exit_reason": exit_reason,
            "direction": event.direction,
            "rvol": float(getattr(event, "rvol", np.nan)),
            "momentum": float(getattr(event, "momentum", np.nan)),
        })

    return trades


def session_returns(
    features: pd.DataFrame,
    entry_start: str,
    entry_end: str,
    exit_mode: str = "close",
    hold_days: int = 0,
    cost_bps: float = 0.0,
    direction_mode: str = "both",
) -> pd.Series:
    """Per-session return of entering at the window open on **every** session.

    This is the reference series the placebo test resamples from: it is the
    payoff of "hold this window every day", indexed by session date. Detection
    is judged on whether the days it *picks* beat this pool.

    Returns:
        Series indexed by session date (float returns). Empty if no data.
    """
    events = detect_events(
        features,
        entry_start=entry_start,
        entry_end=entry_end,
        rvol_threshold=0.0,
        momentum_threshold=0.0,
        direction_mode=direction_mode,
        enter_at="window_start",
    )
    if events.empty:
        return pd.Series(dtype=np.float64)
    trades = simulate_events(
        features, events, exit_mode=exit_mode,
        hold_days=hold_days, cost_bps=cost_bps)
    if not trades:
        return pd.Series(dtype=np.float64)
    return pd.Series(
        [t["return"] for t in trades],
        index=pd.Index([t["entry_date"] for t in trades], name="date"),
        dtype=np.float64,
    )


def placebo_alpha_test(
    features: pd.DataFrame,
    entry_start: str,
    entry_end: str,
    exit_mode: str,
    hold_days: int,
    cost_bps: float,
    direction_mode: str,
    n_selected: int,
    observed_alpha: float,
    n_draws: int = 2000,
    seed: int = 0,
) -> dict:
    """Permutation test: is the detected-day alpha better than random picking?

    The strategy's alpha is the mean per-session excess of the sessions it
    *selects* over all sessions. That number is only meaningful relative to the
    null hypothesis "selecting K days at random has no edge". This resamples
    ``n_selected`` sessions at random from the same session pool ``n_draws``
    times, recomputes the same statistic, and reports where the strategy falls.

    Because the placebo draws from the *same* sessions with the same entry/exit
    convention and costs, market drift, timing and holding period cancel out.
    The resulting p-value is a direct test of whether the RVOL / momentum
    detection carries information.

    Args:
        features: Output of :func:`prepare_features`.
        entry_start/entry_end/exit_mode/hold_days/cost_bps/direction_mode:
            Identical conventions to the strategy being tested.
        n_selected: Number of sessions the strategy actually traded.
        observed_alpha: The strategy's real per-session excess return.
        n_draws: Number of random draws in the null distribution.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with ``p_value``, ``null_mean``, ``null_std``, ``null_p95``,
        ``n_draws``, ``n_selected``, ``n_sessions`` and the ``percentile`` of
        the observed alpha within the null distribution.
    """
    pool = session_returns(
        features, entry_start, entry_end, exit_mode,
        hold_days, cost_bps, direction_mode)

    result = {
        "p_value": None, "null_mean": None, "null_std": None,
        "null_p95": None, "n_draws": 0, "n_selected": int(n_selected),
        "n_sessions": int(len(pool)), "percentile": None,
    }
    # Need at least two sessions to form a distribution and enough sessions to
    # draw from; otherwise the test is undefined rather than "not significant".
    if len(pool) < 3 or n_selected < 1:
        return result

    values = pool.to_numpy()
    baseline_mean = float(values.mean())
    k = min(int(n_selected), len(values))

    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_draws), dtype=np.float64)
    for i in range(int(n_draws)):
        sample = rng.choice(values, size=k, replace=False)
        draws[i] = float(sample.mean()) - baseline_mean

    # One-sided: only better-than-random counts as evidence of skill.
    better = int(np.sum(draws >= observed_alpha))
    result.update({
        "p_value": (better + 1.0) / (len(draws) + 1.0),
        "null_mean": float(draws.mean()),
        "null_std": float(draws.std(ddof=0)),
        "null_p95": float(np.percentile(draws, 95)),
        "n_draws": int(len(draws)),
        "percentile": float((draws < observed_alpha).mean() * 100.0),
    })
    return result
