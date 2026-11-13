"""
Signal math for the backtest-ranked momentum strategy (shared by research and live).

* Per-stock rule (fixed, never optimized): long while ``close > SMA(sma_days)``.
* Score: Sharpe of that rule's daily returns over the trailing window
  ``[d - lookback, d - skip]`` (default 252 -> 21 trading days, i.e. 12-1),
  net of switching costs. Only data up to ``d - skip`` is used.

Research: app/research/momentum_rank_backtest.py (ranking vs random, survivorship)
and app/research/momentum_rank_portfolio.py (portfolio construction variants).
"""
import re

import numpy as np
import pandas as pd


def rule_positions(close: pd.DataFrame, sma_days: int) -> pd.DataFrame:
    """1.0 where the per-stock rule is long at that day's close, else 0.0 (NaN in warm-up)."""
    sma = close.rolling(sma_days, min_periods=sma_days).mean()
    return (close > sma).astype(float).where(sma.notna())


def backtest_scores(close: pd.DataFrame, pos: pd.DataFrame, lookback: int,
                    skip: int, cost_bps: float, min_days_long: int,
                    metric: str = "sharpe") -> pd.DataFrame:
    """Rule performance per stock over the trailing [d-lookback, d-skip].

    ``metric``: "sharpe" (annualized Sharpe of the rule's daily returns) or
    "return" (the rule's summed daily return over the window).

    Daily rule return on day t = pos[t-1] * r[t] - cost * |pos[t-1] - pos[t-2]|
    (decided at the prior close, cost charged when the position changes).
    """
    ret = close.pct_change(fill_method=None)
    held = pos.shift(1)
    switch = held.diff().abs()
    strat = held * ret - switch * (cost_bps / 1e4)
    window = lookback - skip
    mean = strat.rolling(window, min_periods=window).mean()
    std = strat.rolling(window, min_periods=window).std()
    days_long = held.rolling(window, min_periods=window).sum()
    score = (mean / std) * np.sqrt(252.0) if metric == "sharpe" else mean * window
    score = score.where((std > 0) & (days_long >= min_days_long))
    return score.shift(skip)  # window ends `skip` days before d


# Funds, closed-end funds, preferreds/notes and commodity trusts. The rule's
# Sharpe is highest on near-cash funds (T-bill / floating-rate ETFs never close
# below their SMA), so without this the Sharpe ranking parks in bond funds.
_NOT_COMMON_RE = re.compile(
    r"\b(ETFs?|ETNs?|Funds?|iShares|SPDR|ProShares|Direxion|Vanguard|Invesco|Schwab|"
    r"WisdomTree|VanEck|Global X|First Trust|Pacer|Index|Portfolio|Physical|"
    r"Royalty Trust|Income Trust|Municipal|Muni|Bond|Preferred|Depositary|Notes?|"
    r"Debentures?|Warrants?|Units?|Rights?|BlackRock|Gabelli|Nuveen|Eaton Vance|"
    r"Calamos|Cohen & Steers|PIMCO|abrdn|Royce|Grayscale|Bitcoin|Ether)\b", re.I)


def is_common_stock(symbol: str, name: str) -> bool:
    """Operating-company common shares only (name-based; unnamed -> kept)."""
    if "." in symbol or "/" in symbol:
        return False
    return not _NOT_COMMON_RE.search(name or "")
