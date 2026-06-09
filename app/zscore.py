"""
Z-score composite scoring for strategy comparison.

Replaces the legacy (alpha*100 + sharpe + calmar) formula with pool-aware
Z-score normalization so each metric contributes equally regardless of scale.

Usage:
  - Within-symbol: compute_stage_zscores() on a batch of BacktestResult tuples
    from Parallel() to select the best parameter combo for a single symbol.
  - Cross-symbol: compute_cross_symbol_zscores() on the final per-symbol best
    results to make them comparable across symbols.
"""

import logging
from typing import List, Tuple

import numpy as np

# Re-use BacktestResult from strategy to avoid circular imports at runtime.
# StrategyOptimizer imports us, so importing just the dataclass is safe.
# type: ignore  # pylint: disable=import-error
from strategy import BacktestResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Cap Calmar ratio to prevent infinity from zero-drawdown strategies
# dominating the pool.  A calmar of 10 means max drawdown is 1/10 of annual
# return — already excellent.  Values above this are clipped.
CALMAR_CAP: float = 10.0

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_metric_arrays(results: List[BacktestResult]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (alpha, sharpe, capped_calmar) arrays from a list of results."""
    alphas = np.array([r.alpha for r in results], dtype=np.float64)
    sharpes = np.array([r.sharpe_ratio for r in results], dtype=np.float64)
    calmars = np.array(
        [min(r.calmar_ratio, CALMAR_CAP) for r in results], dtype=np.float64
    )
    return alphas, sharpes, calmars


def _compute_pool_stats(alphas: np.ndarray, sharpes: np.ndarray, calmars: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Return ((mean_alpha, std_alpha), (mean_sharpe, std_sharpe), (mean_calmar, std_calmar)).

    If std is 0 (all values identical), std is replaced with 1.0 so z-score
    divides safely and the metric contributes 0.0 to the composite.
    """
    mu_a, sig_a = float(alphas.mean()), float(alphas.std(ddof=0))
    mu_s, sig_s = float(sharpes.mean()), float(sharpes.std(ddof=0))
    mu_c, sig_c = float(calmars.mean()), float(calmars.std(ddof=0))

    if sig_a == 0.0:
        sig_a = 1.0
    if sig_s == 0.0:
        sig_s = 1.0
    if sig_c == 0.0:
        sig_c = 1.0

    return (mu_a, sig_a), (mu_s, sig_s), (mu_c, sig_c)


def _zscore_single(
    alpha: float, sharpe: float, calmar: float,
    stats_a: Tuple[float, float],
    stats_s: Tuple[float, float],
    stats_c: Tuple[float, float],
) -> float:
    """Compute Z-score composite for a single result given precomputed pool stats."""
    z_a = (alpha - stats_a[0]) / stats_a[1]
    z_s = (sharpe - stats_s[0]) / stats_s[1]
    z_c = (min(calmar, CALMAR_CAP) - stats_c[0]) / stats_c[1]
    return z_a + z_s + z_c

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_stage_zscores(
    batch: List[Tuple[BacktestResult, int, int, int]],
) -> List[float]:
    """Compute per-stage Z-scores for a batch of (result, period, lower, upper).

    Pool statistics are computed from *all* results in the batch, then each
    result gets a Z-score relative to that pool.  This is the recommended
    scorer for coarse-grid, fine-grid, and fallback stages in optimize_symbol.

    Returns a list of floats in the same order as the input batch.
    """
    if not batch:
        return []

    # Extract just the BacktestResult portion
    results = [item[0] for item in batch]
    alphas, sharpes, calmars = _extract_metric_arrays(results)
    stats = _compute_pool_stats(alphas, sharpes, calmars)

    scores: List[float] = []
    for result, _, _, _ in batch:
        score = _zscore_single(
            result.alpha, result.sharpe_ratio, result.calmar_ratio,
            stats[0], stats[1], stats[2],
        )
        scores.append(score)

    return scores


def compute_cross_symbol_zscores(results: List[BacktestResult]) -> None:
    """Compute cross-symbol Z-scores and set result.composite_score in place.

    Call this after all per-symbol optimization passes are complete.  It
    normalises alpha, sharpe, and capped-calmar across the full universe of
    best results so that scores are comparable across symbols.

    Mutates each BacktestResult.composite_score.
    """
    if not results:
        return

    alphas, sharpes, calmars = _extract_metric_arrays(results)
    stats = _compute_pool_stats(alphas, sharpes, calmars)

    n_scored = 0
    for result in results:
        score = _zscore_single(
            result.alpha, result.sharpe_ratio, result.calmar_ratio,
            stats[0], stats[1], stats[2],
        )
        result.composite_score = score
        n_scored += 1

    logger.info(
        "Cross-symbol Z-scores computed for %d results "
        "(α μ=%.4f σ=%.4f, Sharpe μ=%.2f σ=%.2f, Calmar μ=%.2f σ=%.2f)",
        n_scored,
        stats[0][0], stats[0][1],
        stats[1][0], stats[1][1],
        stats[2][0], stats[2][1],
    )
