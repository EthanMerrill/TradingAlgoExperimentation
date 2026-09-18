"""Tests for app/zscore.py — composite Z-score scoring (incl. WF parity)."""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.base import BacktestResult  # noqa: E402
import zscore  # noqa: E402
from walk_forward import WalkForwardResult  # noqa: E402


def _result(alpha, sharpe, calmar, symbol="AAPL"):
    return BacktestResult(
        symbol=symbol,
        total_return=alpha,
        buy_and_hold_return=0.0,
        alpha=alpha,
        num_trades=10,
        win_rate=0.5,
        avg_trade_duration=5.0,
        max_drawdown=0.1,
        sharpe_ratio=sharpe,
        profitable=True,
        calmar_ratio=calmar,
    )


class TestComputeMetricTripleZscores(unittest.TestCase):
    """Shared triple scorer used by walk-forward."""

    def test_empty(self):
        self.assertEqual(zscore.compute_metric_triple_zscores([]), [])

    def test_single_metric_neutral(self):
        scores = zscore.compute_metric_triple_zscores([(0.05, 1.0, 2.0)])
        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 0.0, places=6)

    def test_scores_match_backtestresult_path(self):
        """The triple scorer must produce identical scores to the
        BacktestResult-based compute_cross_symbol_zscores for the same data
        (with calmar capped)."""
        triples = [
            (0.01, 0.5, 1.0),
            (0.05, 1.2, 3.0),
            (0.10, 2.0, 15.0),   # calmar above the cap
            (-0.02, -0.3, 0.5),
        ]
        triple_scores = zscore.compute_metric_triple_zscores(triples)

        results = [_result(a, s, c) for a, s, c in triples]
        zscore.compute_cross_symbol_zscores(results)

        for triple_score, result in zip(triple_scores, results):
            self.assertAlmostEqual(triple_score, result.composite_score, places=9)

    def test_calmar_capped(self):
        """Walk-forward previously hardcoded a 10.0 cap; shared scorer must too."""
        low = zscore.compute_metric_triple_zscores([(0.05, 1.0, 2.0), (0.05, 1.0, 5.0)])
        high = zscore.compute_metric_triple_zscores([(0.05, 1.0, 2.0), (0.05, 1.0, 500.0)])
        self.assertAlmostEqual(low[0], high[0], places=9)
        self.assertAlmostEqual(low[1], high[1], places=9)


class TestWalkForwardDelegation(unittest.TestCase):
    """Walk-forward z-scores must match the shared implementation exactly."""

    def _wf(self, alpha, sharpe, calmar, symbol="AAPL"):
        r = WalkForwardResult(symbol=symbol, direction="long")
        r.oos_total_return = alpha
        r.oos_sharpe_ratio = sharpe
        r.oos_calmar_ratio = calmar
        return r

    def test_matches_shared_scorer(self):
        triples = [
            (0.01, 0.5, 1.0),
            (0.05, 1.2, 3.0),
            (0.10, 2.0, 15.0),
            (-0.02, -0.3, 0.5),
        ]
        results = [self._wf(*t, symbol=f"S{i}") for i, t in enumerate(triples)]
        from walk_forward import WalkForwardValidator
        WalkForwardValidator._compute_wf_cross_symbol_zscores(results)

        expected = zscore.compute_metric_triple_zscores(triples)
        for r, exp in zip(results, expected):
            self.assertAlmostEqual(r.composite_score, expected_score(expected, triples, r),
                                   places=9)


def expected_score(expected, triples, r):
    """Locate the expected score for a result by its OOS triple."""
    idx = [i for i, t in enumerate(triples)
           if t == (r.oos_total_return, r.oos_sharpe_ratio, r.oos_calmar_ratio)]
    return expected[idx[0]]


if __name__ == "__main__":
    unittest.main()
