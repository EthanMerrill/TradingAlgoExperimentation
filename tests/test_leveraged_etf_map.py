#!/usr/bin/env python3
"""
Unit tests for the leveraged single-stock ETF → underlying map.
"""
import os
import sys
import unittest

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from strategies.leveraged_single_stock_etfs import (  # noqa: E402
    LEVERAGED_SINGLE_STOCK_ETFS,
    LeveragedEtf,
    available_etfs,
    etfs_for,
    net_flow_direction,
    underlyings,
    validate_available,
)


class TestLeveragedEtfMap(unittest.TestCase):
    """Structural integrity + helper behavior of the ETF map."""

    def test_map_is_populated(self):
        self.assertGreater(len(LEVERAGED_SINGLE_STOCK_ETFS), 10)

    def test_dict_key_matches_ticker(self):
        for key, etf in LEVERAGED_SINGLE_STOCK_ETFS.items():
            self.assertEqual(key, etf.etf)

    def test_entries_well_formed(self):
        for etf in LEVERAGED_SINGLE_STOCK_ETFS.values():
            self.assertTrue(etf.underlying, etf.etf)
            self.assertEqual(etf.underlying, etf.underlying.upper(), etf.etf)
            self.assertNotEqual(etf.leverage, 0.0, etf.etf)
            self.assertTrue(etf.family, etf.etf)
            self.assertTrue(etf.name, etf.etf)
            if etf.aum_usd is not None:
                self.assertGreater(etf.aum_usd, 0, etf.etf)

    def test_is_inverse_property(self):
        self.assertTrue(LeveragedEtf("X", "Y", -2.0, "F", "n").is_inverse)
        self.assertFalse(LeveragedEtf("X", "Y", 2.0, "F", "n").is_inverse)

    def test_underlyings_sorted_and_unique(self):
        symbols = underlyings()
        self.assertEqual(symbols, sorted(set(symbols)))
        self.assertEqual(
            set(symbols),
            {etf.underlying for etf in LEVERAGED_SINGLE_STOCK_ETFS.values()},
        )

    def test_every_underlying_has_a_bull_fund(self):
        for underlying in underlyings():
            self.assertTrue(
                any(e.leverage > 0 for e in etfs_for(underlying)), underlying)

    def test_etfs_for_is_case_insensitive(self):
        self.assertEqual(
            {e.etf for e in etfs_for("nvda")},
            {e.etf for e in etfs_for("NVDA")},
        )
        self.assertTrue(etfs_for("TSLA"))

    def test_unknown_underlying_returns_empty(self):
        self.assertEqual(etfs_for("ZZZZ"), [])

    def test_net_flow_direction_long_only_sign(self):
        # SNDK is mapped to a single long leveraged fund.
        self.assertGreater(net_flow_direction("SNDK", 0.01, use_aum=False), 0)
        self.assertLess(net_flow_direction("SNDK", -0.01, use_aum=False), 0)

    def test_net_flow_direction_matches_manual_sum(self):
        underlying = "TSLA"
        move = 0.02
        expected = sum(
            e.leverage * (e.aum_usd or 1.0) * move for e in etfs_for(underlying))
        self.assertAlmostEqual(
            net_flow_direction(underlying, move, use_aum=True), expected, places=6)

    def test_validate_available_full_and_partial(self):
        all_symbols = set(LEVERAGED_SINGLE_STOCK_ETFS) | set(underlyings()) | {"ZZZ"}
        full = validate_available(all_symbols)
        self.assertEqual(full["missing"], [])
        self.assertEqual(full["underlying_missing"], [])

        partial = validate_available({"TSLL", "TSLA"})
        self.assertIn("NVDL", partial["missing"])
        self.assertIn("NVDA", partial["underlying_missing"])

    def test_available_etfs_filters(self):
        self.assertEqual({e.etf for e in available_etfs({"TSLL"})}, {"TSLL"})
        self.assertEqual(available_etfs({"ZZZZ"}), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)