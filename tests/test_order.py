#!/usr/bin/env python3
"""Unit tests for the order model and client_order_id generator."""
import os
import sys
import unittest
from datetime import datetime

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from order import (  # noqa: E402
    Order,
    TERMINAL_ORDER_STATUSES,
    generate_client_order_id,
)


class TestGenerateClientOrderId(unittest.TestCase):
    """Tests for generate_client_order_id."""

    def test_sanitizes_symbol_and_side(self):
        cid = generate_client_order_id(
            "BRK.B", "buy", datetime(2026, 8, 13, 12, 0, 0, 123456))
        self.assertIn("BRK.B-BUY-", cid)
        self.assertNotIn(" ", cid)
        self.assertNotIn("$", cid)

    def test_length_within_alpaca_limit(self):
        cid = generate_client_order_id("A" * 80, "SELL")
        self.assertLessEqual(len(cid), 48)

    def test_suffix_appended(self):
        cid = generate_client_order_id(
            "AAPL", "SELL", datetime(2026, 1, 1), suffix=2)
        self.assertTrue(cid.endswith("-2"))

    def test_default_timestamp(self):
        cid = generate_client_order_id("AAPL", "BUY")
        self.assertIn("AAPL-BUY-", cid)

    def test_deterministic_given_timestamp(self):
        ts = datetime(2026, 8, 13, 12, 0, 0, 123456)
        self.assertEqual(
            generate_client_order_id("AAPL", "BUY", ts),
            generate_client_order_id("AAPL", "BUY", ts),
        )


class _FakeEnum:
    """Minimal enum stand-in with a .value attribute."""

    value = "LIMIT"


class TestOrderDataclass(unittest.TestCase):
    """Tests for the Order dataclass."""

    def test_normalizes_enum_values_to_lowercase_strings(self):
        o = Order(
            client_order_id="abc",
            symbol="AAPL",
            side=_FakeEnum,
            qty=5.0,
            order_type=_FakeEnum,
            order_class=_FakeEnum,
            status=_FakeEnum,
        )
        self.assertEqual(o.side, "limit")
        self.assertEqual(o.order_type, "limit")
        self.assertEqual(o.order_class, "limit")
        self.assertEqual(o.status, "limit")

    def test_is_terminal(self):
        o = Order(client_order_id="abc", symbol="AAPL", side="buy",
                  qty=1, status="filled")
        self.assertTrue(o.is_terminal)
        o2 = Order(client_order_id="def", symbol="AAPL", side="buy",
                   qty=1, status="new")
        self.assertFalse(o2.is_terminal)

    def test_terminal_statuses_include_expected_values(self):
        self.assertIn("filled", TERMINAL_ORDER_STATUSES)
        self.assertIn("canceled", TERMINAL_ORDER_STATUSES)
        self.assertIn("cancelled", TERMINAL_ORDER_STATUSES)
        self.assertIn("pending_cancel", TERMINAL_ORDER_STATUSES)


if __name__ == '__main__':
    unittest.main()
