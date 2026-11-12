#!/usr/bin/env python3
"""Unit tests for the broker-authoritative PositionsManager.

Model under test: the broker decides what is OPEN (symbol, signed qty, entry
price, current price); ``position_metadata`` only decorates; ``closed_trades``
is written once per round trip.
"""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from positions import (  # noqa: E402
    UNATTRIBUTED, Position, PositionsFetchError, PositionsManager)
from storage.backend import closed_trade_key  # noqa: E402


class FakeStorage:
    """In-memory stand-in for the metadata / closed_trades / snapshot API."""

    def __init__(self):
        self.metadata = {}
        self.closed = []
        self.snapshots = []
        self.metadata_readable = True
        self.closed_readable = True
        self.fail_closed_save = False

    def load_position_metadata(self):
        return [dict(v) for v in self.metadata.values()] \
            if self.metadata_readable else None

    def save_position_metadata(self, rows):
        for r in rows:
            self.metadata[r["symbol"]] = dict(r)
        return True

    def delete_position_metadata(self, symbols):
        for s in symbols:
            self.metadata.pop(s, None)
        return True

    def save_closed_trade(self, trade):
        if self.fail_closed_save:
            return False
        key = closed_trade_key(trade)
        if all(closed_trade_key(t) != key for t in self.closed):
            self.closed.append(dict(trade))
        return True

    def load_closed_trades(self, limit=500):
        if not self.closed_readable:
            return None
        return list(reversed(self.closed))[:limit]

    def save_positions(self, positions, timestamp=None):
        self.snapshots.append((timestamp, list(positions)))
        return True


def _broker(*rows):
    """rows: (symbol, qty, avg_entry, current)"""
    return pd.DataFrame([
        {"symbol": s, "qty": q, "avg_entry_price": e, "current_price": c,
         "market_value": q * c} for s, q, e, c in rows])


def _meta(symbol="AAPL", **kw):
    base = {
        "symbol": symbol, "strategy_name": "rsi_mean_reversion",
        "rsi_period": 10, "rsi_lower": 25, "rsi_upper": 65, "alpha": 0.07,
        "composite_score": 1.5, "stop_loss_price": 140.0,
        "take_profit_price": 165.0, "intraday": False,
        "entry_client_order_id": "AAPL-BUY-1", "entry_order_id": "o1",
        "entry_date": datetime(2025, 6, 1, tzinfo=timezone.utc),
        "current_rsi": 40.0, "side": "long", "quantity": 10.0,
        "entry_price": 150.0}
    base.update(kw)
    return base


def _order(side, price, cid, when, oid, symbol="AAPL"):
    return {"symbol": symbol, "side": side, "filled_qty": 10.0,
            "filled_avg_price": price, "client_order_id": cid,
            "order_id": oid, "submitted_at": when, "filled_at": when}


def _pos(side="long", cid="AAPL-BUY-1", symbol="AAPL"):
    return Position(
        symbol=symbol, quantity=-10.0 if side == "short" else 10.0,
        entry_price=150.0, current_price=151.0, current_rsi=45.0,
        entry_date=datetime(2025, 6, 1, tzinfo=timezone.utc), alpha=0.05,
        rsi_period=14, rsi_lower=30, rsi_upper=70,
        stop_loss_price=140.0, take_profit_price=160.0, client_order_id=cid)


class _Base(unittest.TestCase):
    def setUp(self):
        self.store = FakeStorage()
        self.data = Mock()
        self.data.get_current_positions_df.return_value = _broker()
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame()
        self.data.get_entry_order_for_symbol.return_value = None
        self.data.get_open_orders.return_value = pd.DataFrame()
        self.data.get_active_orders.return_value = pd.DataFrame()
        self.data.get_current_snapshot.return_value = None
        self.manager = PositionsManager(self.store, self.data)

    def broker(self, *rows):
        self.data.get_current_positions_df.return_value = _broker(*rows)


class TestPosition(unittest.TestCase):
    def test_short_side_derived_from_quantity(self):
        self.assertEqual(_pos("short").side, "short")
        self.assertEqual(_pos("long").side, "long")

    def test_optional_fields_default(self):
        p = _pos()
        self.assertFalse(p.exit_price_estimated)
        self.assertIsNone(p.exit_reason)


class TestOpenPositionsFromBroker(_Base):
    def test_empty_broker_returns_empty_list(self):
        self.assertEqual(self.manager.get_and_reconcile_positions(), [])

    def test_broker_only_position_adopted_as_unattributed(self):
        self.broker(("NVOC", 20.0, 10.0, 10.5))
        self.data.get_open_orders.return_value = pd.DataFrame([
            {"symbol": "NVOC", "leg_type": "stop_loss", "stop_price": 9.0},
            {"symbol": "NVOC", "leg_type": "take_profit", "limit_price": 12.0}])
        when = datetime(2025, 6, 2, 14, 30)
        self.data.get_entry_order_for_symbol.return_value = (when, 9.99)

        (pos,) = self.manager.get_and_reconcile_positions()

        self.assertEqual(pos.strategy_name, UNATTRIBUTED)
        self.assertEqual((pos.rsi_period, pos.rsi_lower, pos.rsi_upper),
                         (14, 30, 70))
        self.assertEqual(pos.quantity, 20.0)
        # entry price is the BROKER's, never the order-history price
        self.assertEqual(pos.entry_price, 10.0)
        self.assertEqual((pos.stop_loss_price, pos.take_profit_price),
                         (9.0, 12.0))
        self.assertEqual(pos.entry_date.replace(tzinfo=None), when)
        # adoption is persisted so it is not re-derived next cycle
        self.assertEqual(self.store.metadata["NVOC"]["strategy_name"],
                         UNATTRIBUTED)

    def test_adopted_without_broker_levels_uses_entry_anchored_defaults(self):
        self.broker(("NVOC", 20.0, 10.0, 10.5))
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertAlmostEqual(pos.stop_loss_price, 10.0 * 0.95)
        self.assertAlmostEqual(pos.take_profit_price, 10.0 * 1.15)

    def test_adopted_short_is_negative_with_inverted_defaults(self):
        self.broker(("TSLA", -5.0, 200.0, 195.0))
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertEqual(pos.side, "short")
        self.assertLess(pos.quantity, 0)
        self.assertGreater(pos.stop_loss_price, 200.0)
        self.assertLess(pos.take_profit_price, 200.0)

    def test_no_optimizer_enrichment_on_adoption(self):
        import positions
        self.assertFalse(hasattr(positions, "StrategyOptimizer"))
        self.broker(("NVOC", 20.0, 10.0, 10.5))
        self.manager.get_and_reconcile_positions()
        self.data.get_filled_orders_for_symbol.assert_not_called()

    def test_metadata_joined_by_symbol(self):
        self.store.metadata["AAPL"] = _meta()
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertEqual(pos.strategy_name, "rsi_mean_reversion")
        self.assertEqual((pos.rsi_period, pos.rsi_lower, pos.rsi_upper),
                         (10, 25, 65))
        self.assertEqual((pos.stop_loss_price, pos.take_profit_price),
                         (140.0, 165.0))
        self.assertEqual(pos.client_order_id, "AAPL-BUY-1")
        self.assertEqual(pos.composite_score, 1.5)

    def test_broker_quantity_and_prices_always_win(self):
        self.store.metadata["AAPL"] = _meta(
            quantity=3.0, entry_price=99.0)
        self.broker(("AAPL", 8.0, 151.25, 155.0))
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertEqual(pos.quantity, 8.0)
        self.assertEqual(pos.entry_price, 151.25)
        self.assertEqual(pos.current_price, 155.0)
        # informational copy refreshed from the broker
        self.assertEqual(self.store.metadata["AAPL"]["quantity"], 8.0)
        self.assertEqual(self.store.metadata["AAPL"]["entry_price"], 151.25)

    def test_metadata_never_adds_an_open_position(self):
        self.store.metadata["GHOST"] = _meta("GHOST")
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        syms = {p.symbol for p in self.manager.get_and_reconcile_positions()}
        self.assertEqual(syms, {"AAPL"})

    def test_zero_quantity_broker_row_ignored(self):
        self.broker(("AAPL", 0.0, 150.0, 152.0))
        self.assertEqual(self.manager.get_and_reconcile_positions(), [])

    def test_cache_returns_same_result_without_second_fetch(self):
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        first = self.manager.get_and_reconcile_positions()
        second = self.manager.get_and_reconcile_positions()
        self.assertIs(first, second)
        self.assertEqual(self.data.get_current_positions_df.call_count, 1)

    def test_invalidate_forces_refetch(self):
        self.manager.get_and_reconcile_positions()
        self.manager.invalidate_reconciliation_cache()
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.data.get_current_positions_df.call_count, 2)

    def test_orphan_orders_cancelled_against_broker_symbols(self):
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        self.data.get_active_orders.return_value = pd.DataFrame([
            {"symbol": "AAPL", "order_id": "a1", "order_type": "limit",
             "side": "sell"},
            {"symbol": "MSFT", "order_id": "m1", "order_type": "limit",
             "side": "sell"}])
        self.manager.get_and_reconcile_positions()
        self.data.cancel_order.assert_called_once_with("m1")


class TestBrokerFetchFailure(_Base):
    def test_failed_fetch_raises_and_closes_nothing(self):
        self.store.metadata["AAPL"] = _meta()
        self.manager.positions = [_pos()]
        self.data.get_current_positions_df.side_effect = RuntimeError("503")
        with self.assertRaises(PositionsFetchError):
            self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed, [])
        self.assertIn("AAPL", self.store.metadata)
        self.assertEqual(len(self.manager.positions), 1)
        self.assertFalse(self.manager.reconcile_ok)
        self.data.cancel_order.assert_not_called()

    def test_reconcile_ok_recovers_after_success(self):
        self.data.get_current_positions_df.side_effect = RuntimeError("x")
        with self.assertRaises(PositionsFetchError):
            self.manager.get_and_reconcile_positions()
        self.data.get_current_positions_df.side_effect = None
        self.manager.get_and_reconcile_positions()
        self.assertTrue(self.manager.reconcile_ok)

    def test_non_dataframe_response_is_a_failure(self):
        self.data.get_current_positions_df.return_value = None
        with self.assertRaises(PositionsFetchError):
            self.manager.get_and_reconcile_positions()

    def test_unreadable_metadata_suspends_closes_and_writes(self):
        self.store.metadata["AAPL"] = _meta()
        self.store.metadata_readable = False
        self.broker(("NVOC", 20.0, 10.0, 10.5))   # AAPL absent at broker
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertEqual(pos.symbol, "NVOC")
        self.assertEqual(self.store.closed, [])
        self.assertEqual(set(self.store.metadata), {"AAPL"})  # untouched
        self.manager.persist_positions()
        self.assertEqual(set(self.store.metadata), {"AAPL"})


class TestClosedByBroker(_Base):
    def setUp(self):
        super().setUp()
        self.store.metadata["AAPL"] = _meta()

    def test_close_fill_becomes_closed_trade_and_metadata_deleted(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("sell", 164.5, "oco-leg", datetime(2025, 6, 10), "o2"),
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o1")])
        self.assertEqual(self.manager.get_and_reconcile_positions(), [])

        (t,) = self.store.closed
        self.assertEqual(t["exit_price"], 164.5)
        self.assertFalse(t["exit_price_estimated"])
        self.assertEqual(t["exit_reason"], "oco_take_profit")
        self.assertEqual(t["exit_order_id"], "o2")
        self.assertAlmostEqual(t["realized_return"], 14.5 / 150.0)
        self.assertEqual(t["strategy_name"], "rsi_mean_reversion")
        self.assertEqual(t["entry_client_order_id"], "AAPL-BUY-1")
        self.assertNotIn("AAPL", self.store.metadata)
        # visible in memory as a closed position
        (closed,) = [p for p in self.manager.positions if p.closed]
        self.assertEqual(closed.exit_price, 164.5)

    def test_stop_loss_reason(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("sell", 139.9, "x", datetime(2025, 6, 10), "o2"),
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o1")])
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed[0]["exit_reason"], "oco_stop_loss")

    def test_skips_prior_round_trip_exit(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("sell", 160.0, "cur", datetime(2025, 6, 10), "o4"),
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o3"),
            _order("sell", 120.0, "old", datetime(2025, 5, 20), "o2"),
            _order("buy", 100.0, "OLD", datetime(2025, 5, 1), "o1")])
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed[0]["exit_price"], 160.0)

    def test_first_close_after_entry_beats_later_round_trip(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("sell", 99.0, "later", datetime(2025, 7, 10), "o4"),
            _order("sell", 155.0, "oco", datetime(2025, 6, 10), "o3"),
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o1")])
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed[0]["exit_price"], 155.0)

    def test_short_uses_later_buy(self):
        self.store.metadata["AAPL"] = _meta(side="short", quantity=-10.0)
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("buy", 140.0, "cover", datetime(2025, 6, 10), "o2"),
            _order("sell", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o1")])
        self.manager.get_and_reconcile_positions()
        t = self.store.closed[0]
        self.assertEqual((t["side"], t["exit_price"]), ("short", 140.0))
        self.assertAlmostEqual(t["realized_return"], 10.0 / 150.0)

    def test_entry_only_history_is_estimated_not_entry_price_fill(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o1")])
        self.data.get_current_snapshot.return_value = {"price": 158.0}
        self.manager.get_and_reconcile_positions()
        (t,) = self.store.closed
        self.assertTrue(t["exit_price_estimated"])
        self.assertEqual(t["exit_price"], 158.0)
        self.assertEqual(t["exit_reason"], "broker_closed")
        self.assertNotIn("AAPL", self.store.metadata)

    def test_estimate_prefers_last_seen_price_then_entry(self):
        self.broker(("AAPL", 10.0, 150.0, 157.0))
        self.manager.get_and_reconcile_positions()      # last seen = 157
        self.manager.invalidate_reconciliation_cache()
        self.broker()
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o1")])
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed[0]["exit_price"], 157.0)

    def test_no_history_at_all_is_estimated_at_entry(self):
        self.manager.get_and_reconcile_positions()
        (t,) = self.store.closed
        self.assertTrue(t["exit_price_estimated"])
        self.assertEqual(t["exit_price"], 150.0)

    def test_never_opened_drops_metadata_without_trade(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("sell", 120.0, "unrelated", datetime(2025, 5, 20), "o9")])
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed, [])
        self.assertNotIn("AAPL", self.store.metadata)

    def test_history_lookup_error_defers_close(self):
        self.data.get_filled_orders_for_symbol.side_effect = RuntimeError("x")
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed, [])
        self.assertIn("AAPL", self.store.metadata)

    def test_failed_trade_write_keeps_metadata_for_retry(self):
        self.store.fail_closed_save = True
        self.manager.get_and_reconcile_positions()
        self.assertIn("AAPL", self.store.metadata)
        self.store.fail_closed_save = False
        self.manager.invalidate_reconciliation_cache()
        self.manager.get_and_reconcile_positions()
        self.assertEqual(len(self.store.closed), 1)
        self.assertNotIn("AAPL", self.store.metadata)

    def test_closed_trade_written_only_once(self):
        self.manager.get_and_reconcile_positions()
        self.manager.invalidate_reconciliation_cache()
        self.manager.get_and_reconcile_positions()
        self.assertEqual(len(self.store.closed), 1)


class TestOpenPosition(_Base):
    def test_open_position_upserts_metadata_and_is_visible_same_cycle(self):
        p = _pos()
        p.strategy_name = "momentum"
        self.manager.open_position(p)
        self.assertIn(p, self.manager.positions)
        row = self.store.metadata["AAPL"]
        self.assertEqual(row["strategy_name"], "momentum")
        self.assertEqual(row["stop_loss_price"], 140.0)
        self.assertEqual(row["entry_client_order_id"], "AAPL-BUY-1")

    def test_reopening_same_symbol_replaces_not_duplicates(self):
        self.manager.open_position(_pos())
        newer = _pos()
        newer.strategy_name = "other"
        self.manager.open_position(newer)
        self.assertEqual(
            [p for p in self.manager.positions if not p.closed], [newer])

    def test_next_reconcile_overwrites_from_broker(self):
        p = _pos()
        self.manager.open_position(p)
        self.broker(("AAPL", 12.0, 150.5, 151.0))
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertEqual((pos.quantity, pos.entry_price), (12.0, 150.5))
        self.assertIsNot(pos, p)
        self.assertEqual(self.store.closed, [])

    def test_fill_not_yet_visible_at_broker_is_not_closed(self):
        self.manager.open_position(_pos())
        self.broker()                       # broker lag
        opened = self.manager.get_and_reconcile_positions()
        self.assertEqual([p.symbol for p in opened], ["AAPL"])
        self.assertEqual(self.store.closed, [])
        self.assertIn("AAPL", self.store.metadata)

    def test_after_grace_missing_position_is_closed(self):
        self.manager.open_position(_pos())
        sym, (p, ts) = next(iter(self.manager._pending_opens.items()))
        self.manager._pending_opens[sym] = (
            p, ts - self.manager.OPEN_VISIBILITY_GRACE_SECONDS - 1)
        self.manager.invalidate_reconciliation_cache()
        self.assertEqual(self.manager.get_and_reconcile_positions(), [])
        self.assertEqual(len(self.store.closed), 1)


class TestClosePosition(_Base):
    def test_close_with_fill_writes_trade_once_and_deletes_metadata(self):
        p = _pos()
        self.manager.open_position(p)
        self.manager.close_position("AAPL", exit_price=155.0)
        self.assertTrue(p.closed)
        self.assertEqual(p.exit_price, 155.0)
        self.assertFalse(p.exit_price_estimated)
        self.assertAlmostEqual(p.realized_return, 5.0 / 150.0)
        (t,) = self.store.closed
        self.assertEqual(t["exit_price"], 155.0)
        self.assertFalse(t["exit_price_estimated"])
        self.assertNotIn("AAPL", self.store.metadata)
        self.assertIn(p, self.manager.positions)   # stays as closed

    def test_close_then_reconcile_does_not_duplicate(self):
        self.manager.open_position(_pos())
        self.manager.close_position("AAPL", exit_price=155.0)
        self.broker()
        self.manager.get_and_reconcile_positions()
        self.assertEqual(len(self.store.closed), 1)
        self.assertEqual(
            len([p for p in self.manager.positions if p.closed]), 1)

    def test_close_without_price_uses_history_first_close_after_entry(self):
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("sell", 160.0, "cur", datetime(2025, 6, 10), "o4"),
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o3"),
            _order("sell", 120.0, "old", datetime(2025, 5, 20), "o2")])
        p = _pos()
        self.manager.positions = [p]
        self.manager.close_position("AAPL")
        self.assertEqual(p.exit_price, 160.0)
        self.assertFalse(p.exit_price_estimated)
        self.assertEqual(self.store.closed[0]["exit_order_id"], "o4")

    def test_close_without_price_or_history_is_estimated(self):
        p = _pos()
        self.manager.positions = [p]
        self.manager.close_position("AAPL")
        self.assertEqual(p.exit_price, 151.0)       # current_price
        self.assertTrue(p.exit_price_estimated)
        self.assertTrue(self.store.closed[0]["exit_price_estimated"])

    def test_close_short_return_sign(self):
        p = _pos("short")
        self.manager.positions = [p]
        self.manager.close_position("AAPL", exit_price=140.0)
        self.assertAlmostEqual(p.realized_return, 10.0 / 150.0)
        self.assertEqual(self.store.closed[0]["side"], "short")

    def test_close_unknown_symbol_is_noop(self):
        self.manager.close_position("NOPE", exit_price=1.0)
        self.assertEqual(self.store.closed, [])

    def test_exit_reason_preserved(self):
        p = _pos()
        p.exit_reason = "max_hold_days"
        self.manager.positions = [p]
        self.manager.close_position("AAPL", exit_price=150.0)
        self.assertEqual(self.store.closed[0]["exit_reason"], "max_hold_days")

    def test_failed_trade_write_keeps_metadata(self):
        self.store.fail_closed_save = True
        self.manager.open_position(_pos())
        self.manager.close_position("AAPL", exit_price=155.0)
        self.assertIn("AAPL", self.store.metadata)


class TestCloseConfirmation(_Base):
    """close_position is final only once the broker no longer holds it."""

    def _open_and_close(self, price=155.0):
        p = _pos()
        self.manager.open_position(p)
        self.broker(("AAPL", 10.0, 150.0, 151.0))   # broker still holds it
        self.manager.close_position("AAPL", exit_price=price)
        return p

    def test_still_held_writes_nothing_and_keeps_metadata(self):
        self._open_and_close()
        self.assertEqual(self.store.closed, [])
        self.assertIn("AAPL", self.store.metadata)

    def test_still_held_is_not_readopted_or_listed_open_within_grace(self):
        self._open_and_close()
        self.manager.invalidate_reconciliation_cache()
        self.assertEqual(self.manager.get_and_reconcile_positions(), [])
        self.assertEqual(self.store.metadata["AAPL"]["strategy_name"],
                         "rsi_mean_reversion")
        self.assertEqual(self.store.closed, [])
        # counted once (as the closed copy), not twice
        self.assertEqual(len(self.manager.positions), 1)

    def test_broker_gone_finalizes_with_the_authoritative_fill(self):
        self._open_and_close(price=155.0)
        self.broker()
        self.manager.invalidate_reconciliation_cache()
        self.manager.get_and_reconcile_positions()
        (t,) = self.store.closed
        self.assertEqual(t["exit_price"], 155.0)
        self.assertFalse(t["exit_price_estimated"])
        self.assertNotIn("AAPL", self.store.metadata)
        self.assertEqual(
            len([p for p in self.manager.positions if p.closed]), 1)

    def test_still_held_after_grace_reverts_to_open_with_original_metadata(self):
        p = self._open_and_close()
        self.manager._closing["AAPL"]["ts"] -= (
            self.manager.CLOSE_CONFIRM_GRACE_SECONDS + 1)
        self.manager.invalidate_reconciliation_cache()
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertFalse(pos.closed)
        self.assertEqual(pos.strategy_name, "rsi_mean_reversion")
        self.assertEqual((pos.stop_loss_price, pos.take_profit_price),
                         (140.0, 160.0))
        self.assertEqual(self.store.closed, [])          # nothing voided
        self.assertNotIn(p, self.manager.positions)
        self.assertEqual(
            [x.symbol for x in self.manager.positions if x.closed], [])

    def test_failed_write_retried_at_next_reconcile(self):
        self.store.fail_closed_save = True
        self.manager.open_position(_pos())
        self.manager.close_position("AAPL", exit_price=155.0)   # broker empty
        self.assertEqual(self.store.closed, [])
        self.store.fail_closed_save = False
        self.manager.invalidate_reconciliation_cache()
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self.store.closed[0]["exit_price"], 155.0)

    def test_broker_fetch_failure_at_close_leaves_it_pending(self):
        self.manager.open_position(_pos())
        self.data.get_current_positions_df.side_effect = RuntimeError("x")
        self.manager.close_position("AAPL", exit_price=155.0)
        self.assertEqual(self.store.closed, [])
        self.assertIn("AAPL", self.manager._closing)


class TestStaleMetadataSide(_Base):
    def test_side_mismatch_archives_old_and_adopts_fresh(self):
        self.store.metadata["AAPL"] = _meta(side="long", quantity=10.0)
        self.data.get_filled_orders_for_symbol.return_value = pd.DataFrame([
            _order("sell", 158.0, "exit", datetime(2025, 6, 10), "o2"),
            _order("buy", 150.0, "AAPL-BUY-1", datetime(2025, 6, 1), "o1")])
        self.broker(("AAPL", -4.0, 170.0, 168.0))     # now SHORT at broker
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertEqual(pos.side, "short")
        self.assertEqual(pos.strategy_name, UNATTRIBUTED)
        self.assertEqual(pos.entry_price, 170.0)
        (t,) = self.store.closed
        self.assertEqual((t["side"], t["exit_price"]), ("long", 158.0))
        self.assertFalse(t["exit_price_estimated"])
        self.assertEqual(self.store.metadata["AAPL"]["side"], "short")
        self.assertEqual(self.store.metadata["AAPL"]["strategy_name"],
                         UNATTRIBUTED)

    def test_side_mismatch_without_fill_is_estimated(self):
        self.store.metadata["AAPL"] = _meta(side="long")
        self.broker(("AAPL", -4.0, 170.0, 168.0))
        self.manager.get_and_reconcile_positions()
        self.assertTrue(self.store.closed[0]["exit_price_estimated"])

    def test_matching_side_keeps_metadata(self):
        self.store.metadata["AAPL"] = _meta(side="short", quantity=-4.0)
        self.broker(("AAPL", -4.0, 170.0, 168.0))
        (pos,) = self.manager.get_and_reconcile_positions()
        self.assertEqual(pos.strategy_name, "rsi_mean_reversion")
        self.assertEqual(self.store.closed, [])


class TestClosedTradesInMemoryAndDailyCap(_Base):
    """Positions opened AND closed today must still count against the cap."""

    def _remaining(self):
        from trading_engine import TradingEngine
        from config import globalConfig
        fake = SimpleNamespace(_positions_manager=self.manager)
        return (TradingEngine._new_positions_remaining_today(fake),
                globalConfig.MAX_NEW_POSITIONS_PER_DAY)

    def test_opened_and_closed_today_counted(self):
        now = datetime.now(timezone.utc)
        self.store.closed.append({
            "symbol": "OLD", "side": "long", "quantity": 5.0,
            "entry_price": 10.0, "exit_price": 11.0, "entry_date": now,
            "exit_date": now, "exit_price_estimated": False,
            "entry_client_order_id": "c1", "strategy_name": "x"})
        self.store.closed.append({
            "symbol": "YEST", "side": "long", "quantity": 5.0,
            "entry_price": 10.0, "exit_price": 11.0,
            "entry_date": now - timedelta(days=3),
            "exit_date": now - timedelta(days=2),
            "exit_price_estimated": False,
            "entry_client_order_id": "c2", "strategy_name": "x"})
        self.store.metadata["AAPL"] = _meta(entry_date=now)
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        self.manager.get_and_reconcile_positions()
        remaining, cap = self._remaining()
        self.assertEqual(remaining, max(0, cap - 2))

    def test_close_during_session_still_counts(self):
        p = _pos()
        p.entry_date = datetime.now(timezone.utc)
        self.manager.open_position(p)
        before, cap = self._remaining()
        self.manager.close_position("AAPL", exit_price=150.0)
        after, _ = self._remaining()
        self.assertEqual(before, after)
        self.assertEqual(after, max(0, cap - 1))
        # and survives a reconcile reload from closed_trades
        self.broker()
        self.manager.get_and_reconcile_positions()
        self.assertEqual(self._remaining()[0], after)

    def test_unwritten_closed_trade_kept_in_memory(self):
        self.store.fail_closed_save = True
        p = _pos()
        p.entry_date = datetime.now(timezone.utc)
        self.manager.open_position(p)
        self.manager.close_position("AAPL", exit_price=150.0)
        self.broker()
        self.manager.get_and_reconcile_positions()
        self.assertEqual(
            len([x for x in self.manager.positions if x.closed]), 1)

    def test_closed_trade_estimated_flag_round_trips(self):
        self.store.closed.append({
            "symbol": "X", "side": "short", "quantity": 2.0,
            "entry_price": 10.0, "exit_price": None,
            "entry_date": datetime.now(timezone.utc), "exit_date": None,
            "exit_price_estimated": True, "entry_client_order_id": None,
            "strategy_name": None})
        self.manager.get_and_reconcile_positions()
        (c,) = [p for p in self.manager.positions if p.closed]
        self.assertTrue(c.exit_price_estimated)
        self.assertIsNone(c.exit_price)
        self.assertEqual(c.side, "short")
        self.assertEqual(c.strategy_name, UNATTRIBUTED)


class TestUnattributedInEngine(_Base):
    def test_unattributed_counts_toward_caps_and_notional(self):
        from trading_engine import TradingEngine
        self.broker(("NVOC", 20.0, 10.0, 10.5))
        self.manager.get_and_reconcile_positions()
        fake = SimpleNamespace(_positions_manager=self.manager)
        used = TradingEngine._strategy_notional_used(fake)
        self.assertEqual(used, {UNATTRIBUTED: 200.0})
        # an unknown strategy gets the full-equity budget fallback and does
        # not consume any enabled strategy's budget
        engine = TradingEngine.__new__(TradingEngine)
        budgets = TradingEngine._strategy_budgets(engine, 10000.0)
        self.assertNotIn(UNATTRIBUTED, budgets)
        self.assertEqual(budgets.get(UNATTRIBUTED, 10000.0) - 200.0, 9800.0)

    def test_unattributed_is_daily_and_open(self):
        self.broker(("NVOC", 20.0, 10.0, 10.5))
        (p,) = self.manager.get_and_reconcile_positions()
        self.assertFalse(p.intraday)     # gets the daily OCO refresh
        self.assertFalse(p.closed)
        self.assertIsNotNone(p.entry_date)   # max-hold arithmetic works


class TestPersist(_Base):
    def test_snapshot_contains_open_positions_only(self):
        self.store.metadata["AAPL"] = _meta()
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        self.manager.get_and_reconcile_positions()
        self.manager.begin_session()
        self.manager.persist_positions()
        (_, rows) = self.store.snapshots[-1]
        self.assertEqual([r.symbol for r in rows], ["AAPL"])
        self.assertTrue(all(not r.closed for r in rows))

    def test_persist_writes_in_memory_metadata_edits(self):
        self.store.metadata["AAPL"] = _meta()
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        (pos,) = self.manager.get_and_reconcile_positions()
        pos.stop_loss_price = 145.0
        self.manager.persist_positions()
        self.assertEqual(self.store.metadata["AAPL"]["stop_loss_price"], 145.0)

    def test_persist_without_open_positions_writes_nothing(self):
        self.manager.persist_positions()
        self.assertEqual(self.store.snapshots, [])

    def test_session_timestamp_stable_within_session(self):
        self.broker(("AAPL", 10.0, 150.0, 152.0))
        self.manager.get_and_reconcile_positions()
        self.manager.begin_session()
        self.manager.persist_positions()
        self.manager.persist_positions()
        self.assertEqual(self.store.snapshots[0][0], self.store.snapshots[1][0])


class TestBrokerSlTp(_Base):
    def test_maps_legs(self):
        self.data.get_open_orders.return_value = pd.DataFrame([
            {'symbol': 'NVOC', 'leg_type': 'stop_loss', 'stop_price': 9.5,
             'limit_price': None},
            {'symbol': 'NVOC', 'leg_type': 'take_profit', 'stop_price': None,
             'limit_price': 12.5}])
        self.assertEqual(self.manager._broker_sl_tp_by_symbol(),
                         {'NVOC': (9.5, 12.5)})

    def test_best_effort(self):
        self.data.get_open_orders.side_effect = RuntimeError("x")
        self.assertEqual(self.manager._broker_sl_tp_by_symbol(), {})
        self.assertEqual(
            PositionsManager(self.store, Mock(spec=[]))
            ._broker_sl_tp_by_symbol(), {})


if __name__ == '__main__':
    unittest.main()
