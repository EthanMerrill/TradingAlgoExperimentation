"""
Integration tests for the TradingEngine's order-placement methods, validated
against a live Alpaca **dev paper trading** account.

These tests call ``TradingEngine.place_buy_order()`` and
``TradingEngine.place_market_sell_order()`` — the same code paths used in
production — and then check Alpaca's API to confirm the orders / positions
look correct.  They are meant to be run explicitly:

    pytest tests/test_order_integration.py -v

All tests are skipped when ALPACA_DEV_PAPER_KEY / ALPACA_DEV_PAPER_SECRET
are absent or empty.
"""
import logging
import os
import sys
import time
from typing import List, Optional
from unittest.mock import Mock, patch

import pytest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

# Ensure the app directory is on the path so that existing conftest /
# project-level imports work if needed.
_app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Engine classes under test.
from trading_engine import TradingEngine, TradingOpportunity  # noqa: E402

# Storage serialization helper for position-save validation.
from storage.backend import normalize_position_for_save  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def alpaca_credentials() -> dict:
    """Return dev paper-trading credentials, skipping if they are missing."""
    key = os.getenv("ALPACA_DEV_PAPER_KEY", "").strip()
    secret = os.getenv("ALPACA_DEV_PAPER_SECRET", "").strip()

    if not key or not secret:
        pytest.skip(
            "ALPACA_DEV_PAPER_KEY and / or ALPACA_DEV_PAPER_SECRET "
            "environment variables are not set or are empty."
        )

    return {"api_key": key, "secret_key": secret}


@pytest.fixture(scope="session")
def trading_client(alpaca_credentials: dict) -> TradingClient:
    """Create an Alpaca paper TradingClient and verify connectivity."""
    client = TradingClient(
        api_key=alpaca_credentials["api_key"],
        secret_key=alpaca_credentials["secret_key"],
        paper=True,
    )

    # Smoke-check: the account endpoint must be reachable.
    try:
        account = client.get_account()
        logger.info(
            "Alpaca account %s — status=%s, buying_power=%s",
            getattr(account, "account_number", "?"),
            getattr(account, "status", "?"),
            getattr(account, "buying_power", "?"),
        )
    except Exception as exc:
        pytest.skip(
            f"Alpaca API is unreachable or credentials are invalid: {exc}")

    return client


@pytest.fixture(scope="session")
def test_symbol() -> str:
    """Return the symbol used for integration-test orders."""
    return os.getenv("TEST_SYMBOL", "F")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLACEMENT_VALID_STATUSES = frozenset({
    "accepted",
    "pending_new",
    "new",
    "filled",
    "partially_filled",
})


def _status_value(order: object) -> str:
    """Extract the raw status string from an Alpaca order/enum, e.g. 'pending_new'."""
    val = getattr(order, "status", "")
    return getattr(val, "value", str(val))


def _side_value(order: object) -> str:
    """Extract the raw side string from an Alpaca order/enum, e.g. 'buy'."""
    val = getattr(order, "side", "")
    return getattr(val, "value", str(val))


def _is_filled(status: str) -> bool:
    """Return True when the order status means the trade was executed."""
    return status in ("filled", "partially_filled")


def _wait_for_order_update(client: TradingClient, order_id: str, timeout: float = 6.0) -> object:
    """Re-fetch the order until its status stabilises or *timeout* elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        order = client.get_order_by_id(order_id)
        status = _status_value(order)
        if status not in ("pending_new", "accepted"):
            return order
        time.sleep(1.0)
    return client.get_order_by_id(order_id)


def _make_engine(trading_client: TradingClient) -> TradingEngine:
    """Build a TradingEngine wired to a real Alpaca client, with storage mocked."""
    with patch("trading_engine.data_provider"), patch("trading_engine.storage"):
        engine = TradingEngine()
    engine.trading_client = trading_client
    engine._positions_manager = Mock()
    engine._positions_manager.positions = []
    engine._positions_manager.open_position = Mock()
    engine._positions_manager.close_position = Mock()
    engine.dry_run = False
    return engine


def _make_opportunity(symbol: str) -> TradingOpportunity:
    """Return a 1-share long opportunity with wide bracket prices.

    Stop-loss and take-profit prices are deliberately far from the current
    market so that Alpaca accepts the bracket order regardless of price level.
    """
    return TradingOpportunity(
        symbol=symbol,
        current_rsi=25.0,
        target_rsi_lower=30,
        target_rsi_upper=70,
        rsi_period=14,
        backtest_return=0.15,
        alpha=0.05,
        win_rate=0.8,
        entry_price=10.0,
        stop_loss_price=5.0,       # wide — valid sell stop for any buy bracket
        take_profit_price=50.0,    # wide — valid sell limit for any buy bracket
        num_trades=10,
    )


def _find_open_orders(client: TradingClient, symbol: str) -> List[object]:
    """Return every open order for *symbol* (entry + bracket legs)."""
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    return list(client.get_orders(filter=request))


def _cancel_all_open_for_symbol(client: TradingClient, symbol: str) -> None:
    """Cancel every open order for *symbol* (graceful no-op if none exist)."""
    for order in _find_open_orders(client, symbol):
        order_id = getattr(order, "id", "")
        try:
            client.cancel_order_by_id(order_id)
            logger.info("Cancelled open order %s for %s", order_id, symbol)
        except Exception:
            # 422 = already filled/canceled — that's fine.
            logger.info(
                "Order %s could not be cancelled (already terminal)", order_id)


# ---------------------------------------------------------------------------
# Phase 1 — Connectivity smoke test
# ---------------------------------------------------------------------------

class TestAlpacaConnectivity:
    """Verify the Alpaca paper-trading client works before ordering."""

    def test_get_account_returns_expected_fields(self, trading_client: TradingClient):
        """The account endpoint must return key fields."""
        account = trading_client.get_account()

        assert account is not None
        assert getattr(account, "account_number", None) is not None
        assert getattr(account, "status", "") != ""
        assert getattr(account, "buying_power", None) is not None


# ---------------------------------------------------------------------------
# Phase 2 — place_buy_order verified against Alpaca
# ---------------------------------------------------------------------------

class TestPlaceBuyOrder:
    """Call ``TradingEngine.place_buy_order()`` and validate via Alpaca's API."""

    def test_place_buy_order_and_verify_in_alpaca(
        self, trading_client: TradingClient, test_symbol: str
    ):
        """``place_buy_order()`` returns True and Alpaca shows the order."""
        engine = _make_engine(trading_client)
        opp = _make_opportunity(test_symbol)

        # ---- Act: use the production code path ----
        result = engine.place_buy_order(opp, shares=1)
        assert result is True, "place_buy_order should return True on success"

        # ---- Verify in Alpaca ----
        open_orders = _find_open_orders(trading_client, test_symbol)
        assert len(open_orders) >= 1, (
            f"Expected at least 1 open order for {test_symbol} in Alpaca; "
            f"found {len(open_orders)}"
        )

        # The entry leg should be the first order (or find it by order_class).
        entry = open_orders[0]
        entry_id = str(getattr(entry, "id", ""))

        assert getattr(entry, "symbol", "") == test_symbol
        assert str(getattr(entry, "qty", "")) == "1"
        assert _side_value(entry) == "buy"

        status = _status_value(entry)
        assert status in _PLACEMENT_VALID_STATUSES, (
            f"Unexpected entry order status: {status}"
        )

        logger.info(
            "Engine.place_buy_order → Alpaca order %s, status=%s",
            entry_id, status,
        )

        # Let the order stabilise so the cleanup test has an accurate view.
        updated = _wait_for_order_update(trading_client, entry_id)
        logger.info("Order %s stabilised status=%s",
                    entry_id, _status_value(updated))

    def test_place_buy_order_cleanup(
        self, trading_client: TradingClient, test_symbol: str
    ):
        """Cancel the entry order OR liquidate if it already filled.

        Uses ``place_market_sell_order()`` for liquidation — exercising
        another production code path.
        """
        engine = _make_engine(trading_client)

        open_orders = _find_open_orders(trading_client, test_symbol)

        # Find the entry leg (bracket parent).
        entry_order = None
        for order in open_orders:
            oid = str(getattr(order, "id", ""))
            # entry legs have 'simple' or 'bracket' order_class; bracket legs
            # have 'stop' / 'limit'.  Prefer the first non-leg order.
            cls_val = getattr(getattr(order, "order_class", ""), "value", "")
            if cls_val in ("simple", "bracket"):
                entry_order = order
                break
        if entry_order is None and open_orders:
            entry_order = open_orders[0]  # fallback

        if entry_order is None:
            # The order may have already been cleaned up by a previous run
            # or the market closed it.  Make sure no position lingers.
            positions = trading_client.get_all_positions()
            held = [p for p in positions if getattr(
                p, "symbol", "") == test_symbol]
            assert not held, (
                f"No open orders but {test_symbol} position still exists"
            )
            logger.info(
                "Cleanup: no open orders or positions for %s — already clear.", test_symbol)
            return

        entry_id = str(getattr(entry_order, "id", ""))
        status = _status_value(entry_order)

        if _is_filled(status):
            # ---- Position exists → liquidate via engine ----
            _cancel_all_open_for_symbol(trading_client, test_symbol)
            time.sleep(1.0)

            positions = trading_client.get_all_positions()
            matching = [p for p in positions if getattr(
                p, "symbol", "") == test_symbol]
            if matching:
                qty = str(getattr(matching[0], "qty", "1"))
                logger.info(
                    "Liquidating %s x %s via place_market_sell_order", qty, test_symbol)

                sell_ok = engine.place_market_sell_order(
                    test_symbol, int(
                        float(qty)), "integration_test_liquidation"
                )
                assert sell_ok is True, "place_market_sell_order should return True"
                time.sleep(1.5)

                positions_after = trading_client.get_all_positions()
                still_held = [p for p in positions_after if getattr(
                    p, "symbol", "") == test_symbol]
                assert not still_held, (
                    f"Position for {test_symbol} still exists after place_market_sell_order"
                )
                logger.info(
                    "Liquidation complete — no position held for %s", test_symbol)
        else:
            # ---- Not filled → cancel ----
            _cancel_all_open_for_symbol(trading_client, test_symbol)
            time.sleep(1.0)

            order = trading_client.get_order_by_id(entry_id)
            final_status = _status_value(order)
            # Market orders can fill before the cancel takes effect.
            if _is_filled(final_status):
                logger.info(
                    "Cleanup: order filled during cancel — liquidating via engine")
                positions = trading_client.get_all_positions()
                matching = [p for p in positions if getattr(
                    p, "symbol", "") == test_symbol]
                if matching:
                    qty = str(getattr(matching[0], "qty", "1"))
                    engine.place_market_sell_order(
                        test_symbol, int(float(qty)), "cleanup_liquidation"
                    )
                    time.sleep(1.5)
                positions_after = trading_client.get_all_positions()
                still_held = [p for p in positions_after if getattr(
                    p, "symbol", "") == test_symbol]
                assert not still_held, (
                    f"Position for {test_symbol} still exists after place_market_sell_order"
                )
            else:
                assert final_status == "canceled", (
                    f"Expected canceled, got {final_status}"
                )
                logger.info("Order %s cancelled.", entry_id)


# ---------------------------------------------------------------------------
# Phase 3 — Self-contained full lifecycle via engine methods
# ---------------------------------------------------------------------------

class TestFullOrderLifecycle:
    """Place → verify → clean up, all via TradingEngine methods."""

    def test_full_lifecycle_via_engine(
        self, trading_client: TradingClient, test_symbol: str
    ):
        """End-to-end: place_buy_order, verify Alpaca, then clean up."""
        engine = _make_engine(trading_client)
        opp = _make_opportunity(test_symbol)

        # ---- Place ----
        result = engine.place_buy_order(opp, shares=1)
        assert result is True

        # ---- Find & verify in Alpaca ----
        open_orders = _find_open_orders(trading_client, test_symbol)
        assert len(open_orders) >= 1

        entry = open_orders[0]
        entry_id = str(getattr(entry, "id", ""))
        assert getattr(entry, "symbol", "") == test_symbol
        assert str(getattr(entry, "qty", "")) == "1"
        assert _side_value(entry) == "buy"

        status = _status_value(entry)
        assert status in _PLACEMENT_VALID_STATUSES

        logger.info("Lifecycle: placed %s — status=%s", entry_id, status)

        updated = _wait_for_order_update(trading_client, entry_id)
        status = _status_value(updated)
        logger.info("Lifecycle: stabilised %s — status=%s", entry_id, status)

        # ---- Cleanup ----
        if _is_filled(status):
            _cancel_all_open_for_symbol(trading_client, test_symbol)
            time.sleep(1.0)

            positions = trading_client.get_all_positions()
            matching = [p for p in positions if getattr(
                p, "symbol", "") == test_symbol]
            if matching:
                qty = str(getattr(matching[0], "qty", "1"))
                engine.place_market_sell_order(
                    test_symbol, int(float(qty)), "lifecycle_cleanup"
                )
                time.sleep(1.5)

            positions_after = trading_client.get_all_positions()
            still_held = [p for p in positions_after if getattr(
                p, "symbol", "") == test_symbol]
            assert not still_held, f"{test_symbol} position still held after liquidation"
        else:
            _cancel_all_open_for_symbol(trading_client, test_symbol)
            time.sleep(1.0)

            order = trading_client.get_order_by_id(entry_id)
            final_status = _status_value(order)
            # Market orders can fill before the cancel takes effect.
            if _is_filled(final_status):
                logger.info(
                    "Lifecycle: order filled during cancel — liquidating")
                positions = trading_client.get_all_positions()
                matching = [p for p in positions if getattr(
                    p, "symbol", "") == test_symbol]
                if matching:
                    qty = str(getattr(matching[0], "qty", "1"))
                    engine.place_market_sell_order(
                        test_symbol, int(float(qty)), "lifecycle_cleanup"
                    )
                    time.sleep(1.5)
                positions_after = trading_client.get_all_positions()
                still_held = [p for p in positions_after if getattr(
                    p, "symbol", "") == test_symbol]
                assert not still_held, f"{test_symbol} position still held after liquidation"
            else:
                assert final_status == "canceled", (
                    f"Expected canceled, got {final_status}"
                )


# ---------------------------------------------------------------------------
# Phase 4 — Storage validation: capture Position & compare against Alpaca
# ---------------------------------------------------------------------------

class TestStorageValidation:
    """After ``place_buy_order()``, capture the ``Position`` that would be
    persisted and validate it against Alpaca's ground truth."""

    def test_position_data_matches_alpaca(
        self, trading_client: TradingClient, test_symbol: str
    ):
        """The Position saved to storage matches what Alpaca reports.

        Validates:
        1. Raw Position fields vs Alpaca order response
        2. If filled, Position fields vs Alpaca position response
        3. ``normalize_position_for_save()`` output correctness
        """
        engine = _make_engine(trading_client)
        opp = _make_opportunity(test_symbol)

        # ---- Spy on open_position to capture the Position object ----
        captured_positions: list = []

        def _spy_open_position(pos):
            captured_positions.append(pos)

        engine._positions_manager.open_position = Mock(
            side_effect=_spy_open_position
        )

        # ---- Place order via production code path ----
        result = engine.place_buy_order(opp, shares=1)
        assert result is True
        assert len(captured_positions) == 1, (
            "open_position should have been called exactly once"
        )

        pos = captured_positions[0]

        # ---- 1. Compare Position against Alpaca order ----
        open_orders = _find_open_orders(trading_client, test_symbol)
        assert len(open_orders) >= 1

        entry = open_orders[0]
        entry_id = str(getattr(entry, "id", ""))
        order = _wait_for_order_update(trading_client, entry_id)

        # Symbol must match
        assert pos.symbol == getattr(order, "symbol", ""), (
            f"Position.symbol={pos.symbol}, "
            f"Alpaca order.symbol={getattr(order, 'symbol', '')}"
        )

        # Quantity match
        alpaca_qty = str(getattr(order, "qty", ""))
        assert pos.quantity == 1.0, (
            f"Position.quantity={pos.quantity}, expected 1.0"
        )
        assert alpaca_qty == "1", f"Alpaca order.qty={alpaca_qty}, expected '1'"

        # Side
        assert pos.side == "long", f"Position.side={pos.side}, expected 'long'"
        assert _side_value(order) == "buy"

        # Entry price — Alpaca gives filled_avg_price only when filled
        order_status = _status_value(order)
        if _is_filled(order_status):
            filled_price = getattr(order, "filled_avg_price", None)
            if filled_price is not None:
                assert float(filled_price) > 0, (
                    f"Alpaca filled_avg_price={filled_price} should be positive"
                )
                logger.info(
                    "Order filled: Alpaca filled_avg_price=%s, "
                    "Position.entry_price=%s",
                    filled_price, pos.entry_price,
                )

        # ---- 2. If filled, compare against Alpaca positions API ----
        if _is_filled(order_status):
            time.sleep(1.0)
            positions = trading_client.get_all_positions()
            matching = [
                p for p in positions
                if getattr(p, "symbol", "") == test_symbol
            ]

            if matching:
                alpaca_pos = matching[0]
                alpaca_pos_qty = str(getattr(alpaca_pos, "qty", ""))
                alpaca_entry = getattr(alpaca_pos, "avg_entry_price", None)
                alpaca_current = getattr(alpaca_pos, "current_price", None)

                assert alpaca_pos_qty == "1", (
                    f"Alpaca position qty={alpaca_pos_qty}, expected '1'"
                )
                if alpaca_entry is not None:
                    assert float(alpaca_entry) > 0
                logger.info(
                    "Alpaca position: qty=%s, avg_entry_price=%s, "
                    "current_price=%s",
                    alpaca_pos_qty, alpaca_entry, alpaca_current,
                )
            logger.info(
                "Position capture: symbol=%s, qty=%s, entry_price=%s, side=%s",
                pos.symbol, pos.quantity, pos.entry_price, pos.side,
            )

        # ---- 3. Validate normalize_position_for_save output ----
        save_dict = normalize_position_for_save(pos)

        # Core identity fields
        assert save_dict["symbol"] == test_symbol
        assert save_dict["shares"] == 1.0
        assert save_dict["entry_price"] == pos.entry_price
        assert save_dict["current_price"] == pos.current_price
        assert save_dict["current_rsi"] == pos.current_rsi
        assert save_dict["side"] == "long"
        assert save_dict["closed"] is False

        # Closed-position fields must be None for an open position
        assert save_dict["exit_date"] is None
        assert save_dict["exit_price"] is None
        assert save_dict["realized_return"] is None

        # Bracket prices must be present
        assert save_dict["stop_loss_price"] is not None
        assert save_dict["take_profit_price"] is not None

        # All POSITION_FIELDS from backend.py must be present
        from storage.backend import POSITION_FIELDS  # noqa: E402
        for field in POSITION_FIELDS:
            assert field in save_dict, (
                f"normalize_position_for_save missing field '{field}'"
            )

        logger.info(
            "Storage validation PASSED: %s & Alpaca agree, "
            "save dict has all %d fields",
            test_symbol, len(POSITION_FIELDS),
        )

        # ---- Cleanup ----
        _cancel_all_open_for_symbol(trading_client, test_symbol)
        time.sleep(1.0)

        order = trading_client.get_order_by_id(entry_id)
        final_status = _status_value(order)
        if _is_filled(final_status):
            positions = trading_client.get_all_positions()
            matching = [
                p for p in positions
                if getattr(p, "symbol", "") == test_symbol
            ]
            if matching:
                qty = str(getattr(matching[0], "qty", "1"))
                engine.place_market_sell_order(
                    test_symbol, int(float(qty)), "storage_validation_cleanup"
                )
                time.sleep(1.5)
