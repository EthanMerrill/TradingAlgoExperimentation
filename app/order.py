"""
Order model for the trading engine.

A lightweight dataclass representing a broker order (entry, OCO exit, or
market exit).  Orders are persisted to the storage backend keyed by
``client_order_id`` so that reconciliation can later match positions to
their fills by ID instead of heuristics.
"""
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Order statuses that are fully resolved and no longer "active".
# ``pending_cancel`` is included: although the cancel is still in flight,
# the order can no longer become a fill.
TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "suspended",
    "pending_cancel",
}

# Alpaca's client_order_id limit (characters).
_CLIENT_ORDER_ID_MAX_LEN = 48

# Allowed characters for client_order_id per Alpaca docs:
# alphanumeric plus hyphen, underscore, period, and colon.  We intentionally
# omit the colon to avoid ambiguity in CSV/URL contexts.
_CLIENT_ORDER_ID_ALLOWED = re.compile(r"[^A-Za-z0-9._-]")


@dataclass
class Order:
    """A broker order tracked by the engine.

    ``client_order_id`` is the engine's own idempotency key (set before
    submission); ``order_id`` is the id Alpaca assigns after acceptance.
    """

    client_order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    order_class: str = "simple"
    status: str = "new"
    order_id: Optional[str] = None
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    leg: Optional[str] = None  # "entry", "oco", or "market_exit"

    def __post_init__(self) -> None:
        """Normalise enums to lowercase strings and derive the leg."""
        self.side = _lower(self.side)
        self.order_type = _lower(self.order_type)
        self.order_class = _lower(self.order_class)
        self.status = _lower(self.status)

    @property
    def is_terminal(self) -> bool:
        """True when the order can no longer become a fill."""
        return self.status.lower() in TERMINAL_ORDER_STATUSES


def _lower(value: object) -> str:
    """Coerce a value (possibly an SDK enum) to a lowercase string."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value).lower()
    return str(value).lower()


def generate_client_order_id(
    symbol: str,
    side: str,
    submitted_at: Optional[datetime] = None,
    suffix: int = 0,
) -> str:
    """Build a deterministic, Alpaca-compliant client_order_id.

    Args:
        symbol: Stock symbol (sanitized to allowed characters).
        side: Order side ("buy"/"sell"/"short"/"cover").
        submitted_at: Timestamp to embed (defaults to ``datetime.now()``).
        suffix: Optional integer discriminator appended on collision.

    Returns:
        A string <= 48 chars unique enough for the idempotency key; the
        caller is responsible for guaranteeing uniqueness (see
        ``DataProvider.make_unique_client_order_id``).
    """
    clean_symbol = _CLIENT_ORDER_ID_ALLOWED.sub("", str(symbol)).upper()
    clean_side = _CLIENT_ORDER_ID_ALLOWED.sub("", str(side)).upper()
    if submitted_at is None:
        submitted_at = datetime.now()
    stamp = submitted_at.strftime("%Y%m%d%H%M%S%f")

    base = f"{clean_symbol}-{clean_side}-{stamp}"
    if suffix:
        base = f"{base}-{suffix}"

    # Defensive truncate (in case of an unusually long symbol).
    return base[:_CLIENT_ORDER_ID_MAX_LEN]
