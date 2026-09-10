"""Date/time utilities for the trading algorithm."""
import importlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar, overload

import pandas as pd
from dateutil import parser  # noqa: F401 — kept for caller compatibility

_T = TypeVar('_T')


@overload
def parse_dt(value, default: None = None) -> datetime | None: ...
@overload
def parse_dt(value, default: _T) -> datetime | _T: ...


def parse_dt(value, default=None):
    """Parse a value to datetime, handling strings from CSV loads.

    pd.read_csv without parse_dates returns dates as strings.  This helper
    ensures that any string date is converted to a proper datetime so that
    arithmetic like ``now - entry_date`` works without a TypeError.
    """
    if value is None:
        return default
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    try:
        return pd.to_datetime(value).to_pydatetime()
    except (ValueError, TypeError):
        logging.getLogger(__name__).warning(
            "Could not parse date value: %s, using default", value
        )
        return default


@overload
def ensure_utc(value: None) -> None: ...
@overload
def ensure_utc(value: Any) -> datetime: ...


def ensure_utc(value):
    """Return ``value`` as a timezone-aware UTC ``datetime``.

    Naive datetimes are assumed to be UTC wall-clock (the convention used by
    Alpaca-derived timestamps after ``tzinfo`` is stripped), so they are
    localized to UTC.  Aware datetimes are converted to UTC.

    This is DST-safe: ``now - ensure_utc(entry_date)`` yields a correct
    absolute duration regardless of EDT/EST transitions.
    """
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        value = parse_dt(value)
        if value is None:
            return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def is_trading_day(date: Optional[datetime] = None) -> bool:
    """
    Check if a given date is a trading day (market open).

    Args:
        date: Date to check (defaults to today)

    Returns:
        True if it's a trading day
    """
    if date is None:
        date = datetime.now()

    # Check if it's a weekend
    if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Check if it's a US market holiday
    try:
        holidays_module = importlib.import_module('holidays')
        us_holidays = holidays_module.country_holidays('US', years=date.year)
    except (ImportError, AttributeError):
        us_holidays = set()

    # Additional market-specific holidays
    market_holidays = [
        # Add any additional market holidays that aren't in the standard US holidays
    ]

    if date.date() in us_holidays:
        return False

    for holiday_date in market_holidays:
        if date.date() == holiday_date:
            return False

    return True
