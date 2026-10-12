"""Trading calendar utilities."""
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Optional

import pytz

from .datetime_ import is_trading_day

US_EASTERN = pytz.timezone("US/Eastern")
RTH_OPEN = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)


def to_eastern(dt: Optional[datetime] = None) -> datetime:
    """Return ``dt`` (default: now) as a timezone-aware US/Eastern datetime.

    Naive datetimes are interpreted as the process's *local* time — the same
    assumption ``datetime.now()`` makes — then converted. The previous
    "naive means UTC" assumption silently shifted the RTH window by the
    host's UTC offset whenever the host was not running in UTC.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.astimezone(US_EASTERN)
    return dt.astimezone(US_EASTERN)


def is_regular_trading_hours(dt: Optional[datetime] = None) -> bool:
    """True during US regular trading hours: Mon-Fri, 09:30-16:00 ET."""
    now = to_eastern(dt)
    if now.weekday() >= 5:
        return False
    return RTH_OPEN <= now.time() <= RTH_CLOSE


def is_session_ended(dt: Optional[datetime] = None) -> bool:
    """True on a weekday after the 16:00 ET close."""
    now = to_eastern(dt)
    if now.weekday() >= 5:
        return False
    return now.time() > RTH_CLOSE


class TradingCalendar:
    """Helper class for trading calendar operations."""

    def __init__(self):
        self.us_eastern = pytz.timezone('US/Eastern')
        self.market_open_time = (9, 30)  # 9:30 AM ET
        self.market_close_time = (16, 0)  # 4:00 PM ET

    def is_trading_day(self, date: Optional[datetime] = None) -> bool:
        """Check if it's a trading day."""
        return is_trading_day(date)

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """True during regular trading hours (Mon-Fri, 09:30-16:00 ET)."""
        return is_regular_trading_hours(dt)

    def has_session_ended(self, dt: Optional[datetime] = None) -> bool:
        """True on a weekday after the 16:00 ET close."""
        return is_session_ended(dt)
