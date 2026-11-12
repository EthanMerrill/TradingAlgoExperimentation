"""Date/time utilities for the trading algorithm."""
import logging
import os
import threading
import time as _time
from datetime import date as _date, datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple, TypeVar, overload

import pandas as pd
from dateutil import parser  # noqa: F401 — kept for caller compatibility
from dateutil.easter import easter

logger = logging.getLogger(__name__)

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


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC ``datetime``.

    Use this — not ``datetime.now()`` — for any timestamp that is persisted or
    compared against broker timestamps.

    ``datetime.now()`` returns naive LOCAL time. Persistence paths treat a
    naive value as UTC wall-clock (see ``storage.backend._safe_datetime``), so
    a local timestamp silently lands in the database shifted by the machine's
    UTC offset — e.g. 4 hours during EDT. That produced order rows whose
    ``submitted_at`` was 4h behind the row's own ``created_at``.
    """
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Market sessions (NYSE): Alpaca calendar API with a built-in rule fallback
# ---------------------------------------------------------------------------

_REGULAR_OPEN = dtime(9, 30)
_REGULAR_CLOSE = dtime(16, 0)
_EARLY_CLOSE = dtime(13, 0)

Session = Tuple[dtime, dtime]  # (open, close) in US/Eastern wall-clock


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> _date:
    d = _date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> _date:
    d = _date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: _date) -> Optional[_date]:
    """NYSE observed-day rule: Sat -> Friday, Sun -> Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set:
    """Full-day NYSE closures falling in ``year`` (built-in rule set)."""
    hol = set()
    # New Year's Day: Sunday -> Monday; Saturday -> NOT observed (no closure).
    ny = _date(year, 1, 1)
    if ny.weekday() != 5:
        hol.add(_observed(ny))
    hol.add(_nth_weekday(year, 1, 0, 3))            # MLK
    hol.add(_nth_weekday(year, 2, 0, 3))            # Presidents
    hol.add(easter(year) - timedelta(days=2))       # Good Friday
    hol.add(_last_weekday(year, 5, 0))              # Memorial
    if year >= 2022:
        hol.add(_observed(_date(year, 6, 19)))      # Juneteenth
    hol.add(_observed(_date(year, 7, 4)))           # Independence
    hol.add(_nth_weekday(year, 9, 0, 1))            # Labor
    hol.add(_nth_weekday(year, 11, 3, 4))           # Thanksgiving
    hol.add(_observed(_date(year, 12, 25)))         # Christmas
    # Observed New Year's for next year can fall on Dec 31 only when Jan 1 is
    # a Saturday, which NYSE does not observe, so nothing to add.
    return hol


def _builtin_session(d: _date) -> Optional[Session]:
    if d.weekday() >= 5 or d in nyse_holidays(d.year):
        return None
    early = False
    if d.month == 7 and d.day == 3:                 # day before Independence Day
        early = d.weekday() < 4  # Mon-Thu (July 4 on Tue-Fri)
    elif d.month == 11 and d.weekday() == 4 and d == _nth_weekday(
            d.year, 11, 3, 4) + timedelta(days=1):  # day after Thanksgiving
        early = True
    elif d.month == 12 and d.day == 24:             # Christmas Eve
        early = True
    return (_REGULAR_OPEN, _EARLY_CLOSE if early else _REGULAR_CLOSE)


_calendar_lock = threading.Lock()
_calendar_client: Any = None            # explicitly injected client (tests)
_year_cache: Dict[int, Tuple[_date, Dict[_date, Session]]] = {}
_failed_at: float = 0.0
_RETRY_AFTER_FAILURE_S = 300.0


def set_calendar_client(client: Any) -> None:
    """Inject the Alpaca TradingClient used for calendar lookups (or None)."""
    global _calendar_client
    with _calendar_lock:
        _calendar_client = client
        _year_cache.clear()
        _reset_failure()


def reset_calendar_cache() -> None:
    with _calendar_lock:
        _year_cache.clear()
        _reset_failure()


def _reset_failure() -> None:
    global _failed_at
    _failed_at = 0.0


def _resolve_client() -> Any:
    if _calendar_client is not None:
        return _calendar_client
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None  # never touch the network from tests unless injected
    try:
        from data_provider import data_provider  # type: ignore
        return data_provider.trading_client
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _to_time(v: Any) -> dtime:
    if isinstance(v, datetime):
        return v.time().replace(tzinfo=None)
    if isinstance(v, dtime):
        return v.replace(tzinfo=None)
    return parser.parse(str(v)).time()


def _fetch_year(year: int) -> Optional[Dict[_date, Session]]:
    client = _resolve_client()
    if client is None:
        return None
    try:
        from alpaca.trading.requests import GetCalendarRequest
        days = client.get_calendar(GetCalendarRequest(
            start=_date(year, 1, 1), end=_date(year, 12, 31)))
        out: Dict[_date, Session] = {}
        for day in days:
            d = day.date
            if isinstance(d, datetime):
                d = d.date()
            out[d] = (_to_time(day.open), _to_time(day.close))
        if not out:
            return None
        return out
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Alpaca calendar fetch failed (%s); using built-in "
                       "NYSE rules", e)
        return None


def get_session(d: Any) -> Optional[Session]:
    """(open, close) ET wall-clock times for ``d``, or None if market closed."""
    global _failed_at
    if isinstance(d, datetime):
        d = d.date()
    today = _date.today()
    with _calendar_lock:
        cached = _year_cache.get(d.year)
        if cached is None or cached[0] != today:
            if _time.monotonic() - _failed_at >= _RETRY_AFTER_FAILURE_S \
                    or _failed_at == 0.0:
                data = _fetch_year(d.year)
                if data is not None:
                    _year_cache[d.year] = (today, data)
                    cached = _year_cache[d.year]
                else:
                    _failed_at = _time.monotonic()
        if cached is not None:
            return cached[1].get(d)
    return _builtin_session(d)


def is_trading_day(date: Optional[datetime] = None) -> bool:
    """True if the NYSE has a session on ``date`` (default: today).

    Uses Alpaca's calendar when available, otherwise built-in NYSE rules
    (Columbus/Veterans Day are open; Good Friday is closed).
    """
    if date is None:
        date = datetime.now()
    return get_session(date) is not None
