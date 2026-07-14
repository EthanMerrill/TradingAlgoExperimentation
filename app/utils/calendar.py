"""Trading calendar utilities."""
from datetime import datetime, timedelta
from typing import Optional

import pytz

from .datetime_ import is_trading_day


class TradingCalendar:
    """Helper class for trading calendar operations."""

    def __init__(self):
        self.us_eastern = pytz.timezone('US/Eastern')
        self.market_open_time = (9, 30)  # 9:30 AM ET
        self.market_close_time = (16, 0)  # 4:00 PM ET

    def is_trading_day(self, date: Optional[datetime] = None) -> bool:
        """Check if it's a trading day."""
        return is_trading_day(date)
