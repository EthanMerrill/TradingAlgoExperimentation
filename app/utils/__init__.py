"""
Utility package for the trading algorithm.
Re-exports all symbols for backward compatibility with `from utils import ...`.
"""
from .datetime_ import parse_dt, is_trading_day, ensure_utc
from .logging_ import setup_logging
from .calendar import TradingCalendar
from .metrics import PerformanceMetrics
from .parallelism import resolve_worker_counts
from .progress import ProgressIndicator

__all__ = [
    "parse_dt",
    "is_trading_day",
    "ensure_utc",
    "setup_logging",
    "TradingCalendar",
    "PerformanceMetrics",
    "resolve_worker_counts",
    "ProgressIndicator",
]
