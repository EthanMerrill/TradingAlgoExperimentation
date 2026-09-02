"""
Strategy backtesting module — backward-compatible shim.

Phase B of MULTI_STRATEGY_PLAN.md moved ``BacktestResult`` and ``RSIStrategy``
into the ``strategies`` package. This module re-exports them so existing
``from strategy import BacktestResult, RSIStrategy`` imports keep working
unchanged (trading_engine, optimizer, walk_forward, zscore, storage, tests).
"""
from strategies.base import (
    BacktestResult,
    LiveSignal,
    Strategy,
    StrategyContext,
)
from strategies.rsi import RSIStrategy

__all__ = [
    "BacktestResult",
    "RSIStrategy",
    "Strategy",
    "StrategyContext",
    "LiveSignal",
]
