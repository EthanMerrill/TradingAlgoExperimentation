"""
Multi-strategy framework package (Phase B of MULTI_STRATEGY_PLAN.md).

Importing this package populates the strategy registry with all concrete
strategies (currently ``rsi_mean_reversion``).
"""
from strategies.base import BacktestResult, LiveSignal, Strategy, StrategyContext
from strategies.rsi import RSIStrategy  # noqa: F401 — registers on import
from strategies.leveraged_rebalance import (  # noqa: F401 — registers on import
    LeveragedRebalanceStrategy,
)
from strategies.registry import (
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    register,
)

__all__ = [
    "BacktestResult",
    "LiveSignal",
    "Strategy",
    "StrategyContext",
    "RSIStrategy",
    "LeveragedRebalanceStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "list_strategies",
    "register",
]
