"""
Strategy registry: name → strategy class map (multi-strategy framework).

Strategies self-register by being imported here; ``import strategies`` (or
importing any module that imports this one) populates the registry.
"""
from typing import Dict, List, Type

from strategies.base import Strategy

STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    """Register a Strategy subclass under its ``name`` (also usable as a decorator)."""
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(
            f"Strategy {cls.__name__} must define a non-empty 'name' attribute")
    existing = STRATEGY_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Strategy name '{name}' already registered by {existing.__name__}")
    STRATEGY_REGISTRY[name] = cls
    return cls


def get_strategy(name: str) -> Type[Strategy]:
    """Return the strategy class registered under ``name``."""
    try:
        return STRATEGY_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown strategy '{name}'. Registered strategies: {sorted(STRATEGY_REGISTRY)}"
        ) from None


def list_strategies() -> List[str]:
    """Return sorted registry keys."""
    return sorted(STRATEGY_REGISTRY)


# Import concrete strategies so they self-register. Add new strategies here.
from strategies.rsi import RSIStrategy  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import

register(RSIStrategy)
