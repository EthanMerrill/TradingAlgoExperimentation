"""Portfolio-level risk guard: daily loss limit, drawdown halt, manual halt.

Only gates NEW entries. Exits and protective orders must never consult this.
All checks fail open when data is unavailable (logged at debug).
"""
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_CACHE_TTL_S = 60.0
_HISTORY_LIMIT = 500
_TRUTHY = {'1', 'true', 'yes', 'on'}


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


class RiskGuard:
    """Evaluates whether new entries are currently allowed."""

    def __init__(self, config: Any, get_account_info: Callable[[], Dict[str, Any]],
                 load_session_metadata: Optional[Callable[..., Any]] = None,
                 ttl: float = _CACHE_TTL_S):
        self._config = config
        self._get_account_info = get_account_info
        self._load_meta = load_session_metadata
        self._ttl = ttl
        self._cached_at: Optional[float] = None
        self._cached_reason: Optional[str] = None
        self._last_logged: Optional[str] = None
        self.state: Dict[str, Any] = {}

    def _cfg(self, name: str, default: Any) -> Any:
        return getattr(self._config, name, default)

    def manual_halt_reason(self) -> Optional[str]:
        if os.environ.get('TRADING_HALTED', '').strip().lower() in _TRUTHY:
            return "manual halt: TRADING_HALTED env var is set"
        if self._cfg('TRADING_HALTED', False) is True:
            return "manual halt: trading.halted is true in config"
        return None

    def entry_block_reason(self) -> Optional[str]:
        """Return a reason string if new entries are blocked, else None."""
        reason = self.manual_halt_reason()
        if reason:
            return reason
        now = time.monotonic()
        if self._cached_at is not None and now - self._cached_at < self._ttl:
            return self._cached_reason
        try:
            reason = self._evaluate()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Risk check failed open: %s", e)
            reason = None
        self._cached_at = now
        self._cached_reason = reason
        if reason and reason != self._last_logged:
            logger.warning("RISK HALT — new entries blocked: %s", reason)
        self._last_logged = reason
        return reason

    def _evaluate(self) -> Optional[str]:
        info = self._get_account_info()
        if not isinstance(info, dict) or not info:
            logger.debug("Risk check: account info unavailable")
            return None
        for flag in ('trading_blocked', 'account_blocked',
                     'trade_suspended_by_user'):
            if info.get(flag) is True:
                return f"account status: {flag} is true"

        equity = _num(info.get('equity'))
        if equity is None or equity <= 0:
            return None
        self.state = {'equity': equity}

        last_equity = _num(info.get('last_equity'))
        max_loss = float(self._cfg('MAX_DAILY_LOSS_PCT', 0.03))
        if last_equity is None or last_equity <= 0:
            logger.debug("Risk check: last_equity unavailable")
        else:
            change = (equity - last_equity) / last_equity
            self.state['daily_change_pct'] = change
            if max_loss > 0 and change < -max_loss:
                return (f"daily loss limit: equity ${equity:,.2f} is "
                        f"{-change:.2%} below prior close ${last_equity:,.2f} "
                        f"(limit {max_loss:.2%})")

        max_dd = float(self._cfg('MAX_DRAWDOWN_HALT_PCT', 0.15))
        hwm = self._high_water_mark(equity)
        if hwm:
            dd = (hwm - equity) / hwm
            self.state['high_water_mark'] = hwm
            self.state['drawdown_pct'] = dd
            if max_dd > 0 and dd > max_dd:
                return (f"max drawdown halt: equity ${equity:,.2f} is "
                        f"{dd:.2%} below high-water mark ${hwm:,.2f} "
                        f"(limit {max_dd:.2%})")
        return None

    def _high_water_mark(self, equity: float) -> Optional[float]:
        if self._load_meta is None:
            return None
        try:
            rows = self._load_meta(limit=_HISTORY_LIMIT) or []
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Risk check: no session history: %s", e)
            return None
        values = []
        for row in rows:
            meta = row.get('metadata') if isinstance(row, dict) else None
            v = _num((meta or {}).get('portfolio_value'))
            if v and v > 0:
                values.append(v)
        if not values:
            return None
        return max(max(values), equity)
