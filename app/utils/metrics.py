"""Performance metrics for trading strategies."""
import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Calculate various performance metrics for trading strategies."""

    @staticmethod
    def calculate_max_drawdown(values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return abs(drawdown.min())

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, values: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        annual_return = (1 + returns.mean()) ** 252 - 1
        max_dd = PerformanceMetrics.calculate_max_drawdown(values)

        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0

        return annual_return / max_dd
