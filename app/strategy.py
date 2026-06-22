"""
Strategy backtesting module.
Replaces the legacy backtrader-based approach with a modern vectorized implementation.
"""
import logging
from dataclasses import dataclass
# pylint: disable=broad-exception-caught,logging-fstring-interpolation
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
from data_provider import TechnicalIndicators, data_provider
from utils import PerformanceMetrics

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a single backtest run."""
    symbol: str
    rsi_period: int
    rsi_lower: int
    rsi_upper: int
    total_return: float
    buy_and_hold_return: float
    alpha: float
    num_trades: int
    win_rate: float
    avg_trade_duration: float
    max_drawdown: float
    sharpe_ratio: float
    profitable: bool
    calmar_ratio: float = 0.0
    composite_score: float = 0.0
    # Current RSI value at time of backtest
    current_rsi: Optional[float] = None
    # Add trade details to the result
    trade_details: Optional[List[Dict]] = None
    direction: str = "long"


class RSIStrategy:
    """Vectorized RSI trading strategy."""

    def __init__(self, rsi_period: int, rsi_lower: int, rsi_upper: int, max_hold_days: Optional[int] = None, direction: str = "long"):
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.max_hold_days = max_hold_days or globalConfig.MAX_HOLD_DAYS
        self.direction = direction  # "long" or "short"

    @staticmethod
    def calculate_price_for_target_rsi(data: pd.DataFrame, target_rsi: float, rsi_period: int = 14) -> Optional[float]:
        """
        Calculate the price needed to achieve a target RSI value.

        Args:
            data: DataFrame with OHLCV data (needs at least rsi_period + 1 rows)
            target_rsi: Desired RSI value (0-100)
            rsi_period: RSI calculation period

        Returns:
            Price needed to achieve target RSI, or None if calculation fails
        """
        try:
            if len(data) < rsi_period + 1:
                logger.warning(
                    f"Insufficient data for RSI calculation. Need {rsi_period + 1} rows, got {len(data)}")
                return None

            if not (0 <= target_rsi <= 100):
                logger.warning(
                    f"Invalid RSI target: {target_rsi}. Must be between 0 and 100")
                return None

            # Get the most recent prices (excluding the current/future price we're solving for)
            # Last rsi_period closing prices - handle both 'c' and 'close' column names
            price_col = RSIStrategy._get_price_column(data)
            prices = data[price_col].iloc[-(rsi_period):].values

            if len(prices) < rsi_period:
                return None

            # Calculate price changes for the historical period
            price_changes = np.diff(np.array(prices))

            # Separate gains and losses
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)

            # Calculate current average gain and loss
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0

            # Avoid division by zero
            if avg_loss == 0:
                if target_rsi < 100:
                    # If no losses and target RSI < 100, we need a loss
                    # Return a price lower than current to create a loss
                    return prices[-1] * 0.99
                else:
                    # If no losses and target RSI = 100, any gain works
                    return prices[-1] * 1.01

            # RSI formula: RSI = 100 - (100 / (1 + RS))
            # Where RS = Average Gain / Average Loss
            # Solving for RS: RS = (100 - target_rsi) / target_rsi * 100
            # We need: new_avg_gain / new_avg_loss = rs_target
            # With Wilder's smoothing: new_avg_gain = (old_avg_gain * (period-1) + new_gain) / period
            # Similar for losses

            current_price = prices[-1]

            # Binary search for the target price
            min_price = current_price * 0.5  # 50% down
            max_price = current_price * 1.5  # 50% up
            tolerance = 0.01  # RSI tolerance
            max_iterations = 100

            for _ in range(max_iterations):
                test_price = (min_price + max_price) / 2
                price_change = test_price - current_price

                # Calculate new gains and losses with the test price
                if price_change > 0:
                    new_gain = price_change
                    new_loss = 0
                else:
                    new_gain = 0
                    new_loss = -price_change

                # Calculate new averages using Wilder's smoothing
                new_avg_gain = (avg_gain * (rsi_period - 1) +
                                new_gain) / rsi_period
                new_avg_loss = (avg_loss * (rsi_period - 1) +
                                new_loss) / rsi_period

                if new_avg_loss == 0:
                    calculated_rsi = 100
                else:
                    rs = new_avg_gain / new_avg_loss
                    calculated_rsi = 100 - (100 / (1 + rs))

                # Check if we're close enough
                if abs(calculated_rsi - target_rsi) < tolerance:
                    return test_price

                # Adjust search range
                if calculated_rsi < target_rsi:
                    min_price = test_price
                else:
                    max_price = test_price

            # Return best approximation
            return (min_price + max_price) / 2

        except Exception as e:
            logger.error("Error calculating price for target RSI: %s", e)
            return None

    def backtest(self, data: pd.DataFrame, symbol: str, initial_cash: float = 10000) -> BacktestResult:
        """
        Run vectorized backtest of RSI strategy.

        Args:
            data: DataFrame with OHLCV data
            initial_cash: Starting cash amount
            symbol: Optional symbol name (if not provided, will try to extract from data)

        Returns:
            BacktestResult object with performance metrics
        """
        try:
            if len(data) < self.rsi_period + 10:
                return self._create_null_result(symbol or "UNKNOWN")

            # Determine symbol name - prioritize passed parameter, then try to extract from data
            if symbol is None:
                if 'symbol' in data.columns:
                    symbol = str(data['symbol'].iloc[0])
                elif hasattr(data, 'symbol'):
                    symbol = str(data.symbol)
                else:
                    symbol = "UNKNOWN"

            # Clean symbol name to ensure it's safe for filenames
            symbol = str(symbol).strip().replace('\n', '').replace('\r', '')
            if not symbol or symbol.isspace():
                symbol = "UNKNOWN"

            # Calculate RSI
            price_col = self._get_price_column(data)
            rsi = TechnicalIndicators.calculate_rsi(
                data, self.rsi_period, price_col)

            return self._run_backtest_core(data, symbol, rsi, initial_cash)

        except Exception as e:
            logger.error("Error in backtest: %s", e)
            return self._create_null_result("ERROR")

    def backtest_with_rsi(self, data: pd.DataFrame, symbol: str, rsi: pd.Series, initial_cash: float = 10000) -> BacktestResult:
        """
        Run backtest using a precomputed RSI series (avoids redundant recalculation).

        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol (required — caller must provide it)
            rsi: Precomputed RSI series matching the strategy's rsi_period
            initial_cash: Starting cash amount

        Returns:
            BacktestResult object with performance metrics
        """
        try:
            if len(data) < self.rsi_period + 10:
                return self._create_null_result(symbol)
            return self._run_backtest_core(data, symbol, rsi, initial_cash)
        except Exception as e:
            logger.error("Error in backtest_with_rsi: %s", e)
            return self._create_null_result("ERROR")

    def _run_backtest_core(self, data: pd.DataFrame, symbol: str, rsi: pd.Series, initial_cash: float) -> BacktestResult:
        """Core backtest logic shared by backtest() and backtest_with_rsi()."""
        # Get current RSI (last value)
        current_rsi = rsi.iloc[-1] if not rsi.empty else None

        # Generate signals
        signals = self._generate_signals(data, rsi)

        # Calculate returns
        returns = self._calculate_returns(data, signals, initial_cash)
        # Calculate buy and hold return
        price_col = self._get_price_column(data)
        buy_and_hold_return = (
            data[price_col].iloc[-1] / data[price_col].iloc[0]) - 1

        # Calculate metrics
        total_return = returns['portfolio_value'].iloc[-1] / \
            initial_cash - 1
        alpha = total_return - buy_and_hold_return

        trades_summary, trade_details = self._analyze_trades(signals, data)

        # Precompute risk metrics once to avoid redundant calculations
        sharpe = self._calculate_sharpe_ratio(returns['daily_returns'])
        calmar = PerformanceMetrics.calculate_calmar_ratio(
            returns['daily_returns'], returns['portfolio_value'])

        return BacktestResult(
            symbol=symbol,
            rsi_period=self.rsi_period,
            rsi_lower=self.rsi_lower,
            rsi_upper=self.rsi_upper,
            total_return=total_return,
            buy_and_hold_return=buy_and_hold_return,
            alpha=alpha,
            num_trades=trades_summary['num_trades'],
            win_rate=trades_summary['win_rate'],
            avg_trade_duration=trades_summary['avg_duration'],
            max_drawdown=self._calculate_max_drawdown(
                returns['portfolio_value']),
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            profitable=total_return > 0,
            current_rsi=current_rsi,
            trade_details=trade_details,
            direction=self.direction
        )

    def _generate_signals(self, data: pd.DataFrame, rsi: pd.Series) -> pd.DataFrame:
        """Generate buy/sell signals based on RSI.

        Exit priority (highest to lowest): RSI cross > stop-loss > take-profit > max-hold-days.
        This mirrors live bracket-order OCO logic where stop-loss and take-profit compete.

        For short direction: entry = RSI cross-above rsi_upper, cover = price below
        rsi-implied target (rsi_lower) or RSI cross-below rsi_lower or stop-loss or max hold.
        Position values: 1 = long, -1 = short, 0 = flat.
        """
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = rsi
        signals['position'] = 0
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        signals['sell_reason'] = None

        if self.direction == "long":
            # Buy when RSI crosses below lower threshold
            signals['buy_signal'] = (rsi < self.rsi_lower) & (
                rsi.shift(1) >= self.rsi_lower)
            # Sell when RSI crosses above upper threshold
            signals['sell_signal'] = (rsi > self.rsi_upper) & (
                rsi.shift(1) <= self.rsi_upper)
        else:  # short
            # Short-sell when RSI crosses ABOVE upper threshold
            signals['buy_signal'] = (rsi > self.rsi_upper) & (
                rsi.shift(1) <= self.rsi_upper)
            # Cover when RSI crosses BELOW lower threshold
            signals['sell_signal'] = (rsi < self.rsi_lower) & (
                rsi.shift(1) >= self.rsi_lower)

        # Mark pre-computed RSI-cross sells with their reason
        signals.loc[signals['sell_signal'], 'sell_reason'] = 'rsi_cross'

        # Track position state
        position = 0
        entry_date = None
        entry_price = None
        target_exit_price = None  # RSI-implied take-profit / cover price
        price_col = self._get_price_column(data)

        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i] and position == 0:
                position = 1 if self.direction == "long" else -1
                entry_date = signals.index[i]
                exec_i = min(i + 1, len(data) - 1)
                entry_price = data[price_col].iloc[exec_i]
                # Compute RSI-implied exit price using data available at entry.
                if exec_i >= self.rsi_period:
                    target_rsi_exit = self.rsi_upper if self.direction == "long" else self.rsi_lower
                    target_exit_price = RSIStrategy.calculate_price_for_target_rsi(
                        data.iloc[:exec_i +
                                  1], target_rsi_exit, self.rsi_period
                    )
                else:
                    target_exit_price = None
            elif signals['sell_signal'].iloc[i] and position != 0:
                position = 0
                entry_date = None
                entry_price = None
                target_exit_price = None
            elif position != 0 and entry_date is not None:
                should_exit, reason = self._check_exit_conditions(
                    data, signals, i, entry_price, target_exit_price, entry_date, price_col
                )
                if should_exit:
                    signals.at[signals.index[i], 'sell_signal'] = True
                    signals.at[signals.index[i], 'sell_reason'] = reason
                    position = 0
                    entry_date = None
                    entry_price = None
                    target_exit_price = None
                    signals.at[signals.index[i], 'position'] = position
                    continue

            signals.at[signals.index[i], 'position'] = position

        return signals

    def _check_exit_conditions(self, data: pd.DataFrame, signals: pd.DataFrame, i: int,
                               entry_price: Optional[float], target_exit_price: Optional[float],
                               entry_date: pd.Timestamp, price_col: str) -> Tuple[bool, Optional[str]]:
        """Check all exit conditions for an open position. Returns (should_exit, reason)."""
        current_price = data[price_col].iloc[i]
        stop_loss_pct = globalConfig.STOP_LOSS_PCT
        days_held = (signals.index[i] - entry_date).days

        if self.direction == "long":
            # Stop-loss: price falls below entry * (1 - stop_loss_pct)
            if entry_price is not None and current_price <= entry_price * (1 - stop_loss_pct):
                return True, 'stop_loss'
            # Take-profit: price rises above target
            effective_target = target_exit_price
            if effective_target is None:
                effective_target = entry_price * \
                    (1 + globalConfig.TAKE_PROFIT_PCT) if entry_price else None
            if entry_price is not None and effective_target is not None and current_price >= effective_target:
                reason = 'take_profit' if target_exit_price is not None else 'take_profit_fallback'
                return True, reason
        else:  # short
            # Stop-loss: price rises above entry * (1 + stop_loss_pct)
            if entry_price is not None and current_price >= entry_price * (1 + stop_loss_pct):
                return True, 'stop_loss'
            # Cover: price falls below RSI-implied target (rsi_lower)
            effective_target = target_exit_price
            if effective_target is None:
                effective_target = entry_price * \
                    (1 - globalConfig.TAKE_PROFIT_PCT) if entry_price else None
            if entry_price is not None and effective_target is not None and current_price <= effective_target:
                reason = 'take_profit' if target_exit_price is not None else 'take_profit_fallback'
                return True, reason

        # Max hold days (common to both directions)
        if days_held >= self.max_hold_days:
            return True, 'max_hold_days'

        return False, None

    def _calculate_returns(self, data: pd.DataFrame, signals: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
        """Calculate portfolio returns based on signals.

        For longs (position=1): buy shares with cash, portfolio = cash + shares * price.
        For shorts (position=-1): sell borrowed shares for cash, portfolio = cash - shares * price.
        """
        logger.debug(
            f"Calculating returns with initial cash: {initial_cash}, RSI({self.rsi_period}, {self.rsi_lower}, {self.rsi_upper})")
        returns = pd.DataFrame(index=data.index)
        price_col = self._get_price_column(data)
        returns['price'] = data[price_col]
        # Shift position by 1 bar so execution happens at the bar *after* the signal,
        # eliminating look-ahead bias (signal uses bar-i close; trade fills at bar-i+1).
        returns['position'] = signals['position'].shift(1).fillna(0)

        # Initialize with correct dtypes to avoid FutureWarning
        returns['cash'] = float(initial_cash)
        returns['shares'] = 0.0
        returns['portfolio_value'] = float(initial_cash)

        # Ensure proper dtypes
        returns = returns.astype({
            'cash': 'float64',
            'shares': 'float64',
            'portfolio_value': 'float64'
        })

        cash = float(initial_cash)
        shares = 0.0

        trade_count = 0

        for i in range(len(returns)):
            if i == 0:
                continue

            prev_position = returns['position'].iloc[i-1]
            curr_position = returns['position'].iloc[i]
            price = returns['price'].iloc[i]

            if curr_position == 1 and prev_position == 0:
                # Enter long: use cash to buy shares
                shares = cash / price
                cash = 0.0
                trade_count += 1
            elif curr_position == 0 and prev_position == 1:
                # Exit long: sell shares for cash
                cash = shares * price
                shares = 0.0
            elif curr_position == -1 and prev_position == 0:
                # Enter short: sell borrowed shares, receive cash
                shares = cash / price
                cash = cash + shares * price
                trade_count += 1
            elif curr_position == 0 and prev_position == -1:
                # Cover short: buy back shares with cash
                cash = cash - shares * price
                shares = 0.0
            # portfolio_value = cash + (position * shares * price)
            # position is 1 for long (adds), -1 for short (subtracts)
            returns.at[returns.index[i], 'cash'] = cash
            returns.at[returns.index[i], 'shares'] = shares
            returns.at[returns.index[i],
                       'portfolio_value'] = cash + (curr_position * shares * price)

        # Calculate daily returns
        returns['daily_returns'] = returns['portfolio_value'].pct_change().fillna(0)

        final_portfolio_value = returns['portfolio_value'].iloc[-1]
        total_return_pct = (final_portfolio_value / initial_cash - 1) * 100

        logger.debug(f"Return calculation complete - trades: {trade_count}, "
                     f"final value: {final_portfolio_value:.2f}, "
                     f"return: {total_return_pct:.2f}%")

        return returns

    def _analyze_trades(self, signals: pd.DataFrame, data: pd.DataFrame) -> Tuple[Dict, List[Dict]]:
        """Analyze individual trades and return both summary stats and detailed trade list."""
        trades = []
        entry_price = None
        entry_date = None
        price_col = self._get_price_column(data)

        # Track open-position state so duplicate crosses (while already in position)
        # don't overwrite entry_price and corrupt win_rate / num_trades.
        # Prices use the bar *after* the signal to match _calculate_returns execution.
        in_position = False
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i] and not in_position:
                exec_i = min(i + 1, len(data) - 1)
                entry_price = data[price_col].iloc[exec_i]
                entry_date = data.index[exec_i]
                in_position = True
            elif signals['sell_signal'].iloc[i] and in_position:
                exec_i = min(i + 1, len(data) - 1)
                exit_price = data[price_col].iloc[exec_i]
                exit_date = data.index[exec_i]

                # Read exit reason from signals (set during _generate_signals)
                exit_reason = signals['sell_reason'].iloc[i]
                if exit_reason is None or (hasattr(exit_reason, '__len__') and len(exit_reason) == 0):
                    exit_reason = 'unknown'

                # Return calculation depends on direction
                if self.direction == "long":
                    trade_return = (exit_price / entry_price) - 1
                else:  # short: profit when price drops
                    trade_return = (entry_price / exit_price) - 1

                duration = (exit_date - entry_date).days

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'duration': duration,
                    'exit_reason': exit_reason,
                    'direction': self.direction
                })

                entry_price = None
                entry_date = None
                in_position = False

        if not trades:
            return {'num_trades': 0, 'win_rate': 0, 'avg_duration': 0}, []

        num_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['return'] > 0)
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        avg_duration = np.mean([trade['duration']
                               for trade in trades]) if trades else 0

        summary = {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_duration': avg_duration
        }

        return summary, trades

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min())

    def _calculate_sharpe_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0

        excess_returns = daily_returns - \
            (risk_free_rate / 252)  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / daily_returns.std()

    def _create_null_result(self, symbol: str) -> BacktestResult:
        """Create null result for failed backtests."""
        return BacktestResult(
            symbol=symbol,
            rsi_period=self.rsi_period,
            rsi_lower=self.rsi_lower,
            rsi_upper=self.rsi_upper,
            total_return=0.0,
            buy_and_hold_return=0.0,
            alpha=0.0,
            num_trades=0,
            win_rate=0.0,
            avg_trade_duration=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            composite_score=0.0,
            profitable=False,
            current_rsi=None,
            direction=self.direction
        )

    def build_consolidated_trades_df(self, results: List[BacktestResult]) -> pd.DataFrame:
        """
        Build a consolidated trade DataFrame from multiple backtest results.

        Args:
            results: List of BacktestResult objects containing trade details

        Returns:
            Consolidated trades DataFrame. Empty DataFrame if no trades are available.
        """
        try:
            all_trades = []

            # Collect all trades from all results
            for result in results:
                if result.trade_details:
                    for trade in result.trade_details:
                        # Add strategy and symbol info to each trade
                        trade_record = {
                            'symbol': result.symbol,
                            'rsi_period': result.rsi_period,
                            'rsi_lower': result.rsi_lower,
                            'rsi_upper': result.rsi_upper,
                            'entry_date': trade['entry_date'],
                            'entry_price': trade['entry_price'],
                            'exit_date': trade['exit_date'],
                            'exit_price': trade['exit_price'],
                            'return': trade['return'],
                            'duration': trade['duration'],
                            'exit_reason': trade.get('exit_reason', 'unknown'),
                            'direction': trade.get('direction', result.direction)
                        }
                        all_trades.append(trade_record)

            if not all_trades:
                logger.info("No trades to consolidate")
                return pd.DataFrame()

            logger.info(
                f"Consolidated {len(all_trades)} trades from {len(results)} strategies")

            # Convert to DataFrame
            trades_df = pd.DataFrame(all_trades)

            # Convert datetime to EST timezone and format for readability
            est = pytz.timezone('US/Eastern')

            def convert_to_est(timestamp):
                """Convert timestamp to EST, handling both tz-aware and tz-naive timestamps."""
                if timestamp.tz is None:
                    # If timezone-naive, assume UTC
                    return timestamp.tz_localize('UTC').tz_convert(est).strftime('%Y-%m-%d %H:%M:%S EST')
                else:
                    # If already timezone-aware, just convert
                    return timestamp.tz_convert(est).strftime('%Y-%m-%d %H:%M:%S EST')

            trades_df['entry_date_est'] = trades_df['entry_date'].apply(
                convert_to_est)
            trades_df['exit_date_est'] = trades_df['exit_date'].apply(
                convert_to_est)

            # Round numeric values for readability
            trades_df['entry_price'] = trades_df['entry_price'].round(4)
            trades_df['exit_price'] = trades_df['exit_price'].round(4)
            trades_df['return'] = trades_df['return'].round(6)

            # Reorder columns for better readability
            final_columns = [
                'symbol', 'rsi_period', 'rsi_lower', 'rsi_upper', 'direction',
                'entry_date_est', 'entry_price', 'exit_date_est', 'exit_price',
                'return', 'duration', 'exit_reason'
            ]
            trades_df = trades_df[final_columns]

            # Sort by entry date for chronological order
            trades_df = trades_df.sort_values('entry_date_est')

            return trades_df

        except Exception as e:
            logger.error(f"Error building consolidated trade DataFrame: {e}")
            return pd.DataFrame()

    @staticmethod
    def _get_price_column(data: pd.DataFrame) -> str:
        """Get the correct price column name from the DataFrame."""
        if 'close' in data.columns:
            return 'close'
        elif 'c' in data.columns:
            return 'c'
        else:
            raise ValueError(
                f"No price column found. Expected 'close' or 'c' in data columns: {list(data.columns)}")
