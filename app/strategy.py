"""
Strategy backtesting module.
Replaces the legacy backtrader-based approach with a modern vectorized implementation.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
from data_provider import data_provider, TechnicalIndicators
from config import config

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


class RSIStrategy:
    """Vectorized RSI trading strategy."""
    
    def __init__(self, rsi_period: int, rsi_lower: int, rsi_upper: int, max_hold_days: int = None):
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.max_hold_days = max_hold_days or config.MAX_HOLD_DAYS
    
    def backtest(self, data: pd.DataFrame, initial_cash: float = 10000) -> BacktestResult:
        """
        Run vectorized backtest of RSI strategy.
        
        Args:
            data: DataFrame with OHLCV data
            initial_cash: Starting cash amount
            
        Returns:
            BacktestResult object with performance metrics
        """
        try:
            if len(data) < self.rsi_period + 10:
                return self._create_null_result(data.index[0] if len(data) > 0 else "UNKNOWN")
            
            # Calculate RSI
            rsi = TechnicalIndicators.calculate_rsi(data, self.rsi_period)
            
            # Generate signals
            signals = self._generate_signals(data, rsi)
            
            # Calculate returns
            returns = self._calculate_returns(data, signals, initial_cash)
            
            # Calculate buy and hold return
            buy_and_hold_return = (data['c'].iloc[-1] / data['c'].iloc[0]) - 1
            
            # Calculate metrics
            total_return = returns['portfolio_value'].iloc[-1] / initial_cash - 1
            alpha = total_return - buy_and_hold_return
            
            trades = self._analyze_trades(signals, data)
            
            return BacktestResult(
                symbol=data.get('symbol', 'UNKNOWN'),
                rsi_period=self.rsi_period,
                rsi_lower=self.rsi_lower,
                rsi_upper=self.rsi_upper,
                total_return=total_return,
                buy_and_hold_return=buy_and_hold_return,
                alpha=alpha,
                num_trades=trades['num_trades'],
                win_rate=trades['win_rate'],
                avg_trade_duration=trades['avg_duration'],
                max_drawdown=self._calculate_max_drawdown(returns['portfolio_value']),
                sharpe_ratio=self._calculate_sharpe_ratio(returns['daily_returns']),
                profitable=total_return > 0
            )
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return self._create_null_result("ERROR")
    
    def _generate_signals(self, data: pd.DataFrame, rsi: pd.Series) -> pd.DataFrame:
        """Generate buy/sell signals based on RSI."""
        signals = pd.DataFrame(index=data.index)
        signals['rsi'] = rsi
        signals['position'] = 0
        signals['buy_signal'] = False
        signals['sell_signal'] = False
        
        # Buy when RSI crosses below lower threshold
        signals['buy_signal'] = (rsi < self.rsi_lower) & (rsi.shift(1) >= self.rsi_lower)
        
        # Sell when RSI crosses above upper threshold or max hold period reached
        signals['sell_signal'] = (rsi > self.rsi_upper) & (rsi.shift(1) <= self.rsi_upper)
        
        # Track position state
        position = 0
        entry_date = None
        
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i] and position == 0:
                position = 1
                entry_date = signals.index[i]
            elif signals['sell_signal'].iloc[i] and position == 1:
                position = 0
                entry_date = None
            elif position == 1 and entry_date is not None:
                # Check max hold period
                days_held = (signals.index[i] - entry_date).days
                if days_held >= self.max_hold_days:
                    signals.at[signals.index[i], 'sell_signal'] = True
                    position = 0
                    entry_date = None
            
            signals.at[signals.index[i], 'position'] = position
        
        return signals
    
    def _calculate_returns(self, data: pd.DataFrame, signals: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
        """Calculate portfolio returns based on signals."""
        returns = pd.DataFrame(index=data.index)
        returns['price'] = data['c']
        returns['position'] = signals['position']
        
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
        
        for i in range(len(returns)):
            if i == 0:
                continue
                
            prev_position = returns['position'].iloc[i-1]
            curr_position = returns['position'].iloc[i]
            price = returns['price'].iloc[i]
            
            # Buy signal
            if curr_position == 1 and prev_position == 0:
                shares = cash / price
                cash = 0.0
            
            # Sell signal
            elif curr_position == 0 and prev_position == 1:
                cash = shares * price
                shares = 0.0
            
            returns.at[returns.index[i], 'cash'] = cash
            returns.at[returns.index[i], 'shares'] = shares
            returns.at[returns.index[i], 'portfolio_value'] = cash + (shares * price)
        
        # Calculate daily returns
        returns['daily_returns'] = returns['portfolio_value'].pct_change().fillna(0)
        
        return returns
    
    def _analyze_trades(self, signals: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """Analyze individual trades."""
        trades = []
        entry_price = None
        entry_date = None
        
        for i in range(len(signals)):
            if signals['buy_signal'].iloc[i]:
                entry_price = data['c'].iloc[i]
                entry_date = signals.index[i]
            elif signals['sell_signal'].iloc[i] and entry_price is not None:
                exit_price = data['c'].iloc[i]
                exit_date = signals.index[i]
                
                trade_return = (exit_price / entry_price) - 1
                duration = (exit_date - entry_date).days
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'duration': duration
                })
                
                entry_price = None
                entry_date = None
        
        if not trades:
            return {'num_trades': 0, 'win_rate': 0, 'avg_duration': 0}
        
        num_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['return'] > 0)
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        avg_duration = np.mean([trade['duration'] for trade in trades]) if trades else 0
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_duration': avg_duration
        }
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0
        
        excess_returns = daily_returns - (risk_free_rate / 252)  # Daily risk-free rate
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
            profitable=False
        )


class StrategyOptimizer:
    """Optimize RSI strategy parameters for multiple symbols."""
    
    def __init__(self):
        self.rsi_periods = list(range(*config.RSI_PERIOD_RANGE))
        self.rsi_lowers = list(range(*config.RSI_LOWER_RANGE))
        self.rsi_uppers = list(range(*config.RSI_UPPER_RANGE))
    
    def optimize_symbol(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[BacktestResult]:
        """
        Optimize RSI parameters for a single symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Best BacktestResult or None if optimization fails
        """
        try:
            # Get historical data
            data = data_provider.get_single_stock_bars(symbol, start_date, end_date)
            
            if data.empty or len(data) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            best_result = None
            best_score = -float('inf')
            
            # Grid search optimization
            for rsi_period in self.rsi_periods:
                for rsi_lower in self.rsi_lowers:
                    for rsi_upper in self.rsi_uppers:
                        if rsi_lower >= rsi_upper:
                            continue
                        
                        strategy = RSIStrategy(rsi_period, rsi_lower, rsi_upper)
                        result = strategy.backtest(data, config.BACKTEST_INIT_CASH)
                        result.symbol = symbol
                        
                        # Score based on alpha (risk-adjusted return vs buy-and-hold)
                        score = result.alpha
                        
                        if score > best_score and result.profitable:
                            best_score = score
                            best_result = result
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error optimizing {symbol}: {e}")
            return None
    
    async def optimize_universe(self, symbols: List[str], start_date: datetime, end_date: datetime) -> List[BacktestResult]:
        """
        Optimize RSI parameters for multiple symbols concurrently.
        
        Args:
            symbols: List of stock symbols
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            List of BacktestResult objects
        """
        results = []
        
        # Use ThreadPoolExecutor for I/O bound operations
        loop = asyncio.get_event_loop()
        
        # Process symbols in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            tasks = []
            for symbol in batch:
                task = loop.run_in_executor(
                    None, 
                    self.optimize_symbol, 
                    symbol, 
                    start_date, 
                    end_date
                )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, BacktestResult):
                    results.append(result)
                elif result is not None:
                    logger.error(f"Error in batch processing: {result}")
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        return [r for r in results if r is not None]
    
    def filter_results(self, results: List[BacktestResult]) -> List[BacktestResult]:
        """
        Filter backtest results for trading opportunities.
        
        Args:
            results: List of BacktestResult objects
            
        Returns:
            Filtered list of profitable strategies with positive alpha
        """
        filtered = []
        
        for result in results:
            # Filter criteria from legacy get_entries function
            if (result.alpha > 0 and 
                result.profitable and 
                result.num_trades > 0 and
                result.win_rate > 0.3):  # At least 30% win rate
                filtered.append(result)
        
        # Sort by alpha (descending)
        filtered.sort(key=lambda x: x.alpha, reverse=True)
        
        return filtered


def run_backtest_for_symbol(args):
    """Helper function for parallel processing."""
    symbol, start_date, end_date = args
    optimizer = StrategyOptimizer()
    return optimizer.optimize_symbol(symbol, start_date, end_date)
