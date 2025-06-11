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
import pytz
from data_provider import data_provider, TechnicalIndicators
from config import config
from utils import ProgressIndicator

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
    trade_details: List[Dict] = None  # Add trade details to the result


class RSIStrategy:
    """Vectorized RSI trading strategy."""
    
    def __init__(self, rsi_period: int, rsi_lower: int, rsi_upper: int, max_hold_days: int = None):
        self.rsi_period = rsi_period
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.max_hold_days = max_hold_days or config.MAX_HOLD_DAYS
    
    def backtest(self, data: pd.DataFrame, initial_cash: float = 10000, symbol: str = None) -> BacktestResult:
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
            
            trades_summary, trade_details = self._analyze_trades(signals, data)
            
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
                max_drawdown=self._calculate_max_drawdown(returns['portfolio_value']),
                sharpe_ratio=self._calculate_sharpe_ratio(returns['daily_returns']),
                profitable=total_return > 0,
                trade_details=trade_details
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
        logger.debug(f"Calculating returns with initial cash: {initial_cash}, RSI({self.rsi_period}, {self.rsi_lower}, {self.rsi_upper})")
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
        
        trade_count = 0
        
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
                trade_count += 1
                logger.debug(f"Buy signal at {returns.index[i]}, price: {price:.2f}, shares: {shares:.2f}")
            
            # Sell signal
            elif curr_position == 0 and prev_position == 1:
                old_value = shares * price
                cash = shares * price
                shares = 0.0
                logger.debug(f"Sell signal at {returns.index[i]}, price: {price:.2f}, value: {old_value:.2f}")
            
            returns.at[returns.index[i], 'cash'] = cash
            returns.at[returns.index[i], 'shares'] = shares
            returns.at[returns.index[i], 'portfolio_value'] = cash + (shares * price)
        
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
            return {'num_trades': 0, 'win_rate': 0, 'avg_duration': 0}, []
        
        num_trades = len(trades)
        winning_trades = sum(1 for trade in trades if trade['return'] > 0)
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        avg_duration = np.mean([trade['duration'] for trade in trades]) if trades else 0
        
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
    
    # Individual trade logging removed - using consolidated approach instead
    
    def save_all_trades_to_cloud(self, results: List[BacktestResult]) -> None:
        """
        Save all trades from multiple backtest results to a single CSV file in cloud storage.
        
        Args:
            results: List of BacktestResult objects containing trade details
        """
        try:
            from cloud_storage import cloud_storage
            
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
                            'duration': trade['duration']
                        }
                        all_trades.append(trade_record)
            
            if not all_trades:
                logger.info("No trades to save")
                return
            
            logger.info(f"Saving {len(all_trades)} trades from {len(results)} strategies to consolidated CSV")
            
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
            
            trades_df['entry_date_est'] = trades_df['entry_date'].apply(convert_to_est)
            trades_df['exit_date_est'] = trades_df['exit_date'].apply(convert_to_est)
            
            # Round numeric values for readability
            trades_df['entry_price'] = trades_df['entry_price'].round(4)
            trades_df['exit_price'] = trades_df['exit_price'].round(4)
            trades_df['return'] = trades_df['return'].round(6)
            
            # Reorder columns for better readability
            final_columns = [
                'symbol', 'rsi_period', 'rsi_lower', 'rsi_upper',
                'entry_date_est', 'entry_price', 'exit_date_est', 'exit_price',
                'return', 'duration'
            ]
            trades_df = trades_df[final_columns]
            
            # Sort by entry date for chronological order
            trades_df = trades_df.sort_values('entry_date_est')
            
            # Save consolidated trades to cloud storage
            cloud_storage.save_consolidated_trades(trades_df)
            
        except Exception as e:
            logger.error(f"Error saving consolidated trade log: {e}")

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
                logger.debug(f"⚠️  {symbol}: Insufficient data ({len(data)} rows)")
                return None
            
            # Calculate total parameter combinations
            total_combinations = 0
            for rsi_period in self.rsi_periods:
                for rsi_lower in self.rsi_lowers:
                    for rsi_upper in self.rsi_uppers:
                        if rsi_lower < rsi_upper:  # Valid combination
                            total_combinations += 1
            
            logger.debug(f"🔍 {symbol}: Testing {total_combinations} parameter combinations...")
            
            best_result = None
            best_score = -float('inf')
            tested_combinations = 0
            
            # Grid search optimization
            for rsi_period in self.rsi_periods:
                for rsi_lower in self.rsi_lowers:
                    for rsi_upper in self.rsi_uppers:
                        if rsi_lower >= rsi_upper:
                            continue
                        
                        tested_combinations += 1
                        strategy = RSIStrategy(rsi_period, rsi_lower, rsi_upper)
                        result = strategy.backtest(data, config.BACKTEST_INIT_CASH, symbol)
                        
                        # Score based on alpha (risk-adjusted return vs buy-and-hold)
                        score = result.alpha
                        
                        if score > best_score and result.profitable:
                            best_score = score
                            best_result = result
                            logger.debug(f"🎯 {symbol}: New best strategy found - "
                                       f"RSI({rsi_period}, {rsi_lower}, {rsi_upper}) "
                                       f"Alpha: {score:.2%}")
            
            if best_result:
                logger.debug(f"✅ {symbol}: Optimization complete - "
                           f"Best: RSI({best_result.rsi_period}, {best_result.rsi_lower}, {best_result.rsi_upper}) "
                           f"Alpha: {best_result.alpha:.2%}, Trades: {best_result.num_trades}")
            else:
                logger.debug(f"❌ {symbol}: No profitable strategies found from {tested_combinations} combinations")
            
            return best_result
            
        except Exception as e:
            logger.error(f"💥 Error optimizing {symbol}: {e}")
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
        processed_count = 0
        successful_count = 0
        total_symbols = len(symbols)
        
        # Track timing for progress estimates
        import time
        start_time = time.time()
        
        logger.info(f"🚀 Starting backtest optimization for {total_symbols} symbols")
        logger.info(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"⚙️  RSI Parameters - Periods: {self.rsi_periods}, Lower: {self.rsi_lowers}, Upper: {self.rsi_uppers}")
        logger.info("=" * 60)
        
        # Initialize progress indicator
        progress = ProgressIndicator(total_symbols, "🔍 Optimizing strategies")
        
        # Use ThreadPoolExecutor for I/O bound operations
        loop = asyncio.get_event_loop()
        
        # Process symbols in batches to avoid overwhelming the API
        batch_size = 10
        total_batches = (total_symbols + batch_size - 1) // batch_size
        
        for batch_num, i in enumerate(range(0, len(symbols), batch_size), 1):
            batch = symbols[i:i + batch_size]
            batch_start_time = time.time()
            
            logger.info(f"📊 Processing batch {batch_num}/{total_batches} ({len(batch)} symbols): {', '.join(batch)}")
            
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
            
            batch_successful = 0
            for result in batch_results:
                processed_count += 1
                progress.update(1, f"Batch {batch_num}/{total_batches}")
                
                if isinstance(result, BacktestResult):
                    results.append(result)
                    successful_count += 1
                    batch_successful += 1
                elif result is not None:
                    logger.error(f"Error in batch processing: {result}")
            
            # Calculate progress and time estimates
            batch_time = time.time() - batch_start_time
            total_elapsed = time.time() - start_time
            completion_pct = (processed_count / total_symbols) * 100
            
            if processed_count > 0:
                avg_time_per_symbol = total_elapsed / processed_count
                remaining_symbols = total_symbols - processed_count
                estimated_remaining_time = avg_time_per_symbol * remaining_symbols
                
                # Clear progress line and show batch summary
                print()  # New line after progress bar
                logger.info(f"✅ Batch {batch_num} complete: {batch_successful}/{len(batch)} successful (took {batch_time:.1f}s)")
                logger.info(f"📈 Progress: {processed_count}/{total_symbols} ({completion_pct:.1f}%) | "
                          f"Successful: {successful_count} | "
                          f"ETA: {estimated_remaining_time/60:.1f} min")
                logger.info("─" * 60)
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        # Finish progress indicator
        progress.finish("All symbols processed!")
        
        # Final summary
        total_time = time.time() - start_time
        success_rate = (successful_count / total_symbols) * 100 if total_symbols > 0 else 0
        
        logger.info("=" * 60)
        logger.info("🎯 BACKTEST OPTIMIZATION COMPLETE!")
        logger.info(f"📊 Results Summary:")
        logger.info(f"   • Total symbols processed: {processed_count}/{total_symbols}")
        logger.info(f"   • Successful optimizations: {successful_count} ({success_rate:.1f}%)")
        logger.info(f"   • Total time: {total_time/60:.1f} minutes")
        logger.info(f"   • Average time per symbol: {total_time/total_symbols:.1f}s")
        if successful_count > 0:
            profitable_count = len([r for r in results if r.profitable])
            logger.info(f"   • Profitable strategies: {profitable_count}/{successful_count}")
        logger.info("=" * 60)
        
        # Save all trades to consolidated CSV
        if results:
            logger.info("💾 Saving trade details to cloud storage...")
            self.save_all_trades(results)
        
        return [r for r in results if r is not None]
    
    def save_all_trades(self, results: List[BacktestResult]) -> None:
        """
        Save all trades from optimization results to cloud storage.
        
        Args:
            results: List of BacktestResult objects
        """
        strategy = RSIStrategy(14, 30, 70)  # Dummy strategy instance for the method
        strategy.save_all_trades_to_cloud(results)
    
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
