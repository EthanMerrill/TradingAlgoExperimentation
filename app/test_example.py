"""
Example script to test the trading algorithm components.
This can be used for development and testing purposes.
"""
import asyncio
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import config
from app.data_provider import data_provider, TechnicalIndicators
from app.strategy import StrategyOptimizer, RSIStrategy
from app.utils import setup_logging, TradingCalendar


async def test_data_provider():
    """Test the data provider functionality."""
    print("=" * 50)
    print("Testing Data Provider")
    print("=" * 50)
    
    # Test account info
    account_info = data_provider.get_account_info()
    print(f"Account info: {account_info}")
    
    # Test current positions
    positions = data_provider.get_current_positions()
    print(f"Current positions: {len(positions)} positions")
    
    # Test historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\nTesting data for {symbol}:")
        data = data_provider.get_single_stock_bars(symbol, start_date, end_date)
        
        if not data.empty:
            print(f"  Rows: {len(data)}")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            print(f"  Latest close: ${data['c'].iloc[-1]:.2f}")
            
            # Test RSI calculation
            rsi = TechnicalIndicators.calculate_rsi(data, 14)
            if not rsi.empty:
                print(f"  Current RSI: {rsi.iloc[-1]:.2f}")
        else:
            print(f"  No data available")


async def test_strategy():
    """Test the strategy backtesting functionality."""
    print("=" * 50)
    print("Testing Strategy Backtesting")
    print("=" * 50)
    
    # Test with a single symbol
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)  # 6 months
    
    print(f"Testing strategy optimization for {symbol}")
    
    optimizer = StrategyOptimizer()
    result = optimizer.optimize_symbol(symbol, start_date, end_date)
    
    if result:
        print(f"Optimization successful!")
        print(f"  Symbol: {result.symbol}")
        print(f"  RSI Period: {result.rsi_period}")
        print(f"  RSI Lower: {result.rsi_lower}")
        print(f"  RSI Upper: {result.rsi_upper}")
        print(f"  Total Return: {result.total_return:.2%}")
        print(f"  Buy & Hold Return: {result.buy_and_hold_return:.2%}")
        print(f"  Alpha: {result.alpha:.2%}")
        print(f"  Win Rate: {result.win_rate:.2%}")
        print(f"  Number of Trades: {result.num_trades}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    else:
        print("Optimization failed")


def test_config():
    """Test the configuration system."""
    print("=" * 50)
    print("Testing Configuration")
    print("=" * 50)
    
    print(f"Paper Trading: {config.PAPER_TRADE}")
    print(f"Max Positions: {config.MAX_POSITIONS}")
    print(f"Position Size: {config.POSITION_SIZE_PCT:.1%}")
    print(f"Min Cash: {config.MIN_CASH_PCT:.1%}")
    print(f"Backtest Init Cash: ${config.BACKTEST_INIT_CASH:,}")
    print(f"RSI Period Range: {config.RSI_PERIOD_RANGE}")
    print(f"RSI Lower Range: {config.RSI_LOWER_RANGE}")
    print(f"RSI Upper Range: {config.RSI_UPPER_RANGE}")
    
    alpaca_config = config.get_alpaca_config()
    print(f"Alpaca Base URL: {alpaca_config['base_url']}")


def test_utils():
    """Test utility functions."""
    print("=" * 50)
    print("Testing Utilities")
    print("=" * 50)
    
    calendar = TradingCalendar()
    
    print(f"Is today a trading day? {calendar.is_trading_day()}")
    print(f"Is market open now? {calendar.is_market_open()}")
    print(f"Next trading day: {calendar.next_trading_day()}")
    print(f"Previous trading day: {calendar.previous_trading_day()}")


async def run_simple_backtest():
    """Run a simple backtest example."""
    print("=" * 50)
    print("Running Simple Backtest Example")
    print("=" * 50)
    
    # Get some test data
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    data = data_provider.get_single_stock_bars(symbol, start_date, end_date)
    
    if data.empty:
        print("No data available for backtest")
        return
    
    # Create a simple RSI strategy
    strategy = RSIStrategy(rsi_period=14, rsi_lower=30, rsi_upper=70)
    
    # Run backtest
    result = strategy.backtest(data, initial_cash=10000)
    
    print(f"Backtest Results for {symbol}:")
    print(f"  Total Return: {result.total_return:.2%}")
    print(f"  Buy & Hold Return: {result.buy_and_hold_return:.2%}")
    print(f"  Alpha: {result.alpha:.2%}")
    print(f"  Number of Trades: {result.num_trades}")
    print(f"  Win Rate: {result.win_rate:.2%}")
    print(f"  Max Drawdown: {result.max_drawdown:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")


async def main():
    """Run all tests."""
    setup_logging('INFO')
    
    print("Trading Algorithm Test Suite")
    print("=" * 50)
    
    try:
        # Test configuration
        test_config()
        
        # Test utilities
        test_utils()
        
        # Test data provider (requires API keys)
        try:
            await test_data_provider()
        except Exception as e:
            print(f"Data provider test failed (probably missing API keys): {e}")
        
        # Test strategy
        try:
            await test_strategy()
        except Exception as e:
            print(f"Strategy test failed: {e}")
        
        # Run simple backtest
        try:
            await run_simple_backtest()
        except Exception as e:
            print(f"Simple backtest failed: {e}")
        
        print("\n" + "=" * 50)
        print("Test suite completed")
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
