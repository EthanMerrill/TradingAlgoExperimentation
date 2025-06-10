#!/usr/bin/env python3
"""
Test script to demonstrate consolidated trade logging functionality.
Tests multiple symbols and saves all trades to a single CSV file.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the app directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy import RSIStrategy, StrategyOptimizer
from data_provider import data_provider
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_consolidated_trade_logging():
    """Test the consolidated trade logging functionality with multiple symbols."""
    try:
        logger.info("Testing consolidated trade logging functionality...")
        
        # Test with multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Use recent dates for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months of data
        
        logger.info(f"Testing with symbols: {symbols}")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        all_results = []
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            
            # Get historical data
            data = data_provider.get_single_stock_bars(symbol, start_date, end_date)
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                continue
            
            logger.info(f"Received {len(data)} rows of data for {symbol}")
            
            # Create a strategy instance
            strategy = RSIStrategy(rsi_period=14, rsi_lower=30, rsi_upper=70)
            
            # Run backtest
            result = strategy.backtest(data, symbol=symbol)
            
            logger.info(f"Backtest completed for {symbol}:")
            logger.info(f"  Total Return: {result.total_return:.2%}")
            logger.info(f"  Number of Trades: {result.num_trades}")
            logger.info(f"  Win Rate: {result.win_rate:.2%}")
            logger.info(f"  Alpha: {result.alpha:.2%}")
            
            if result.num_trades > 0:
                all_results.append(result)
                logger.info(f"  ✅ {len(result.trade_details)} trades captured for {symbol}")
            else:
                logger.info(f"  ⚠️ No trades generated for {symbol}")
        
        if all_results:
            # Save all trades to consolidated CSV
            logger.info(f"\nSaving all trades from {len(all_results)} symbols to consolidated CSV...")
            strategy = RSIStrategy(14, 30, 70)  # Create instance for the method
            strategy.save_all_trades_to_cloud(all_results)
            
            # Count total trades
            total_trades = sum(len(result.trade_details) for result in all_results if result.trade_details)
            logger.info(f"✅ Consolidated trade logging test completed successfully!")
            logger.info(f"Total trades saved: {total_trades}")
            logger.info("Check your Google Cloud Storage bucket in the 'trades' folder for the consolidated CSV file.")
        else:
            logger.warning("⚠️ No trades were generated for any symbols. Try different symbols or date ranges.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in consolidated trade logging test: {e}")
        return False

def test_with_optimizer():
    """Test consolidated trade logging using the StrategyOptimizer."""
    try:
        logger.info("\n" + "="*60)
        logger.info("Testing with StrategyOptimizer (finds best parameters per symbol)...")
        
        symbols = ["AAPL", "TSLA"]  # Smaller list for faster testing
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months for faster optimization
        
        optimizer = StrategyOptimizer()
        results = []
        
        for symbol in symbols:
            logger.info(f"Optimizing {symbol}...")
            result = optimizer.optimize_symbol(symbol, start_date, end_date)
            if result:
                results.append(result)
                logger.info(f"  Best strategy for {symbol}: RSI({result.rsi_period}, {result.rsi_lower}, {result.rsi_upper})")
                logger.info(f"  Trades: {result.num_trades}, Alpha: {result.alpha:.2%}")
        
        if results:
            # Save all optimized trades
            logger.info(f"\nSaving optimized trades from {len(results)} symbols...")
            optimizer.save_all_trades(results)
            
            total_trades = sum(len(result.trade_details) for result in results if result.trade_details)
            logger.info(f"✅ Optimizer test completed! Total optimized trades saved: {total_trades}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in optimizer test: {e}")
        return False

if __name__ == "__main__":
    # Test basic consolidated logging
    success1 = test_consolidated_trade_logging()
    
    # Test with optimizer
    success2 = test_with_optimizer()
    
    if success1 and success2:
        print("\n🎉 All consolidated trade logging tests completed successfully!")
        print("📁 Check your Google Cloud Storage bucket 'trades' folder for:")
        print("   - all_trades_YYYYMMDD_HHMMSS.csv (consolidated trades)")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
