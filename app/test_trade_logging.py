#!/usr/bin/env python3
"""
Test script to verify trade logging functionality.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the app directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy import RSIStrategy
from data_provider import data_provider
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_trade_logging():
    """Test the trade logging functionality."""
    try:
        logger.info("Testing trade logging functionality...")
        
        # Test with a sample symbol
        symbol = "AAPL"
        
        # Use recent dates for testing
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # 3 months of data
        
        logger.info(f"Getting data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get historical data
        data = data_provider.get_single_stock_bars(symbol, start_date, end_date)
        
        if data.empty:
            logger.error(f"No data received for {symbol}")
            return False
        
        logger.info(f"Received {len(data)} rows of data for {symbol}")
        
        # Add symbol to the data DataFrame so it can be used in the backtest
        data['symbol'] = symbol
        
        # Create a strategy instance
        strategy = RSIStrategy(rsi_period=14, rsi_lower=30, rsi_upper=70)
        
        # Run backtest
        logger.info("Running backtest...")
        try:
            result = strategy.backtest(data, symbol=symbol)
            logger.info("Backtest completed successfully")
        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            return False
        
        logger.info(f"Backtest completed for {symbol}:")
        logger.info(f"  Total Return: {result.total_return:.2%}")
        logger.info(f"  Number of Trades: {result.num_trades}")
        logger.info(f"  Win Rate: {result.win_rate:.2%}")
        logger.info(f"  Alpha: {result.alpha:.2%}")
        
        if result.num_trades > 0:
            logger.info("✅ Trade logging test completed successfully!")
            logger.info("Check your Google Cloud Storage bucket in the 'trades' folder for the trade log CSV file.")
        else:
            logger.warning("⚠️ No trades were generated in this backtest period. Try a different symbol or date range.")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in trade logging test: {e}")
        return False

if __name__ == "__main__":
    success = test_trade_logging()
    if success:
        print("\n🎉 Trade logging test completed!")
    else:
        print("\n❌ Trade logging test failed!")
        sys.exit(1)
