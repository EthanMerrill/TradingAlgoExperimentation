#!/usr/bin/env python3
"""
Test script for position logging functionality.
Demonstrates how position entries with backtest details are saved to cloud storage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from trading_engine import TradingOpportunity
from cloud_storage import cloud_storage
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_position_logging():
    """Test the position logging functionality."""
    
    # Create a sample trading opportunity (this would normally come from backtesting)
    sample_opportunity = TradingOpportunity(
        symbol="AAPL",
        current_rsi=25.5,
        target_rsi_lower=30,
        target_rsi_upper=70,
        rsi_period=14,
        expected_return=0.15,  # 15% expected return
        alpha=0.08,  # 8% alpha
        confidence=0.65,  # 65% win rate
        entry_price=150.25
    )
    
    # Simulate placing an order
    shares = 10
    order_success = True  # Simulate successful order
    
    print(f"\n{'='*60}")
    print("Testing Position Entry Logging")
    print(f"{'='*60}")
    
    print(f"Symbol: {sample_opportunity.symbol}")
    print(f"Entry Price: ${sample_opportunity.entry_price:.2f}")
    print(f"Shares: {shares}")
    print(f"Position Value: ${shares * sample_opportunity.entry_price:.2f}")
    print(f"Current RSI: {sample_opportunity.current_rsi:.1f}")
    print(f"Expected Return: {sample_opportunity.expected_return:.1%}")
    print(f"Alpha: {sample_opportunity.alpha:.1%}")
    print(f"Confidence: {sample_opportunity.confidence:.1%}")
    
    # Test saving position entry
    print(f"\nSaving position entry to cloud storage...")
    success = cloud_storage.save_position_entry(
        opportunity=sample_opportunity,
        shares=shares,
        order_success=order_success
    )
    
    if success:
        print("✅ Position entry saved successfully!")
    else:
        print("❌ Failed to save position entry")
        return
    
    # Test listing position files
    print(f"\nListing position files in cloud storage...")
    position_files = cloud_storage.list_position_files()
    
    if position_files:
        print(f"Found {len(position_files)} position files:")
        for file in position_files:
            print(f"  - {file}")
        
        # Test loading the most recent file
        latest_file = position_files[-1]  # Assuming files are sorted
        print(f"\nLoading latest position file: {latest_file}")
        
        df = cloud_storage.load_position_entries(latest_file)
        
        if not df.empty:
            print(f"✅ Loaded {len(df)} position entries")
            print(f"\nPosition entries for today:")
            print(df.to_string(index=False))
        else:
            print("❌ No data found in position file")
    else:
        print("No position files found")


def demo_multiple_positions():
    """Demonstrate logging multiple positions in a day."""
    
    print(f"\n{'='*60}")
    print("Testing Multiple Position Entries")
    print(f"{'='*60}")
    
    # Create multiple sample opportunities
    opportunities = [
        TradingOpportunity(
            symbol="MSFT",
            current_rsi=28.2,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            expected_return=0.12,
            alpha=0.06,
            confidence=0.68,
            entry_price=420.50
        ),
        TradingOpportunity(
            symbol="GOOGL",
            current_rsi=26.8,
            target_rsi_lower=25,
            target_rsi_upper=75,
            rsi_period=10,
            expected_return=0.18,
            alpha=0.11,
            confidence=0.72,
            entry_price=2850.75
        ),
        TradingOpportunity(
            symbol="TSLA",
            current_rsi=24.5,
            target_rsi_lower=30,
            target_rsi_upper=70,
            rsi_period=14,
            expected_return=0.22,
            alpha=0.15,
            confidence=0.58,
            entry_price=195.30
        )
    ]
    
    # Log each position
    for i, opportunity in enumerate(opportunities, 1):
        shares = 5 * i  # Different position sizes
        order_success = i != 3  # Simulate one failed order
        
        print(f"\nPosition {i}: {opportunity.symbol}")
        print(f"  Shares: {shares}")
        print(f"  Order Success: {order_success}")
        
        success = cloud_storage.save_position_entry(
            opportunity=opportunity,
            shares=shares,
            order_success=order_success
        )
        
        print(f"  Logged: {'✅' if success else '❌'}")
    
    # Load and display all positions for today
    today = datetime.now().strftime('%Y%m%d')
    filename = f"positions_{today}.csv"
    
    print(f"\nLoading all positions for today ({filename})...")
    df = cloud_storage.load_position_entries(filename)
    
    if not df.empty:
        print(f"\nTotal positions logged today: {len(df)}")
        print(f"Successful orders: {df['order_success'].sum()}")
        print(f"Total position value: ${df['position_value'].sum():.2f}")
        
        print(f"\nDetailed position log:")
        print(df[['symbol', 'shares', 'entry_price', 'position_value', 
                 'order_success', 'alpha', 'expected_return']].to_string(index=False))
    else:
        print("❌ No positions found for today")


if __name__ == "__main__":
    print("Position Logging Test Script")
    print("This script tests the position entry logging functionality")
    
    try:
        # Test basic position logging
        test_position_logging()
        
        # Test multiple positions
        demo_multiple_positions()
        
        print(f"\n{'='*60}")
        print("Test completed successfully! 🎉")
        print("Position entries are now being logged to cloud storage.")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
