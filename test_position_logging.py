#!/usr/bin/env python3
"""
Test script for the new position logging functionality.
Tests the save_position_entry method without requiring live trading.
"""

import sys
import os
from datetime import datetime
from dataclasses import dataclass

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from cloud_storage import cloud_storage
from config import config

# Mock TradingOpportunity for testing
@dataclass
class MockTradingOpportunity:
    """Mock trading opportunity for testing."""
    symbol: str
    current_rsi: float
    target_rsi_lower: int
    target_rsi_upper: int
    rsi_period: int
    expected_return: float
    alpha: float
    confidence: float
    entry_price: float


def test_position_logging():
    """Test the position logging functionality."""
    print("Testing position logging functionality...")
    print(f"Environment: {config.ENVIRONMENT}")
    print(f"GCS Bucket: {config.GCS_BUCKET_NAME}")
    
    # Create a mock trading opportunity
    mock_opportunity = MockTradingOpportunity(
        symbol="AAPL",
        current_rsi=25.5,
        target_rsi_lower=30,
        target_rsi_upper=70,
        rsi_period=14,
        expected_return=0.15,
        alpha=0.08,
        confidence=0.65,
        entry_price=150.25
    )
    
    # Test parameters
    shares = 100
    order_success = True
    test_date = datetime.now().strftime('%Y%m%d')
    
    print(f"\nAttempting to log position entry:")
    print(f"Symbol: {mock_opportunity.symbol}")
    print(f"Shares: {shares}")
    print(f"Entry Price: ${mock_opportunity.entry_price}")
    print(f"Position Value: ${shares * mock_opportunity.entry_price:,.2f}")
    print(f"Expected Return: {mock_opportunity.expected_return:.2%}")
    print(f"Alpha: {mock_opportunity.alpha:.2%}")
    print(f"Test Date: {test_date}")
    
    # Test the save_position_entry method
    try:
        success = cloud_storage.save_position_entry(
            opportunity=mock_opportunity,
            shares=shares,
            order_success=order_success,
            date=test_date
        )
        
        if success:
            print("✅ Position entry logged successfully!")
            
            # Test listing position files
            print("\nListing position files in cloud storage:")
            position_files = cloud_storage.list_position_files()
            for file in position_files:
                print(f"  - {file}")
            
            # Test loading the position entries we just saved
            filename = f"positions_{test_date}.csv"
            if filename in position_files:
                print(f"\nLoading position entries from {filename}:")
                df = cloud_storage.load_position_entries(filename)
                if not df.empty:
                    print(f"Successfully loaded {len(df)} position entries")
                    print("\nPosition entry details:")
                    print(df.to_string(index=False))
                else:
                    print("No position entries found in the file")
            else:
                print(f"File {filename} not found in position files list")
        else:
            print("❌ Failed to log position entry")
            
    except Exception as e:
        print(f"❌ Error during position logging test: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_positions():
    """Test logging multiple positions to the same daily file."""
    print("\n" + "="*60)
    print("Testing multiple position entries for the same day...")
    
    # Create multiple mock opportunities
    opportunities = [
        MockTradingOpportunity("MSFT", 28.0, 30, 70, 14, 0.12, 0.05, 0.68, 420.50),
        MockTradingOpportunity("GOOGL", 32.5, 35, 75, 14, 0.18, 0.09, 0.62, 2800.75),
        MockTradingOpportunity("TSLA", 22.0, 25, 65, 14, 0.25, 0.12, 0.58, 180.30)
    ]
    
    test_date = datetime.now().strftime('%Y%m%d')
    
    for i, opportunity in enumerate(opportunities, 1):
        shares = 50 * i  # Different share amounts
        order_success = i % 2 == 1  # Alternate success/failure
        
        print(f"\nLogging position {i}: {opportunity.symbol}")
        success = cloud_storage.save_position_entry(
            opportunity=opportunity,
            shares=shares,
            order_success=order_success,
            date=test_date
        )
        
        if success:
            print(f"✅ {opportunity.symbol} logged successfully")
        else:
            print(f"❌ Failed to log {opportunity.symbol}")
    
    # Load and display all entries for today
    filename = f"positions_{test_date}.csv"
    print(f"\nLoading all position entries from {filename}:")
    df = cloud_storage.load_position_entries(filename)
    if not df.empty:
        print(f"Total entries: {len(df)}")
        print("\nAll position entries for today:")
        print(df[['timestamp', 'symbol', 'shares', 'entry_price', 'order_success', 'alpha']].to_string(index=False))
        
        # Calculate summary statistics
        total_value = df['position_value'].sum()
        successful_orders = df['order_success'].sum()
        avg_alpha = df['alpha'].mean()
        
        print(f"\nSummary Statistics:")
        print(f"Total Position Value: ${total_value:,.2f}")
        print(f"Successful Orders: {successful_orders}/{len(df)}")
        print(f"Average Alpha: {avg_alpha:.2%}")
    else:
        print("No position entries found")


if __name__ == "__main__":
    print("Position Logging Test Script")
    print("="*60)
    
    # Test single position logging
    test_position_logging()
    
    # Test multiple position logging
    test_multiple_positions()
    
    print("\n" + "="*60)
    print("Position logging tests completed!")
