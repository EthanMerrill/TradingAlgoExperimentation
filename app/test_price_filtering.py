#!/usr/bin/env python3
"""
Test script for the price filtering functionality in data_provider.py
Tests the _filter_symbols_by_price method to ensure it's working correctly.
"""
import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Starting price filtering test script...")

try:
    from data_provider import data_provider
    from config import config
    print("✅ Successfully imported data_provider and config")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_price_filtering():
    """Test the price filtering functionality."""
    print("=" * 60)
    print("Testing Price Filtering Functionality")
    print("=" * 60)
    
    # Display current configuration
    print(f"Current Configuration:")
    print(f"  Environment: {config.ENVIRONMENT}")
    print(f"  Paper Trading: {config.PAPER_TRADE}")
    print(f"  Min Price: ${config.MIN_PRICE}")
    print(f"  Max Price: ${config.MAX_PRICE}")
    print(f"  Min Volume: {config.MIN_VOLUME:,}")
    print(f"  Rate Limit Delay: {config.API_RATE_LIMIT_DELAY}s")
    print()
    
    # Check if data provider is properly initialized
    print(f"Historical client status: {data_provider.historical_client is not None}")
    if not data_provider.historical_client:
        print("❌ Data provider not initialized - missing API credentials")
        print("Please set ALPACA_PAPER_KEY and ALPACA_PAPER_SECRET environment variables")
        return
    
    print("✅ Data provider initialized successfully")
    print()
    
    # Test with a small set of known symbols
    test_symbols = [
        'AAPL',   # Should pass (typically ~$180)
        'GOOGL',  # Should pass (typically ~$140)
        'TSLA',   # Should pass (typically ~$200)
        'MSFT',   # Should pass (typically ~$420) - might fail if MAX_PRICE is low
        'NVDA',   # Should pass (typically ~$900) - might fail if MAX_PRICE is low
        'SPY',    # Should pass (typically ~$540) - might fail if MAX_PRICE is low
        'QQQ',    # Should pass (typically ~$380) - might fail if MAX_PRICE is low
        'AMZN',   # Should pass (typically ~$170)
    ]
    
    print(f"Testing price filtering with {len(test_symbols)} symbols:")
    for symbol in test_symbols:
        print(f"  - {symbol}")
    print()
    
    try:
        # Test the price filtering function directly
        print("Running price filtering...")
        start_time = datetime.now()
        
        filtered_symbols = await data_provider._filter_symbols_by_price(test_symbols)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Price filtering completed in {duration:.2f} seconds")
        print()
        
        # Display results
        print("Results:")
        print(f"  Original symbols: {len(test_symbols)}")
        print(f"  Filtered symbols: {len(filtered_symbols)}")
        print(f"  Filter rate: {len(filtered_symbols)/len(test_symbols)*100:.1f}%")
        print()
        
        print("Symbols that passed filtering:")
        for symbol in filtered_symbols:
            print(f"  ✅ {symbol}")
        
        print()
        print("Symbols that were filtered out:")
        filtered_out = set(test_symbols) - set(filtered_symbols)
        for symbol in filtered_out:
            print(f"  ❌ {symbol}")
        
        if not filtered_out:
            print("  (None - all symbols passed)")
        
        print()
        
        # Test individual snapshots to show actual prices and volumes
        print("Individual symbol price and volume checks:")
        for symbol in test_symbols:
            print(f"  Checking {symbol}...", end=" ")
            try:
                snapshot = data_provider.get_current_snapshot(symbol)
                if snapshot:
                    price = snapshot.get('price', 'N/A')
                    volume = snapshot.get('prev_daily_volume', 'N/A')
                    
                    if price != 'N/A' and volume != 'N/A':
                        price_pass = config.MIN_PRICE <= price <= config.MAX_PRICE
                        volume_pass = volume >= config.MIN_VOLUME
                        overall_pass = price_pass and volume_pass
                        
                        status = "✅ PASS" if overall_pass else "❌ FAIL"
                        price_status = "✅" if price_pass else "❌"
                        volume_status = "✅" if volume_pass else "❌"
                        
                        print(f"${price:.2f} {price_status} | {volume:,} vol {volume_status} | {status}")
                    else:
                        # Try alternative price sources
                        bid_price = snapshot.get('bid_price', 'N/A')
                        daily_close = snapshot.get('daily_close', 'N/A')
                        if bid_price != 'N/A':
                            price_pass = config.MIN_PRICE <= bid_price <= config.MAX_PRICE
                            volume_pass = volume >= config.MIN_VOLUME if volume != 'N/A' else False
                            overall_pass = price_pass and volume_pass
                            
                            status = "✅ PASS" if overall_pass else "❌ FAIL"
                            print(f"${bid_price:.2f} (bid) | {volume:,} vol | {status}")
                        elif daily_close != 'N/A':
                            price_pass = config.MIN_PRICE <= daily_close <= config.MAX_PRICE
                            volume_pass = volume >= config.MIN_VOLUME if volume != 'N/A' else False
                            overall_pass = price_pass and volume_pass
                            
                            status = "✅ PASS" if overall_pass else "❌ FAIL"
                            print(f"${daily_close:.2f} (daily) | {volume:,} vol | {status}")
                        else:
                            print("No price data available")
                else:
                    print("Failed to get snapshot")
            except Exception as e:
                print(f"Error: {e}")
        
    except Exception as e:
        print(f"❌ Error during price filtering test: {e}")
        import traceback
        traceback.print_exc()


async def test_full_universe_filtering():
    """Test the complete stock universe filtering."""
    print("\n" + "=" * 60)
    print("Testing Full Stock Universe Filtering")
    print("=" * 60)
    
    try:
        print("Fetching and filtering full stock universe...")
        print("This may take several minutes due to API rate limits...")
        
        start_time = datetime.now()
        universe = await data_provider.get_stock_universe()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Universe filtering completed in {duration:.2f} seconds")
        print()
        
        if not universe.empty:
            print("Universe Statistics:")
            print(f"  Total filtered symbols: {len(universe)}")
            print(f"  Processing time: {duration:.2f} seconds")
            print(f"  Rate: {len(universe)/duration:.1f} symbols/second")
            print()
            
            # Show sample of filtered symbols
            print("Sample of filtered symbols:")
            sample_size = min(10, len(universe))
            for i, row in universe.head(sample_size).iterrows():
                print(f"  {row['symbol']} - {row['name']} ({row['exchange']})")
            
            if len(universe) > sample_size:
                print(f"  ... and {len(universe) - sample_size} more")
            
        else:
            print("❌ No symbols returned from universe filtering")
            
    except Exception as e:
        print(f"❌ Error during universe filtering test: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("Price Filtering Test Script")
    print(f"Current time: {datetime.now()}")
    print()
    
    # Test basic price filtering
    await test_price_filtering()
    
    # Ask user if they want to test full universe (takes longer)
    print("\n" + "=" * 60)
    response = input("Do you want to test full universe filtering? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        await test_full_universe_filtering()
    else:
        print("Skipping full universe test")
    
    print("\n" + "=" * 60)
    print("Price filtering tests completed!")


if __name__ == "__main__":
    # Run the async test
    asyncio.run(main())
