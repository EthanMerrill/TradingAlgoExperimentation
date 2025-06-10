#!/usr/bin/env python3
"""
Debug script to examine the actual structure of Alpaca snapshot responses.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_provider import data_provider
from alpaca.data.requests import StockSnapshotRequest
import pprint

def debug_snapshot_response():
    """Debug the actual snapshot response structure."""
    print("Debugging Alpaca snapshot response structure...")
    
    if not data_provider.historical_client:
        print("❌ Historical client not initialized")
        return
    
    symbol = 'AAPL'
    print(f"Testing snapshot request for {symbol}...")
    
    try:
        request = StockSnapshotRequest(symbol_or_symbols=[symbol])
        response = data_provider.historical_client.get_stock_snapshot(request)
        
        print(f"Response type: {type(response)}")
        print(f"Response dir: {dir(response)}")
        
        if hasattr(response, 'data'):
            print(f"Response.data type: {type(response.data)}")
            print(f"Response.data: {response.data}")
        
        if isinstance(response, dict):
            print("Response is a dictionary:")
            # Only show keys to avoid overwhelming output
            print(f"Keys: {list(response.keys())}")
            
            if symbol in response:
                snapshot_data = response[symbol]
                print(f"\nData for {symbol}:")
                print(f"Type: {type(snapshot_data)}")
                print(f"Dir: {dir(snapshot_data)}")
                
                # Try to access data different ways
                if hasattr(snapshot_data, 'latest_trade'):
                    print(f"Has latest_trade attribute: {snapshot_data.latest_trade}")
                if hasattr(snapshot_data, 'get'):
                    print("Has 'get' method - it's dict-like")
                else:
                    print("No 'get' method - it's an object")
                    
                # Try accessing trade price
                try:
                    if hasattr(snapshot_data, 'latest_trade') and snapshot_data.latest_trade:
                        print(f"Trade price (attribute): {snapshot_data.latest_trade.price}")
                except Exception as e:
                    print(f"Error accessing trade price as attribute: {e}")
                    
                try:
                    if isinstance(snapshot_data, dict) and 'latest_trade' in snapshot_data:
                        print(f"Trade price (dict): {snapshot_data['latest_trade']['price']}")
                except Exception as e:
                    print(f"Error accessing trade price as dict: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_snapshot_response()
