"""
Data provider module for fetching market data from various sources.
Replaces the legacy networking.py with modern async/await patterns.
"""
import asyncio
import aiohttp
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from config import config

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """Data class for stock bar information."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class DataProvider:
    """Modern data provider using Alpaca's latest API."""
    
    def __init__(self):
        alpaca_config = config.get_alpaca_config()
        
        # Check if we have valid credentials
        if not alpaca_config['api_key'] or not alpaca_config['secret_key']:
            logger.warning("No Alpaca API credentials found. Data provider will have limited functionality.")
            self.historical_client = None
            self.trading_client = None
        else:
            try:
                self.historical_client = StockHistoricalDataClient(
                    api_key=alpaca_config['api_key'],
                    secret_key=alpaca_config['secret_key']
                )
                self.trading_client = TradingClient(
                    api_key=alpaca_config['api_key'],
                    secret_key=alpaca_config['secret_key'],
                    paper=config.PAPER_TRADE
                )
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca clients: {e}")
                self.historical_client = None
                self.trading_client = None
        
        self._rate_limit_delay = config.API_RATE_LIMIT_DELAY
    
    async def get_historical_bars(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        timeframe: TimeFrame = TimeFrame.Day
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bar data for multiple symbols using Alpaca's current API.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe (Day, Hour, Minute)
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            bars = self.historical_client.get_stock_bars(request_params)
            
            # Convert to DataFrame format
            result = {}
            for symbol in symbols:
                if symbol in bars.data:
                    symbol_bars = bars.data[symbol]
                    df_data = []
                    for bar in symbol_bars:
                        df_data.append({
                            'timestamp': bar.timestamp,
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        df['symbol'] = symbol
                        df.set_index('timestamp', inplace=True)
                        result[symbol] = df
                    else:
                        logger.warning(f"No data found for symbol {symbol}")
                        result[symbol] = pd.DataFrame()
                else:
                    logger.warning(f"Symbol {symbol} not found in response")
                    result[symbol] = pd.DataFrame()
            
            # Rate limiting
            await asyncio.sleep(self._rate_limit_delay)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return {symbol: pd.DataFrame() for symbol in symbols}
    
    def get_single_stock_bars(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get historical data for a single stock (synchronous version).
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.historical_client:
            logger.error("Historical client not initialized - missing API credentials")
            return pd.DataFrame()
        
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            bars = self.historical_client.get_stock_bars(request_params)
            
            if symbol in bars.data and bars.data[symbol]:
                df_data = []
                for bar in bars.data[symbol]:
                    df_data.append({
                        't': bar.timestamp,  # Keep legacy column name for compatibility
                        'o': bar.open,
                        'h': bar.high,
                        'l': bar.low,
                        'c': bar.close,
                        'v': bar.volume
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('t', inplace=True)
                return df
            else:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
        
        finally:
            time.sleep(self._rate_limit_delay)
    
    def get_current_positions(self) -> pd.DataFrame:
        """Get current portfolio positions."""
        try:
            positions = self.trading_client.get_all_positions()
            
            if not positions:
                return pd.DataFrame()
            
            position_data = []
            for position in positions:
                position_data.append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'market_value': float(position.market_value),
                    'avg_entry_price': float(position.avg_entry_price),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'current_price': float(position.current_price)
                })
            
            return pd.DataFrame(position_data)
            
        except Exception as e:
            logger.error(f"Error fetching current positions: {e}")
            return pd.DataFrame()
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including cash and equity."""
        try:
            account = self.trading_client.get_account()
            
            return {
                'cash': float(account.cash),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value)
            }
            
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {}
    
    async def get_stock_universe(self, date: datetime = None) -> pd.DataFrame:
        """
        Get filtered universe of stocks for trading.
        Replaces the polygon-based approach with Alpaca screener.
        
        Args:
            date: Date for universe (defaults to today)
            
        Returns:
            DataFrame with filtered stock universe
        """
        if date is None:
            date = datetime.now()
        
        try:
            # Get all active assets
            assets = self.trading_client.get_all_assets()
            
            # Filter for tradable stocks
            # Check what attributes are available and filter accordingly
            tradable_stocks = []
            for asset in assets:
                if (asset.tradable and 
                    asset.status == 'active' and 
                    hasattr(asset, 'exchange') and
                    asset.exchange in ['NASDAQ', 'NYSE', 'ARCA', 'BATS']):
                    tradable_stocks.append(asset)
            
            symbols = [asset.symbol for asset in tradable_stocks]
            
            # Get market data for filtering
            # Note: This is a simplified approach. In production, you might want to use
            # a market data service that provides volume and market cap data
            logger.info(f"Found {len(symbols)} tradable stocks")
            
            # For now, return a subset for testing
            # In production, you'd want to implement proper filtering based on:
            # - Market cap
            # - Volume
            # - Price range
            
            universe_data = []
            for asset in tradable_stocks[:100]:  # Limit for testing
                universe_data.append({
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'exchange': asset.exchange,
                    'tradable': asset.tradable
                })
            
            df = pd.DataFrame(universe_data)
            logger.info(f"Returning universe of {len(df)} stocks")
            return df
            
        except Exception as e:
            logger.error(f"Error getting stock universe: {e}")
            return pd.DataFrame()


class TechnicalIndicators:
    """Technical analysis indicators."""
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: str = 'c') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with price data
            period: RSI period
            price_col: Column name for price data
            
        Returns:
            Series with RSI values
        """
        try:
            if len(data) < period + 1:
                return pd.Series(index=data.index, dtype=float)
            
            delta = data[price_col].diff()
            up = delta.copy()
            down = delta.copy()
            
            up[up < 0] = 0
            down[down > 0] = 0
            
            # Use exponential moving average
            rUp = up.ewm(com=period - 1, adjust=False).mean()
            rDown = down.ewm(com=period - 1, adjust=False).mean().abs()
            
            # Avoid division by zero
            rDown = rDown.replace(0, np.nan)
            rs = rUp / rDown
            rsi = 100 - (100 / (1 + rs))
            
            return rsi.fillna(50)  # Fill NaN with neutral RSI value
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()


# Global data provider instance
data_provider = DataProvider()
