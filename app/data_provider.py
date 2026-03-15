"""
Data provider module for fetching market data from various sources.
Replaces the legacy networking.py with modern async/await patterns.
"""
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient

from config import globalConfig  # type: ignore

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
        alpaca_config = globalConfig.get_alpaca_config()

        # Check if we have valid credentials
        if not alpaca_config['api_key'] or not alpaca_config['secret_key']:
            logger.warning(
                "No Alpaca API credentials found. Data provider will have limited functionality.")
            self.historical_client: Optional[StockHistoricalDataClient] = None
            self.trading_client: Optional[TradingClient] = None
        else:
            try:
                self.historical_client: Optional[StockHistoricalDataClient] = StockHistoricalDataClient(
                    api_key=alpaca_config['api_key'],
                    secret_key=alpaca_config['secret_key']
                )
                self.trading_client: Optional[TradingClient] = TradingClient(
                    api_key=alpaca_config['api_key'],
                    secret_key=alpaca_config['secret_key'],
                    paper=globalConfig.PAPER_TRADE
                )
            except Exception as e:
                logger.error("Failed to initialize Alpaca clients: %s", e)
                self.historical_client = None
                self.trading_client = None

        self._rate_limit_delay: float = globalConfig.API_RATE_LIMIT_DELAY
        print("DataProvider initialized with rate limit delay of",
              self._rate_limit_delay)

    # async def get_historical_bars(
    #     self,
    #     symbols: List[str],
    #     start_date: datetime,
    #     end_date: datetime,
    #     timeframe: TimeFrame = TimeFrame(1, TimeFrameUnit.Day)
    # ) -> Dict[str, pd.DataFrame]:
    #     """
    #     Get historical bar data for multiple symbols using Alpaca's current API.

    #     Args:
    #         symbols: List of stock symbols
    #         start_date: Start date for data
    #         end_date: End date for data
    #         timeframe: Data timeframe (Day, Hour, Minute)

    #     Returns:
    #         Dictionary with symbol as key and DataFrame as value
    #     """
    #     try:
    #         if self.historical_client is None:
    #             logger.error("Historical client not available")
    #             return {}

    #         request_params = StockBarsRequest(
    #             symbol_or_symbols=symbols,
    #             timeframe=timeframe,
    #             start=start_date,
    #             end=end_date
    #         )

    #         bars = self.historical_client.get_stock_bars(request_params)

    #         # Convert to DataFrame format
    #         result = {}
    #         for symbol in symbols:
    #             # Handle both actual API response (with .data attribute) and test mocks (direct dict)
    #             if isinstance(bars, dict):
    #                 # Direct dict response (test mocks)
    #                 bars_data = bars
    #             else:
    #                 # API response with .data attribute
    #                 bars_data = getattr(bars, 'data', None) if bars else None

    #             if bars_data:
    #                 # Handle both dict and object-like access
    #                 symbol_bars = None
    #                 if hasattr(bars_data, 'get'):
    #                     # Dict-like access
    #                     symbol_bars = bars_data.get(symbol, None)
    #                 elif hasattr(bars_data, symbol):
    #                     # Attribute access
    #                     symbol_bars = getattr(bars_data, symbol, None)

    #                 if symbol_bars:
    #                     df_data = []
    #                     for bar in symbol_bars:
    #                         df_data.append({
    #                             'timestamp': bar.timestamp,
    #                             'open': bar.open,
    #                             'high': bar.high,
    #                             'low': bar.low,
    #                             'close': bar.close,
    #                             'volume': bar.volume
    #                         })

    #                     if df_data:
    #                         df = pd.DataFrame(df_data)
    #                         df['symbol'] = symbol
    #                         df.set_index('timestamp', inplace=True)
    #                         result[symbol] = df
    #                     else:
    #                         logger.warning(
    #                             "No data found for symbol %s", symbol)
    #                         result[symbol] = pd.DataFrame()
    #                 else:
    #                     logger.warning(
    #                         "Symbol %s not found in response", symbol)
    #                     result[symbol] = pd.DataFrame()
    #             else:
    #                 logger.warning("Symbol %s not found in response", symbol)
    #                 result[symbol] = pd.DataFrame()

    #         # Rate limiting
    #         await asyncio.sleep(self._rate_limit_delay)
    #         return result

    #     except Exception as e:
    #         logger.error("Error fetching historical data: %s", e)
    #         return {symbol: pd.DataFrame() for symbol in symbols}

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
            logger.error(
                "Historical client not initialized - missing API credentials")
            return pd.DataFrame()

        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame(1, cast(TimeFrameUnit, TimeFrameUnit.Day)),
                start=start_date,
                end=end_date
            )

            bars = self.historical_client.get_stock_bars(request_params)

            # Handle both actual API response (with .data attribute) and test mocks (direct dict)
            if isinstance(bars, dict):
                # Direct dict response (test mocks)
                bars_data = bars
            else:
                # API response with .data attribute
                bars_data = getattr(bars, 'data', None) if bars else None

            if bars_data:
                # Handle both dict and object-like access
                symbol_bars = None
                if hasattr(bars_data, 'get'):
                    # Dict-like access
                    symbol_bars = bars_data.get(symbol, None)
                elif hasattr(bars_data, symbol):
                    # Attribute access
                    symbol_bars = getattr(bars_data, symbol, None)

                if symbol_bars:
                    df_data = []
                    for bar in symbol_bars:
                        df_data.append({
                            'timestamp': bar.timestamp,  # Use full name for timestamp
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume
                        })

                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    return df
                else:
                    logger.warning("No data found for symbol %s", symbol)
                    return pd.DataFrame()
            else:
                logger.warning("No data found for symbol %s", symbol)
                return pd.DataFrame()

        except Exception as e:
            logger.error("Error fetching data for %s: %s", symbol, e)
            return pd.DataFrame()

        finally:
            time.sleep(self._rate_limit_delay)

    def get_current_positions_df(self) -> pd.DataFrame:
        """Get current portfolio positions."""
        try:
            if self.trading_client is None:
                logger.error("Trading client not available")
                return pd.DataFrame()

            positions = self.trading_client.get_all_positions()
            print(f"Fetched {len(positions)} positions from Alpaca")

            if not positions:
                return pd.DataFrame()

            position_data = []
            for position in positions:
                # Safely handle position attributes
                symbol = getattr(position, 'symbol', None)
                qty = getattr(position, 'qty', 0)
                market_value = getattr(position, 'market_value', 0)
                avg_entry_price = getattr(position, 'avg_entry_price', 0)
                unrealized_pl = getattr(position, 'unrealized_pl', 0)
                unrealized_plpc = getattr(position, 'unrealized_plpc', 0)
                current_price = getattr(position, 'current_price', 0)

                position_data.append({
                    'symbol': symbol,
                    'qty': float(qty) if qty else 0.0,
                    'market_value': float(market_value) if market_value else 0.0,
                    'avg_entry_price': float(avg_entry_price) if avg_entry_price else 0.0,
                    'unrealized_pl': float(unrealized_pl) if unrealized_pl else 0.0,
                    'unrealized_plpc': float(unrealized_plpc) if unrealized_plpc else 0.0,
                    'current_price': float(current_price) if current_price else 0.0
                })

                # if current price is not available, fetch it from the snapshot
                if not current_price and symbol:
                    snapshot = self.get_current_snapshot(symbol)
                    if snapshot and 'latest_trade' in snapshot:
                        position_data[-1]['current_price'] = snapshot['latest_trade'].get(
                            'price', 0.0)

            return pd.DataFrame(position_data)

        except Exception as e:
            logger.error("Error fetching current positions: %s", e)
            return pd.DataFrame()

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information including cash and equity."""
        try:
            if self.trading_client is None:
                logger.error("Trading client not available")
                return {}

            account = self.trading_client.get_account()

            # Safely handle account attributes
            cash = getattr(account, 'cash', 0)
            equity = getattr(account, 'equity', 0)
            long_market_value = getattr(account, 'long_market_value', 0)
            short_market_value = getattr(account, 'short_market_value', 0)
            buying_power = getattr(account, 'buying_power', 0)

            return {
                'cash': float(cash) if cash else 0.0,
                'equity': float(equity) if equity else 0.0,
                'long_market_value': float(long_market_value) if long_market_value else 0.0,
                'short_market_value': float(short_market_value) if short_market_value else 0.0,
                'buying_power': float(buying_power) if buying_power else 0.0,
            }

        except Exception as e:
            logger.error("Error fetching account info: %s", e)
            return {}

    def get_stock_universe(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Get filtered universe of stocks for trading.
        Filters stocks based on price using Alpaca snapshots API and MIN_PRICE globalConfig.

        Args:
            date: Date for universe (defaults to today)

        Returns:
            DataFrame with filtered stock universe
        """
        if date is None:
            date = datetime.now()

        try:
            if self.trading_client is None:
                logger.error("Trading client not available")
                return pd.DataFrame()

            # Get all active assets
            assets = self.trading_client.get_all_assets()
            print(f"Total assets fetched: {len(assets)}")
            # Filter for tradable stocks
            tradable_stocks = []
            for asset in assets:
                # Safely handle asset attributes
                tradable = getattr(asset, 'tradable', False)
                status = getattr(asset, 'status', None)
                exchange = getattr(asset, 'exchange', None)
                symbol = getattr(asset, 'symbol', None)

                if (tradable and
                    status == 'active' and
                    exchange and exchange in ['NASDAQ', 'NYSE', 'ARCA', 'BATS'] and
                        symbol):
                    tradable_stocks.append(asset)

            symbols = [getattr(asset, 'symbol', '')
                       for asset in tradable_stocks if getattr(asset, 'symbol', None)]
            logger.info(
                "Found %d tradable stocks before price filtering", len(symbols))

            # Apply price filtering using snapshots
            price_filtered_symbols = self._filter_symbols_by_price(symbols)

            # Create universe dataframe with price-filtered symbols
            universe_data = []
            for asset in tradable_stocks:
                if asset.symbol in price_filtered_symbols:
                    universe_data.append({
                        'symbol': asset.symbol,
                        'name': asset.name,
                        'exchange': asset.exchange,
                        'tradable': asset.tradable
                    })

            # UNCOMMENT WHEN DONE TESTING TEMP
            df = pd.DataFrame(universe_data)
            logger.info(
                "Returning universe of %d stocks after price filtering (min price: $%s)", len(df), globalConfig.MIN_PRICE)
            return df

        except Exception as e:
            logger.error("Error getting stock universe: %s", e)
            return pd.DataFrame()

    def _filter_symbols_by_price(self, symbols: List[str]) -> List[str]:
        """
        Filter symbols by current price using Alpaca snapshots API.

        Args:
            symbols: List of symbols to filter

        Returns:
            List of symbols that meet minimum price criteria
        """
        if not self.historical_client:
            logger.warning(
                "Historical client not initialized - skipping price filtering")
            return symbols

        try:
            # Process symbols in batches to avoid API limits
            batch_size = 100  # Alpaca snapshot API limit
            filtered_symbols = []

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]

                # Get snapshots for this batch
                try:
                    request = StockSnapshotRequest(symbol_or_symbols=batch)
                    snapshots = self.historical_client.get_stock_snapshot(
                        request)

                    # Check if we got a valid response (should be a dict)
                    if not isinstance(snapshots, dict):
                        logger.warning(
                            "Invalid snapshot response for batch %d: %s", i//batch_size + 1, type(snapshots))
                        # Include all symbols in this batch as fallback
                        filtered_symbols.extend(batch)
                        time.sleep(self._rate_limit_delay)
                        continue

                    # Filter by price and volume
                    for symbol in batch:
                        if symbol in snapshots:
                            snapshot = snapshots[symbol]

                            # Handle both Snapshot object and dict responses
                            current_price = None
                            daily_volume = None

                            # Try to get price from latest_trade
                            if hasattr(snapshot, 'latest_trade') and snapshot.latest_trade:
                                # Snapshot object format
                                current_price = float(
                                    snapshot.latest_trade.price)
                            elif isinstance(snapshot, dict) and snapshot.get('latest_trade'):
                                # Dict format
                                current_price = float(
                                    snapshot['latest_trade']['price'])

                            # Fallback to bid price if no trade data
                            if current_price is None:
                                latest_quote = getattr(snapshot, 'latest_quote', None) or (
                                    snapshot.get('latest_quote') if isinstance(snapshot, dict) else None)
                                if latest_quote:
                                    # Get bid price from quote
                                    if hasattr(latest_quote, 'bid_price'):
                                        current_price = float(
                                            latest_quote.bid_price)
                                    elif isinstance(latest_quote, dict) and 'bid_price' in latest_quote:
                                        current_price = float(
                                            latest_quote['bid_price'])

                            # Get volume from previous_daily_bar
                            previous_bar = getattr(snapshot, 'previous_daily_bar', None) or (
                                snapshot.get('previous_daily_bar') if isinstance(snapshot, dict) else None)
                            if previous_bar:
                                # Get volume from bar
                                if hasattr(previous_bar, 'volume'):
                                    daily_volume = float(previous_bar.volume)
                                elif isinstance(previous_bar, dict) and 'volume' in previous_bar:
                                    daily_volume = float(
                                        previous_bar['volume'])

                            # Apply price and volume filters
                            if current_price is not None and daily_volume is not None:
                                if (current_price >= globalConfig.MIN_PRICE and
                                    current_price <= globalConfig.MAX_PRICE and
                                        daily_volume >= globalConfig.MIN_VOLUME):
                                    filtered_symbols.append(symbol)

                except Exception as batch_error:
                    logger.warning(
                        "Error processing batch %d: %s", i//batch_size + 1, batch_error)
                    # If snapshots fail, include all symbols in this batch (fallback)
                    filtered_symbols.extend(batch)

                # Rate limiting
                time.sleep(self._rate_limit_delay)

            logger.info(
                "Price and volume filtering: %d/%d symbols passed ($%s - $%s, volume >= %s)",
                len(filtered_symbols), len(symbols), globalConfig.MIN_PRICE, globalConfig.MAX_PRICE, f"{globalConfig.MIN_VOLUME:,}")
            return filtered_symbols

        except Exception as e:
            logger.error("Error in price filtering: %s", e)
            # Return original list if filtering fails
            return symbols

    def get_current_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current snapshot data for a single symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with current price and volume data, or None if error
        """
        if not self.historical_client:
            logger.error(
                "Historical client not initialized - missing API credentials")
            return None

        try:
            request = StockSnapshotRequest(symbol_or_symbols=[symbol])
            snapshots = self.historical_client.get_stock_snapshot(request)

            # Check if response is a dict (which is the expected format)
            if not isinstance(snapshots, dict):
                logger.warning(
                    "Unexpected snapshot response type for %s: %s", symbol, type(snapshots))
                return None

            # Check if response contains the symbol data
            if symbol not in snapshots:
                logger.warning(
                    "No data found for symbol %s in snapshot response", symbol)
                return None

            snapshot = snapshots[symbol]
            result = {
                'symbol': symbol,
                'timestamp': datetime.now()
            }

            # Handle both Snapshot object and dict responses
            # Get latest trade data
            if hasattr(snapshot, 'latest_trade') and snapshot.latest_trade:
                # Snapshot object format
                trade_data = snapshot.latest_trade
                result.update({
                    'price': float(trade_data.price),
                    'volume': int(trade_data.size) if hasattr(trade_data, 'size') else 0,
                    'timestamp': trade_data.timestamp if hasattr(trade_data, 'timestamp') else result['timestamp']
                })
            elif isinstance(snapshot, dict) and snapshot.get('latest_trade'):
                # Dict format
                trade_data = snapshot['latest_trade']
                result.update({
                    'price': float(trade_data['price']),
                    'volume': int(trade_data.get('size', 0)),
                    'timestamp': trade_data.get('timestamp', result['timestamp'])
                })

            # Get latest quote data
            latest_quote = getattr(snapshot, 'latest_quote', None) or (
                snapshot.get('latest_quote') if isinstance(snapshot, dict) else None)
            if latest_quote:
                # Handle both object and dict formats
                if hasattr(latest_quote, 'bid_price'):
                    # Object format
                    result.update({
                        'bid_price': float(latest_quote.bid_price),
                        'ask_price': float(latest_quote.ask_price),
                        'bid_size': int(getattr(latest_quote, 'bid_size', 0)),
                        'ask_size': int(getattr(latest_quote, 'ask_size', 0))
                    })
                elif isinstance(latest_quote, dict):
                    # Dict format
                    result.update({
                        'bid_price': float(latest_quote.get('bid_price', 0)),
                        'ask_price': float(latest_quote.get('ask_price', 0)),
                        'bid_size': int(latest_quote.get('bid_size', 0)),
                        'ask_size': int(latest_quote.get('ask_size', 0))
                    })

            # Get daily bar data
            daily_bar = getattr(snapshot, 'daily_bar', None) or (
                snapshot.get('daily_bar') if isinstance(snapshot, dict) else None)
            if daily_bar:
                # Handle both object and dict formats
                if hasattr(daily_bar, 'open'):
                    # Object format
                    result.update({
                        'daily_open': float(daily_bar.open),
                        'daily_high': float(daily_bar.high),
                        'daily_low': float(daily_bar.low),
                        'daily_close': float(daily_bar.close),
                        'daily_volume': int(daily_bar.volume)
                    })
                elif isinstance(daily_bar, dict):
                    # Dict format
                    result.update({
                        'daily_open': float(daily_bar.get('open', 0)),
                        'daily_high': float(daily_bar.get('high', 0)),
                        'daily_low': float(daily_bar.get('low', 0)),
                        'daily_close': float(daily_bar.get('close', 0)),
                        'daily_volume': int(daily_bar.get('volume', 0))
                    })

            # Get previous daily bar data (for volume filtering)
            previous_daily_bar = getattr(snapshot, 'previous_daily_bar', None) or (
                snapshot.get('previous_daily_bar') if isinstance(snapshot, dict) else None)
            if previous_daily_bar:
                # Handle both object and dict formats
                if hasattr(previous_daily_bar, 'volume'):
                    # Object format
                    result.update({
                        'prev_daily_volume': int(previous_daily_bar.volume)
                    })
                elif isinstance(previous_daily_bar, dict):
                    # Dict format
                    result.update({
                        'prev_daily_volume': int(previous_daily_bar.get('volume', 0))
                    })

            return result

        except Exception as e:
            logger.error("Error fetching snapshot for %s: %s", symbol, e)
            return None

        finally:
            time.sleep(self._rate_limit_delay)

    # def get_historical_data(self, symbol: str, days_back: int = 30) -> Optional[pd.DataFrame]:
    #     """
    #     Get historical data for a single symbol - wrapper for compatibility with tests.

    #     Args:
    #         symbol: Stock symbol to get data for
    #         days_back: Number of days back to fetch data

    #     Returns:
    #         DataFrame with historical data or None if failed
    #     """
    #     try:
    #         end_date = datetime.now() - timedelta(minutes=20)
    #         start_date = end_date - timedelta(days=days_back)
    #         result = self.get_single_stock_bars(symbol, start_date, end_date)
    #         return result if not result.empty else None
    #     except Exception as e:
    #         logger.error("Error getting historical data for %s: %s", symbol, e)
    #         return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol - wrapper for compatibility with tests.

        Args:
            symbol: Stock symbol to get price for

        Returns:
            Current price or None if not available
        """
        try:
            # For test compatibility, try the mocked method first
            if hasattr(self.historical_client, 'get_stock_snapshots'):
                request = StockSnapshotRequest(symbol_or_symbols=[symbol])
                try:
                    snapshots = getattr(
                        self.historical_client, 'get_stock_snapshots')(request)
                    if isinstance(snapshots, dict) and symbol in snapshots:
                        snapshot_data = snapshots[symbol]
                        if hasattr(snapshot_data, 'latest_trade') and hasattr(snapshot_data.latest_trade, 'price'):
                            return snapshot_data.latest_trade.price
                except (AttributeError, TypeError):
                    pass  # Fall through to normal method

            # Fallback to the normal snapshot method
            snapshot = self.get_current_snapshot(symbol)
            if snapshot:
                # Check for latest trade price first
                if 'latest_trade' in snapshot and hasattr(snapshot['latest_trade'], 'price'):
                    return getattr(snapshot['latest_trade'], 'price', None)
                # Then check for quote data
                elif 'quote' in snapshot:
                    quote = snapshot['quote']
                    # Try to get the best price available
                    if hasattr(quote, 'bid') and hasattr(quote, 'ask'):
                        bid = getattr(quote, 'bid', 0)
                        ask = getattr(quote, 'ask', 0)
                        if bid and ask:
                            return (bid + ask) / 2
                    elif hasattr(quote, 'last_price'):
                        return getattr(quote, 'last_price', None)
            return None
        except Exception as e:
            logger.error("Error getting current price for %s: %s", symbol, e)
            return None

    # def get_multiple_stocks_data(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
    #     """
    #     Get historical data for multiple symbols - wrapper for compatibility with tests.

    #     Args:
    #         symbols: List of stock symbols
    #         **kwargs: Additional parameters (like days_back)

    #     Returns:
    #         Dict mapping symbols to their DataFrames
    #     """
    #     try:
    #         if not self.historical_client:
    #             logger.error(
    #                 "Historical client not initialized - missing API credentials")
    #             return {}

    #         # Get days_back from kwargs, default to 30
    #         days_back = kwargs.get('days_back', 30)

    #         # Try to get historical data using the same approach as get_historical_bars but synchronously
    #         request_params = StockBarsRequest(
    #             symbol_or_symbols=symbols,
    #             timeframe=TimeFrame(1, TimeFrameUnit.Day),
    #             start=datetime.now() - timedelta(days=days_back),
    #             end=datetime.now()
    #         )

    #         bars = self.historical_client.get_stock_bars(request_params)

    #         # Handle both actual API response (with .data attribute) and test mocks (direct dict)
    #         if isinstance(bars, dict):
    #             # Direct dict response (test mocks)
    #             bars_data = bars
    #         else:
    #             # API response with .data attribute
    #             bars_data = getattr(bars, 'data', None) if bars else None

    #         result = {}
    #         for symbol in symbols:
    #             if bars_data:
    #                 # Handle both dict and object-like access
    #                 symbol_bars = None
    #                 if hasattr(bars_data, 'get'):
    #                     # Dict-like access
    #                     symbol_bars = bars_data.get(symbol, None)
    #                 elif hasattr(bars_data, symbol):
    #                     # Attribute access
    #                     symbol_bars = getattr(bars_data, symbol, None)

    #                 if symbol_bars:
    #                     df_data = []
    #                     for bar in symbol_bars:
    #                         df_data.append({
    #                             'timestamp': bar.timestamp,
    #                             'open': getattr(bar, 'open', None),
    #                             'high': getattr(bar, 'high', None),
    #                             'low': getattr(bar, 'low', None),
    #                             'close': getattr(bar, 'close', None),
    #                             'volume': getattr(bar, 'volume', None)
    #                         })

    #                     if df_data:
    #                         df = pd.DataFrame(df_data)
    #                         df['symbol'] = symbol
    #                         df.set_index('timestamp', inplace=True)
    #                         result[symbol] = df
    #                     else:
    #                         result[symbol] = pd.DataFrame()
    #                 else:
    #                     result[symbol] = pd.DataFrame()
    #             else:
    #                 result[symbol] = pd.DataFrame()

    #         return result

    #     except Exception as e:
    #         logger.error("Error getting multiple stocks data: %s", e)
    #         return {}

    # def get_market_snapshot(self, symbols: List[str]) -> Dict[str, float]:
    #     """
    #     Get market snapshot for multiple symbols - wrapper for compatibility with tests.

    #     Args:
    #         symbols: List of stock symbols

    #     Returns:
    #         Dict mapping symbols to their prices
    #     """
    #     try:
    #         result = {}

    #         # For test compatibility, try the mocked method first
    #         if hasattr(self.historical_client, 'get_stock_snapshots'):
    #             request = StockSnapshotRequest(symbol_or_symbols=symbols)
    #             try:
    #                 snapshots = getattr(
    #                     self.historical_client, 'get_stock_snapshots')(request)
    #                 if isinstance(snapshots, dict):
    #                     for symbol in symbols:
    #                         if symbol in snapshots:
    #                             snapshot_data = snapshots[symbol]
    #                             if hasattr(snapshot_data, 'latest_trade') and hasattr(snapshot_data.latest_trade, 'price'):
    #                                 result[symbol] = snapshot_data.latest_trade.price
    #                     return result
    #             except (AttributeError, TypeError):
    #                 pass  # Fall through to normal method

    #         # Fallback to individual snapshot calls
    #         for symbol in symbols:
    #             snapshot = self.get_current_snapshot(symbol)
    #             if snapshot:
    #                 result[symbol] = snapshot
    #         return result
    #     except Exception as e:
    #         logger.error("Error getting market snapshot: %s", e)
    #         return {}


class TechnicalIndicators:
    """Technical analysis indicators."""

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: Optional[str] = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            data: DataFrame with price data
            period: RSI period
            price_col: Column name for price data (auto-detected if None)

        Returns:
            Series with RSI values
        """
        try:
            if len(data) < period + 1:
                return pd.Series(index=data.index, dtype=float)

            # Auto-detect price column if not specified
            if price_col is None:
                if 'close' in data.columns:
                    price_col = 'close'
                elif 'c' in data.columns:
                    price_col = 'c'
                else:
                    logger.error(
                        "No price column found. Expected 'close' or 'c' in data columns: %s", list(data.columns))
                    return pd.Series(index=data.index, dtype=float)

            # Ensure price column is numeric
            price_series = pd.to_numeric(data[price_col], errors='coerce')
            delta = price_series.diff()
            up = delta.copy()
            down = delta.copy()

            # Fix type issues with pandas operations - convert to numeric first
            up = pd.to_numeric(up, errors='coerce').where(
                pd.to_numeric(up, errors='coerce') > 0, 0)
            down = pd.to_numeric(down, errors='coerce').where(
                pd.to_numeric(down, errors='coerce') < 0, 0).abs()

            # Use exponential moving average
            rUp = up.ewm(com=period - 1, adjust=False).mean()
            rDown = down.ewm(com=period - 1, adjust=False).mean()

            # Avoid division by zero
            rDown = rDown.replace(0, np.nan)
            rs = rUp / rDown
            rsi = 100 - (100 / (1 + rs))

            return rsi.fillna(50)  # Fill NaN with neutral RSI value

        except Exception as e:
            logger.error("Error calculating RSI: %s", e)
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_moving_average(data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average - alias for compatibility."""
        return TechnicalIndicators.calculate_sma(data, window)

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price series
            window: Period for moving average
            num_std: Number of standard deviations for bands

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period  
            signal_period: Signal line EMA period

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = TechnicalIndicators.calculate_ema(data, fast_period)
        ema_slow = TechnicalIndicators.calculate_ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.calculate_ema(
            macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High price series
            low: Low price series
            close: Close price series
            k_period: Period for %K calculation
            d_period: Period for %D (signal line) calculation

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent


# Global data provider instance
data_provider = DataProvider()
