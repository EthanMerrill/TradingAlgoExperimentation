"""
Trading execution module.
Handles order placement, position management, and portfolio updates.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from data_provider import data_provider, TechnicalIndicators
from strategy import BacktestResult
from config import config
from cloud_storage import cloud_storage

logger = logging.getLogger(__name__)


@dataclass
class TradingOpportunity:
    """Trading opportunity based on strategy results."""
    symbol: str
    current_rsi: float
    target_rsi_lower: int
    target_rsi_upper: int
    rsi_period: int
    expected_return: float
    alpha: float
    confidence: float
    entry_price: float


@dataclass
class Position:
    """Current position information."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_date: datetime
    rsi_period: int
    rsi_lower: int
    rsi_upper: int
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None


class TradingEngine:
    """Main trading execution engine."""
    
    def __init__(self):
        self.trading_client = data_provider.trading_client
        self._positions_cache = {}
        self._last_position_update = None
    
    def get_current_positions(self) -> List[Position]:
        """Get current positions with strategy metadata."""
        try:
            alpaca_positions = self.trading_client.get_all_positions()
            positions = []
            
            # Load RSI metadata from the most recent positions CSV
            position_metadata = self._get_position_metadata_from_csv()
            
            for pos in alpaca_positions:
                # Get RSI metadata for this symbol, fall back to defaults if not found
                metadata = position_metadata.get(pos.symbol, {})
                
                position = Position(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    entry_price=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    entry_date=datetime.now(),  # Placeholder - would need to track separately
                    rsi_period=metadata.get('rsi_period', 14),  # From CSV or default
                    rsi_lower=metadata.get('target_rsi_lower', 30),   # From CSV or default
                    rsi_upper=metadata.get('target_rsi_upper', 70)    # From CSV or default
                )
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []
    
    def identify_buying_opportunities(self, backtest_results: List[BacktestResult]) -> List[TradingOpportunity]:
        """
        Identify current buying opportunities based on backtest results.
        
        Args:
            backtest_results: List of profitable backtest results
            
        Returns:
            List of current trading opportunities
        """
        opportunities = []
        
        for result in backtest_results:
            try:
                # Get current RSI for the symbol
                current_rsi = self._get_current_rsi(result.symbol, result.rsi_period)
                
                if current_rsi is None:
                    continue
                
                # Check if current RSI indicates a buy signal
                if current_rsi < result.rsi_lower:
                    # Get current price
                    current_price = self._get_current_price(result.symbol)
                    
                    if current_price is None:
                        continue
                    
                    opportunity = TradingOpportunity(
                        symbol=result.symbol,
                        current_rsi=current_rsi,
                        target_rsi_lower=result.rsi_lower,
                        target_rsi_upper=result.rsi_upper,
                        rsi_period=result.rsi_period,
                        expected_return=result.total_return,
                        alpha=result.alpha,
                        confidence=result.win_rate,
                        entry_price=current_price
                    )
                    opportunities.append(opportunity)
                    
            except Exception as e:
                logger.error(f"Error evaluating opportunity for {result.symbol}: {e}")
                continue
        
        # Sort by alpha (best opportunities first)
        opportunities.sort(key=lambda x: x.alpha, reverse=True)
        
        return opportunities
    
    def identify_exit_opportunities(self, positions: List[Position]) -> List[Position]:
        """
        Identify positions that should be exited.
        
        Args:
            positions: Current positions
            
        Returns:
            List of positions to exit
        """
        exit_positions = []
        
        for position in positions:
            try:
                # Get current RSI
                current_rsi = self._get_current_rsi(position.symbol, position.rsi_period)
                
                if current_rsi is None:
                    continue
                
                should_exit = False
                exit_reason = ""
                
                # RSI exit signal
                if current_rsi > position.rsi_upper:
                    should_exit = True
                    exit_reason = "RSI_UPPER"
                
                # Time-based exit (max hold period)
                days_held = (datetime.now() - position.entry_date).days
                if days_held >= config.MAX_HOLD_DAYS:
                    should_exit = True
                    exit_reason = "MAX_HOLD_DAYS"
                
                # Stop loss exit
                current_loss = (position.current_price - position.entry_price) / position.entry_price
                if current_loss < -config.STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                
                # Take profit exit
                current_gain = (position.current_price - position.entry_price) / position.entry_price
                if current_gain > config.TAKE_PROFIT_PCT:
                    should_exit = True
                    exit_reason = "TAKE_PROFIT"
                
                if should_exit:
                    logger.info(f"Exit signal for {position.symbol}: {exit_reason}")
                    exit_positions.append(position)
                    
            except Exception as e:
                logger.error(f"Error evaluating exit for {position.symbol}: {e}")
                continue
        
        return exit_positions
    
    def calculate_position_sizes(self, opportunities: List[TradingOpportunity]) -> List[Tuple[TradingOpportunity, int]]:
        """
        Calculate position sizes for trading opportunities.
        
        Args:
            opportunities: List of trading opportunities
            
        Returns:
            List of (opportunity, shares) tuples
        """
        try:
            account_info = data_provider.get_account_info()
            current_positions = self.get_current_positions()
            
            if not account_info:
                return []
            
            cash = account_info['cash']
            equity = account_info['equity']
            
            # Check if we have enough cash to trade
            cash_pct = cash / equity if equity > 0 else 0
            if cash_pct < config.MIN_CASH_PCT:
                logger.info(f"Insufficient cash percentage: {cash_pct:.2%}")
                return []
            
            # Calculate how many new positions we can take
            current_position_count = len(current_positions)
            max_new_positions = min(
                config.MAX_NEW_POSITIONS_PER_DAY,
                config.MAX_POSITIONS - current_position_count
            )
            
            if max_new_positions <= 0:
                logger.info("No new positions allowed")
                return []
            
            # Select top opportunities up to max new positions
            selected_opportunities = opportunities[:max_new_positions]
            
            # Calculate position size for each opportunity
            position_allocations = []
            available_cash = cash * (1 - config.MIN_CASH_PCT)  # Reserve minimum cash
            
            for opportunity in selected_opportunities:
                # Equal weight allocation
                position_value = available_cash * config.POSITION_SIZE_PCT
                shares = int(position_value / opportunity.entry_price)
                
                if shares > 0:
                    position_allocations.append((opportunity, shares))
            
            return position_allocations
            
        except Exception as e:
            logger.error(f"Error calculating position sizes: {e}")
            return []
    
    def place_buy_order(self, opportunity: TradingOpportunity, shares: int) -> bool:
        """
        Place a buy order for a trading opportunity.
        
        Args:
            opportunity: Trading opportunity
            shares: Number of shares to buy
            
        Returns:
            True if order was placed successfully
        """
        order_success = False
        try:
            # Calculate stop loss and take profit prices
            stop_loss_price = opportunity.entry_price * (1 - config.STOP_LOSS_PCT)
            take_profit_price = opportunity.entry_price * (1 + config.TAKE_PROFIT_PCT)

            # Round stop loss and take profit prices to two decimal places
            stop_loss_price = round(stop_loss_price, 2)
            take_profit_price = round(take_profit_price, 2)
            
            # Create bracket order (buy with stop loss and take profit)
            order_request = MarketOrderRequest(
                symbol=opportunity.symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=stop_loss_price),
                take_profit=TakeProfitRequest(limit_price=take_profit_price)
            )
            
            order = self.trading_client.submit_order(order_request)
            
            logger.info(f"Buy order placed for {shares} shares of {opportunity.symbol} at ${opportunity.entry_price:.2f}")
            logger.info(f"Stop loss: ${stop_loss_price:.2f}, Take profit: ${take_profit_price:.2f}")
            
            order_success = True
            
        except Exception as e:
            logger.error(f"Error placing buy order for {opportunity.symbol}: {e}")
        
        # Log position entry details to cloud storage regardless of order success
        try:
            cloud_storage.save_position_entry(opportunity, shares, order_success)
        except Exception as e:
            logger.error(f"Error saving position entry log for {opportunity.symbol}: {e}")
        
        return order_success
    
    def place_sell_order(self, position: Position) -> bool:
        """
        Place a sell order to exit a position.
        
        Args:
            position: Position to exit
            
        Returns:
            True if order was placed successfully
        """
        try:
            order_request = MarketOrderRequest(
                symbol=position.symbol,
                qty=abs(position.quantity),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            logger.info(f"Sell order placed for {position.quantity} shares of {position.symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error placing sell order for {position.symbol}: {e}")
            return False
    
    def execute_trading_session(self, backtest_results: List[BacktestResult]) -> Dict[str, any]:
        """
        Execute a complete trading session.
        
        Args:
            backtest_results: Results from strategy backtesting
            
        Returns:
            Dictionary with session summary
        """
        session_summary = {
            'timestamp': datetime.now(),
            'opportunities_found': 0,
            'orders_placed': 0,
            'positions_exited': 0,
            'errors': []
        }
        
        try:
            # Get current positions
            current_positions = self.get_current_positions()
            
            # Check for exit opportunities
            exit_positions = self.identify_exit_opportunities(current_positions)
            
            # Execute exit orders
            for position in exit_positions:
                if self.place_sell_order(position):
                    session_summary['positions_exited'] += 1
            
            # Identify buying opportunities
            opportunities = self.identify_buying_opportunities(backtest_results)
            session_summary['opportunities_found'] = len(opportunities)
            
            # Calculate position sizes
            position_allocations = self.calculate_position_sizes(opportunities)
            
            # Execute buy orders
            for opportunity, shares in position_allocations:
                if self.place_buy_order(opportunity, shares):
                    session_summary['orders_placed'] += 1
            
            logger.info(f"Trading session complete: {session_summary}")
            
        except Exception as e:
            error_msg = f"Error in trading session: {e}"
            logger.error(error_msg)
            session_summary['errors'].append(error_msg)
        
        return session_summary
    
    def _get_current_rsi(self, symbol: str, period: int) -> Optional[float]:
        """Get current RSI value for a symbol."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period * 3)  # Buffer for weekends/holidays
            
            data = data_provider.get_single_stock_bars(symbol, start_date, end_date)
            
            if data.empty or len(data) < period:
                return None
            
            rsi = TechnicalIndicators.calculate_rsi(data, period)
            return rsi.iloc[-1] if not rsi.empty else None
            
        except Exception as e:
            logger.error(f"Error getting RSI for {symbol}: {e}")
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)
            
            data = data_provider.get_single_stock_bars(symbol, start_date, end_date)
            
            if data.empty:
                return None
            
            return data['c'].iloc[-1]
            
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def _get_position_metadata_from_csv(self) -> Dict[str, Dict]:
        """
        Load position metadata from the most recent positions CSV file.
        
        Returns:
            Dictionary mapping symbol to metadata (rsi_period, target_rsi_lower, target_rsi_upper)
        """
        try:
            # Import cloud_storage here to avoid circular imports
            from cloud_storage import cloud_storage
            
            # Get list of position files
            position_files = cloud_storage.list_position_files()
            
            if not position_files:
                logger.warning("No position files found in cloud storage")
                return {}
            
            # Sort files by name (assumes YYYYMMDD format) and get the most recent
            position_files.sort(reverse=True)
            most_recent_file = position_files[0]
            
            logger.info(f"Loading position metadata from {most_recent_file}")
            
            # Load the CSV data
            df = cloud_storage.load_position_entries(most_recent_file)
            
            if df.empty:
                logger.warning(f"No data found in {most_recent_file}")
                return {}
            
            # Create metadata dictionary mapping symbol to RSI parameters
            metadata = {}
            for _, row in df.iterrows():
                symbol = row['symbol']
                metadata[symbol] = {
                    'rsi_period': int(row.get('rsi_period', 14)),
                    'target_rsi_lower': int(row.get('target_rsi_lower', 30)),
                    'target_rsi_upper': int(row.get('target_rsi_upper', 70))
                }
            
            logger.info(f"Loaded metadata for {len(metadata)} symbols from {most_recent_file}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error loading position metadata from CSV: {e}")
            return {}


# Global trading engine instance
trading_engine = TradingEngine()
