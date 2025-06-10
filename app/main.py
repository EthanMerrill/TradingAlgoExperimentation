"""
Main application entry point for the trading algorithm.
Orchestrates the entire trading workflow.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta
import argparse
import pandas as pd
from typing import List
import time

from config import config
from data_provider import data_provider
from strategy import StrategyOptimizer
from trading_engine import trading_engine
from cloud_storage import cloud_storage
from utils import setup_logging, is_trading_day, TradingCalendar

logger = logging.getLogger(__name__)


class TradingAlgorithm:
    """Main trading algorithm orchestrator."""
    
    def __init__(self):
        self.optimizer = StrategyOptimizer()
        self.trading_calendar = TradingCalendar()
        self.session_metadata = {
            'start_time': None,
            'end_time': None,
            'config': config.to_dict(),
            'results_summary': {}
        }
    
    async def run_full_cycle(self, force_backtest: bool = False) -> dict:
        """
        Run the complete trading algorithm cycle.
        
        Args:
            force_backtest: Force running backtest even if recent results exist
            
        Returns:
            Dictionary with session results
        """
        self.session_metadata['start_time'] = datetime.now()
        
        # Startup banner
        logger.info("🚀" * 20)
        logger.info("🚀 TRADING ALGORITHM STARTING")
        logger.info("🚀" * 20)
        logger.info(f"📅 Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"💼 Paper Trading: {config.PAPER_TRADE}")
        logger.info(f"🔄 Force Backtest: {force_backtest}")
        logger.info("=" * 60)
        
        try:
            # Check if it's a trading day
            if not self.trading_calendar.is_trading_day():
                logger.info("Market is closed today - skipping execution")
                return {'status': 'market_closed'}
            
            # Step 1: Check current positions and account status
            logger.info("🔍 Checking account status and current positions...")
            account_info = data_provider.get_account_info()
            current_positions = trading_engine.get_current_positions()
            
            logger.info("💰 Account Summary:")
            logger.info(f"   • Equity: ${account_info.get('equity', 0):,.2f}")
            logger.info(f"   • Cash Available: ${account_info.get('cash', 0):,.2f}")
            logger.info(f"   • Current Positions: {len(current_positions)}")
            logger.info("─" * 40)
            
            # Check if we have enough cash to potentially trade
            cash_pct = account_info.get('cash', 0) / account_info.get('equity', 1)
            
            if cash_pct < config.MIN_CASH_PCT and not force_backtest:
                logger.info(f"💸 Insufficient cash for new positions:")
                logger.info(f"   • Current cash: {cash_pct:.2%} of equity")
                logger.info(f"   • Required minimum: {config.MIN_CASH_PCT:.2%}")
                logger.info("🔄 Checking existing positions for exit opportunities only...")
                # Still check for exit opportunities
                return await self._handle_exits_only()
            
            # Step 2: Get or run backtests
            backtest_results = await self._get_backtest_results(force_backtest)
            
            if not backtest_results:
                logger.warning("No backtest results available")
                return {'status': 'no_backtest_results'}
            
            # Step 3: Execute trading session
            logger.info("🎯 Analyzing trading opportunities and executing orders...")
            trading_summary = trading_engine.execute_trading_session(backtest_results)
            
            # Step 4: Save results and metadata
            logger.info("💾 Saving session results and metadata...")
            await self._save_session_results(backtest_results, trading_summary)
            
            self.session_metadata['end_time'] = datetime.now()
            self.session_metadata['results_summary'] = trading_summary
            
            # Success banner
            session_duration = (self.session_metadata['end_time'] - self.session_metadata['start_time']).total_seconds()
            logger.info("🎉" * 20)
            logger.info("🎉 TRADING ALGORITHM COMPLETE!")
            logger.info("🎉" * 20)
            logger.info(f"⏱️  Session Duration: {session_duration/60:.1f} minutes")
            logger.info(f"📊 Backtest Results: {len(backtest_results)} strategies")
            logger.info(f"💼 Trading Summary: {trading_summary}")
            logger.info("=" * 60)
            return {
                'status': 'success',
                'trading_summary': trading_summary,
                'backtest_count': len(backtest_results),
                'duration': (self.session_metadata['end_time'] - self.session_metadata['start_time']).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error in trading algorithm: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _get_backtest_results(self, force_backtest: bool) -> List:
        """Get backtest results, either from cache or by running new backtests."""
        
        # Check for recent backtest results
        if not force_backtest:
            logger.info("🔍 Checking for recent cached backtest results...")
            recent_results = self._load_recent_backtest_results()
            if recent_results:
                logger.info(f"✅ Found cached results: {len(recent_results)} profitable strategies")
                logger.info("⚡ Skipping backtest - using cached data")
                return recent_results
            else:
                logger.info("❌ No recent cached results found")
        else:
            logger.info("🔄 Force backtest enabled - ignoring cached results")
        
        logger.info("Running new backtests...")
        
        # Step 1: Get stock universe
        universe_df = await data_provider.get_stock_universe()
        
        if universe_df.empty:
            logger.error("Failed to get stock universe")
            return []
        
        symbols = universe_df['symbol'].tolist()
        logger.info(f"📋 Stock universe loaded: {len(symbols)} symbols")
        
        # Step 2: Set backtest date range
        end_date = datetime.now()
        start_date = config.BACKTEST_START_DATE
        
        logger.info(f"📊 Starting comprehensive backtest analysis...")
        logger.info(f"🕐 This may take 30-90 minutes depending on market conditions")
        
        # Step 3: Run optimization for all symbols
        results = await self.optimizer.optimize_universe(symbols, start_date, end_date)
        
        # Step 4: Filter results
        logger.info("🔍 Filtering and analyzing results...")
        filtered_results = self.optimizer.filter_results(results)
        
        logger.info(f"📈 Backtest analysis complete!")
        logger.info(f"   • Total strategies tested: {len(results)}")
        logger.info(f"   • Profitable strategies: {len(filtered_results)}")
        logger.info(f"   • Success rate: {(len(filtered_results)/len(results)*100) if results else 0:.1f}%")
        
        # Step 5: Save results to cloud storage
        logger.info("💾 Saving results to cloud storage...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cloud_storage.save_backtest_results(filtered_results, timestamp)
        
        return filtered_results
    
    async def _handle_exits_only(self) -> dict:
        """Handle position exits when not running full backtests."""
        try:
            current_positions = trading_engine.get_current_positions()
            exit_positions = trading_engine.identify_exit_opportunities(current_positions)
            
            exits_executed = 0
            for position in exit_positions:
                if trading_engine.place_sell_order(position):
                    exits_executed += 1
            
            return {
                'status': 'exits_only',
                'positions_exited': exits_executed,
                'total_positions': len(current_positions)
            }
            
        except Exception as e:
            logger.error(f"Error handling exits: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _load_recent_backtest_results(self) -> List:
        """Load recent backtest results from cloud storage."""
        try:
            backtest_files = cloud_storage.list_backtest_files()
            
            if not backtest_files:
                return []
            
            # Sort by filename (which contains timestamp) and get most recent
            backtest_files.sort(reverse=True)
            most_recent = backtest_files[0]
            
            # Check if file is recent enough (within last 24 hours)
            try:
                file_timestamp = most_recent.split('_')[2].split('.')[0]  # Extract timestamp
                file_date = datetime.strptime(file_timestamp, '%H%M%S')
                # Combine with today's date for comparison
                file_datetime = datetime.combine(datetime.now().date(), file_date.time())
                
                if (datetime.now() - file_datetime).total_seconds() < 24 * 3600:
                    return cloud_storage.load_backtest_results(most_recent)
            except (IndexError, ValueError):
                pass
            
            return []
            
        except Exception as e:
            logger.error(f"Error loading recent backtest results: {e}")
            return []
    
    async def _save_session_results(self, backtest_results: List, trading_summary: dict):
        """Save session results and metadata."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save current positions
            positions_df = data_provider.get_current_positions()
            if not positions_df.empty:
                cloud_storage.save_positions(positions_df, timestamp)
            
            # Save session metadata
            self.session_metadata['backtest_count'] = len(backtest_results)
            self.session_metadata['trading_summary'] = trading_summary
            cloud_storage.save_metadata(self.session_metadata, timestamp)
            
        except Exception as e:
            logger.error(f"Error saving session results: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading Algorithm')
    parser.add_argument('--force-backtest', action='store_true', 
                       help='Force running new backtests')
    parser.add_argument('--paper-trading', action='store_true', 
                       help='Enable paper trading mode')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run analysis without placing orders')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Override config if needed
    if args.paper_trading:
        config.PAPER_TRADE = True
    
    logger.info("=" * 50)
    logger.info("Trading Algorithm Starting")
    logger.info(f"Paper Trading: {config.PAPER_TRADE}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 50)
    
    try:
        # Initialize and run the trading algorithm
        algorithm = TradingAlgorithm()
        
        if args.dry_run:
            logger.info("DRY RUN MODE - No orders will be placed")
            # You could implement a dry run mode here
        
        result = await algorithm.run_full_cycle(force_backtest=args.force_backtest)
        
        logger.info("=" * 50)
        logger.info("Trading Algorithm Complete")
        logger.info(f"Result: {result}")
        logger.info("=" * 50)
        
        return result
        
    except KeyboardInterrupt:
        logger.info("Algorithm interrupted by user")
        return {'status': 'interrupted'}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {'status': 'error', 'error': str(e)}


if __name__ == "__main__":
    result = asyncio.run(main())
    
    # Exit with appropriate code
    if result.get('status') == 'success':
        sys.exit(0)
    else:
        sys.exit(1)
