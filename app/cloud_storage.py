"""
Cloud storage module for persisting data and results.
Handles Google Cloud Storage operations for backtests and positions.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
import logging
import io
import json
from google.cloud import storage
from config import config
from strategy import BacktestResult

if TYPE_CHECKING:
    from trading_engine import TradingOpportunity

logger = logging.getLogger(__name__)


class CloudStorage:
    """Google Cloud Storage handler for trading data."""
    
    def __init__(self):
        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(config.GCS_BUCKET_NAME)
        except Exception as e:
            logger.error(f"Error initializing cloud storage: {e}")
            self.client = None
            self.bucket = None
    
    def _round_floats(self, data):
        """
        Round all float values in data structure to 2 decimal places.
        
        Args:
            data: Dictionary, DataFrame, or other data structure
            
        Returns:
            Data with floats rounded to 2 decimal places
        """
        if isinstance(data, dict):
            return {k: round(v, 2) if isinstance(v, (float, np.float64, np.float32)) else v 
                   for k, v in data.items()}
        elif isinstance(data, pd.DataFrame):
            return data.round(2)
        elif isinstance(data, (list, tuple)):
            return [self._round_floats(item) for item in data]
        else:
            return round(data, 2) if isinstance(data, (float, np.float64, np.float32)) else data
    
    def save_backtest_results(self, results: List[BacktestResult], timestamp: str = None) -> bool:
        """
        Save backtest results to cloud storage.
        
        Args:
            results: List of BacktestResult objects
            timestamp: Optional timestamp string for filename
            
        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False
        
        try:
            # Convert results to DataFrame
            results_data = []
            for result in results:
                result_dict = {
                    'symbol': result.symbol,
                    'rsi_period': result.rsi_period,
                    'rsi_lower': result.rsi_lower,
                    'rsi_upper': result.rsi_upper,
                    'total_return': result.total_return,
                    'buy_and_hold_return': result.buy_and_hold_return,
                    'alpha': result.alpha,
                    'num_trades': result.num_trades,
                    'win_rate': result.win_rate,
                    'avg_trade_duration': result.avg_trade_duration,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'profitable': result.profitable
                }
                results_data.append(self._round_floats(result_dict))
            
            df = pd.DataFrame(results_data)
            
            # Generate filename with environment-specific path
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"{config.get_environment_path('Backtests')}/backtest_results_{timestamp}.csv"
            
            # Upload to cloud storage
            blob = self.bucket.blob(filename)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            blob.upload_from_string(stream.getvalue(), content_type='text/csv')
            
            logger.info(f"Saved {len(results)} backtest results to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False
    
    def load_backtest_results(self, filename: str) -> List[BacktestResult]:
        """
        Load backtest results from cloud storage.
        
        Args:
            filename: Filename in cloud storage
            
        Returns:
            List of BacktestResult objects
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return []
        
        try:
            blob = self.bucket.blob(f"{config.get_environment_path('Backtests')}/{filename}")
            
            if not blob.exists():
                logger.error(f"File {filename} not found in cloud storage")
                return []
            
            csv_string = blob.download_as_text()
            df = pd.read_csv(io.StringIO(csv_string))
            
            # Convert DataFrame back to BacktestResult objects
            results = []
            for _, row in df.iterrows():
                result = BacktestResult(
                    symbol=row['symbol'],
                    rsi_period=int(row['rsi_period']),
                    rsi_lower=int(row['rsi_lower']),
                    rsi_upper=int(row['rsi_upper']),
                    total_return=float(row['total_return']),
                    buy_and_hold_return=float(row['buy_and_hold_return']),
                    alpha=float(row['alpha']),
                    num_trades=int(row['num_trades']),
                    win_rate=float(row['win_rate']),
                    avg_trade_duration=float(row['avg_trade_duration']),
                    max_drawdown=float(row['max_drawdown']),
                    sharpe_ratio=float(row['sharpe_ratio']),
                    profitable=bool(row['profitable'])
                )
                results.append(result)
            
            logger.info(f"Loaded {len(results)} backtest results from {filename}")
            return results
            
        except Exception as e:
            logger.error(f"Error loading backtest results: {e}")
            return []
    
    def save_positions(self, positions_df: pd.DataFrame, timestamp: str = None) -> bool:
        """
        Save current positions to cloud storage.
        
        Args:
            positions_df: DataFrame with position data
            timestamp: Optional timestamp string for filename
            
        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False
        
        try:
            # Generate filename with environment-specific path
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"{config.get_environment_path('Positions')}/positions_{timestamp}.csv"
            
            # Round floats before uploading
            rounded_df = self._round_floats(positions_df)
            
            # Upload to cloud storage
            blob = self.bucket.blob(filename)
            stream = io.StringIO()
            rounded_df.to_csv(stream, index=False)
            blob.upload_from_string(stream.getvalue(), content_type='text/csv')
            
            logger.info(f"Saved positions to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
            return False
    
    def load_positions(self, filename: str) -> pd.DataFrame:
        """
        Load positions from cloud storage.
        
        Args:
            filename: Filename in cloud storage
            
        Returns:
            DataFrame with position data
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return pd.DataFrame()
        
        try:
            blob = self.bucket.blob(f"{config.get_environment_path('Positions')}/{filename}")
            
            if not blob.exists():
                logger.error(f"File {filename} not found in cloud storage")
                return pd.DataFrame()
            
            csv_string = blob.download_as_text()
            df = pd.read_csv(io.StringIO(csv_string))
            
            logger.info(f"Loaded positions from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            return pd.DataFrame()
    
    def save_metadata(self, metadata: dict, timestamp: str = None) -> bool:
        """
        Save algorithm metadata and configuration.
        
        Args:
            metadata: Dictionary with metadata
            timestamp: Optional timestamp string for filename
            
        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False
        
        try:
            # Generate filename with environment-specific path
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"{config.get_environment_path('Metadata')}/metadata_{timestamp}.json"
            
            # Upload to cloud storage
            blob = self.bucket.blob(filename)
            metadata_json = json.dumps(metadata, indent=2, default=str)
            blob.upload_from_string(metadata_json, content_type='application/json')
            
            logger.info(f"Saved metadata to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return False
    
    def list_backtest_files(self) -> List[str]:
        """List all backtest files in cloud storage."""
        if not self.bucket:
            return []
        
        try:
            # Use environment-specific path
            prefix = f"{config.get_environment_path('Backtests')}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name.replace(prefix, '') for blob in blobs if blob.name.endswith('.csv')]
        except Exception as e:
            logger.error(f"Error listing backtest files: {e}")
            return []
    
    def save_consolidated_trades(self, trades_df: pd.DataFrame, timestamp: str = None) -> bool:
        """
        Save consolidated trade log from multiple symbols to cloud storage.
        
        Args:
            trades_df: DataFrame with all trade details from multiple symbols
            timestamp: Optional timestamp string for filename
            
        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False
        
        try:
            # Generate filename with environment-specific path
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"{config.get_environment_path('trades')}/consolidated_trades_{timestamp}.csv"
            
            # Round floats before converting to CSV
            rounded_df = self._round_floats(trades_df)
            
            # Convert DataFrame to CSV string
            csv_buffer = io.StringIO()
            rounded_df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Upload to cloud storage
            blob = self.bucket.blob(filename)
            blob.upload_from_string(csv_string, content_type='text/csv')
            
            logger.info(f"Saved {len(trades_df)} consolidated trades to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving consolidated trade log: {e}")
            return False

    def list_trade_files(self) -> List[str]:
        """List all trade log files in cloud storage."""
        if not self.bucket:
            return []
        
        try:
            # Use environment-specific path
            prefix = f"{config.get_environment_path('trades')}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name.replace(prefix, '') for blob in blobs if blob.name.endswith('.csv')]
        except Exception as e:
            logger.error(f"Error listing trade files: {e}")
            return []
    
    def save_position_entry(self, opportunity: 'TradingOpportunity', shares: int, order_success: bool, date: str = None) -> bool:
        """
        Save position entry details with backtest information to a daily CSV file.
        
        Args:
            opportunity: TradingOpportunity with backtest details
            shares: Number of shares purchased
            order_success: Whether the order was successfully placed
            date: Optional date string (YYYYMMDD format), defaults to today
            
        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False
        
        try:
            # Use provided date or today's date
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
            
            # Create position entry record
            position_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': opportunity.symbol,
                'shares': shares,
                'entry_price': opportunity.entry_price,
                'order_success': order_success,
                'current_rsi': opportunity.current_rsi,
                'rsi_period': opportunity.rsi_period,
                'target_rsi_lower': opportunity.target_rsi_lower,
                'target_rsi_upper': opportunity.target_rsi_upper,
                'backtest_return': opportunity.backtest_return,
                'alpha': opportunity.alpha,
                'win_rate': opportunity.win_rate,
                'position_value': shares * opportunity.entry_price
            }
            
            # Round floats in position entry
            position_entry = self._round_floats(position_entry)
            
            # Generate filename for daily positions file
            filename = f"{config.get_environment_path('Positions')}/positions_{date}.csv"
            
            # Check if file already exists and load existing data
            blob = self.bucket.blob(filename)
            existing_df = pd.DataFrame()
            
            if blob.exists():
                try:
                    csv_string = blob.download_as_text()
                    existing_df = pd.read_csv(io.StringIO(csv_string))
                except Exception as e:
                    logger.warning(f"Could not load existing positions file {filename}: {e}")
                    existing_df = pd.DataFrame()
            
            # Create new DataFrame with this position entry
            new_entry_df = pd.DataFrame([position_entry])
            
            # Combine with existing data and round all floats
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_entry_df], ignore_index=True)
            else:
                combined_df = new_entry_df
            
            # Round all floats in the combined DataFrame
            combined_df = self._round_floats(combined_df)
            
            # Upload updated CSV to cloud storage
            stream = io.StringIO()
            combined_df.to_csv(stream, index=False)
            blob.upload_from_string(stream.getvalue(), content_type='text/csv')
            
            logger.info(f"Saved position entry for {opportunity.symbol} to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving position entry: {e}")
            return False

    def list_position_files(self) -> List[str]:
        """List all position entry files in cloud storage."""
        if not self.bucket:
            return []
        
        try:
            # Use environment-specific path
            prefix = f"{config.get_environment_path('Positions')}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name.replace(prefix, '') for blob in blobs if blob.name.endswith('.csv')]
        except Exception as e:
            logger.error(f"Error listing position files: {e}")
            return []
    
    def load_position_entries(self, filename: str) -> pd.DataFrame:
        """
        Load position entries from a specific daily file.
        
        Args:
            filename: Filename in cloud storage (e.g., 'positions_20240610.csv')
            
        Returns:
            DataFrame with position entry data
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return pd.DataFrame()
        
        try:
            blob = self.bucket.blob(f"{config.get_environment_path('Positions')}/{filename}")
            
            if not blob.exists():
                logger.error(f"Position file {filename} not found in cloud storage")
                return pd.DataFrame()
            
            csv_string = blob.download_as_text()
            df = pd.read_csv(io.StringIO(csv_string))
            
            logger.info(f"Loaded position entries from {filename}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading position entries: {e}")
            return pd.DataFrame()

# Global cloud storage instance
cloud_storage = CloudStorage()
