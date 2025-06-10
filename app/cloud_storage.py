"""
Cloud storage module for persisting data and results.
Handles Google Cloud Storage operations for backtests and positions.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List
import logging
import io
import json
from google.cloud import storage
from config import config
from strategy import BacktestResult

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
                results_data.append({
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
                })
            
            df = pd.DataFrame(results_data)
            
            # Generate filename
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"Backtests/backtest_results_{timestamp}.csv"
            
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
            blob = self.bucket.blob(f"Backtests/{filename}")
            
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
            # Generate filename
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"Positions/positions_{timestamp}.csv"
            
            # Upload to cloud storage
            blob = self.bucket.blob(filename)
            stream = io.StringIO()
            positions_df.to_csv(stream, index=False)
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
            blob = self.bucket.blob(f"Positions/{filename}")
            
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
            # Generate filename
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"Metadata/metadata_{timestamp}.json"
            
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
            blobs = self.bucket.list_blobs(prefix='Backtests/')
            return [blob.name.replace('Backtests/', '') for blob in blobs if blob.name.endswith('.csv')]
        except Exception as e:
            logger.error(f"Error listing backtest files: {e}")
            return []
    
    def list_position_files(self) -> List[str]:
        """List all position files in cloud storage."""
        if not self.bucket:
            return []
        
        try:
            blobs = self.bucket.list_blobs(prefix='Positions/')
            return [blob.name.replace('Positions/', '') for blob in blobs if blob.name.endswith('.csv')]
        except Exception as e:
            logger.error(f"Error listing position files: {e}")
            return []


# Global cloud storage instance
cloud_storage = CloudStorage()
