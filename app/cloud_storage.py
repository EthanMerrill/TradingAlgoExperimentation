"""
Cloud storage module for persisting data and results.
Handles Google Cloud Storage operations for backtests and positions.
"""
import importlib
import io
import logging
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import numpy as np
# pylint: disable=broad-exception-caught
import pandas as pd
from strategy import BacktestResult

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)


class CloudStorage:
    """Google Cloud Storage handler for trading data."""

    def __init__(self):
        try:
            storage_module = importlib.import_module("google.cloud.storage")
            self.client = storage_module.Client()
            self.bucket = self.client.bucket(globalConfig.GCS_BUCKET_NAME)
        except Exception as e:
            logger.error("Error initializing cloud storage: %s", e)
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
            return {k: round(v, 2) if isinstance(v, (float, np.floating)) else v
                    for k, v in data.items()}
        elif isinstance(data, pd.DataFrame):
            return data.round(2)
        elif isinstance(data, (list, tuple)):
            return [self._round_floats(item) for item in data]
        else:
            return round(data, 2) if isinstance(data, (float, np.floating)) else data

    def save_backtest_results(self, results: List[BacktestResult], timestamp: Optional[str] = None) -> bool:
        """
        Save backtest results to cloud storage.

        Args:
            results: List of BacktestResult objects
            timestamp: Optional timestamp string for filename (if None, uses current date)

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
                    'profitable': result.profitable,
                    'current_rsi': result.current_rsi
                }
                results_data.append(self._round_floats(result_dict))

            df = pd.DataFrame(results_data)

            # Generate filename with timestamp
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            filename = f"{globalConfig.get_environment_path('Backtests')}/backtest_results_{timestamp}.csv"

            # Upload to cloud storage
            blob = self.bucket.blob(filename)
            stream = io.StringIO()
            df.to_csv(stream, index=False)
            blob.upload_from_string(stream.getvalue(), content_type='text/csv')

            logger.info("Saved %d backtest results to %s",
                        len(results), filename)
            return True

        except Exception as e:
            logger.error("Error saving backtest results: %s", e)
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
            blob = self.bucket.blob(
                f"{globalConfig.get_environment_path('Backtests')}/{filename}")

            if not blob.exists():
                logger.error("File %s not found in cloud storage", filename)
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
                    profitable=bool(row['profitable']),
                    current_rsi=float(row['current_rsi']) if 'current_rsi' in row and pd.notna(
                        row['current_rsi']) else None
                )
                results.append(result)

            logger.info("Loaded %d backtest results from %s",
                        len(results), filename)
            return results

        except Exception as e:
            logger.error("Error loading backtest results: %s", e)
            return []

    def save_positions(self, positions_data, _run_number: Optional[int] = None, timestamp: Optional[str] = None) -> bool:
        """
        Save positions to cloud storage.

        Args:
            positions_data: DataFrame with position data or List of Position objects
            runNumber: Optional run number (deprecated, maintained for compatibility)
            timestamp: Optional timestamp string for filename (if None, uses current date)

        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False

        try:
            # Convert list of Position objects to DataFrame if needed
            if isinstance(positions_data, list):
                if not positions_data:
                    # Empty list, create empty DataFrame
                    positions_df = pd.DataFrame()
                else:
                    # Convert Position objects to dict format
                    positions_list = []
                    for pos in positions_data:
                        exit_price = pos.exit_price
                        if exit_price is None and pos.closed:
                            exit_price = pos.current_price

                        realized_return = pos.realized_return
                        if realized_return is None and pos.closed and pos.entry_price:
                            realized_return = (
                                (exit_price - pos.entry_price) / pos.entry_price
                                if exit_price is not None else None
                            )

                        pos_dict = {
                            'symbol': pos.symbol,
                            'shares': pos.quantity,
                            'entry_price': pos.entry_price,
                            'current_price': pos.current_price,
                            'current_rsi': pos.current_rsi,
                            'entry_date': pos.entry_date,
                            'rsi_period': pos.rsi_period,
                            'rsi_lower': pos.rsi_lower,
                            'rsi_upper': pos.rsi_upper,
                            'alpha': pos.alpha,
                            'stop_loss_price': pos.stop_loss_price,
                            'take_profit_price': pos.take_profit_price,
                            'closed': pos.closed,
                            'exit_date': pos.exit_date,
                            'exit_price': exit_price,
                            'realized_return': realized_return
                        }
                        positions_list.append(pos_dict)
                    positions_df = pd.DataFrame(positions_list)
            else:
                positions_df = positions_data

            # Generate filename with timestamp
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            filename = f"{globalConfig.get_environment_path('Positions')}/positions_{timestamp}.csv"

            # Round floats before uploading
            if isinstance(positions_df, pd.DataFrame):
                rounded_df = positions_df.round(2)
            else:
                rounded_df = positions_df

            # Upload to cloud storage
            blob = self.bucket.blob(filename)
            stream = io.StringIO()
            rounded_df.to_csv(stream, index=False)
            blob.upload_from_string(stream.getvalue(), content_type='text/csv')

            logger.info("Saved positions to %s", filename)
            return True

        except Exception as e:
            logger.error("Error saving positions: %s", e)
            return False

    def save_metadata(self, metadata: dict, timestamp: Optional[str] = None) -> bool:
        """
        Save algorithm metadata and configuration to cloud storage.
        Appends to existing metadata.csv file instead of overwriting.

        Args:
            metadata: Dictionary with metadata
            timestamp: Optional timestamp string (deprecated, only used for adding timestamp to metadata)

        Returns:
            True if successful
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False

        try:
            filename = f"{globalConfig.get_environment_path('Metadata')}/metadata.csv"

            # Add timestamp to metadata
            metadata_with_timestamp = metadata.copy()
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metadata_with_timestamp['timestamp'] = timestamp

            # Round any float values
            metadata_with_timestamp = self._round_floats(
                metadata_with_timestamp)

            # Check if file already exists
            blob = self.bucket.blob(filename)
            existing_df = pd.DataFrame()

            if blob.exists():
                # Load existing data
                csv_string = blob.download_as_text()
                existing_df = pd.read_csv(io.StringIO(csv_string))

            # Convert new metadata to DataFrame
            new_metadata_df = pd.DataFrame([metadata_with_timestamp])

            # Append to existing data
            if not existing_df.empty:
                combined_df = pd.concat(
                    [existing_df, new_metadata_df], ignore_index=True)
            else:
                combined_df = new_metadata_df

            # Upload combined data to cloud storage
            stream = io.StringIO()
            combined_df.to_csv(stream, index=False)
            blob.upload_from_string(stream.getvalue(), content_type='text/csv')

            logger.info("Appended metadata to %s", filename)
            return True

        except Exception as e:
            logger.error("Error saving metadata: %s", e)
            return False

    def list_backtest_files(self) -> List[str]:
        """List all backtest files in cloud storage."""
        if not self.bucket:
            return []

        try:
            # Use environment-specific path
            prefix = f"{globalConfig.get_environment_path('Backtests')}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name.replace(prefix, '') for blob in blobs if blob.name.endswith('.csv')]
        except Exception as e:
            logger.error("Error listing backtest files: %s", e)
            return []

    def list_position_files(self) -> List[str]:
        """List all position entry files in cloud storage."""
        if not self.bucket:
            return []

        try:
            # Use environment-specific path
            prefix = f"{globalConfig.get_environment_path('Positions')}/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            return [blob.name.replace(prefix, '') for blob in blobs if blob.name.endswith('.csv')]
        except Exception as e:
            logger.error("Error listing position files: %s", e)
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
            blob = self.bucket.blob(
                f"{globalConfig.get_environment_path('Positions')}/{filename}")

            if not blob.exists():
                logger.error(
                    "Position file %s not found in cloud storage", filename)
                return pd.DataFrame()

            csv_string = blob.download_as_text()
            df = pd.read_csv(io.StringIO(csv_string))

            logger.info("Loaded position entries from %s", filename)
            return df

        except Exception as e:
            logger.error("Error loading position entries: %s", e)
            return pd.DataFrame()

    def get_latest_position_file(self) -> Optional[str]:
        """
        Get the most recent position file based on filename.

        Returns:
            Filename of the most recent position file, or None if no files found
        """
        position_files = self.list_position_files()
        if not position_files:
            return None

        # Sort files by name (assumes YYYYMMDD format) and get the most recent
        position_files.sort(reverse=True)
        return position_files[0]

    def get_latest_positions_df(self, openPosition=True) -> pd.DataFrame:
        """
        Get the most recent position DataFrame.
        Args:
            openPosition: If True, return only open positions; if False, return only closed positions

        Returns:
            DataFrame with the latest position entries, or empty DataFrame if no files found
        """
        latest_file = self.get_latest_position_file()
        if not latest_file:
            logger.warning("No position files found in cloud storage")
            return pd.DataFrame()
        positions_df = self.load_position_entries(latest_file)
        # Filter out closed positions
        if not positions_df.empty and 'closed' in positions_df.columns and openPosition:
            return positions_df[positions_df['closed'] != True]
        elif not positions_df.empty and 'closed' in positions_df.columns and not openPosition:
            return positions_df[positions_df['closed'] == True]
        return positions_df


# Global cloud storage instance
cloud_storage = CloudStorage()
