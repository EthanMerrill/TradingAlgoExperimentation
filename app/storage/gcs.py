"""
Cloud storage module for persisting data and results.
Handles Google Cloud Storage operations for backtests and positions.
"""
import base64
import importlib
import io
import json
import logging
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
# pylint: disable=broad-exception-caught
import pandas as pd
from storage.backend import StorageBackend, backtest_result_to_dict, dict_to_backtest_result, normalize_position_for_save, order_to_dict, dict_to_order
from strategy import BacktestResult

from config import globalConfig  # type: ignore

logger = logging.getLogger(__name__)

# Environment variable name for JSON-based service account credentials.
# When set, takes priority over file-based GOOGLE_APPLICATION_CREDENTIALS.
_GCS_JSON_CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS_JSON"


class GcsStorage(StorageBackend):
    """Google Cloud Storage handler for trading data."""

    def __init__(self):
        try:
            storage_module = importlib.import_module("google.cloud.storage")

            # Step A: JSON credentials in env var (Coolify secret, etc.)
            # Supports both raw JSON and base64-encoded JSON (for platforms
            # like Coolify where raw JSON breaks .env parsing).
            creds_json = os.environ.get(_GCS_JSON_CREDENTIALS_ENV)
            if creds_json:
                creds_info = self._parse_credentials_json(creds_json)
                self.client = storage_module.Client.from_service_account_info(
                    creds_info
                )
                logger.info("Cloud storage initialized via %s",
                            _GCS_JSON_CREDENTIALS_ENV)
            else:
                # Step B: File-based credentials path (local dev)
                creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                if creds_path and not os.path.exists(creds_path):
                    logger.warning(
                        "GOOGLE_APPLICATION_CREDENTIALS=%s does not exist; "
                        "unsetting and falling back to Application Default Credentials",
                        creds_path,
                    )
                    # Unset so the GCS client falls through to ADC instead of
                    # failing on a missing file.
                    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

                # Step C: Default ADC (metadata server, gcloud config, etc.)
                self.client = storage_module.Client()

            self.bucket = self.client.bucket(globalConfig.GCS_BUCKET_NAME)
        except Exception as e:
            logger.error("Error initializing cloud storage: %s", e)
            self.client = None
            self.bucket = None

    @staticmethod
    def _parse_credentials_json(raw_value: str) -> dict:
        """Parse GOOGLE_APPLICATION_CREDENTIALS_JSON from env var.

        Tries raw JSON first, then falls back to base64 decoding.
        Strips whitespace/newlines (Coolify .env may inject them) and
        removes the ``type`` metadata field (it is not a credential
        keyword and can break from_service_account_info on some versions).
        """
        # Sanitize: strip whitespace, newlines, and invisible chars that
        # Coolify / .env parsers may inject around the value.
        cleaned = raw_value.strip()

        # Try raw JSON first
        creds = None
        try:
            creds = json.loads(cleaned)
            logger.debug("Parsed %s as raw JSON", _GCS_JSON_CREDENTIALS_ENV)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fall back to base64-decoded JSON
        if creds is None:
            try:
                # Remove whitespace that Coolify may inject into the
                # base64 string.
                b64_clean = re.sub(r'[\s]', '', cleaned)
                decoded = base64.b64decode(b64_clean).decode("utf-8")
                creds = json.loads(decoded)
                logger.debug("Parsed %s as base64-encoded JSON",
                             _GCS_JSON_CREDENTIALS_ENV)
            except Exception as e:
                preview = cleaned[:80] if len(cleaned) > 80 else cleaned
                raise ValueError(
                    f"{_GCS_JSON_CREDENTIALS_ENV} is neither valid JSON "
                    f"nor valid base64-encoded JSON: {e}. "
                    f"Raw value preview: {preview!r}"
                ) from e

        # Strip the "type" field — it is metadata ("service_account"),
        # not a credential parameter, and can cause:
        #   "unexpected keyword argument 'type'"
        # on some google-auth library versions.
        creds.pop("type", None)

        return creds

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
            # Convert results to DataFrame using shared serialization helper
            results_data = [backtest_result_to_dict(r) for r in results]
            results_data = [self._round_floats(d) for d in results_data]
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
                row_dict = row.to_dict()
                result = dict_to_backtest_result(row_dict)
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
                    # Convert Position objects to dict format using shared helper
                    positions_list = []
                    for pos in positions_data:
                        pos_dict = normalize_position_for_save(pos)
                        positions_list.append(pos_dict)
                    positions_df = pd.DataFrame(positions_list)
            else:
                positions_df = positions_data

            # Generate filename with timestamp
            if timestamp is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            filename = f"{globalConfig.get_environment_path('Positions')}/positions_{timestamp}.csv"

            # Round price/quantity columns to 2 decimal places before uploading,
            # but preserve full precision for ratio columns (realized_return,
            # alpha, current_rsi) so the UI can display exact percentages.
            PRICE_COLS = {
                'entry_price', 'exit_price', 'current_price',
                'stop_loss_price', 'take_profit_price', 'shares',
            }
            if isinstance(positions_df, pd.DataFrame):
                rounded_df = positions_df.copy()
                for col in rounded_df.select_dtypes(include='number').columns:
                    if col in PRICE_COLS:
                        rounded_df[col] = rounded_df[col].round(2)
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

    def _orders_path(self) -> str:
        """Blob path for the single append+upsert orders ledger."""
        return f"{globalConfig.get_environment_path('Orders')}/orders.csv"

    def save_orders(self, orders, timestamp: Optional[str] = None) -> bool:
        """Persist broker orders to a single append+upsert CSV.

        Keyed by ``client_order_id``: incoming rows replace existing rows with
        the same key, so status lifecycle updates do not create duplicates.
        """
        if not self.bucket:
            logger.error("Cloud storage not initialized")
            return False

        if not orders:
            return True

        try:
            incoming = pd.DataFrame([order_to_dict(o) for o in orders])

            blob = self.bucket.blob(self._orders_path())
            if blob.exists():
                existing = pd.read_csv(
                    io.StringIO(blob.download_as_text()))
                # Upsert: drop existing rows whose client_order_id is incoming,
                # then append the incoming rows (incoming wins).
                if not existing.empty and 'client_order_id' in existing.columns:
                    keys = set(incoming['client_order_id'].astype(str))
                    existing = existing[
                        ~existing['client_order_id'].astype(str).isin(keys)
                    ]
                combined = pd.concat([existing, incoming], ignore_index=True)
            else:
                combined = incoming

            # Round only numeric price/qty columns; leave datetime columns
            # (submitted_at / filled_at) untouched to avoid pandas warnings.
            order_price_cols = {'qty', 'stop_price', 'limit_price'}
            for col in combined.select_dtypes(include='number').columns:
                if col in order_price_cols:
                    combined[col] = combined[col].round(2)

            stream = io.StringIO()
            combined.to_csv(stream, index=False)
            blob.upload_from_string(
                stream.getvalue(), content_type='text/csv')
            logger.info(
                "Saved %d orders to %s", len(orders), self._orders_path())
            return True

        except Exception as e:
            logger.error("Error saving orders: %s", e)
            return False

    def load_orders(
        self, symbol: Optional[str] = None, status: Optional[str] = None
    ) -> List[Any]:
        """Load orders from the orders ledger, optionally filtered."""
        if not self.bucket:
            return []

        try:
            blob = self.bucket.blob(self._orders_path())
            if not blob.exists():
                return []

            df = pd.read_csv(io.StringIO(blob.download_as_text()))
            if df.empty:
                return []

            orders: List[Any] = [
                dict_to_order(row.to_dict()) for _, row in df.iterrows()
            ]
            if symbol is not None:
                orders = [
                    o for o in orders
                    if str(o.symbol).upper() == str(symbol).upper()
                ]
            if status is not None:
                s = status.lower()
                orders = [o for o in orders if o.status.lower() == s]
            return orders

        except Exception as e:
            logger.error("Error loading orders: %s", e)
            return []

    def get_open_orders_stored(self, symbol: Optional[str] = None) -> List[Any]:
        """Load persisted orders that are not in a terminal status."""
        return [o for o in self.load_orders(symbol) if not o.is_terminal]

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
            # Parse date columns when present; fall back gracefully for legacy
            # files that don't have entry_date / exit_date columns yet.
            date_cols = ['entry_date', 'exit_date']
            try:
                df = pd.read_csv(io.StringIO(csv_string),
                                 parse_dates=date_cols)
            except ValueError:
                df = pd.read_csv(io.StringIO(csv_string))

            # Normalize 'closed' from CSV strings ("True"/"False") to real
            # booleans so downstream comparisons (== True / != True) work.
            if 'closed' in df.columns:
                df['closed'] = df['closed'].apply(
                    lambda x: str(x).strip().lower() in ('true', '1')
                )

            logger.info(
                "load_position_entries(%s): %d rows, columns=%s, "
                "closed_counts: open=%d closed=%d",
                filename, len(df), list(df.columns),
                int((~df['closed'].astype(bool)).sum()
                    ) if 'closed' in df.columns else -1,
                int(df['closed'].astype(bool).sum()
                    ) if 'closed' in df.columns else -1,
            )
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
        logger.debug(
            "get_latest_position_file: %d total files, picked %s "
            "(top 5: %s)", len(position_files), position_files[0],
            position_files[:5])
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
        if 'closed' in positions_df.columns:
            if openPosition:
                return positions_df[positions_df['closed'] != True]
            else:
                return positions_df[positions_df['closed'] == True]
        else:
            # Legacy CSV without 'closed' column — treat all rows as open
            if not openPosition:
                logger.warning(
                    "Positions file '%s' is missing 'closed' column. "
                    "Returning empty DataFrame for closed positions query.",
                    latest_file
                )
                return pd.DataFrame()
            logger.info(
                "Positions file '%s' is missing 'closed' column. "
                "Treating all rows as open (legacy format).",
                latest_file
            )
            return positions_df
