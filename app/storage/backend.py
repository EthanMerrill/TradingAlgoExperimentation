"""
Abstract storage backend interface.
All persistence operations (GCS, Postgres, etc.) must implement this ABC.
"""
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

if TYPE_CHECKING:
    from config import Config
    from strategy import BacktestResult

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for all storage backends (GCS, Postgres, etc.)."""

    @abstractmethod
    def save_backtest_results(
        self, results: "List[BacktestResult]", timestamp: Optional[str] = None
    ) -> bool:
        """Save backtest results to the backend."""

    @abstractmethod
    def load_backtest_results(self, filename: str) -> "List[BacktestResult]":
        """Load backtest results from the backend."""

    @abstractmethod
    def save_positions(
        self,
        positions_data,
        _run_number: Optional[int] = None,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Save positions snapshot to the backend."""

    @abstractmethod
    def save_metadata(
        self, metadata: dict, timestamp: Optional[str] = None
    ) -> bool:
        """Save session metadata to the backend."""

    @abstractmethod
    def list_backtest_files(self) -> List[str]:
        """List all backtest file identifiers in the backend."""

    @abstractmethod
    def list_position_files(self) -> List[str]:
        """List all position file identifiers in the backend."""

    @abstractmethod
    def load_position_entries(self, filename: str) -> pd.DataFrame:
        """Load position entries from a specific file in the backend."""

    @abstractmethod
    def get_latest_position_file(self) -> Optional[str]:
        """Get the most recent position file identifier."""

    @abstractmethod
    def get_latest_positions_df(self, openPosition: bool = True) -> pd.DataFrame:
        """Get the most recent position DataFrame."""

    @staticmethod
    def create(config: "Config") -> "StorageBackend":
        """Factory: return the correct StorageBackend for the active config."""
        backend = getattr(config, "STORAGE_BACKEND", "gcs")

        if backend == "gcs":
            from storage.gcs import GcsStorage  # pylint: disable=import-outside-toplevel
            return GcsStorage()

        if backend == "postgres":
            from storage.postgres import (  # pylint: disable=import-outside-toplevel
                PostgresStorage,
            )
            return PostgresStorage()

        raise ValueError(
            f"Unknown STORAGE_BACKEND '{backend}'. "
            f"Expected 'gcs' or 'postgres'."
        )
