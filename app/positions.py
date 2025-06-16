from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Current position information."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    current_rsi: float
    entry_date: datetime
    alpha: float
    rsi_period: int
    rsi_lower: int
    rsi_upper: int
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    closed: bool = False


class PositionsManager:
    """
    Tracks state of the positions in the trading engine.
    This class is responsible for managing position entries, saving them to cloud storage,
    and providing methods to retrieve and analyze position data.
    """

    def __init__(self, cloud_storage_instance, data_provider_instance):
        self.cloud_storage = cloud_storage_instance
        self.data_provider = data_provider_instance
        # Initialize as empty list of Position objects
        self.positions: List[Position] = []

    def get_and_reconcile_positions(self) -> List[Position]:
        """
        Retrieves positions from cloud storage and alpaca and updates prices
        Returns a list of Position objects.
        """

        alpaca_positions = self.data_provider.get_current_positions_df()
        cloud_positions = self.cloud_storage.get_latest_open_positions_df()

        # Print both sets of positions for debugging
        print("Alpaca positions: %s", alpaca_positions)
        print("Cloud positions: %s", cloud_positions)

        # Reconcile positions from both sources
        # Check for symbols in Alpaca that don't exist in cloud positions
        if not alpaca_positions.empty and not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            alpaca_only_symbols = set(
                alpaca_positions['symbol']) - set(cloud_positions['symbol'])
            if alpaca_only_symbols:
                logger.warning(
                    "Found %d symbols in Alpaca that are not in cloud storage: %s",
                    len(alpaca_only_symbols), alpaca_only_symbols)
        # Check for symbols in cloud positions that don't exist in Alpaca
        if not alpaca_positions.empty and not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            cloud_only_symbols = set(
                cloud_positions['symbol']) - set(alpaca_positions['symbol'])
            if cloud_only_symbols:
                logger.warning(
                    "Found %d symbols in cloud storage that are not in Alpaca: %s. Marking as closed",
                    len(cloud_only_symbols), cloud_only_symbols)
                # Mark cloud-only symbols as closed
                for symbol in cloud_only_symbols:
                    cloud_positions.loc[cloud_positions['symbol']
                                        == symbol, 'closed'] = True
                    cloud_positions['closed'] = cloud_positions['closed'].astype(
                        bool)

                logger.info(
                    "Marked %d positions as closed in cloud storage", len(cloud_only_symbols))
        # update the current prices, shares in cloud positions using the alpaca positions df
        if not alpaca_positions.empty and not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            for index, row in cloud_positions.iterrows():
                symbol = row['symbol']
                if symbol in alpaca_positions['symbol'].values:
                    cloud_positions.at[index, 'current_price'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'current_price'].values[0]
                    cloud_positions.at[index, 'shares'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'qty'].values[0]
                    cloud_positions.at[index, 'position_value'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'market_value'].values[0]
                    cloud_positions.at[index, 'entry_price'] = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol, 'avg_entry_price'].values[0]
        # if cloud positions are empty, initialize with alpaca positions
        if cloud_positions.empty:
            logger.info(
                "Cloud positions are empty. Initializing with Alpaca positions.")
            cloud_positions = alpaca_positions.copy()
            cloud_positions['closed'] = False

        print("Updated cloud positions: %s", cloud_positions)
        # Save the updated cloud positions back to storage
        self.cloud_storage.save_positions(cloud_positions)
        # Convert cloud positions DataFrame to a list of Position objects
        self.positions = []
        if not cloud_positions.empty:
            for _, row in cloud_positions.iterrows():
                if 'closed' in row and row['closed']:
                    continue  # Skip closed positions

                position = Position(
                    symbol=row['symbol'],
                    quantity=float(row['shares']) if 'shares' in row else 0,
                    entry_price=float(row['entry_price']
                                      ) if 'entry_price' in row else 0,
                    current_price=float(
                        row['current_price']) if 'current_price' in row else 0,
                    entry_date=row['entry_date'] if 'entry_date' in row else datetime.now(
                    ),
                    current_rsi=float(row['current_rsi']
                                      ) if 'current_rsi' in row else 0.0,
                    rsi_period=int(row['rsi_period']
                                   ) if 'rsi_period' in row else 14,
                    rsi_lower=int(row['rsi_lower']
                                  ) if 'rsi_lower' in row else 30,
                    rsi_upper=int(row['rsi_upper']
                                  ) if 'rsi_upper' in row else 70,
                    alpha=float(row['alpha']) if 'alpha' in row and pd.notna(
                        row['alpha']) else 0.0,
                    stop_loss_price=float(row['stop_loss_price']) if 'stop_loss_price' in row and pd.notna(
                        row['stop_loss_price']) else None,
                    take_profit_price=float(row['take_profit_price']) if 'take_profit_price' in row and pd.notna(
                        row['take_profit_price']) else None,
                    closed=row['closed'] if 'closed' in row else False,
                    exit_date=row['exit_date'] if 'exit_date' in row else None
                )
                self.positions.append(position)

        return self.positions

    def close_position(self, symbol: str):
        """
        Removes a position by symbol.
        Args:
            symbol (str): The symbol of the position to remove.
        """
        # Find and remove the position from self.positions
        position_found = False
        for i, position in enumerate(self.positions):
            if position.symbol == symbol:
                self.positions.pop(i)
                position_found = True
                break

        if not position_found:
            logger.warning(
                "Position with symbol %s not found in current positions.", symbol)
            return
        # Get current cloud positions to mark as closed
        cloud_positions = self.cloud_storage.get_latest_open_positions_df()
        if not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            cloud_positions.loc[cloud_positions['symbol']
                                == symbol, 'closed'] = True
            cloud_positions['closed'] = cloud_positions['closed'].astype(bool)
            cloud_positions['closed_date'] = datetime.now()
            self.cloud_storage.save_positions(cloud_positions)
        # Log the removal
        logger.info("Removed position for %s.", symbol)

    def open_position(self, position: Position):
        """
        Adds a new position to the current positions.
        Args:
            position (Position): The position to add.
        """
        # Check if the position already exists
        for existing_position in self.positions:
            if existing_position.symbol == position.symbol:
                logger.warning(
                    "Position for %s already exists. Use update_position to modify it.", position.symbol)
                return

        # Add the new position
        self.positions.append(position)
        # Save the updated positions to cloud storage
        self.cloud_storage.save_positions(self.positions)
        logger.info(
            "Opened new position for %s. Full saved positions: %s", position.symbol, self.positions)
