import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

from config import globalConfig  # type: ignore

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
    exit_price: Optional[float] = None
    realized_return: Optional[float] = None
    exit_reason: Optional[str] = None
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
        Returns a list of open Position objects.
        """

        alpaca_positions = self.data_provider.get_current_positions_df()
        cloud_positions = self.cloud_storage.get_latest_positions_df(True)
        alpaca_symbols_list = (
            sorted(alpaca_positions['symbol'].astype(str).tolist())
            if not alpaca_positions.empty and 'symbol' in alpaca_positions.columns
            else []
        )
        cloud_symbols_list = (
            sorted(cloud_positions['symbol'].astype(str).tolist())
            if not cloud_positions.empty and 'symbol' in cloud_positions.columns
            else []
        )
        logger.info(
            "Reconciliation start: alpaca_open_rows=%d, cloud_open_rows=%d, alpaca_symbols=%s, cloud_symbols=%s",
            len(alpaca_positions), len(
                cloud_positions), alpaca_symbols_list, cloud_symbols_list
        )

        # Normalize cloud schema when there is no prior snapshot but Alpaca has open positions.
        if cloud_positions.empty:
            if not alpaca_positions.empty:
                logger.info(
                    "Cloud positions are empty. Initializing with Alpaca positions.")
                cloud_positions = pd.DataFrame({
                    'symbol': alpaca_positions['symbol'],
                    'shares': pd.to_numeric(alpaca_positions.get('qty', 0), errors='coerce').fillna(0.0),
                    'entry_price': pd.to_numeric(alpaca_positions.get('avg_entry_price', 0), errors='coerce').fillna(0.0),
                    'current_price': pd.to_numeric(alpaca_positions.get('current_price', 0), errors='coerce').fillna(0.0),
                    'position_value': pd.to_numeric(alpaca_positions.get('market_value', 0), errors='coerce').fillna(0.0),
                    'current_rsi': 0.0,
                    'entry_date': datetime.now(),
                    'rsi_period': 14,
                    'rsi_lower': 30,
                    'rsi_upper': 70,
                    'alpha': 0.0,
                    'stop_loss_price': np.nan,
                    'take_profit_price': np.nan,
                    'exit_date': pd.NaT,
                    'exit_price': np.nan,
                    'realized_return': np.nan,
                    'exit_reason': None,
                    'closed': False,
                })
            else:
                cloud_positions = pd.DataFrame()

        # Reconcile positions from both sources
        newly_closed_positions = pd.DataFrame()
        if not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            alpaca_symbols = set(alpaca_positions['symbol']) if (
                not alpaca_positions.empty and 'symbol' in alpaca_positions.columns
            ) else set()
            cloud_symbols = set(cloud_positions['symbol'])
            logger.info(
                "Reconciliation symbol sets: alpaca_open=%d, cloud_open=%d",
                len(alpaca_symbols), len(cloud_symbols)
            )

            # Add Alpaca-only symbols as open positions (broker is source of truth for open holdings).
            alpaca_only_symbols = alpaca_symbols - cloud_symbols
            if alpaca_only_symbols:
                logger.warning(
                    "Found %d symbols in Alpaca that are not in cloud storage: %s. Adding as open positions.",
                    len(alpaca_only_symbols), alpaca_only_symbols)

                optimizer = None
                try:
                    # Local import avoids unnecessary startup cost and potential import cycles.
                    from strategy import StrategyOptimizer
                    optimizer = StrategyOptimizer()
                except (ImportError, AttributeError) as e:
                    logger.warning(
                        "Could not initialize strategy optimizer for Alpaca-only positions: %s", e)

                new_rows = []
                for symbol in alpaca_only_symbols:
                    alpaca_row = alpaca_positions.loc[
                        alpaca_positions['symbol'] == symbol].iloc[0]

                    current_rsi = 0.0
                    rsi_period = 14
                    rsi_lower = 30
                    rsi_upper = 70
                    alpha = 0.0

                    if optimizer is not None:
                        try:
                            end_date = datetime.now() - timedelta(minutes=20)
                            start_date = globalConfig.BACKTEST_START_DATE
                            backtest_result = optimizer.optimize_symbol(
                                symbol, start_date, end_date)
                            if backtest_result is not None:
                                current_rsi = float(
                                    backtest_result.current_rsi) if backtest_result.current_rsi is not None else 0.0
                                rsi_period = int(backtest_result.rsi_period)
                                rsi_lower = int(backtest_result.rsi_lower)
                                rsi_upper = int(backtest_result.rsi_upper)
                                alpha = float(backtest_result.alpha)
                        except (ValueError, TypeError, KeyError, RuntimeError) as e:
                            logger.warning(
                                "Backtest enrichment failed for Alpaca-only symbol %s: %s", symbol, e)

                    entry_price = float(alpaca_row.get(
                        'avg_entry_price', 0) or 0)
                    new_rows.append({
                        'symbol': symbol,
                        'shares': float(alpaca_row.get('qty', 0) or 0),
                        'entry_price': entry_price,
                        'current_price': float(alpaca_row.get('current_price', 0) or 0),
                        'position_value': float(alpaca_row.get('market_value', 0) or 0),
                        'current_rsi': current_rsi,
                        'entry_date': datetime.now(),
                        'rsi_period': rsi_period,
                        'rsi_lower': rsi_lower,
                        'rsi_upper': rsi_upper,
                        'alpha': alpha,
                        'stop_loss_price': (entry_price * (1 - globalConfig.STOP_LOSS_PCT)) if entry_price > 0 else np.nan,
                        'take_profit_price': (entry_price * (1 + globalConfig.TAKE_PROFIT_PCT)) if entry_price > 0 else np.nan,
                        'exit_date': pd.NaT,
                        'exit_price': np.nan,
                        'realized_return': np.nan,
                        'exit_reason': None,
                        'closed': False,
                    })
                cloud_positions = pd.concat(
                    [cloud_positions, pd.DataFrame(new_rows)], ignore_index=True)
                logger.info(
                    "Added %d Alpaca-only symbols to in-memory positions snapshot",
                    len(new_rows)
                )

            # Any symbol still in cloud open positions but not in Alpaca is treated as closed.
            cloud_only_symbols = set(
                cloud_positions['symbol']) - alpaca_symbols
            if cloud_only_symbols:
                logger.warning(
                    "Found %d symbols in cloud storage that are not in Alpaca: %s. Marking as closed",
                    len(cloud_only_symbols), cloud_only_symbols)
                newly_closed_positions = cloud_positions[
                    cloud_positions['symbol'].isin(cloud_only_symbols)
                ].copy()
                # Mark cloud-only symbols as closed
                for symbol in cloud_only_symbols:
                    symbol_mask = cloud_positions['symbol'] == symbol
                    # Use latest known price as exit price when position disappears from broker holdings.
                    if 'current_price' in cloud_positions.columns:
                        cloud_positions.loc[symbol_mask,
                                            'exit_price'] = cloud_positions.loc[symbol_mask, 'current_price']
                    elif 'entry_price' in cloud_positions.columns:
                        cloud_positions.loc[symbol_mask,
                                            'exit_price'] = cloud_positions.loc[symbol_mask, 'entry_price']

                    if 'exit_date' not in cloud_positions.columns:
                        cloud_positions['exit_date'] = pd.NaT
                    cloud_positions.loc[symbol_mask,
                                        'exit_date'] = datetime.now()

                    if 'entry_price' in cloud_positions.columns:
                        if 'realized_return' not in cloud_positions.columns:
                            cloud_positions['realized_return'] = np.nan
                        entry_prices = pd.to_numeric(
                            cloud_positions.loc[symbol_mask, 'entry_price'], errors='coerce')
                        exit_prices = pd.to_numeric(
                            cloud_positions.loc[symbol_mask, 'exit_price'], errors='coerce')
                        valid_mask = entry_prices > 0
                        cloud_positions.loc[symbol_mask, 'realized_return'] = np.where(
                            valid_mask,
                            (exit_prices - entry_prices) / entry_prices,
                            np.nan
                        )

                    cloud_positions.loc[cloud_positions['symbol']
                                        == symbol, 'closed'] = True
                    cloud_positions.loc[cloud_positions['symbol']
                                        == symbol, 'exit_reason'] = 'broker_closed'
                    cloud_positions['closed'] = cloud_positions['closed'].astype(
                        bool)

                logger.info(
                    "Marked %d positions as closed in cloud storage", len(cloud_only_symbols))

                # Keep the updated closed rows so they can be re-added to self.positions
                # and persisted in the next positions snapshot save.
                newly_closed_positions = cloud_positions[
                    cloud_positions['symbol'].isin(cloud_only_symbols)
                ].copy()
        # Update cloud open positions with current broker values for symbols that are currently open in Alpaca.
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
                    exit_price=float(row['exit_price']) if 'exit_price' in row and pd.notna(
                        row['exit_price']) else None,
                    realized_return=float(row['realized_return']) if 'realized_return' in row and pd.notna(
                        row['realized_return']) else None,
                    exit_reason=str(row['exit_reason']) if 'exit_reason' in row and pd.notna(
                        row['exit_reason']) and row['exit_reason'] is not None else None,
                    closed=row['closed'] if 'closed' in row else False,
                    exit_date=row['exit_date'] if 'exit_date' in row else None
                )
                self.positions.append(position)

        # add the closed positions
        closed_positions = self.cloud_storage.get_latest_positions_df(False)
        if not closed_positions.empty:
            for _, row in closed_positions.iterrows():
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
                    exit_price=float(row['exit_price']) if 'exit_price' in row and pd.notna(
                        row['exit_price']) else None,
                    realized_return=float(row['realized_return']) if 'realized_return' in row and pd.notna(
                        row['realized_return']) else None,
                    exit_reason=str(row['exit_reason']) if 'exit_reason' in row and pd.notna(
                        row['exit_reason']) and row['exit_reason'] is not None else None,
                    closed=row['closed'] if 'closed' in row else False,
                    exit_date=row['exit_date'] if 'exit_date' in row else None
                )
                self.positions.append(position)

        if not newly_closed_positions.empty:
            for _, row in newly_closed_positions.iterrows():
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
                    exit_price=float(row['exit_price']) if 'exit_price' in row and pd.notna(
                        row['exit_price']) else None,
                    realized_return=float(row['realized_return']) if 'realized_return' in row and pd.notna(
                        row['realized_return']) else None,
                    exit_reason=str(row['exit_reason']) if 'exit_reason' in row and pd.notna(
                        row['exit_reason']) and row['exit_reason'] is not None else None,
                    closed=True,
                    exit_date=row['exit_date'] if 'exit_date' in row else datetime.now(
                    )
                )
                self.positions.append(position)

        open_positions = [pos for pos in self.positions if not pos.closed]
        closed_positions_count = len(self.positions) - len(open_positions)
        logger.info(
            "Reconciliation complete: open_positions=%d, closed_positions=%d, total_in_memory=%d",
            len(open_positions), closed_positions_count, len(self.positions)
        )
        return open_positions

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
        cloud_positions = self.cloud_storage.get_latest_positions_df(True)
        if not cloud_positions.empty and 'symbol' in cloud_positions.columns:
            symbol_mask = cloud_positions['symbol'] == symbol
            if 'current_price' in cloud_positions.columns:
                cloud_positions.loc[symbol_mask,
                                    'exit_price'] = cloud_positions.loc[symbol_mask, 'current_price']
            elif 'entry_price' in cloud_positions.columns:
                cloud_positions.loc[symbol_mask,
                                    'exit_price'] = cloud_positions.loc[symbol_mask, 'entry_price']

            if 'entry_price' in cloud_positions.columns:
                entry_prices = pd.to_numeric(
                    cloud_positions.loc[symbol_mask, 'entry_price'], errors='coerce')
                exit_prices = pd.to_numeric(
                    cloud_positions.loc[symbol_mask, 'exit_price'], errors='coerce')
                cloud_positions.loc[symbol_mask, 'realized_return'] = np.where(
                    entry_prices > 0,
                    (exit_prices - entry_prices) / entry_prices,
                    np.nan
                )

            cloud_positions.loc[symbol_mask, 'exit_date'] = datetime.now()
            cloud_positions.loc[symbol_mask, 'closed'] = True
            cloud_positions['closed'] = cloud_positions['closed'].astype(bool)
            # self.cloud_storage.save_positions(cloud_positions) SAVE AT END
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
            if existing_position.symbol == position.symbol and not existing_position.closed:
                logger.warning(
                    "Position for %s already exists. Use update_position to modify it.", position.symbol)
                return

        # Add the new position
        self.positions.append(position)
        # Save the updated positions to cloud storage
        # self.cloud_storage.save_positions(self.positions) SAVE AT END
        logger.info(
            "Opened new position for %s. Full saved positions: %s", position.symbol, self.positions)
