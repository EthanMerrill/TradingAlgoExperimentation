# Modern Trading Algorithm

This directory contains a complete rewrite of the legacy trading algorithm using modern Python practices and the latest Alpaca API.

## Architecture Overview

The application is organized into several modular components:

### Core Modules

- **`config.py`** - Configuration management and environment variables
- **`data_provider.py`** - Modern data fetching using Alpaca's latest API
- **`strategy.py`** - Vectorized RSI strategy backtesting and optimization
- **`trading_engine.py`** - Order execution and position management
- **`cloud_storage.py`** - Google Cloud Storage for data persistence
- **`utils.py`** - Utility functions and helper classes
- **`main.py`** - Main application orchestrator

### Key Improvements Over Legacy Code

1. **Modern API Integration**

   - Uses Alpaca's latest Python SDK (`alpaca-py`)
   - Proper async/await patterns for efficient data fetching
   - Better error handling and rate limiting

2. **Vectorized Backtesting**

   - Replaced Backtrader with custom vectorized implementation
   - Significantly faster execution
   - More flexible and maintainable

3. **Better Code Organization**

   - Clear separation of concerns
   - Type hints throughout
   - Comprehensive logging
   - Modular design for easy testing and extension

4. **Enhanced Risk Management**

   - Configurable position sizing
   - Stop-loss and take-profit orders
   - Maximum holding period limits
   - Correlation checking for diversification

5. **Robust Configuration**
   - Environment variable support
   - Command-line arguments
   - Paper trading mode
   - Dry run capability

## Setup Instructions

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables** (Optional)
   You can also set these environment variables:
   ```bash
   export PAPER_TRADE=True
   export MAX_POSITIONS=10
   export POSITION_SIZE_PCT=0.1
   export GCS_BUCKET_NAME=your-bucket-name
   ```

## Usage

### Run the Full Algorithm

```bash
python main.py
```

### Command Line Options

```bash
python main.py --help
python main.py --force-backtest  # Force new backtests
python main.py --paper-trading   # Enable paper trading
python main.py --dry-run         # Analysis only, no orders
python main.py --log-level DEBUG # Set logging level
python main.py --test-mode       # Run in test mode with limited universe and backtests
```

### Test the Components

```bash
python test_example.py
```

## Algorithm Workflow

1. **Position Check** - Evaluate current positions and account status
2. **Exit Signals** - Check existing positions for exit opportunities
3. **Universe Selection** - Get tradable stock universe (filtered by volume, price, etc.)
4. **Backtesting** - Run RSI strategy optimization across the universe
5. **Entry Signals** - Identify new buying opportunities based on current RSI
6. **Position Sizing** - Calculate appropriate position sizes based on risk management
7. **Order Execution** - Place bracket orders (buy with stop-loss and take-profit)
8. **Data Persistence** - Save results to Google Cloud Storage

## Strategy Details

### RSI Strategy

- **Entry Signal**: RSI crosses below the optimized lower threshold
- **Exit Signals**:
  - RSI crosses above the optimized upper threshold
  - Maximum holding period reached
  - Stop-loss triggered (configurable % loss)
  - Take-profit triggered (configurable % gain)

### Optimization

- Grid search across RSI periods, upper bounds, and lower bounds
- Evaluates strategies based on alpha (excess return vs buy-and-hold)
- Filters for profitable strategies with positive alpha and minimum win rate

### Risk Management

- Maximum number of positions (default: 10)
- Maximum new positions per day (default: 2)
- Position sizing based on portfolio percentage (default: 10% each)
- Minimum cash reserve (default: 10%)
- Stop-loss and take-profit orders on all positions

## Configuration Parameters

All parameters can be configured via environment variables:

| Parameter           | Default | Description                     |
| ------------------- | ------- | ------------------------------- |
| `PAPER_TRADE`       | True    | Use paper trading account       |
| `MAX_POSITIONS`     | 10      | Maximum total positions         |
| `MAX_NEW_POSITIONS` | 2       | Max new positions per day       |
| `POSITION_SIZE_PCT` | 0.1     | Position size as % of portfolio |
| `MIN_CASH_PCT`      | 0.1     | Minimum cash reserve            |
| `STOP_LOSS_PCT`     | 0.05    | Stop loss percentage            |
| `TAKE_PROFIT_PCT`   | 0.15    | Take profit percentage          |
| `MAX_HOLD_DAYS`     | 30      | Maximum holding period          |
| `BACKTEST_MONTHS`   | 6       | Backtest lookback period        |
| `MIN_VOLUME`        | 1000000 | Minimum daily volume filter     |
| `MIN_PRICE`         | 15.0    | Minimum stock price             |
| `MAX_PRICE`         | 200.0   | Maximum stock price             |

## Data Storage

The application uses Google Cloud Storage for persistence, with environment-prefixed paths:

- `dev/Backtests`, `qa/Backtests`, `prod/Backtests`
- `dev/Positions`, `qa/Positions`, `prod/Positions`
- `dev/Metadata`, `qa/Metadata`, `prod/Metadata`
- `dev/trades`, `qa/trades`, `prod/trades`

### Write Pattern (New File vs Append)

| Storage Area | File Pattern | Write Behavior | Cadence |
| ------------ | ------------ | -------------- | ------- |
| `Backtests/` | `backtest_results_YYYYMMDD_HHMMSS.csv` | New file each save (overwrite only if same exact timestamp is reused) | Usually when a fresh backtest is run; cached files can be reused for up to 24 hours |
| `Positions/` | `positions_YYYYMMDD_HHMMSS.csv` | New file each save (snapshot) | Saved at the end of each trading session (non-dry-run), so potentially multiple files per day |
| `Metadata/` | `metadata.csv` | Appended row-by-row (read existing CSV, concat, write back) | One appended row per full-cycle run that reaches metadata save |

Note: Files are not strictly "one per day." Timestamps are second-level, so multiple files can be created in the same day for backtests and positions.

### Backtests (`backtest_results_*.csv`)

Produced from optimized `BacktestResult` records and used later for opportunity selection and trading.

| Column | Description | Used In Later Steps? |
| ------ | ----------- | -------------------- |
| `symbol` | Ticker for the strategy result. | Yes. Used to fetch live RSI/price and place potential orders. |
| `rsi_period` | RSI lookback period used during optimization. | Yes. Used when recalculating live RSI for entries. |
| `rsi_lower` | RSI buy threshold from the optimized strategy. | Yes. Compared against live RSI for entry checks. |
| `rsi_upper` | RSI sell threshold from the optimized strategy. | Yes. Stored into positions and later used for exit price logic. |
| `total_return` | Strategy return over the backtest window. | Yes. Carried into `TradingOpportunity.backtest_return` for ranking/visibility. |
| `buy_and_hold_return` | Benchmark return over same window. | No direct runtime use after persistence (analytics/reference). |
| `alpha` | `total_return - buy_and_hold_return`. | Yes. Used for sorting/filtering opportunities and metadata context. |
| `num_trades` | Number of completed historical trades in the backtest. | Yes. Used to filter low-sample opportunities. |
| `win_rate` | Fraction of winning trades in the backtest. | Yes. Used to filter opportunities by minimum win-rate rules. |
| `avg_trade_duration` | Mean trade duration (days). | No direct runtime use after persistence (analytics/reference). |
| `max_drawdown` | Maximum observed drawdown in backtest equity curve. | No direct runtime use after persistence (analytics/reference). |
| `sharpe_ratio` | Risk-adjusted return metric. | No direct runtime use after persistence (analytics/reference). |
| `profitable` | Boolean indicating positive strategy return. | No direct runtime use after persistence (already applied in filtering before/after optimization). |
| `current_rsi` | RSI at backtest time (snapshot field). | No direct runtime use for entries; live RSI is recalculated during trading. |

### Positions (`positions_*.csv`)

Saved from the in-memory `Position` list as session snapshots. The latest file is loaded and reconciled with broker positions.

| Column | Description | Used In Later Steps? |
| ------ | ----------- | -------------------- |
| `symbol` | Open position symbol. | Yes. Primary key for reconciliation, order updates, and filtering duplicate entries. |
| `shares` | Position size in shares. | Yes. Used when submitting OCO sell/update orders and reconciliation. |
| `entry_price` | Fill/entry price for the position. | Yes. Used in stop-loss/take-profit calculations and reconciliation. |
| `current_price` | Last known price for the position snapshot. | Yes. Updated during reconciliation and used in portfolio reporting. |
| `current_rsi` | RSI captured around entry/update. | Limited. Mostly informational in current flow. |
| `entry_date` | Position entry timestamp. | Limited. Persisted for tracking/history; not heavily used in live decision logic. |
| `rsi_period` | RSI period attached to this position's strategy. | Yes. Used to compute target exit pricing logic. |
| `rsi_lower` | Entry threshold for this position's strategy. | Limited. Mainly historical context after entry. |
| `rsi_upper` | Exit threshold for this position's strategy. | Yes. Used in RSI-derived target price calculations for exits. |
| `alpha` | Strategy alpha at time of entry selection. | Limited. Informational/context field in current flow. |
| `stop_loss_price` | Current stop-loss level for the position. | Yes. Used when placing/updating OCO orders. |
| `take_profit_price` | Current take-profit level for the position. | Yes. Used when placing/updating OCO orders. |
| `exit_price` | Realized exit price when a position is closed. | Yes. Used for realized P/L tracking in persisted snapshots. |
| `realized_return` | Calculated realized return for closed positions: `(exit_price - entry_price) / entry_price`. | Yes. Used for post-trade performance analysis in position history. |
| `closed` | Position state flag. | Yes. Used to filter active vs closed records when loading latest snapshots. |

### Metadata (`metadata.csv`)

Session-level audit log. This is the only storage artifact that is appended (new row per run).

| Column | Description | Used In Later Steps? |
| ------ | ----------- | -------------------- |
| `timestamp` | Run timestamp added at save time (`YYYYMMDD_HHMMSS`). | No direct runtime use; audit/log key. |
| `start_time` | Run start datetime from orchestrator. | No direct runtime use; observability. |
| `end_time` | Run end datetime in session metadata. | No direct runtime use; observability. |
| `config` | Serialized configuration dictionary used for the run. | No direct runtime use; reproducibility/audit. |
| `portfolio_value` | Account equity captured for the run. | No direct runtime use; performance tracking. |
| `results_summary` | Serialized run summary dictionary. | No direct runtime use; audit/debug. |
| `backtest_count` | Number of backtest results used in run. | No direct runtime use; run diagnostics. |
| `long_market_value` | Account long market value snapshot. | No direct runtime use; diagnostics. |
| `short_market_value` | Account short market value snapshot. | No direct runtime use; diagnostics. |
| `dry_run` | Whether the run was in dry-run mode. | No direct runtime use; audit/debug. |
| `trading_timestamp` | Timestamp from trading session summary. | No direct runtime use; audit/debug. |
| `trading_opportunities_found` | Count of opportunities identified. | No direct runtime use; diagnostics. |
| `trading_new_positions` | Count of new positions opened. | No direct runtime use; diagnostics. |
| `trading_orders_placed` | Count of orders submitted. | No direct runtime use; diagnostics. |
| `trading_positions_exited` | Count of exited positions (if tracked). | No direct runtime use; diagnostics. |
| `trading_errors` | Serialized list of session errors. | No direct runtime use; diagnostics. |
| `trading_dry_run` | Trading-engine dry-run flag. | No direct runtime use; diagnostics. |

Note: Metadata columns can expand over time if additional keys are added to session metadata or trading summary.

### Trades (Runtime Only)

Trade-level history is still consolidated into a DataFrame during optimization, but it is no longer saved to cloud storage.

| Column | Description | Used In Later Steps? |
| ------ | ----------- | -------------------- |
| `symbol` | Ticker for the historical trade. | No direct runtime use; analytics/debug. |
| `rsi_period` | RSI period used by the strategy that produced the trade. | No direct runtime use; analytics/debug. |
| `rsi_lower` | Strategy buy threshold. | No direct runtime use; analytics/debug. |
| `rsi_upper` | Strategy sell threshold. | No direct runtime use; analytics/debug. |
| `entry_date_est` | Entry timestamp converted to US/Eastern display string. | No direct runtime use; analytics/debug. |
| `entry_price` | Historical entry price. | No direct runtime use; analytics/debug. |
| `exit_date_est` | Exit timestamp converted to US/Eastern display string. | No direct runtime use; analytics/debug. |
| `exit_price` | Historical exit price. | No direct runtime use; analytics/debug. |
| `return` | Trade-level return. | No direct runtime use; analytics/debug. |
| `duration` | Trade duration in days. | No direct runtime use; analytics/debug. |

## Logging

Comprehensive logging is implemented throughout:

- Daily log files in `logs/` directory
- Configurable log levels
- Structured logging for key events
- Error tracking and debugging information

## Testing

The `test_example.py` script provides:

- Configuration validation
- Data provider connectivity tests
- Strategy backtesting examples
- Utility function verification

## Future Enhancements

Potential improvements identified from the legacy TODO list:

1. **Advanced Optimization**

   - Machine learning-based parameter optimization
   - Walk-forward analysis
   - Multi-objective optimization (return vs risk)

2. **Additional Indicators**

   - Moving averages
   - Bollinger Bands
   - MACD
   - Volume-based indicators

3. **Portfolio Management**

   - Risk-adjusted position sizing
   - Sector diversification
   - Correlation-based position limits

4. **Real-time Features**

   - Intraday trading signals
   - Real-time RSI monitoring
   - Dynamic stop-loss adjustment

5. **Performance Analytics**
   - Advanced performance attribution
   - Benchmark comparison
   - Risk metrics dashboard

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure JSON files are in the correct format and location
2. **Rate Limiting**: The app includes built-in rate limiting, but increase delays if needed
3. **Data Issues**: Check market hours and ensure stocks are tradable
4. **Cloud Storage**: Verify GCS bucket exists and credentials are correct

### Debug Mode

Run with debug logging to see detailed execution:

```bash
python main.py --log-level DEBUG
```
