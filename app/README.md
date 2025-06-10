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

The application uses Google Cloud Storage for persistence:

- **Backtests/**: Strategy optimization results
- **Positions/**: Historical position snapshots
- **Metadata/**: Algorithm run metadata and configuration

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
