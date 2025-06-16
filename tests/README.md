# Trading Algorithm Unit Tests

This directory contains comprehensive unit tests for all modules in the trading algorithm project.

## Test Structure

### Individual Module Tests
- `test_positions_manager.py` - Tests for position management functionality
- `test_config.py` - Tests for configuration loading and validation
- `test_utils.py` - Tests for utility functions and helpers
- `test_strategy.py` - Tests for trading strategy and backtesting logic
- `test_data_provider.py` - Tests for market data retrieval and processing
- `test_cloud_storage.py` - Tests for Google Cloud Storage operations
- `test_trading_engine.py` - Tests for order execution and portfolio management
- `test_main.py` - Tests for main application logic and workflow
- `test_integration.py` - Integration tests for complete system workflows

### Test Runner
- `run_tests.py` - Comprehensive test runner for all modules

## Running Tests

### Prerequisites
1. Make sure you're in the project root directory
2. Activate your virtual environment: `source venv/bin/activate`
3. Install requirements: `pip install -r requirements.txt`
4. Set up environment variables (create `.env` file with required API keys)

### Run All Tests
```bash
cd tests
python run_tests.py
```

### Run Specific Test Module
```bash
cd tests
python run_tests.py positions_manager
python run_tests.py strategy
python run_tests.py trading_engine
# etc.
```

### Run Individual Test File
```bash
cd tests
python test_positions_manager.py
python test_strategy.py
# etc.
```

## Test Coverage

### Positions Manager (`test_positions_manager.py`)
- ✅ Position dataclass creation and validation
- ✅ PositionsManager initialization
- ✅ Alpaca positions retrieval (mocked)
- ✅ Google Cloud Storage positions retrieval
- ✅ Position reconciliation logic
- ✅ DataFrame conversion functionality
- ✅ Error handling for various scenarios

### Configuration (`test_config.py`)  
- ✅ Environment variable loading
- ✅ JSON configuration file parsing
- ✅ Multi-environment support (dev/qa/prod)
- ✅ Alpaca API configuration
- ✅ Missing variable validation
- ✅ Invalid configuration handling

### Utilities (`test_utils.py`)
- ✅ Logging setup functionality
- ✅ Trading day detection
- ✅ Market hours validation
- ✅ RSI calculation
- ✅ Portfolio metrics calculation
- ✅ Currency and percentage formatting
- ✅ Symbol validation
- ✅ Progress indicator functionality

### Strategy (`test_strategy.py`)
- ✅ BacktestResult dataclass functionality
- ✅ RSI strategy initialization and configuration
- ✅ RSI calculation methods
- ✅ Signal generation logic
- ✅ Returns calculation
- ✅ Backtest execution with various data scenarios
- ✅ Strategy backtester parallel processing
- ✅ Parameter optimization

### Data Provider (`test_data_provider.py`)
- ✅ BarData dataclass functionality
- ✅ DataProvider initialization with/without credentials
- ✅ Historical data retrieval from Alpaca API
- ✅ Multiple stocks data retrieval
- ✅ Current price fetching
- ✅ Market snapshot functionality
- ✅ Technical indicators (RSI, Moving Average, Bollinger Bands, MACD, Stochastic)
- ✅ Error handling for API failures

### Cloud Storage (`test_cloud_storage.py`)
- ✅ CloudStorage initialization and authentication
- ✅ Float rounding utility functions
- ✅ Backtest results upload/download
- ✅ Position entries upload/download
- ✅ File listing operations
- ✅ Data serialization (JSON/CSV)
- ✅ Old file cleanup functionality
- ✅ Error handling for storage operations

### Trading Engine (`test_trading_engine.py`)
- ✅ TradingOpportunity dataclass functionality
- ✅ TradingEngine initialization
- ✅ Dry run vs live trading modes
- ✅ Current positions retrieval with strategy metadata
- ✅ Buying opportunity identification
- ✅ Position sizing calculations
- ✅ Order placement (buy/sell) in both modes
- ✅ Exit condition monitoring
- ✅ Portfolio value retrieval
- ✅ Position metadata management

### Main Application (`test_main.py`)
- ✅ Main execution flow control
- ✅ Trading hours and market day validation
- ✅ Backtesting workflow execution
- ✅ Trade execution workflow
- ✅ Exit signal monitoring
- ✅ Configuration validation
- ✅ Exception handling

### Integration Tests (`test_integration.py`)
- ✅ Complete data flow testing
- ✅ Module import validation
- ✅ Error handling across system
- ✅ Risk management integration
- ✅ Backtesting workflow integration
- ✅ Cloud storage integration
- ✅ Logging integration
- ✅ Performance testing scenarios

## Test Features

### Mocking and Patching
All tests use comprehensive mocking to isolate units and avoid external dependencies:
- Alpaca API calls are mocked
- Google Cloud Storage operations are mocked
- File system operations are mocked
- Network requests are mocked
- Environment variables are patched

### Data Generation
Tests include realistic data generation for:
- Historical price data
- RSI calculations
- Portfolio metrics
- Market snapshots
- Backtest results

### Error Scenarios
Tests cover various error scenarios:
- API failures and timeouts
- Invalid data inputs
- Missing configuration
- File system errors
- Network connectivity issues

### Edge Cases
Tests include edge case handling:
- Empty datasets
- Insufficient historical data
- Invalid symbols
- Market holidays
- After-hours trading

## Expected Test Results

### Import Errors
Many tests will show import errors initially because:
1. The test files expect to import the actual modules
2. Some modules may have missing dependencies
3. Environment variables may not be set up

These import errors are expected and can be resolved by:
1. Installing all required dependencies
2. Setting up proper environment variables
3. Ensuring the project structure is correct

### Successful Test Execution
Once dependencies are properly configured, you should see:
- All unit tests passing
- Comprehensive coverage reports
- Clear success/failure indicators
- Detailed test output for debugging

## Adding New Tests

When adding new functionality to the trading algorithm:

1. **Create corresponding test files** for new modules
2. **Add test cases** for new functions and classes
3. **Include error handling tests** for new code paths
4. **Update integration tests** if the change affects system workflow
5. **Mock external dependencies** to keep tests isolated
6. **Follow naming conventions**: `test_<module_name>.py`
7. **Update this README** with new test descriptions

## Test Best Practices

- ✅ Use descriptive test method names
- ✅ Include docstrings explaining what each test validates
- ✅ Mock all external dependencies
- ✅ Test both success and failure scenarios
- ✅ Use setUp/tearDown methods for common test fixtures
- ✅ Assert specific expected values, not just truthy/falsy
- ✅ Test edge cases and boundary conditions
- ✅ Keep tests focused on single functionality
- ✅ Use subTest for parametrized testing

## Troubleshooting

### Common Issues

1. **Import Errors**: Install requirements and check Python path
2. **Mock Failures**: Ensure mock patches match actual module structure
3. **Assertion Errors**: Check expected vs actual values in test output
4. **Environment Issues**: Verify virtual environment activation
5. **Path Issues**: Run tests from correct directory

### Debug Tips

1. Add print statements to understand test flow
2. Use `python -m pytest -v` for verbose output
3. Run individual test methods: `python -m unittest test_module.TestClass.test_method`
4. Check mock call arguments: `mock_object.assert_called_with(...)`
5. Use debugger: `import pdb; pdb.set_trace()`

This comprehensive test suite ensures the reliability, correctness, and maintainability of the trading algorithm codebase.
