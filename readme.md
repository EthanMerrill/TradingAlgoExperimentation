# Trading Algorithm

An RSI-based trading algorithm for US common stocks, with pluggable storage
backends (GCS / Postgres), walk-forward validation, and short-selling support.

## Architecture Overview

```
app/
├── config.py              Configuration & env vars (dev/qa/prod JSON)
├── data_provider.py        Alpaca API — OHLCV, positions, orders, snapshots
├── strategy.py             Vectorized RSI backtester & BacktestResult
├── optimizer.py            Grid-search optimization across stock universe
├── walk_forward.py         Walk-forward validation (IS/OOS windows)
├── trading_engine.py       Order execution, position sizing, OCO orders
├── positions.py            Position reconciliation (GCS/Postgres ↔ Alpaca)
├── zscore.py               Cross-symbol Z-score normalization
├── utils.py                Trading calendar, logging, date helpers
├── health_server.py        Lightweight HTTP health check server
├── main.py                 Orchestrator — full trading cycle runner
└── storage/
    ├── __init__.py          Shared singleton (auto-selects GCS or Postgres)
    ├── backend.py           Abstract StorageBackend ABC + factory
    ├── gcs.py               GCS backend — CSV blobs (implements StorageBackend)
    └── postgres.py          Postgres backend — relational tables (implements StorageBackend)
```

### Key Improvements Over Legacy Code

- **Modern Alpaca SDK** (`alpaca-py`) with async/await patterns
- **Vectorized backtesting** — replaced Backtrader, much faster
- **Pluggable storage** — swap between GCS and Postgres with a config toggle
- **Walk-forward validation** — IS/OOS window evaluation to reduce overfitting
- **Short selling** — RSI-based short signals with leverage caps
- **Cross-symbol Z-scores** — composite ranking across alpha, Sharpe, and Calmar
- **Type hints throughout**, comprehensive logging, modular design

---

## Setup

### Prerequisites

- **Python 3.13+**
- **TA-Lib** (C library — `brew install ta-lib` on macOS; `apt-get install ta-lib` on Linux)
- **Alpaca API keys** — free paper trading at [alpaca.markets](https://alpaca.markets)

### 1. Create & activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install TA-Lib

```bash
# macOS
brew install ta-lib

# Linux
sudo apt-get install ta-lib
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file in the project root:

```bash
# Required — environment & Alpaca keys
export ENVIRONMENT=dev
export ALPACA_DEV_PAPER_KEY=your_paper_key
export ALPACA_DEV_PAPER_SECRET=your_paper_secret

# Optional — Google Cloud Storage (default backend)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
export GCS_BUCKET_NAME=trading-algo-data

# Optional — Postgres (alternative backend, toggle via config JSON)
export DATABASE_URL=postgresql://user:password@host:5432/dbname

# Optional — container lifecycle & health
export KEEP_ALIVE=true       # keep container alive after cycle
export HEALTH_PORT=8080      # health check server port
```

| Environment | Key Variable | Secret Variable |
|-------------|-------------|-----------------|
| `dev` | `ALPACA_DEV_PAPER_KEY` | `ALPACA_DEV_PAPER_SECRET` |
| `qa` | `ALPACA_QA_PAPER_KEY` | `ALPACA_QA_PAPER_SECRET` |
| `prod` | `ALPACA_LIVE_KEY` | `ALPACA_LIVE_SECRET` |

---

## Usage

```bash
cd app
python main.py
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--dry-run` | Analysis only — no orders placed |
| `--test-mode` | Limited stock universe for fast validation |
| `--force-backtest` | Force new backtests (ignore 24h cache) |
| `--paper-trading` | Explicitly enable paper trading mode |
| `--log-level DEBUG` | Verbose logging |

```bash
python main.py --test-mode --dry-run --log-level DEBUG
```

### Makefile

```bash
make run           # python app/main.py
make test          # run full test suite
make install       # pip install -r requirements.txt
make lint          # flake8
make clean         # remove caches & logs
make docker-build  # build Docker image
make docker-run    # run in Docker
```

### Re-running the Trading Cycle (KEEP_ALIVE mode)

When running in a container with `KEEP_ALIVE=true`, the health/dashboard server stays
alive after the initial cycle completes. Cron jobs or schedulers should **not** spawn a
new `python app/main.py` process — that would fail with `Address already in use`
because the server port is already bound.

Instead, trigger a new cycle via HTTP:

```bash
# Trigger a new trading cycle (use cached backtest results)
curl -X POST -u admin:$DASHBOARD_PASSWORD http://localhost:8080/api/run-cycle

# Force fresh backtests (skip 24h cache)
curl -X POST -u admin:$DASHBOARD_PASSWORD "http://localhost:8080/api/run-cycle?force_backtest=true"

# Dry run only (analysis, no orders)
curl -X POST -u admin:$DASHBOARD_PASSWORD "http://localhost:8080/api/run-cycle?dry_run=true"

# Test mode with limited universe
curl -X POST -u admin:$DASHBOARD_PASSWORD "http://localhost:8080/api/run-cycle?test_mode=true"
```

The endpoint returns `200` on success, or `409` if a cycle is already in progress.

Check cycle status and last-run results at `GET /health` (no auth required):

```bash
curl http://localhost:8080/health
```

Example cron entry (weekdays, one hour after market open):

```
30 10 * * 1-5 curl -X POST -u admin:$DASHBOARD_PASSWORD http://localhost:8080/api/run-cycle
```

---

## Running Tests

### Prerequisites

1. Activate your virtual environment: `source .venv/bin/activate`
2. Install requirements: `pip install -r requirements.txt`
3. Set up environment variables (create `.env` file with required API keys)

### Run All Tests

```bash
# via pytest
python -m pytest tests/ -v

# Custom runner
python tests/run_tests.py
```

### Run Specific Tests

```bash
# Single file via pytest
python -m pytest tests/test_strategy.py -v

# Via custom runner (by module name)
python tests/run_tests.py strategy
python tests/run_tests.py trading_engine
python tests/run_tests.py positions_manager

# Run individual file directly
python tests/test_positions_manager.py
```

### Test Suites

| Test File | Coverage |
|-----------|----------|
| `test_cloud_storage.py` | GCS backend — init, upload/download, file listing, error handling |
| `test_postgres_storage.py` | Postgres backend — all 9 StorageBackend methods, date parsing, ABC compliance |
| `test_config.py` | Config loading, env vars, multi-environment, invalid JSON |
| `test_data_provider.py` | Alpaca API — bars, positions, orders, snapshots, technical indicators |
| `test_strategy.py` | RSI backtesting, BacktestResult, signal generation, parameter optimization |
| `test_trading_engine.py` | Order placement, position sizing, OCO orders, dry-run mode, short selling |
| `test_positions_manager.py` | Position reconciliation (cloud ↔ broker), open/close logic, enrichment |
| `test_positions_reconcile_regression.py` | Regression tests — reconciliation always returns a list |
| `test_utils.py` | Trading calendar, logging, date parsing |
| `test_main.py` | Full orchestration — backtest → filter → trade → save cycle |
| `test_integration.py` | End-to-end data flow, error handling, risk management, cloud storage |
| `test_order_integration.py` | **Live Alpaca paper-trading** — order placement, cancellation, liquidation, storage validation |

### Order Integration Tests (`test_order_integration.py`)

These tests connect to the **dev Alpaca paper trading** account and validate the
production order-placement code paths (`TradingEngine.place_buy_order()`,
`TradingEngine.place_market_sell_order()`) against Alpaca's API ground truth.

**Prerequisites:**

```bash
export ALPACA_DEV_PAPER_KEY=your_dev_paper_key
export ALPACA_DEV_PAPER_SECRET=your_dev_paper_secret
# Optional: override the test symbol (default: F — Ford, ~$12)
export TEST_SYMBOL=F
```

**Run:**

```bash
# Via make (auto-loads credentials from .env if present)
make test-integration

# Or manually
export $(grep -v '^#' .env | grep ALPACA_DEV_PAPER | xargs)
.venv/bin/python -m pytest tests/test_order_integration.py -v
```

Tests are automatically skipped when credentials are missing.

| Test Class | What It Validates |
|-----------|-------------------|
| `TestAlpacaConnectivity` | Paper trading client reaches Alpaca; account fields present |
| `TestPlaceBuyOrder` | `engine.place_buy_order()` → order visible in Alpaca; cleanup via cancel or `place_market_sell_order()` liquidation |
| `TestFullOrderLifecycle` | Self-contained end-to-end: place → verify → cleanup |
| `TestStorageValidation` | Spies on `open_position()` to capture the `Position` object, then cross-validates against Alpaca's order API, positions API, and `normalize_position_for_save()` output (all `POSITION_FIELDS` present) |

**Design:**

- 1 share per order on `F` (or `$TEST_SYMBOL`) — minimal paper account impact
- All tests clean up after themselves (cancel unfilled orders, liquidate filled positions)
- Handles variable order statuses based on market open/closed
- Not part of `make test` / `tests/run_tests.py` — intended for explicit CI or manual runs

### Test Features

- **Mocking & Patching** — All tests mock external dependencies (Alpaca API, GCS, filesystem, network) to keep units isolated
- **Realistic Data Generation** — Historical prices, RSI values, portfolio metrics, market snapshots, backtest results
- **Error Scenarios** — API failures, timeouts, invalid inputs, missing config, filesystem errors, network issues
- **Edge Cases** — Empty datasets, insufficient history, invalid symbols, market holidays, after-hours trading

### Adding New Tests

1. Create `test_<module_name>.py` for new modules
2. Add test cases for new functions and classes
3. Include error handling tests for new code paths
4. Update integration tests if the change affects system workflow
5. Mock external dependencies to keep tests isolated

### Test Best Practices

- Use descriptive test method names with docstrings
- Mock all external dependencies
- Test both success and failure scenarios
- Use `setUp`/`tearDown` for common fixtures
- Assert specific expected values, not just truthy/falsy
- Test edge cases and boundary conditions
- Keep tests focused on a single functionality
- Use `subTest` for parametrized testing

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Import errors | Install requirements and check Python path |
| Mock failures | Ensure mock patches match actual module structure |
| Assertion errors | Check expected vs actual values in test output |
| Environment issues | Verify virtual environment activation |
| Path issues | Run tests from project root directory |

For debugging: use `python -m pytest -v` for verbose output, run individual test methods with `python -m unittest test_module.TestClass.test_method`, or insert `import pdb; pdb.set_trace()` to drop into a debugger.

---

## Strategy Details

### RSI Strategy

- **Entry (long)**: RSI crosses below the optimized lower threshold
- **Entry (short)**: RSI crosses above the optimized upper threshold
- **Exit signals**:
  - RSI crosses above/below the opposite threshold
  - Maximum holding period reached
  - Stop-loss triggered (configurable %)
  - Take-profit / cover target triggered (configurable %)

### Optimization

- Grid search across RSI periods, lower bounds, and upper bounds
- Optional two-stage optimization (coarse + fine tuning)
- Walk-forward validation splits data into IS/OOS windows
- Evaluates on alpha (excess return vs buy-and-hold), win rate, Sharpe, Calmar
- Cross-symbol Z-score composite ranking

### Risk Management

- Maximum positions: 10 (default)
- Maximum new positions per day: 2 (default)
- Position sizing: 10% of equity each (default)
- Minimum cash reserve: 10% (default)
- Stop-loss: 5% (default)
- Take-profit: 15% (default)
- Max hold days: 30 (default)
- Short selling with leverage cap (`max_short_long_ratio`: 0.30 default)

---

## Configuration

All parameters live in `config/{dev,qa,prod}.json`:

### Trading

| Parameter | Default | Description |
|-----------|---------|-------------|
| `paper_trade` | `true` | Use paper trading account |
| `max_positions` | 10 | Maximum total positions |
| `max_new_positions` | 2 | Max new positions per day |
| `position_size_pct` | 0.1 | Position size as % of equity |
| `min_cash_pct` | 0.1 | Minimum cash reserve |
| `stop_loss_pct` | 0.05 | Stop-loss percentage |
| `take_profit_pct` | 0.15 | Take-profit percentage |
| `max_hold_days` | 15–30 | Maximum holding period |
| `min_win_rate` | 0.7–0.8 | Minimum backtest win rate |
| `enable_short_selling` | `true`/`false` | Enable short selling |
| `max_short_long_ratio` | 0.3 | Max short notional / equity |

### Backtesting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_cash` | 10000 | Initial cash for backtest |
| `months` | 6–12 | Lookback period |

### RSI Optimization

| Parameter | Description |
|-----------|-------------|
| `fine_tuning_enabled` | Enable two-stage (coarse + fine) optimization |
| `period_range` | `{start, stop, step}` for RSI period grid |
| `lower_range` | `{start, stop, step}` for RSI lower bound grid |
| `upper_range` | `{start, stop, step}` for RSI upper bound grid |

### Walk-Forward

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true`/`false` | Enable walk-forward validation |
| `is_months` | 4–6 | In-sample window months |
| `oos_months` | 2 | Out-of-sample window months |
| `step_months` | 2 | Step size between windows |
| `min_windows` | 3 | Minimum required windows |

### Data Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_volume` | 200K–1M | Minimum daily volume |
| `max_volume` | varies | Maximum daily volume |
| `min_price` | 5–15 | Minimum stock price |
| `max_price` | 200–350 | Maximum stock price |
| `min_market_cap` | 100M–500M | Minimum market cap |
| `max_market_cap` | varies | Maximum market cap |

### Storage

| Parameter | Default | Description |
|-----------|---------|-------------|
| `storage_backend` | `"gcs"` | `"gcs"` or `"postgres"` |

---

## Data Persistence (Storage Backend)

The app uses a pluggable storage backend system. Toggle in `config/{env}.json`:

```json
"storage_backend": "gcs"       // Google Cloud Storage (default)
"storage_backend": "postgres"  // Postgres database
```

All persistence goes through the `StorageBackend` ABC (`storage/backend.py`).
A shared singleton (`storage/__init__.py`) auto-selects the correct backend at startup.
Both backends expose 9 identical methods — callers never know which is active.

### GCS (`storage/gcs.py`)

CSV blobs under environment-prefixed paths in your GCS bucket:

| Path | Content | Write Pattern |
|------|---------|---------------|
| `{env}/Backtests/backtest_results_{timestamp}.csv` | Optimized BacktestResult records | New file per backtest run |
| `{env}/Positions/positions_{timestamp}.csv` | Position snapshot | New file per session |
| `{env}/Metadata/metadata.csv` | Session metadata | Append row per cycle |

### Postgres (`storage/postgres.py`)

Set `DATABASE_URL` + toggle the config. Tables auto-create on first use:

| Table | Equivalent GCS Path | Key Columns |
|-------|---------------------|-------------|
| `backtest_results` | `{env}/Backtests/*.csv` | `run_timestamp`, `environment`, all 17 BacktestResult fields |
| `position_snapshots` | `{env}/Positions/*.csv` | `snapshot_timestamp`, `environment`, all 17 Position fields |
| `session_metadata` | `{env}/Metadata/metadata.csv` | `timestamp`, `environment`, `metadata` (JSONB) |

All tables have an `environment` column — dev/qa/prod data stays isolated.

### Backtest Results Schema

| Column | Used In Trading? |
|--------|:---:|
| `symbol` | ✅ Fetch live RSI/price, place orders |
| `rsi_period`, `rsi_lower`, `rsi_upper` | ✅ Live RSI recalculation & entry checks |
| `total_return` | ✅ `TradingOpportunity.backtest_return` |
| `alpha` | ✅ Sort/filter opportunities |
| `num_trades`, `win_rate` | ✅ Filter low-sample / low-win opportunities |
| `buy_and_hold_return`, `avg_trade_duration`, `max_drawdown`, `sharpe_ratio` | Reference / analytics only |
| `calmar_ratio`, `composite_score` | Composite ranking |
| `profitable`, `current_rsi` | Applied during filtering, not runtime |

---

## Algorithm Workflow

1. **Position Check** — Reconcile cloud/broker positions, update prices
2. **Exit Signals** — Check existing positions for exit conditions
3. **Universe Selection** — Filter stocks by price, volume, market cap
4. **Backtesting** — Grid-optimize RSI strategies across the universe (optionally with walk-forward validation)
5. **Entry Signals** — Identify long & short opportunities based on live RSI vs optimized thresholds
6. **Position Sizing** — Calculate shares based on risk parameters & leverage caps
7. **Order Execution** — Place bracket/OCO orders (entry + stop-loss + take-profit)
8. **Data Persistence** — Save results, positions, and metadata to the active storage backend

## Deployment

### GCP Cloud Run

**Do not** set `KEEP_ALIVE`. The container exits after each cycle, allowing scale-to-zero.

### Coolify

Set `KEEP_ALIVE=true` to keep the container running after the cycle for SSH access.

| Platform | `KEEP_ALIVE` | Behavior |
|----------|:---:|---|
| GCP Cloud Run | unset / `false` | Exit → scale to zero |
| Coolify | `true` | Idle indefinitely → SSH stays open |

### Health Check

The health server listens on port **8080** by default. Override with `HEALTH_PORT` if needed.

---

## Todo

- [ ] Clean up trading engine (limits/stops calculated multiple times)
- [ ] Improve portfolio allocation methodology
- [ ] Check that NYSE volumes are not doubled
- [ ] Place OCO order types at day start for existing positions
- [ ] Pass 10% of equity to backtester for more accurate position sizing
- [ ] Improve optimization strategies (look at win rate, not just ROI)
- [ ] Add better position reconciliation for manual orders
