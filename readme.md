# Basic Algo

Some Experimenting with a trading algorithm for US Common Stock.

# Running Locally

## Prerequisites

- **Python 3.13+**
- **TA-Lib** (C library — install via `brew install ta-lib` on macOS)
- **Alpaca API keys** — sign up for free paper trading at [alpaca.markets](https://alpaca.markets)

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the TA-Lib C library (macOS)

```bash
brew install ta-lib
```

> On Linux: `apt-get install ta-lib` or build from source. See [TA-Lib docs](https://github.com/TA-Lib/ta-lib).

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

The app reads configuration from environment variables. Create a `.env` file in the project root (or export them directly):

```bash
# Required: Environment and corresponding Alpaca API keys
export ENVIRONMENT=dev                              # dev, qa, or prod
export ALPACA_DEV_PAPER_KEY=your_paper_key
export ALPACA_DEV_PAPER_SECRET=your_paper_secret

# Optional: Google Cloud Storage (for data persistence)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Optional: Keep container alive after completion (for SSH/debugging in Coolify)
export KEEP_ALIVE=true     # default: false
```

Environment-specific API key pairs:
| Environment | Key Variable | Secret Variable |
|---|---|---|
| `dev` | `ALPACA_DEV_PAPER_KEY` | `ALPACA_DEV_PAPER_SECRET` |
| `qa` | `ALPACA_QA_PAPER_KEY` | `ALPACA_QA_PAPER_SECRET` |
| `prod` | `ALPACA_LIVE_KEY` | `ALPACA_LIVE_SECRET` |

### 5. Run the algorithm

```bash
cd app
python main.py
```

### Command-line options

| Flag | Description |
|---|---|
| `--dry-run` | Analysis only — no orders placed |
| `--test-mode` | Limited stock universe for fast validation |
| `--force-backtest` | Forces new backtests (ignores cached results) |
| `--paper-trading` | Explicitly enables paper trading mode |
| `--log-level DEBUG` | Sets verbose logging |

Example:

```bash
python main.py --test-mode --dry-run --log-level DEBUG
```

You can also use the **Makefile** from the project root:

```bash
make run       # runs `python app/main.py`
```

## Running Tests

### Option A: Using the Makefile

```bash
make test
```

This runs `python tests/run_tests.py`, which auto-discovers all `test_*.py` files.

### Option B: Using pytest directly

```bash
python -m pytest tests/
```

Run a specific test file:

```bash
python -m pytest tests/test_strategy.py -v
```

### Option C: The custom test runner

```bash
python tests/run_tests.py
```

### Available test suites

| Test File | What it tests |
|---|---|
| `test_config.py` | Configuration loading and environment variables |
| `test_data_provider.py` | Alpaca data fetching and API calls |
| `test_strategy.py` | RSI strategy backtesting and optimization |
| `test_trading_engine.py` | Order execution and position management |
| `test_positions_manager.py` | Position reconciliation logic |
| `test_positions_reconcile_regression.py` | Regression tests for reconciliation |
| `test_cloud_storage.py` | Google Cloud Storage persistence |
| `test_utils.py` | Utility functions (calendar, helpers) |
| `test_main.py` | Full orchestration workflow |
| `test_integration.py` | End-to-end integration tests |

### Other Makefile commands

```bash
make install     # Install Python dependencies
make lint        # Run flake8 linter
make clean       # Remove Python cache and logs
make docker-build  # Build the Docker image
make docker-run    # Run the app in Docker
```

## Deployment

### GCP Cloud Run (default behavior)

On Google Cloud Run, **do not** set `KEEP_ALIVE`. The container exits after each trading cycle, allowing Cloud Run to scale to zero and save costs. Cloud Run will spin up a new container when the next scheduled job triggers.

### Coolify (SSH/debugging)

If you deploy via Coolify and need to SSH into the container (e.g., for debugging or checking logs), set the environment variable `KEEP_ALIVE=true`. After the trading cycle completes, the container will **idle indefinitely** instead of exiting, keeping the SSH connection open.

| Platform | `KEEP_ALIVE` | Container behavior after cycle |
|----------|---------------|-------------------------------|
| GCP Cloud Run | unset or `false` | Exits → scale to zero (saves costs) |
| Coolify | `true` | Sleeps indendently → stays running for SSH |

## Processes Overview:

### Positions

1. Get all previous positions in the portfolio by querying the google cloud datastore. This will return a picked dataframe.
1. Get all current positions in the <a href='alpaca.markets'>alpaca.markets</a> portfolio. Updated the picked dataframe with the new positions data. This updates the close price in the dataframe to the most recent close price.
1. Get cash and long market value amounts. If the cash is greater than 10% of the equity (cash+long market value), run the backtester:

### Backtest

1. Filter universe of all stocks to those with bounded price and volume, plus market-cap filters when available, resulting in a focused tradable set
2. For each security in this universe, backtest RSI Strategies over the past _6 months_. These are backtested with different combinations of `rsi upper bound`, `rsi lower bound`, and `rsi period`. These three parameters are optimized for each security using grid optimization. An improved optimization strategy is on the roadmap. This operation creates a dataframe with all the securities, their optimized parameters, the strategy return, and the return of a buy and hold strategy for the given security.
3. Every 50 strategy generations, or securities run through the backests/parameter optimization, the dataframe is appended to a locally saved version. This is saved in a format called a pickle. This is a way of reducing the amount of ram required by the program.
4. The backtest dataframe and is saved to google cloud storage once all securities have been processed.

### Buying Opportunities

1. Once the Backtester is finished (2-5 hours) the results are filtered using the get entries function. This function creates the alpha column, then filters for positive alpha items (where the ROI is greater than the buy and hold).
2. Next, only profitable strategies are selected. A strategy can be better than buy and hold, but still have lost money.
3. After these filters are finished, the current rsi is calculated for each strategy based the strategies' specified rsi period. If the current RSI is lower than the rsi_lower_bound specified in the strategy, add to the buying opportunities DF
4. Determine how many different shares can be purchased by calculating the amount of cash as a percentage of equity. The portfolio is Equally weighted with 10 positions, so if 20% cash is available, buy 2 new securities. This can also be overridden with a **MAX_NEW_POSITIONS** value, if this is less than the number of securities capable of being purchased with cash, just purchase the [max new positions] number of securities. This was added to limit the amount of shares purchased on a single day. The portfolio allocation methodology could be significantly improved.
5.

### Get Exits

_Positions DF:_
symbol|rsi*period|rsi_lower|rsi_upper|current_rsi|modeled_returns|alpha|entry_date|entry_price|exit_date|exit|price
Backtests DF*
symbol|rsi_period|rsi_lower|rsi_upper|current_rsi|profit|ROI|Buy_and_hold

# New Todo:

- [ ] Clean up the trading engine (limits and stops calculated multiple times)
- [ ] Add short selling strategies
- [x] either use or remove the metadata json file. it may be replaced by the positions csv file.
  - [x] update metadata as a csv file

todo:

- [x] Add a better time estimator, this will be useful as we move to backtesting multiple equities at once.
- [x] Remove all dependency on yfinance
- [x] Make algo built on backtrader not fastquant on top of backtrader. (cut out the middle package!)

- [x] make a function which evaluates and updates trailing stops
- [x] compare current rsi to rsi entry limits in entry calculator function
- [x] function to place buy and sell orders (switch for paper vs non paper trading)
- [x] simple comparator for rsi exit conditions
- [x] Make the Paths Portable: https://docs.python.org/3/library/os.path.html
- [x] optimize ram usage during backtesting

- [x] Clean up the key/path/variable management to use only environ variables.
- [x] Manually update positions df on buy and sell orders.
  - [ ] may want to build more error handling in in the future to handle manual buy and sell orders amongst other things. Just a better way to reconcile strategies and positions
- [x] More complex buy ordering. Limit orders, not market orders
  - [x] Implement oco order on initial position
  - [x] create a class for order types
- [x] Set up containerization
  - [x] Create docker file
  - [x] Make a working build
- [x] Add error handling to polygon request, if it returns a blank set throw error

- [ ] Check that NYSE volumes are not 2x!!!

- [ ] put all variables in one place. and make a log of this data as a sort of metadata file. Add arguments on run, at least argument for PAPER_TRADING
  - stop price multiplier
  - volatility stop multiple
  - rsi_optimizer inputs
  - number of positions
- [x] handle the situation when 10% of portfolio isn't enough to purchase even one share.

- [ ] Check the RMA EMA functions, something is a little off there.

- [ ] Pass 10% of equity to backtester for a more accurate test

- [x] GCP Cloud bucket!
- [ ] add a max time period for trades: https://community.backtrader.com/topic/2150/sell-a-position-after-2-days
  - [x] add to backtests
  - [ ] add to live_trader
- [x] add correlation checking for new positions https://machinelearningmastery.com/gentle-introduction-autocorrelation-partial-autocorrelation/

- [x] fix to make backtester less error prone (problem is in the buy and hold analyzer I think)
  - it was the RSI strategy. safediv was off by default. Replaced with RSI_safe strategy
- [ ] have the script turn off the system when done.
- [ ] place oco order types at day start

- ## Long term Features:
- [x] Use a format better than pickle for long term storage
- [x] CI/CD!! :o https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build
- [ ] improve optimization strategies
  - look at win rate, not just roi: https://backtest-rookies.com/2017/06/11/using-analyzers-backtrader/
- [ ] integrate facebook prophet

pip freeze > requirements.txt
pip install -r requirements.txt

source venv/bin/activate
