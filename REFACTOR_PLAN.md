# Plan: Refactor TradingAlgoExperimentation — Dead Code, Duplication & Structure

**TL;DR** — Seven-phase refactoring to remove ~220 lines of dead code, deduplicate ~600 lines of shared logic, fix 3 critical bugs, and decompose 3 oversized files into focused modules. Each phase is independently verifiable with run_tests.py.

**Executed:** 2026-07-13 through 2026-07-14 on branch `test/dev-branch`.

**Final outcome:** 961 net lines removed (443 added, 1404 deleted). 203 tests pass.

> **Follow-up (2026-09-01):** The multi-strategy framework (Phases A–D of
> [`MULTI_STRATEGY_PLAN.md`](./MULTI_STRATEGY_PLAN.md)) built on top of this refactor:
> `strategy.py` is now a re-export shim over `app/strategies/`, `optimizer.py`
> delegates to per-strategy `Strategy.optimize()`, `trading_engine.py` gained
> strategy-aware dispatch + per-strategy allocation, and a new `bar_engine.py`
> handles intraday strategies. The deferred **Phase 6** engine-split remains
> deferred, but the engine is now partially decomposed along strategy lines
> (registry + strategy-provided live signals) rather than purely by
> price/opportunity/sizer/executor concerns.

---

### Phase 1: Dead Code Removal (safe, no behavioral changes)

Remove 16 truly dead items. None are called by production code.

**Step 1.1 — Delete 8 items with zero callers (no test impact):**
- `utils.py`: delete `calculate_sortino_ratio` (L216-228), `next_trading_day` (L161-170), `previous_trading_day` (L172-180)
- `trading_engine.py`: delete `_get_current_rsi` (L1196-1208), `get_current_positions` (L1141-1149), `_last_position_update` field (L56)
- `config.py`: delete `globalConfig = None` vestigial line (L323)
- `optimizer.py`: delete `PerformanceMetrics` import (L18)

**Step 1.2 — Delete 8 items kept alive only by tests (delete tests alongside):**
- `utils.py`: delete `DataValidator` class + delete `TestDataValidator` in test_utils.py
- `utils.py`: delete `RiskManager` class + delete `TestRiskManager` in test_utils.py
- `utils.py`: delete `format_currency`, `format_percentage` + delete their tests
- `utils.py`: delete `PerformanceMetrics.calculate_sharpe_ratio` (keep calmar_ratio + max_drawdown) + delete test
- `utils.py`: delete `TradingCalendar.is_market_open` + delete its tests (keep `is_trading_day` which IS used)
- `data_provider.py`: delete `BarData` dataclass + delete its test
- `optimizer.py`: delete `_composite_score` + `_composite_score_from_parts` + delete their tests

**Verification:** Run `python tests/run_tests.py` — all tests must pass with deleted dead tests removed.

---

### Phase 2: Fix Critical Bugs

**Step 2.1 — Add `side` field to GCS position saves** (`storage/gcs.py` L297-321)
- Add `"side": getattr(pos, "side", "long")` to the `pos_dict` in `save_positions` — mirror `postgres.py` L370.
- *depends on Phase 1 complete*

**Step 2.2 — Fix broken integration tests** (`tests/test_integration.py`)
- Replace mock of `get_historical_data` with `get_single_stock_bars` (L62, L88)
- Remove empty `pass`-only test bodies in `test_module_imports` (L33-51) and `test_error_handling_integration` (L112-131) — either implement real assertions or delete the tests.
- *depends on Phase 1 complete; parallel with 2.1*

**Step 2.3 — Fix async test methods in test_main.py**
- Change `test_main_execution_during_trading_hours`, `test_run_backtests_function`, `test_execute_trades_function` to use `unittest.IsolatedAsyncioTestCase` so async defs actually execute.
- *depends on Phase 1 complete; parallel with 2.1, 2.2*

**Verification:** `python tests/run_tests.py` — integration tests must assert real behavior; async tests must actually run.

---

### Phase 3: Extract Shared Storage Serialization (~170 lines of GCS/Postgres duplication eliminated)

**Step 3.1 — Create shared serialization helpers in `storage/backend.py`**
- Add `BACKTEST_FIELDS` ordered list constant
- Add `backtest_result_to_dict(result: BacktestResult) -> dict` — converts to flat dict with rounding
- Add `dict_to_backtest_result(d: dict) -> BacktestResult` — reconstructs from dict
- Add `POSITION_FIELDS` ordered list constant  
- Add `normalize_position_for_save(pos) -> dict` — computes exit_price/realized_return, returns dict with all fields including `side`

**Step 3.2 — Refactor `gcs.py` to use shared helpers**
- `save_backtest_results`: call `backtest_result_to_dict` for each result, then build DataFrame & upload CSV
- `load_backtest_results`: after reading CSV, call `dict_to_backtest_result` per row
- `save_positions`: call `normalize_position_for_save` per position, then build DataFrame & upload
- *depends on 3.1*

**Step 3.3 — Refactor `postgres.py` to use shared helpers**
- Same pattern: call shared dict converters, only differ in SQL INSERT / asyncpg fetch transport
- *depends on 3.1; parallel with 3.2*

**Verification:** Existing storage tests (`test_cloud_storage.py`, `test_postgres_storage.py`) must pass unchanged. GCS CSV output must now include `side` column. Postgres output must remain identical.

---

### Phase 4: Merge Duplicate Functions in Trading Engine (~220 lines deduplicated)

**Step 4.1 — Unify `identify_buying_opportunities` + `identify_shorting_opportunities`**
- Extract into `_identify_opportunities(backtest_results, direction)` in `trading_engine.py`
- Keeps both public methods as thin wrappers calling the unified helper
- *depends on Phase 1 complete*

**Step 4.2 — Unify `place_buy_order` + `place_short_order`**
- Extract into `_place_order(opportunity, shares, side: OrderSide, quantity_sign, log_label, profit_label)` in `trading_engine.py`
- Keep both public methods as thin wrappers
- *depends on Phase 1 complete; parallel with 4.1*

**Step 4.3 — Unify `_compute_rsi_take_profit` + `_compute_rsi_cover_price`**
- Extract into `_compute_rsi_target_price(symbol, rsi_target, rsi_period, entry_price, direction)` in `trading_engine.py`
- Keep both public methods as thin wrappers
- *depends on Phase 1 complete; parallel with 4.1, 4.2*

**Verification:** `python tests/run_tests.py` — all trading engine tests must pass. Opportunity identification, order placement, and RSI target computation must produce identical outputs.

---

### Phase 5: Break Down Large Functions & Split Utils

**Step 5.2 — Split `utils.py` into `app/utils/` package** ✅ Completed
- Created 5 sub-modules: `datetime_.py`, `logging_.py`, `calendar.py`, `metrics.py`, `progress.py`
- `app/utils/__init__.py` re-exports all symbols for backward compatibility
- Deleted dead classes after Phase 1 removal
- *depends on Phase 1 complete*

**Step 5.1-lite — Deduplicate position-from-DataFrame blocks** ✅ Completed
- Extracted `_df_row_to_position()` helper replacing 3× duplicated ~25-line Position construction blocks
- ~70 lines of duplication eliminated

**Step 5.3 — OCO exit price logic** ❌ Deferred
- Inline block and helper method operate on different data shapes (DataFrame values vs Position attributes)

**Verification:** `python tests/run_tests.py` — all position reconciliation tests and utils tests must pass. Utils imports from all callers (`from utils import parse_dt`, etc.) must continue working.

---

### Phase 6: Split `trading_engine.py` into Focused Modules — ❌ DEFERRED

**Reason:** 7+ methods are tightly patched in tests via `patch.object(self.engine, ...)`. Module extraction would require rewriting those tests for dependency injection. File is already down to 1119 lines (from ~1400).

**Planned but deferred steps:**
- `app/price_service.py` — OHLCV + RSI service (inject DataProvider)
- `app/opportunity_finder.py` — signal detection (inject PriceService)
- `app/position_sizer.py` — account-aware sizing (inject PositionsManager)
- `app/order_executor.py` — broker order placement (inject trading_client + data_provider)
- Fix duplicate `PositionsManager` bug: inject shared instance from `TradingAlgorithm`

---

### Phase 7: Docker & Config Hygiene ✅ Completed

**Step 7.1 — Create `.dockerignore`**
- Exclude: `__pycache__`, `*.pyc`, `*.pyo`, `.pytest_cache`, `.env`, `.git`, `logs/`, `*.log`, `.venv`, `.vscode`

**Step 7.2 — Split `requirements.txt` into runtime + dev**
- `requirements.txt`: runtime deps only + added `TA-Lib>=0.4.0`
- `requirements-dev.txt`: pytest, pytest-asyncio, black, flake8, mypy
- Dockerfile updated to install only `requirements.txt`

**Step 7.3 — Multi-stage Docker build + non-root user**
- Stage 1 (builder): install TA-Lib C library, compile wheels
- Stage 2 (runtime): copy wheels + app code, create non-root `appuser`, run as `appuser`

**Step 7.4 — Fix Dockerfile HEALTHCHECK**
- Changed from `exit 0` (always passes) to actual health check: `curl -sf http://localhost:8080/health || exit 1`

**Step 7.5 — Fix Makefile**
- `make run`: `.venv/bin/python app/main.py` → `python -m app.main`
- `make clean`: now cleans both `app/logs/` and top-level `logs/`

**Verification:** `docker build -t trading-algo . && docker run --rm trading-algo` — container starts, health check passes, runs as non-root.

---

### Relevant Files Modified

- `app/trading_engine.py` — duplicate long/short, buy/short, RSI target methods unified; dead code removed
- `app/positions.py` — `_df_row_to_position()` helper added; 3× construction blocks deduplicated
- `app/utils.py` → `app/utils/` package — split into 5 sub-modules + `__init__.py` re-exports
- `storage/backend.py` — new shared serialization helpers (`backtest_result_to_dict`, `dict_to_backtest_result`, `normalize_position_for_save`)
- `storage/gcs.py` — uses shared helpers; added `side` field to position saves
- `storage/postgres.py` — uses shared helpers
- `app/config.py` — removed `globalConfig = None` vestige
- `app/data_provider.py` — removed `BarData` dataclass
- `app/optimizer.py` — removed `_composite_score` + dead import
- `tests/test_integration.py` — fixed broken mocks and always-pass tests; replaced `get_historical_data` with `get_single_stock_bars`
- `tests/test_main.py` — fixed async test methods (`TestCase` → `IsolatedAsyncioTestCase`)
- `tests/test_utils.py` — removed tests for deleted classes; updated patch targets for new sub-modules
- `tests/test_data_provider.py` — removed `TestBarData`
- `tests/test_optimizer.py` — removed deprecated composite score tests
- `tests/test_cloud_storage.py` — added missing mock attributes
- `Dockerfile` — multi-stage, non-root, HEALTHCHECK fix
- `requirements.txt` — split dev deps out, added TA-Lib
- `Makefile` — portable `make run`, improved `make clean`

### New Files Created

- `app/utils/__init__.py`, `app/utils/datetime_.py`, `app/utils/logging_.py`, `app/utils/calendar.py`, `app/utils/metrics.py`, `app/utils/progress.py`
- `.dockerignore`
- `requirements-dev.txt`

---

### Verification Checklist

1. ✅ `python tests/run_tests.py` after every phase — all 203 tests pass
2. ✅ `git diff --stat` shows net negative lines: **-961 net** (443 added, 1404 deleted)
3. ⬜ Phase 6: dry-run trading session produces identical portfolio results (deferred)
4. ⬜ Phase 7: `docker build` succeeds, container runs as non-root, HEALTHCHECK returns valid status (verify in CI)

---

### Decisions

- **Phase ordering**: Phases 1-3 were low-risk cleanup; Phases 4-5 were moderate refactoring; Phase 6 was deferred as highest-risk structural change.
- **Scope EXCLUDED**: The 7 README TODO items (clean up stops calculated multiple times, improve portfolio allocation, OCO order types at day start, 10% equity to backtester, win-rate optimization, position reconciliation for manual orders) are OUT OF SCOPE. They are feature-level improvements, not structural refactoring.
- **Backward compatibility**: All existing public API signatures preserved. Phase 5.2 uses `__init__.py` re-exports so `from utils import parse_dt` continues working.
- **`strategy_critique.md` findings**: The critique recommends switching from alpha to Sharpe ratio scoring. This is a strategic change, out of scope for this structural refactoring. File remains as a design document.
- **Rate-limit sleep in `finally` block** (`data_provider.py` L135): Known bottleneck documented in repo memory. Out of scope — requires behavioral change, not just structural.
- **Optimizer grid search dedup** (optimizer.py L166-275): Not included — the 3 stages (coarse/fine/fallback) differ more in parameters than structure and extracting risks introducing subtle grid-search bugs. Deferred to a performance-focused PR.
- **Walk-forward / optimizer universe batching dedup** (optimizer.py + walk_forward.py): Not included — `WalkForwardValidator` will eventually be refactored separately.

---

### DEFERRED ITEMS (Post-Phase-7)

1. **Phase 6 — Split trading_engine.py**: Deferred because 7+ methods are tightly patched in tests via `patch.object(self.engine, ...)`. Module extraction would require rewriting those tests for dependency injection. File is already down to 1119 lines (from ~1400).
2. **Full decomposition of `get_and_reconcile_positions()`** (~450 lines now): Too risky mechanically due to interwoven local variables and DataFrame mutation patterns.
3. **OCO exit price dedup** (Phase 5.3): Inline block and helper method operate on different data shapes (DataFrame values vs Position attributes).
4. **Optimizer grid-search dedup** (optimizer.py): 3 stages differ enough to risk subtle bugs.
5. **Walk-forward/optimizer universe batching dedup**: WalkForwardValidator refactoring deferred.
6. **Rate-limit sleep in `finally` block** (data_provider.py): Requires behavioral change.
7. **7 README TODO items**: Feature-level improvements (OCO at day start, portfolio allocation, Sharpe ratio scoring, etc.).

---

### Final Outcome Summary

| Metric | Value |
|--------|-------|
| Net lines changed | **-961** (443 added, 1404 deleted) |
| Dead code removed | ~360 lines |
| Duplication eliminated | ~390 lines |
| Bugs fixed | 3 (GCS side field, broken integration tests, non-executing async tests) |
| New modules | 6 (utils package with 5 sub-modules + `__init__.py`) |
| Tests passing | 203 passed, 0 failed |
| Docker hygiene | `.dockerignore`, multi-stage build, non-root user, real HEALTHCHECK |
