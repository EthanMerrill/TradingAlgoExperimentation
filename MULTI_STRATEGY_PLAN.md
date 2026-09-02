# Multi-Strategy Framework Plan

**Status:** Partially implemented — Phases A–D done (2026-09-01); E & F pending
**Date:** 2026-08-27
**Scope decisions (confirmed with Ethan):****
1. **Simultaneous multi-strategy portfolio** — multiple strategies run per cycle, positions tagged by strategy, capital allocated across strategies.
2. **Full intraday bar-loop engine** — a bar-driven execution path alongside the existing daily session engine.
3. **Framework only** — this plan covers the framework; the RVOL-ORB strategy (Section 1) is a follow-up implementation once the framework exists.

---

## 1. Reference Strategy: 5m RVOL + Opening Range Break (from Ben Patreon post)

Source: [Full Intraday Strategy Breakdown & Backtest (Sharpe 2.5+)](https://www.patreon.com/givebenyourmoney/posts/full-intraday-2-161477771), June 19.

This is the motivating use case for the framework (intraday, cross-sectional ranking, simultaneous signals across many symbols). It is **not** part of this implementation phase — it will be the first strategy built on the framework.

### 1.1 Thesis
- Extreme opening participation = informed/big flow. Price moves regardless of whether there's a sane reason (e.g. RGTI, BYND, GME).
- A trader without size constraints can get in front of big players who must accumulate slowly. The signal doesn't need to explain *why* — it just waits for the market to reveal the most important stocks of the day, then trades momentum ("yolo in the direction of flow").

### 1.2 Core signal — 5m Relative Volume (RVOL)
```
5m RVOL = (first 5m bar volume today) / (avg first-5m-bar volume over past 14 days)
```
- Computed per stock per day, after the first 5m bar closes (~9:35 ET).
- Ranking metric across the eligible universe. Extreme = high participation/orderflow.

### 1.3 Daily universe filters (evaluated pre-market on prior-day data)
| Filter | Value |
|---|---|
| Price | > $5 |
| Avg volume (14d) | > 1,500,000 shares/day |
| Daily ATR (14d) | > $0.50 |
| 5m RVOL | ≥ 1.0 |
| Data quality | drop missing/weird bars |

Then rank by 5m RVOL descending and take the **top 20** stocks.

### 1.4 Execution — Opening Range Break (ORB)
- Wait for the first 5m bar to close (9:35 ET).
- If in top-20 list **and** first 5m bar is **green** → trade only the break of the range **high** (long).
- If in top-20 list **and** first 5m bar is **red** → trade only the break of the range **low** (short).
- "There's nothing special about ORB as an execution strategy — the signal (participation) is the edge."

### 1.5 Exit variants
| Variant | Exit rules |
|---|---|
| **Simple** | Exit on the close, no stop. |
| **Paper (ATR stop)** | Stop at **10% of daily ATR** below entry (long) / above entry (short). E.g. 14d ATR = $5, entry long $100 → stop $99.50. Exit at close otherwise. |

### 1.6 Critical ATR nuance (from the post — avoids lookahead/incorrect stops)
- **Wrong:** 10% of *previous day's* 14d ATR computed on yesterday's close. High-RVOL stocks gap/expand volatility overnight, so this yields artificially tight stops.
- **Correct (what the paper did):** compute ATR incorporating the **gap from yesterday's close to today's open**, so the ATR reflects today's volatility.

### 1.7 Claims & caveats
- Paper claims Sharpe 2.5+; tested first on S&P stocks in 2026, then expanded to all stocks meeting criteria over a longer window.
- Attachments: `Cleaned_ORB_Sheet_19_filtered_corrected.csv`, `orb_range_break_atr_stop_compounded_return_trades.csv`.
- Post is paywalled; unknown details (to confirm against the paper): exact bar-time alignment, gap-ATR formula (True Range including overnight gap vs. open-to-open), entry execution style (stop order at range break vs. market on confirmation), per-day max position counts. These become tunable params in the follow-up implementation.

### 1.8 Parameterization (for the follow-up)
```
rvol_lookback_days: 14
top_n: 20
atr_lookback_days: 14
atr_stop_fraction: 0.10        # 0.0 => simple "exit at close" variant
bar_size: 5m
entry_after_first_bar: True     # first bar close = 9:35 ET
include_gap_in_atr: True
exit_policy: "close"            # always exit at market close
```

---

## 2. Framework Goals & Non-Goals

### 2.1 Goals
- **Pluggable strategy registry** — a common `Strategy` interface; strategies registered by name; config-driven enablement.
- **Simultaneous execution** — multiple strategies can be enabled in one cycle; results merged; positions tagged with the owning strategy.
- **Capital allocation** — per-strategy budget/weight; position sizing respects both per-strategy and portfolio limits.
- **Intraday execution path** — a bar-loop engine that evaluates signals on bar close and schedules intraday exits, coexisting with the existing daily session engine.
- **Backward compatibility** — default config behaves exactly as today (RSI strategy only); existing tests keep passing; no strategy-selection key required to run.

### 2.2 Non-goals (this phase)
- Implementing RVOL-ORB (follow-up, Section 1.8).
- Portfolio-level optimizer for allocation weights (start with static config weights).
- Cross-strategy correlation / overlap analytics.
- Position-level capital rebalancing across strategies mid-cycle.

---

## 3. Current Architecture (as-is)

| Component | Today | Multi-strategy impact |
|---|---|---|
| `app/strategy.py` | Single `RSIStrategy`; vectorized backtest → `BacktestResult` (RSI fields hardcoded) | Becomes one implementation of `Strategy`; `BacktestResult` gains `strategy_name`/`params` |
| `app/optimizer.py` | `StrategyOptimizer` hardcodes RSI grid + `RSIStrategy` instantiation; per-symbol grid search; `optimize_universe` (async) | Becomes generic over a `Strategy` (param grid from strategy, backtest via strategy) |
| `app/walk_forward.py` | `WalkForwardValidator(optimizer)`; `_run_oos_backtest` constructs `RSIStrategy` directly | Follows the optimizer generalization |
| `app/trading_engine.py` | Session-based (no bar loop). `_identify_opportunities` hardcodes RSI cross detection + RSI take-profit math | Live signal evaluation moves into strategy/executor hooks; engine dispatches per `strategy_name` |
| `app/positions.py` | `Position` stores RSI params; reconciliation enriches via `optimizer.optimize_symbol` | `Position` gains `strategy_name`; enrichment routes to the owning strategy's optimizer |
| `app/data_provider.py` | `get_single_stock_bars` hardcodes `TimeFrame(1, Day)` | Timeframe parameterized (needed for 5m bars) |
| `app/storage/backend.py` + `postgres.py` | Flat dicts/columns keyed on RSI fields | Add `strategy_name` (+ optional `params` JSON); Postgres migration |
| `app/zscore.py` | Cross-symbol composite scoring | Scored within strategy stage, then merged; optional cross-strategy normalization |
| `app/config.py` + `config/*.json` | `rsi_optimization` section; no strategy key | New `strategies` section (enabled list + allocation + per-strategy config), backward-compatible defaults |
| `app/main.py` | `TradingAlgorithm.run_full_cycle` → optimizer/WF → engine session | Iterates enabled strategies; runs bar loop when intraday strategies enabled |

---

## 4. Proposed Design

### 4.1 Strategy interface — new `app/strategies/base.py`

```python
class Strategy(ABC):
    name: str                        # registry key, e.g. "rsi_mean_reversion"
    execution_style: str             # "session" (daily) | "bar_loop" (intraday)
    bar_size: Optional[str]          # e.g. "5m" when bar_loop
    directions: Tuple[str, ...]      # ("long",) ("short",) ("long", "short")

    @abstractmethod
    def get_param_grid(self) -> List[Dict[str, Any]]: ...
    @abstractmethod
    def backtest(self, data, symbol, params: Dict[str, Any], initial_cash) -> BacktestResult: ...
    @abstractmethod
    def evaluate_live_signals(self, ctx: StrategyContext) -> List[LiveSignal]: ...
    def build_consolidated_trades(self, results) -> pd.DataFrame: ...   # default impl
```

- `StrategyContext` = shared, read-only per-cycle context: `data_provider`, `positions_manager`, `globalConfig`, `as_of`, cached OHLCV.
- `LiveSignal` = generalized `TradingOpportunity`: `symbol`, `direction`, `entry_price`, `stop_loss`, `take_profit`, `backtest_return`, `alpha`, `win_rate`, `composite_score`, `num_trades`, `strategy_name`, `extra: Dict` (strategy-specific payload).
- RSI-specific fields on `TradingOpportunity` (`current_rsi`, `target_rsi_lower/upper`, `rsi_period`) move to `extra` (or stay as deprecated optional fields) — engine no longer interprets them.

### 4.2 Registry — new `app/strategies/registry.py`

- `STRATEGY_REGISTRY: Dict[str, Type[Strategy]]`
- `register(cls)`, `get_strategy(name)`, `list_strategies()`
- `RSIStrategy` moves into `app/strategies/rsi.py`; `app/strategy.py` keeps a re-export (`RSIStrategy`, `BacktestResult`) so existing imports/tests don't churn.

### 4.3 Data model changes

**`BacktestResult`** — add with defaults (no breakage):
```python
strategy_name: str = "rsi_mean_reversion"
params: Dict[str, Any] = field(default_factory=dict)   # replaces rsi_* over time
```
Existing `rsi_period/lower/upper` stay (deprecated); `params` is populated by the RSI strategy for forward compat.

**`Position`** — add `strategy_name: str = "rsi_mean_reversion"`. Existing RSI fields stay.

**Storage** — `backtest_result_to_dict` / `dict_to_backtest_result` / `normalize_position_for_save` emit/read `strategy_name` (and `params` as JSON string). Postgres: `ALTER TABLE ... ADD COLUMN strategy_name TEXT NOT NULL DEFAULT 'rsi_mean_reversion'` (+ `params JSONB`). Old rows default to the RSI strategy.

### 4.4 Config — new `strategies` section (backward compatible)

```json
"strategies": {
  "enabled": ["rsi_mean_reversion"],
  "allocation": { "rsi_mean_reversion": 1.0 },
  "rsi_mean_reversion": { ...moved existing rsi_optimization keys... }
}
```
- Missing `strategies` key ⇒ defaults to `["rsi_mean_reversion"]` with today's behavior (existing `rsi_optimization` section).
- Allocation weights normalized to sum 1.0; each strategy's budget = equity × weight.

### 4.5 Optimizer & walk-forward generalization

- `StrategyOptimizer(strategy: Strategy)` — grid from `strategy.get_param_grid()`, backtest via `strategy.backtest(...)`, drop the `RSIStrategy` import.
- Keep `_test_single_combo` static; pass `params: Dict` instead of `(rsi_period, rsi_lower, rsi_upper)`.
- `WalkForwardValidator(optimizer)` unchanged in shape — it already delegates to the optimizer; only `_run_oos_backtest` changes to use `params` from the IS result.
- `main._get_backtest_results()` → loop over enabled strategies: per-strategy optimizer → filter → zscore (within strategy) → tag with `strategy_name` → merge lists. Backtest cache key becomes `(strategy_name, symbol, params)`.

### 4.6 Engine dispatch

- `TradingEngine.execute_trading_session(backtest_results)` gains a strategy-aware funnel:
  - Group results by `strategy_name`; for each, call `strategy.evaluate_live_signals(ctx)` instead of the hardcoded RSI cross block in `_identify_opportunities`.
  - `update_portfolio_orders` / `calculate_todays_stop_loss_and_take_profit` dispatch per `position.strategy_name` (RSI path keeps `RSIStrategy.calculate_price_for_target_rsi`).
- **Capital allocation** in `calculate_position_sizes`: per-strategy budget cap (weight × equity), then portfolio caps (`max_positions`, `max_new_positions`, `max_short_long_ratio`) enforced globally.
- **Overlap policy**: two strategies wanting the same symbol → priority by config order (or higher composite_score); single position wins, loser skipped (documented decision).

### 4.7 Intraday bar-loop engine — new `app/bar_engine.py`

```
BarLoopEngine
├── schedule: poll interval (e.g. every 60s) over market hours, or per-strategy event times
├── on_bar_close(strategy): fetch recent bars (timeframe = strategy.bar_size)
│   → strategy.evaluate_live_signals(ctx) → size → place day-TIF entry orders (market/stop)
├── on_session_close(strategy): force-close intraday positions tagged strategy_name
│   (intraday positions flagged `intraday=True` in PositionsManager; excluded from
│    daily max-hold/OCO refresh logic)
└── reuses TradingEngine order placement; respects dry_run
```

Prerequisites:
- `data_provider.get_single_stock_bars(symbol, start, end, timeframe=...)` — parameterize the hardcoded `TimeFrame(1, Day)`.
- Intraday positions persist with `strategy_name` + `intraday` flag; `update_portfolio_orders` skips them (their exits are bar-engine-managed).
- Scheduling: daily pre-market cycle computes universes/backtests; bar loop runs during RTH (health server / keep-alive mode stays the host process).

### 4.8 Testing plan

- **Registry**: registration, unknown-strategy error, config-driven selection defaults.
- **Serialization**: round-trip `BacktestResult`/`Position` with `strategy_name`; Postgres migration path.
- **Generic optimizer**: fake `Strategy` double exercises `get_param_grid`/`backtest` through `optimize_symbol`/`optimize_universe`.
- **Engine dispatch**: mocked strategies returning `LiveSignal`s; per-strategy allocation caps; overlap dedup.
- **Bar engine**: mocked clock + bar data; signal evaluation, day-TIF ordering, close-force-exit, dry-run.
- **Performance tracker**: daily per-strategy aggregation from a mocked positions snapshot + account equity; idempotent re-run for the same day; storage round-trip.
- **Regression**: existing `test_*.py` untouched where possible — new fields have defaults; `strategy.py` re-exports preserve imports; default config path is RSI-only.
- Known risk: `REFACTOR_PLAN.md` Phase 6 (engine split) was deferred because tests patch engine methods via `patch.object(self.engine, ...)` — engine-dispatch changes will require updating those tests. Budget for it.

### 4.9 Frontend — strategy visibility (dashboard)

The dashboard (`frontend/static/dashboard.js`) renders positions in a Tabulator table grouped into three column groups: **Position Record** (CSV/DB), **Live (Alpaca)**, and **Unrealized P&L**. Since Phase A, stored positions carry `strategy_name` and `normalize_position_for_save` emits it; `health_server._df_row_to_dict` passes unknown columns through generically, so `/api/positions` already includes `strategy_name` for new rows.

Changes (implemented with Phase C, 2026-09-01):

1. **API default** — in `app/health_server.py`, `_df_row_to_dict`: add `d.setdefault('strategy_name', 'rsi_mean_reversion')` so legacy rows (no column in CSV/DB) render as the RSI strategy instead of `undefined`.
2. **New “Strategy” column** in the Position Record group (place right after `Symbol`):
   - `field: 'strategy_name'`, `sorter: 'string'`, `headerFilter: 'list'` (values from the configured strategies or a static friendly map).
   - Friendly label + color badge formatter, e.g. `rsi_mean_reversion` → “RSI Mean Reversion” (blue), `rvol_orb` → “RVOL ORB” (purple), unknown → raw name in gray, missing → “—”.
3. **Filtering** — the strategy header filter slices the table by strategy; composes with the existing All/Open/Closed toggle.
4. **Position count** — extend `#position-count` to optionally show per-strategy counts (e.g. “12 positions · RSI 8 · RVOL 4”) when more than one strategy is enabled.
5. **CSS** — add a `.badge-strategy-*` family alongside existing badge styles in `frontend/static/style.css`.

Kept as-is for now: the RSI param columns (`rsi_period/lower/upper`, `current_rsi`) — they're still meaningful for RSI positions; a future pass can fold strategy-specific params into an expandable sub-row for non-RSI strategies.

### 4.10 Strategy-level daily performance tracking

**Today there is no per-strategy performance tracking** — only per-symbol backtest results, a session-level `trading_summary` (saved to `session_metadata`), and position snapshots (which carry `strategy_name` since Phase A but nothing aggregates it). This section adds a daily, strategy-level P&L ledger with a dashboard view.

**Computation — new `app/performance_tracker.py`**

Run at the end of each trading session (from `main._save_session_results`), after positions are reconciled. For each `strategy_name` present in today's positions (plus enabled strategies with no positions, recorded as zeros):

| Field | Definition |
|---|---|
| `date` | Trading day (ET) |
| `strategy_name` | Registry key |
| `realized_pnl` | Σ over positions closed today of `realized_return × entry notional` |
| `unrealized_pnl` | MTM over open positions: `(current − entry) × qty` (long) / inverse (short) |
| `total_pnl` | `realized + unrealized` |
| `return_pct` | `total_pnl ÷ strategy budget` (budget = allocation weight × account equity at day start) |
| `open_positions` / `closed_trades` | Counts |
| `win_rate_today` | Realized wins ÷ closed trades (null when 0 closed) |
| `cumulative_pnl` | Running sum per strategy (computed from prior ledger rows) |
| `avg_composite_score` | Mean `composite_score` of the strategy's active backtest results that day — links live P&L to the signal quality that produced it |

**Persistence — new `StrategyDailyPerformance` storage**

- Postgres: `strategy_daily_performance` table (PK `(date, strategy_name, environment)`, idempotent upsert so re-runs of a day are safe) + `ALTER TABLE … IF NOT EXISTS` migration pattern.
- GCS: `strategy_performance_YYYYMMDD.csv` per day, merged on re-run.
- `StorageBackend` gains `save_strategy_performance(rows, timestamp)` / `load_strategy_performance()` — same abstract-method pattern as positions/backtests, implemented by both `GcsStorage` and `PostgresStorage`.
- Idempotency: a session re-run overwrites that day's rows (upsert / rewrite file), never duplicates.

**Backfill** (future, out of initial scope): recompute historical daily rows from stored position snapshots + account equity history.

**Frontend**

New dashboard section below the positions table, fed by a new authenticated endpoint `GET /api/strategy-performance` (same auth pattern as `/api/positions`):

1. **Per-strategy summary cards** — one per enabled strategy: cumulative P&L, today's P&L, return %, open positions (reuses the `strategy_name` badge styling from §4.9).
2. **“Strategy Performance” Tabulator table** — one row per `(date, strategy)`: Date, Strategy, Realized, Unrealized, Total, Return %, Open, Closed, Win Rate, Avg Composite. Sortable + filterable; the Strategy column reuses the §4.9 badge/filter values.
3. **Cumulative P&L sparkline** per strategy — lightweight inline SVG (no new chart dependency; dashboard currently only pulls Tabulator from CDN). Optional first pass: skip the sparkline, ship the table + cards.
4. **CSS** — `.perf-*` styles in `frontend/static/style.css`; cards reuse the existing badge palette.

---

## 5. Implementation Phases

| Phase | Scope | Key files | Exit criteria |
|---|---|---|---|
| **A — Data model** ✅ done | `strategy_name` + `params` on `BacktestResult`/`Position`; storage + Postgres migration; config `strategies` section with defaults | `strategy.py`, `positions.py`, `storage/backend.py`, `storage/postgres.py`, `config.py`, `config/*.json`, `main.py` (cache key) | All existing tests pass; serialization round-trips; old rows default correctly |
| **B — Registry + interface** ✅ done | `strategies/base.py`, `strategies/registry.py`, move `RSIStrategy` → `strategies/rsi.py` (re-export from `strategy.py`); generalize optimizer/WF/zscore over `Strategy` | new `app/strategies/`, `optimizer.py`, `walk_forward.py`, `zscore.py` | Grid/backtest results identical to today for RSI; optimizer tests green |
| **C — Engine dispatch + allocation** ✅ done | Strategy-aware opportunity funnel; per-strategy + global caps; overlap policy; position tagging on entry; **dashboard strategy column + filter + badge styles** (see §4.9) | `trading_engine.py`, `positions.py`, `main.py`, `frontend/static/dashboard.js`, `frontend/index.html`, `frontend/static/style.css`, `app/health_server.py` (default) | Multiple strategies run one session; allocations respected; engine tests updated; dashboard shows strategy tags/filters |
| **D — Bar-loop engine** ✅ done | Timeframe-parameterized data fetch; `bar_engine.py`; intraday position lifecycle; intraday scheduling | `data_provider.py`, new `app/bar_engine.py`, `main.py`, `health_server.py` | Bar-loop harness test with mocked bars/clock; dry-run end-to-end |
| **F — Strategy performance tracking** | Daily per-strategy P&L ledger (§4.10): `performance_tracker.py`, storage table/CSV + backend methods, session-end hook, `/api/strategy-performance`, dashboard cards + table (+ optional sparkline) | new `app/performance_tracker.py`, `storage/backend.py`, `storage/postgres.py`, `storage/gcs.py`, `main.py`, `health_server.py`, `frontend/static/dashboard.js`, `frontend/index.html`, `frontend/static/style.css` | Daily rows persist idempotently; dashboard shows per-strategy P&L; tests green |
| **E — (follow-up) RVOL-ORB** | First bar-loop strategy per Section 1.8 | `app/strategies/rvol_orb.py`, config | Backtest matches paper-era results; dry-run signals |

---

## 6. Open Decisions

1. **Overlap policy** when two strategies target the same symbol (priority order vs. composite score).
2. Whether `BacktestResult.rsi_*` fields get removed after `params` lands (deprecation window).
3. Bar-loop poll granularity vs. event-driven (Alpaca websocket) for intraday data.
4. Gap-ATR exact formula for RVOL-ORB (True Range incl. overnight gap) — confirm against the paper in Phase E.
5. **Performance baseline for `return_pct`**: strategy budget (allocation weight × equity) vs. actual capital deployed. Budget is simpler and consistent with §4.4; note that unused budget (soft-cap allocation) slightly distorts per-strategy returns. Phase F defaults to budget, flagged in the ledger.
