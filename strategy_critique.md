# Trading Strategy Critique

## Strategy Category

This is a **systematic mean-reversion strategy** driven by RSI thresholds, trading single stocks on daily bars via Alpaca Markets.

- Entry logic: buy when RSI crosses *below* a lower bound (oversold cross-below).
- Exit logic: RSI crosses *above* an upper bound (overbought), stop-loss (fixed %), RSI-implied take-profit, or max hold days — executed as bracket/OCO orders.
- Structure: single-indicator RSI, grid-search optimized per symbol, fixed-percent position sizing.

## Codebase Audit — What's Already Solid

Before listing issues, it's worth noting what the implementation already gets right:

- **Backtest/live exit parity is strong.** The `calculate_price_for_target_rsi()` method computes an RSI-implied take-profit price used in *both* `_generate_signals()` (backtest) and `calculate_todays_stop_loss_and_take_profit()` (live). Stop-loss, take-profit, and max-hold-days exit logic is mirrored across environments.
- **Cross-based entry signals.** Entry requires RSI to cross *below* the lower threshold (was above, now below), not just be below it. The live engine's `identify_buying_opportunities()` replicates this cross-below check with a fallback to level-based if prior RSI is unavailable.
- **Exit reason tracking.** Every trade in the backtest records why it exited (`rsi_cross`, `stop_loss`, `take_profit`, `max_hold_days`) — valuable for diagnosing strategy behavior.
- **Bracket/OCO orders in live trading.** Orders are placed with attached stop-loss and take-profit legs; daily OCO refresh via `update_portfolio_orders()`.
- **Dry-run mode, position reconciliation, and environment-specific configs** (dev/qa/prod) are all in place.
- **Performance metrics exist** in `utils.py`: `PerformanceMetrics` has Sharpe, Sortino, and Calmar ratio calculations ready to use — they just aren't wired into the optimizer yet.

## Key Issues With This Strategy

### 1. Mean-reversion fragility on single stocks
RSI mean-reversion is generally more robust on broad indices/ETFs than on individual names. Single stocks can remain "oversold" for fundamental reasons (earnings deterioration, guidance cuts, legal/regulatory events), where reversion may not occur quickly.

### 2. In-sample optimization bias (CRITICAL)
`StrategyOptimizer.optimize_symbol()` runs a grid search over RSI parameters on the *entire* backtest window and picks the best alpha. There is no train/test split or walk-forward validation. The parameter ranges are wide (periods 3–34, lower 20–60, upper 50–85 in dev), producing many combinations. This is pure in-sample optimization — the selected parameters can capture noise rather than a durable edge.

### 3. Single-indicator signal quality limits
RSI alone is the only signal generator. `_generate_signals()` uses RSI cross events exclusively. Without confirmation filters (trend regime, volume, volatility state), false positives are frequent, especially during strong trends where RSI can stay oversold for extended periods.

### 4. Regime dependence — no gating
Mean-reversion underperforms in momentum/trending regimes. A static RSI strategy with fixed thresholds degrades when volatility and trend structure change. There is currently no regime filter of any kind — no SMA gate, no volatility guardrail, no market-state check.

### 5. Cost and slippage blindness
`_calculate_returns()` assumes zero-cost execution — every trade fills at the exact price on the bar following the signal, with no spread, slippage, commission, or market-impact deduction. For daily-bar trading on single stocks, this overstates net returns, especially for lower-liquidity names.

### 6. Objective function is raw alpha (high-impact, low-effort fix)
`StrategyOptimizer.optimize_symbol()` scores strategies on `result.alpha` (total_return − buy_and_hold_return). This ignores risk entirely — it can favor strategies with high variance, large drawdowns, or sparse trade counts. The `PerformanceMetrics` class in `utils.py` already implements Sharpe, Sortino, and Calmar ratios, making this one of the easiest high-impact fixes in the codebase.

### 7. Position sizing is fixed-percent, not risk-based
`calculate_position_sizes()` allocates `equity * POSITION_SIZE_PCT` per position regardless of the symbol's volatility, signal quality, or correlation with existing holdings. A 10% allocation to a high-volatility biotech stock carries vastly different risk than the same allocation to a low-volatility utility.

### 8. Parameter and filter consistency risk
If optimization thresholds differ from live-trading acceptance thresholds, many "good" backtest candidates may never be traded. The optimizer selects on `alpha > 0` + `profitable`, while live trading adds `alpha > 0`, `win_rate >= MIN_WIN_RATE`, and `num_trades >= MIN_NUM_TRADES`. These post-hoc filters are applied after optimization, so the optimizer may select parameters that would later be rejected.

### 9. No live-vs-backtest monitoring
There is no systematic comparison of live execution metrics (realized slippage, win rate, signal frequency) against backtest expectations. Without this feedback loop, parameter drift and model decay go undetected.

## Practical Improvements (High Impact)

1. **Add walk-forward validation** (Issue #2): Split the backtest window into in-sample and out-of-sample periods. Optimize on IS, validate on OOS. Roll forward and repeat.
2. **Wire up risk-adjusted scoring** (Issue #6): Change the optimizer score from `result.alpha` to a composite using existing `PerformanceMetrics` (e.g., `result.sharpe_ratio` or Sortino).
3. **Add a trend filter** (Issue #4/#5): Gate entries with a 200-day SMA or dual-MA regime classifier.
4. **Add transaction-cost model** (Issue #5): Model spread + slippage in `_calculate_returns()`.
5. **Introduce volatility-based sizing** (Issue #7): Replace fixed-percent with ATR-targeted position sizing.
6. **Add live-vs-backtest monitoring** (Issue #9): Track key metrics on a rolling basis with alert thresholds.

## Bottom Line

Category-wise, this is a **technical mean-reversion strategy**. The execution infrastructure (bracket orders, backtest/live parity for exits, dry-run mode, position reconciliation) is solid. The core weakness is the *alpha research pipeline* — in-sample grid search with a risk-blind objective, no regime filters, and no cost model. The highest-ROI fixes are: (1) wire up risk-adjusted scoring using the `PerformanceMetrics` class that already exists, (2) add a simple trend filter, and (3) introduce a train/test split in the optimizer. These three changes can be implemented without restructuring the architecture.

---

## Deep Dive: Signal Quality & Regime Awareness

The sections below provide concrete, code-aware guidance for improving signal quality and adding regime awareness to the existing `RSIStrategy` and `StrategyOptimizer` classes.

### Current State (What the Code Does Today)

`RSIStrategy._generate_signals()` produces buy signals when RSI crosses below `rsi_lower` and sell signals when RSI crosses above `rsi_upper`, with additional exits for stop-loss, take-profit, and max-hold-days. The optimizer then grid-searches `(rsi_period, rsi_lower, rsi_upper)` per symbol. There is no pre-signal filtering or regime check.

### Trend Filters — Eliminating "Falling Knife" Entries

The core problem: RSI alone fires oversold signals during freefalls. A trend filter gates entries so mean-reversion trades only fire *with* the prevailing trend.

**Implementation approach (minimal changes to `_generate_signals`):**

Add a `regime` column to the backtest data *before* signal generation. In `RSIStrategy.backtest()`, compute a simple SMA regime and pass it through. The signal loop already has a `position` state machine — adding a regime gate is a single extra condition on the buy branch:

```python
# In _generate_signals, the buy condition becomes:
if signals['buy_signal'].iloc[i] and position == 0 and regime_allows_entry:
    position = 1
    ...
```

**Recommended starting point — 200-day SMA gate (lowest complexity, high impact):**
- Add `sma_200` to `TechnicalIndicators` (or compute inline with `data['close'].rolling(200).mean()`).
- Define regime: `bull` when `close > sma_200`, `bear` when `close < sma_200`.
- In bull regime, take long oversold signals. In bear regime, suppress all entries.
- This requires no new dependencies and minimal code. Wire it into `RSIStrategy.backtest()` and `StrategyOptimizer` — the optimizer will naturally favor parameters that work with the filtered signal set.

**If you want more nuance — dual-MA regime classifier:**
- Add a `calculate_regime(data)` static method to `RSIStrategy` that returns `'trending_up'`, `'trending_down'`, or `'sideways'` based on 50-day vs 200-day MA relationship.
- Take mean-reversion signals only in `'sideways'` regime. In `'trending_up'`, skip (or if you later add short capability, flip the signal direction).
- This can be toggled via a new config key (`REGIME_FILTER_MODE: 'sma200' | 'dual_ma' | 'none'`).

### Volatility Guardrails

High-volatility environments produce wider RSI swings and more false signals.

**ATR-based gate (easy addition to `TechnicalIndicators`):**
- Add `calculate_atr(data, period=14)` to `TechnicalIndicators` if not already present.
- Compute `atr_ratio = atr / atr.rolling(50).mean()`.
- Gate entries when `atr_ratio > 1.5` (volatility spike). Add this as a required condition alongside the trend filter.

**Config-driven approach:**
Add to `config/dev.json` (and other environments):
```json
"signal_filters": {
    "trend_filter": "sma200",
    "volatility_gate": true,
    "volatility_threshold": 1.5
}
```

These gates should be applied in the *backtest* so the optimizer sees the filtered universe, not just at live entry time. Otherwise you're optimizing on signals that will never trade.

### Multi-Indicator Confirmation

RSI plus one uncorrelated confirmation indicator reduces noise. Add optional confirmation scoring to `_generate_signals()`.

**Bollinger Band %B (low-correlation with RSI):**
- `%B = (close − lower_band) / (upper_band − lower_band)` with 20-period, 2-std bands.
- Confirm RSI oversold only when `%B < 0.1` (price is actually at the band, not just showing RSI divergence).

**Volume confirmation:**
- Require `volume > volume.rolling(20).mean()` on the signal bar. Low-volume oversold signals are often noise.
- This data is already available — `data_provider` returns OHLCV with volume.

**Signal scoring pattern (optional enhancement to `_generate_signals`):**

Add a `signal_score` that accumulates confirmation points. A signal fires only if `signal_score >= MIN_CONFIRMATION_SCORE` (configurable). This is more flexible than hard-gating and allows the optimizer to be run with different confirmation thresholds to measure the impact.

### Updating the Optimizer Objective Function

This is the single highest-impact, lowest-effort change. In `StrategyOptimizer.optimize_symbol()`, replace:

```python
score = result.alpha  # current: risk-blind
```

With a risk-adjusted composite using the `PerformanceMetrics` class already in `utils.py`:

```python
# Composite: reward Sharpe, penalize drawdown and sparse trades
score = (
    result.sharpe_ratio * 2.0
    - result.max_drawdown * 0.5
    + (result.num_trades / 20.0)  # bonus for sufficient sample
)
```

Or simpler: just use `result.sharpe_ratio` directly as the score. The `BacktestResult` dataclass already carries `sharpe_ratio` and `max_drawdown` — no schema changes needed.

Better yet, expose the scoring method via config so you can A/B test:
```json
"optimization": {
    "scoring_method": "sharpe",
    "min_trades_for_score": 5
}
```

### Practical Testing Framework

When you modify signal logic or the objective function, compare these metrics on both the full backtest window and a held-out period:

| Metric | Why It Matters |
|---|---|
| **Signal count per year** | Too few = low statistical confidence; too many = likely overfit |
| **Win rate** | Should improve as noise is filtered |
| **Average win / average loss ratio** | Good filters reduce loser size more than winner size |
| **Max drawdown** | Regime filters should materially reduce this |
| **Sharpe ratio** | The ultimate summary — should improve if filters add real edge |
| **Performance by regime** | Break out returns in bull/bear/sideways separately |
| **Turnover** | Gating entries during bad regimes should reduce turnover and implied costs |

### Recommended Implementation Order

1. **Change optimizer score to Sharpe ratio** — one-line change in `optimize_symbol()`, uses existing data.
2. **Add SMA-200 trend filter** — ~20 lines in `_generate_signals()` + `TechnicalIndicators`, immediate regime protection.
3. **Add ATR volatility gate** — ~15 lines, prevents entries during volatility spikes.
4. **Add train/test split** — split the backtest window in `StrategyOptimizer`, optimize on first 70%, validate on last 30%.
5. **Add simple cost model** — apply a configurable slippage factor in `_calculate_returns()` (e.g., 5 bps per trade).
6. **Wire up Sortino or composite scoring** — once you have multiple metrics being tracked, move to a composite.

Each step is independently testable and can be rolled back via config flags.
