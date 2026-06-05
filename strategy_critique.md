# Trading Strategy Critique

## Strategy Category

This is a **systematic mean-reversion strategy** driven by RSI thresholds.

- Entry logic: buy when RSI crosses below a lower bound (oversold).
- Exit logic: sell when RSI crosses above an upper bound (overbought) or max hold days are reached.
- Structure: single-indicator, rules-based technical strategy.

## Key Issues With This Strategy

### 1. Mean-reversion fragility on single stocks
RSI mean-reversion is generally more robust on broad indices/ETFs than on individual names. Single stocks can remain "oversold" for fundamental reasons (earnings deterioration, guidance cuts, legal/regulatory events), where reversion may not occur quickly.

### 2. In-sample optimization bias
Grid-searching RSI parameters on the same data used for evaluation risks overfitting. The selected parameter set can capture noise rather than durable edge without walk-forward or out-of-sample validation.

### 3. Backtest/live behavior mismatch
If live execution uses stop-loss/take-profit logic that is not fully represented in backtests, optimization targets become misaligned with real execution. This can materially distort expected performance.

### 4. Single-indicator signal quality limits
RSI alone is often noisy. Without confirmation filters (trend regime, volume, volatility state), false positives can be frequent, especially during strong trends.

### 5. Regime dependence
Mean-reversion tends to underperform in momentum/trending regimes. A static RSI strategy with fixed thresholds can degrade sharply when volatility and trend structure change.

### 6. Cost and slippage blindness
Backtests that ignore spread, slippage, market-impact, and execution timing assumptions overstate real-world returns, especially for frequent turnover strategies.

### 7. Objective metric weaknesses
Using raw alpha (`strategy return - buy-and-hold`) as the main optimizer score can favor unstable high-variance outcomes. Risk-adjusted criteria (Sharpe/Sortino, drawdown constraints, turnover penalties) are often better optimization targets.

### 8. Position sizing may be non-risk-based
Fixed-percent sizing is simple but does not account for symbol volatility, signal confidence, or correlation. This can produce uneven risk and fragile portfolio behavior.

### 9. Parameter and filter consistency risk
If optimization thresholds differ from live-trading acceptance thresholds, many "good" backtest candidates may never be traded, reducing system coherence.

## Practical Improvements (High Impact)

1. Add robust validation: train/test split + walk-forward optimization.
2. Enforce backtest/live parity: replicate stop-loss/take-profit and execution assumptions exactly.
3. Add regime filters: e.g., trend filter (200-day MA), volatility guardrails, or market-state gating.
4. Add transaction-cost model: spread + slippage assumptions by liquidity bucket.
5. Improve objective function: optimize for risk-adjusted return and drawdown, not raw alpha alone.
6. Introduce risk-based sizing: ATR/volatility-targeted sizing and portfolio exposure caps.

## Bottom Line

Category-wise, this is a **technical mean-reversion strategy**. It can work in specific market regimes, but without stronger validation, regime controls, and realistic execution modeling, it is vulnerable to overfitting and live underperformance.
