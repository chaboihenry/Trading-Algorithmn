# Walk-Forward Validation Methodology

## Overview
This system uses expanding-window walk-forward validation to ensure strategies are tested on truly unseen data, preventing overfitting and providing realistic performance estimates.

## Process

### 1. Training Phase
- Use all historical data up to signal generation date
- Train models on this data
- Models learn patterns from past price movements, fundamentals, sentiment, and technical indicators
- No future information is included in training data

### 2. Signal Generation
- Models generate signals for current date
- Signals represent predictions about future price movements
- Each signal includes:
  - Direction (BUY/SELL/HOLD)
  - Strength (0-1 confidence score)
  - Entry price
  - Stop loss level
  - Take profit target
  - Meta-confidence (for ensemble)

### 3. Forward Testing
- Wait N days (30/60/90) for actual price movements
- Calculate returns using actual observed prices
- No peeking at future data during signal generation
- Positions are evaluated based on real market outcomes

### 4. Performance Calculation
- Aggregate all forward-tested trades
- Calculate risk metrics on out-of-sample returns
- Validate against minimum thresholds
- Statistical significance testing

## Example Timeline

```
Day 0: Train model on Jan 1 - Dec 31, 2024
Day 0: Generate signal to BUY AAPL at $195.50
Day 30: Observe actual AAPL price = $203.20
Day 30: Calculate return = +3.9%
```

## Why This Prevents Overfitting

1. **Temporal Separation**: Signal generation happens BEFORE observing future prices
2. **No Data Leakage**: Models cannot "see" validation period data
3. **Real-World Mimicry**: Exactly mimics how the system would operate in production
4. **Out-of-Sample Testing**: Every trade is evaluated on unseen future data

## Validation Criteria

Strategy passes validation if it meets ALL of the following thresholds:

### Core Metrics
- **Sharpe Ratio** > 1.0 (risk-adjusted returns)
- **Win Rate** > 55% (percentage of profitable trades)
- **Max Drawdown** < 20% (maximum peak-to-trough decline)
- **Minimum Trades** > 30 (for statistical significance)

### Advanced Risk Metrics
- **Calmar Ratio** > 0.5 (annualized return / max drawdown)
- **Sortino Ratio** > 1.0 (penalizes only downside volatility)
- **MAR Ratio** > 1.0 (similar to Calmar, uses absolute values)
- **VaR (95%)** > -5% (maximum expected loss at 95% confidence)
- **CVaR (95%)** > -7% (expected loss when VaR is exceeded)

### Statistical Significance
- **Sharpe 95% CI** lower bound > 0 (bootstrap confidence interval)
- **Max DD p-value** < 0.10 (Monte Carlo simulation shows DD is not extreme)
- **T-statistic** > 2.0 (returns significantly different from zero)
- **P-value** < 0.05 (statistical significance of returns)

## Industry Standard Benchmarks

| Metric | Good | Excellent | Target |
|--------|------|-----------|--------|
| Sharpe Ratio | > 1.0 | > 2.0 | > 1.5 |
| Sortino Ratio | > 1.0 | > 2.0 | > 1.5 |
| Calmar Ratio | > 0.5 | > 1.0 | > 0.8 |
| Win Rate | > 55% | > 65% | > 60% |
| Max Drawdown | < 20% | < 10% | < 15% |
| VaR (95%) | > -5% | > -3% | > -4% |
| Profit Factor | > 1.5 | > 2.0 | > 1.8 |

## Expanding Window Methodology

### Window Structure
```
Window 1:  [Train: Jan-Dec 2022] -> [Test: Jan-Mar 2023]
Window 2:  [Train: Jan 2022-Mar 2023] -> [Test: Apr-Jun 2023]
Window 3:  [Train: Jan 2022-Jun 2023] -> [Test: Jul-Sep 2023]
...
```

### Why Expanding (Not Rolling)?
- **More Data Over Time**: Later windows have more training data
- **Realistic**: Mimics how model is actually retrained in production
- **Stable**: Model sees all historical patterns, not just recent ones

## Multi-Strategy Ensemble

### Base Strategies
1. **Sentiment Trading**: Uses news sentiment, social media, analyst ratings
2. **Pairs Trading**: Mean-reversion on cointegrated stock pairs
3. **Volatility Trading**: Exploits volatility regime changes (GARCH models)

### Stacked Ensemble (Meta-Learner)
- Takes signals from all 3 base strategies as features
- Uses XGBoost meta-model to combine predictions
- Learns which strategy works best in different market conditions
- **Meta-Confidence**: Probability that ensemble prediction is correct

### Why This Works
- **Diversification**: Different strategies work in different market regimes
- **Non-Correlation**: Strategies use different data sources and logic
- **Adaptive Weighting**: Ensemble learns optimal combination over time

## Risk Management

### Position Sizing
- Uses fractional Kelly Criterion (25% of full Kelly)
- Caps maximum position size at 10% of capital
- Accounts for liquidity (minimum $1M daily volume)
- Adjusts for market volatility (VIX proxy)

### Stop Loss / Take Profit
- Fixed 2:1 reward/risk ratio
- Stop loss: -2% from entry
- Take profit: +4% from entry
- Adjusted for short positions

## Continuous Improvement

### Incremental Learning
- Model is retrained daily with new data
- Maintains versioning (v1, v2, v3...)
- Tracks test accuracy over time
- Degrades gracefully if performance declines

### Performance Monitoring
- Daily validation reports
- Email notifications with top 5 trades
- Advanced risk metrics tracked
- Statistical significance continuously evaluated

## References

- **Sharpe Ratio**: Sharpe, W.F. (1994). "The Sharpe Ratio"
- **Sortino Ratio**: Sortino, F.A. & Price, L.N. (1994). "Performance Measurement in a Downside Risk Framework"
- **Calmar Ratio**: Young, T.W. (1991). "Calmar Ratio: A Smoother Tool"
- **Value at Risk**: Jorion, P. (2006). "Value at Risk: The New Benchmark for Managing Financial Risk"
- **Kelly Criterion**: Kelly, J.L. (1956). "A New Interpretation of Information Rate"
- **Walk-Forward Analysis**: Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"

---

**Last Updated**: 2025-12-02
**Maintained By**: Integrated Trading Agent System
