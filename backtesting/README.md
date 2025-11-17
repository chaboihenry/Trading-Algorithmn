# Streamlined Backtesting System

Simple, focused backtesting for trading strategies using industry-standard metrics and Kelly Criterion position sizing.

## Features

✅ **Industry-Standard Metrics** - Sharpe, Sortino, Calmar, Information Ratio, MCC, etc.
✅ **Walk-Forward Validation** - 1 year training, 3 months testing windows
✅ **Kelly Criterion Scoring** - Risk-adjusted position sizing
✅ **Top 5 Trade Selection** - Automatic ranking and selection
✅ **Statistical Significance** - T-statistics and p-values
✅ **Validation Thresholds** - Minimum Sharpe 1.0, Max DD -15%, Win Rate 55%

## Quick Start

```bash
# Run complete backtest workflow
python backtesting/run_backtest.py

# Or use the executable directly
./backtesting/run_backtest.py
```

This will:
1. Validate all strategies
2. Analyze portfolio performance
3. Select top 5 trades
4. Export trades to CSV

## Usage Examples

### Validate All Strategies

```bash
python backtesting/run_backtest.py --validate
```

Output:
- Sharpe ratio, returns, drawdowns for each strategy
- Pass/fail status against minimum thresholds
- Summary of which strategies are viable

### Analyze Portfolio Performance

```bash
python backtesting/run_backtest.py --performance
```

Output:
- Overall portfolio metrics (Sharpe, Sortino, Calmar, etc.)
- Per-strategy breakdown
- Statistical significance tests
- Win rate, profit factor, drawdown analysis

### Select Top Trades

```bash
# Default: Top 5 trades with $100k capital
python backtesting/run_backtest.py --trades

# Custom: Top 10 trades with $250k capital
python backtesting/run_backtest.py --trades --num-trades 10 --capital 250000 --export
```

Output:
- Ranked list of signals by composite score
- Position sizes (Kelly Criterion at 25% fractional)
- Stop-loss and take-profit levels (2:1 reward/risk)
- Total capital allocation
- CSV export for execution

### Quick Validation (Single Strategy)

```bash
python backtesting/run_backtest.py --quick pairs_trading
python backtesting/run_backtest.py --quick sentiment_trading
python backtesting/run_backtest.py --quick volatility_trading
```

Output:
- Fast validation using single train/test split
- Key metrics and pass/fail status
- Useful for development and debugging

### Full Workflow with Custom Settings

```bash
python backtesting/run_backtest.py \
  --num-trades 10 \
  --capital 250000 \
  --export \
  --output-dir backtesting/results
```

## Files

| File | Purpose |
|------|---------|
| `run_backtest.py` | Simple runner script with CLI |
| `backtest_engine.py` | Main orchestrator |
| `metrics_calculator.py` | Industry-standard validation metrics |
| `walk_forward_validator.py` | Time series validation |
| `trade_ranker.py` | Kelly Criterion trade ranking |
| `results/` | Output directory for CSV exports |

## Metrics Calculated

### Trading Performance
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside-only risk adjustment
- **Calmar Ratio** - Return vs max drawdown
- **Information Ratio** - Excess return vs tracking error
- **Max Drawdown** - Largest peak-to-trough decline
- **Win Rate** - Percentage of profitable trades
- **Profit Factor** - Gross profit / gross loss

### Classification Metrics (for ML models)
- **Precision** - True positives / predicted positives
- **Recall** - True positives / actual positives
- **F1 Score** - Harmonic mean of precision and recall
- **Accuracy** - Correct predictions / total predictions
- **MCC** - Matthews Correlation Coefficient

### Regression Metrics (for price prediction)
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of determination
- **Directional Accuracy** - Correctly predicted direction

### Statistical Significance
- **T-Statistic** - Test if returns differ from zero
- **P-Value** - Probability of results by chance

## Validation Thresholds

Strategies must pass these minimum thresholds:

| Metric | Minimum | Notes |
|--------|---------|-------|
| Sharpe Ratio | 1.0 | Industry standard for viable strategy |
| Max Drawdown | -15% | Risk management requirement |
| Win Rate | 55% | Must be better than random |
| Profit Factor | 1.5 | Gross profit must exceed losses |
| T-Statistic | 2.0 | Statistical significance |
| P-Value | 0.05 | 95% confidence level |

## Kelly Criterion Position Sizing

**Formula:**
```
Kelly% = (p * b - q) / b
```

Where:
- `p` = Win rate
- `q` = Loss rate (1 - p)
- `b` = Win/loss ratio (avg_win / avg_loss)

**Application:**
- Use **fractional Kelly at 25%** to reduce risk
- Cap individual positions at **10% of capital**
- Adjust for signal strength and Sharpe ratio
- Reduce size in high volatility environments

**Example:**
```
Win rate: 60%
Avg win: 3%
Avg loss: 2%

Kelly% = (0.60 * 1.5 - 0.40) / 1.5 = 33%
Fractional (25%): 33% * 0.25 = 8.25%
Position size: 8.25% of capital
```

## Trade Ranking Composite Score

Trades are ranked by composite score:

```
Score = Kelly% × Signal_Strength × Sharpe_Adjustment × Liquidity_Factor × Vol_Adjustment
```

**Components:**
1. **Kelly%** - Historical win rate and risk/reward
2. **Signal Strength** - Model confidence (0-1)
3. **Sharpe Adjustment** - Strategy quality multiplier
4. **Liquidity Factor** - Penalize illiquid stocks
5. **Volatility Adjustment** - Reduce size in high VIX

## Walk-Forward Validation

**Process:**
1. Train on 12 months of data
2. Test on next 3 months
3. Roll forward by 3 months
4. Repeat across all available history

**Benefits:**
- Prevents lookahead bias
- Tests across different market regimes
- Validates consistency over time
- Industry-standard for time series

**Example Timeline:**
```
Window 1: Train Jan-Dec 2023, Test Jan-Mar 2024
Window 2: Train Apr 2023-Mar 2024, Test Apr-Jun 2024
Window 3: Train Jul 2023-Jun 2024, Test Jul-Sep 2024
...
```

## Output Format

### CSV Export (for execution)

```csv
symbol,action,num_shares,entry_price,stop_loss_price,take_profit_price,position_size,score,signal_strength,strategy_name,kelly_fraction,strategy_win_rate,strategy_sharpe
AAPL,BUY,500,150.00,147.00,156.00,0.0825,0.0825,0.85,sentiment_trading,0.33,0.62,1.85
MSFT,BUY,300,380.00,372.40,395.20,0.0756,0.0756,0.78,pairs_trading,0.28,0.58,1.42
...
```

### Console Output

```
================================================================================
TRADE RANKING & SELECTION
================================================================================

Total Signals: 47
Selecting Top: 5

────────────────────────────────────────────────────────────────────────────────
TOP TRADES
────────────────────────────────────────────────────────────────────────────────

#1 AAPL (LONG) - sentiment_trading
  Score:          0.0825
  Signal Strength: 0.85
  Position Size:   8.25% ($8,250)
  Shares:          55
  Entry Price:     $150.00
  Stop Loss:       $147.00 (-2.0%)
  Take Profit:     $156.00 (+4.0%)
  Kelly Fraction:  33.00%
  Strategy Stats:  WR=62.0%, Sharpe=1.85

...

────────────────────────────────────────────────────────────────────────────────
Total Capital Allocated: 42.50% ($42,500)
Cash Remaining: 57.50% ($57,500)
================================================================================
```

## Integration with Automated Pipeline

The backtesting system integrates seamlessly with the automated data collection pipeline:

1. **Daily Runner** (9:30 AM) - Collects fundamental data, news, sentiment
2. **Intraday Runner** (continuous) - Updates prices, technical indicators
3. **Strategy Execution** (9:55 AM) - Generates trading signals
4. **Backtesting** (on demand) - Validates strategies and selects top trades

## Development Workflow

1. **Develop Strategy** - Create new strategy in `strategies/`
2. **Generate Signals** - Run `python strategies/run_strategies.py`
3. **Quick Validation** - `python backtesting/run_backtest.py --quick my_strategy`
4. **Full Validation** - `python backtesting/run_backtest.py --validate`
5. **Select Trades** - `python backtesting/run_backtest.py --trades`
6. **Execute** - Use exported CSV for order execution

## Advanced Usage

### Custom Database Path

```bash
python backtesting/run_backtest.py --db /path/to/custom.db
```

### Programmatic Use

```python
from backtesting.backtest_engine import BacktestEngine

# Create engine
engine = BacktestEngine()

# Validate single strategy
result = engine.quick_validation('pairs_trading')

# Analyze portfolio
performance = engine.analyze_portfolio_performance(lookback_days=90)

# Get top trades
top_trades = engine.get_top_trades(num_trades=5, total_capital=100000)

# Complete workflow
results = engine.run_complete_backtest(
    num_trades=10,
    total_capital=250000,
    export_csv=True
)
```

### Custom Metrics

```python
from backtesting.metrics_calculator import MetricsCalculator

calc = MetricsCalculator(risk_free_rate=0.04)

# Classification metrics
metrics = calc.classification_metrics(y_true, y_pred)

# Trading metrics
metrics = calc.trading_metrics(returns, periods_per_year=252)

# Check thresholds
passes, reason = calc.passes_thresholds(metrics)
```

## Troubleshooting

### No Signals Found

**Problem:** `❌ No signals found for this strategy`

**Solution:** Ensure strategy has generated signals:
```bash
python strategies/run_strategies.py
```

### Insufficient Data

**Problem:** `Insufficient data for validation (need 3 windows)`

**Solution:** Need at least 18 months of historical data (12 month train + 3 month test × 2 windows minimum). Collect more data or use `--quick` validation.

### Database Not Found

**Problem:** `unable to open database file`

**Solution:** Check external drive is mounted:
```bash
ls /Volumes/Vault
# Should show: 85_assets_prediction.db
```

### All Strategies Fail Validation

**Problem:** All strategies fail minimum thresholds

**Solution:**
1. Review strategy parameters and regularization
2. Check for overfitting (train vs test performance gap)
3. Verify data quality and completeness
4. Consider relaxing thresholds for development

## Best Practices

1. **Always validate before trading** - Don't trade unvalidated strategies
2. **Use fractional Kelly** - Full Kelly is too aggressive, use 25%
3. **Respect position limits** - Never exceed 10% per position
4. **Monitor drawdowns** - Stop trading if drawdown exceeds -15%
5. **Walk-forward validate** - Test across different market conditions
6. **Export trades to CSV** - Keep audit trail of all selections
7. **Review before executing** - Final manual check of top trades

---

**Created:** 2025-11-17
**System:** Streamlined Backtesting Engine
**Database:** `/Volumes/Vault/85_assets_prediction.db`
