# Backtesting Quick Start Guide

## 5-Minute Setup

### Step 1: Run Your First Backtest

```bash
cd backtesting
python run_backtest.py
```

This will:
- Backtest all your strategies
- Calculate 50+ performance metrics
- Generate visual reports
- Save results to `backtest_results/YYYYMMDD_HHMMSS/`

### Step 2: View Results

After running, check these files:

```
backtest_results/YYYYMMDD_HHMMSS/
├── backtest_report.png          ← Open this for visual overview
├── performance_metrics.txt      ← Read this for detailed stats
├── signal_accuracy.txt          ← Check signal quality
└── trades_detailed.csv          ← All trades in spreadsheet
```

### Step 3: Interpret Key Metrics

Look at these critical numbers in the terminal output:

```
Win Rate: XX%          → % of profitable trades (want >50%)
Total Return: XX%      → Overall profit (higher is better)
Sharpe Ratio: X.XX     → Risk-adjusted return (want >1.0)
Max Drawdown: XX%      → Worst loss period (lower is better)
Profit Factor: X.XX    → Wins/Losses ratio (want >1.5)
```

## Common Use Cases

### Test a Specific Strategy

```bash
python run_backtest.py --strategy PairsTradingStrategy
```

### Test Recent Performance (Last 3 Months)

```bash
python run_backtest.py --start-date 2024-10-01
```

### Test with Lower Risk

```bash
python run_backtest.py --max-position 0.05 --max-holding-days 10
```

### Compare All Strategies

```bash
# Test each strategy separately
python run_backtest.py --strategy PairsTradingStrategy --output ./results/pairs
python run_backtest.py --strategy SentimentTradingStrategy --output ./results/sentiment
python run_backtest.py --strategy VolatilityTradingStrategy --output ./results/volatility

# Compare the performance_metrics.txt files
```

## Understanding Your Results

### Good Strategy Indicators ✅
- Win rate > 50%
- Sharpe ratio > 1.0
- Max drawdown < 20%
- Profit factor > 1.5
- Consistent returns over time

### Red Flags 🚩
- Win rate < 40%
- Sharpe ratio < 0.5
- Max drawdown > 30%
- Profit factor < 1.0
- Only a few trades (< 30)

## What to Do Next

### If Results Look Good ✅
1. Test on different time periods
2. Verify with out-of-sample data
3. Check strategy correlation
4. Consider live paper trading

### If Results Look Bad 🚩
1. Check signal accuracy metrics
2. Analyze exit reasons (stop loss vs take profit)
3. Review strategy-specific performance
4. Consider adjusting parameters
5. Check if strategy logic needs improvement

## Advanced Usage

### Programmatic Access

```python
from backtesting import BacktestEngine, PerformanceMetrics

# Run backtest
engine = BacktestEngine(initial_capital=100000)
results = engine.run_backtest()
trades_df = engine.get_trades_df()

# Get metrics
metrics = PerformanceMetrics(
    trades_df,
    results['portfolio_value'],
    100000
)

# Access specific metrics
all_metrics = metrics.calculate_all_metrics()
print(f"Sharpe Ratio: {all_metrics['sharpe_ratio']:.3f}")
print(f"Win Rate: {all_metrics['win_rate']:.2f}%")
```

### Custom Analysis

```python
# Filter trades by strategy
pairs_trades = trades_df[trades_df['strategy'] == 'PairsTradingStrategy']

# Calculate custom metrics
avg_return = pairs_trades['pnl'].mean()
win_rate = (pairs_trades['pnl'] > 0).sum() / len(pairs_trades) * 100

# Analyze by symbol
symbol_performance = trades_df.groupby('symbol')['pnl'].sum()
best_symbols = symbol_performance.nlargest(5)
```

## Troubleshooting

### "No signals found"
→ Run your strategies first to generate signals:
```bash
cd ../strategies
python run_strategies.py
```

### "No trades executed"
→ Check your date range and ensure signals exist:
```bash
sqlite3 /Volumes/Vault/85_assets_prediction.db "SELECT COUNT(*) FROM trading_signals;"
```

### "Permission denied"
→ Make scripts executable:
```bash
chmod +x run_backtest.py example_usage.py
```

## Tips for Better Results

1. **Always include transaction costs** - Use realistic commission (0.1-0.2%) and slippage (0.05-0.1%)

2. **Test multiple time periods** - Don't just test one date range

3. **Check signal strength correlation** - Better signals should perform better

4. **Monitor exit reasons** - If most trades hit stop loss, adjust strategy

5. **Compare to benchmark** - Is your strategy better than buy-and-hold?

6. **Consider position sizing** - Don't risk too much per trade (max 10%)

7. **Watch drawdown patterns** - Long drawdowns can be psychologically difficult

## Next Steps

1. ✅ Run basic backtest
2. ✅ Understand key metrics
3. ✅ Compare strategies
4. ⬜ Optimize parameters (careful of overfitting!)
5. ⬜ Test on out-of-sample data
6. ⬜ Implement best strategy in paper trading

## Get Help

- Read full documentation: [README.md](README.md)
- Run examples: `python example_usage.py`
- Check code comments in source files

---

**Remember:** Past performance doesn't guarantee future results. Always paper trade before risking real capital!
