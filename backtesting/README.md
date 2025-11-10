# Backtesting Framework

A comprehensive backtesting system for evaluating trading strategy performance and signal accuracy.

## Overview

This backtesting framework provides realistic simulation of trading strategies with:

- **Realistic execution modeling** - Commission costs, slippage, and position sizing
- **Comprehensive performance metrics** - 50+ metrics including Sharpe ratio, max drawdown, win rate, etc.
- **Signal accuracy analysis** - Evaluate prediction quality and reliability
- **Visual analytics** - Automated generation of performance charts and reports
- **Strategy comparison** - Compare multiple strategies side-by-side

## Features

### 1. Backtest Engine ([backtest_engine.py](backtest_engine.py))

Core backtesting functionality with realistic trade execution:

- **Order Execution**
  - Commission costs (default: 0.1%)
  - Slippage modeling (default: 0.05%)
  - Position sizing based on signal strength
  - Stop loss and take profit handling

- **Risk Management**
  - Maximum position size limits
  - Maximum holding period enforcement
  - Capital allocation tracking

- **Trade Management**
  - Automatic position closing on stop loss/take profit
  - Time-based exit after max holding period
  - Portfolio value tracking over time

### 2. Performance Metrics ([performance_metrics.py](performance_metrics.py))

Calculate 50+ comprehensive performance metrics:

#### Basic Metrics
- Total return ($ and %)
- Average return per trade
- Best/worst trades
- Median return

#### Return Metrics
- Annualized return
- Monthly return (estimated)
- Cumulative returns

#### Risk Metrics
- Volatility (annualized)
- Downside volatility
- Value at Risk (95%)
- Conditional VaR

#### Risk-Adjusted Metrics
- **Sharpe Ratio** - Risk-adjusted returns
- **Sortino Ratio** - Downside risk-adjusted returns
- **Calmar Ratio** - Return to max drawdown
- **Information Ratio** - Excess return per unit of tracking error

#### Win/Loss Analysis
- Win rate
- Profit factor
- Average win/loss
- Win/loss ratio
- Consecutive win/loss streaks
- Exit reason breakdown

#### Drawdown Analysis
- Maximum drawdown ($ and %)
- Drawdown duration
- Average drawdown
- Recovery time

#### Trade Statistics
- Average holding period
- Position size analysis
- Trades per symbol
- Unique symbols traded

### 3. Signal Accuracy Analyzer ([signal_accuracy.py](signal_accuracy.py))

Evaluate the quality and reliability of trading signals:

#### Overall Accuracy
- Percentage of profitable signals
- Total signals tested
- Profitable vs unprofitable breakdown

#### Signal Strength Analysis
- Correlation between signal strength and profitability
- Win rate by strength bins (weak/moderate/strong/very strong)
- Performance segmentation

#### Strategy-Specific Accuracy
- Per-strategy win rates
- Strategy ranking by accuracy
- Comparative performance metrics

#### Signal Type Analysis
- BUY vs SELL signal accuracy
- Directional bias detection

#### Exit Reason Analysis
- Performance by exit type (stop loss, take profit, time exit)
- Exit reason distribution

#### Prediction Horizon Analysis
- Accuracy by holding period (0-3d, 3-7d, 7-14d, etc.)
- Optimal holding period identification

#### Symbol-Level Analysis
- Top performing symbols
- Bottom performing symbols
- Per-symbol statistics

#### Directional Accuracy
- 1-day, 5-day, 10-day directional accuracy
- Forward returns analysis
- Signal reliability over time horizons

### 4. Visualizations ([visualizations.py](visualizations.py))

Automated generation of professional charts:

#### Comprehensive Report
Single-page dashboard with:
- Portfolio value over time
- Drawdown chart
- Returns distribution
- Strategy performance comparison
- Exit reasons pie chart
- Cumulative returns
- Monthly returns
- Trade duration analysis

#### Individual Charts
Separate high-resolution charts for:
- Portfolio value evolution
- Drawdown timeline
- Returns histogram
- Strategy win rates
- Cumulative P&L

All charts are publication-ready with proper formatting, labels, and annotations.

## Usage

### Basic Usage

Run backtest for all strategies:

```bash
python run_backtest.py
```

### Advanced Usage

#### Backtest Specific Strategy

```bash
python run_backtest.py --strategy PairsTradingStrategy
```

#### Backtest with Date Range

```bash
python run_backtest.py --start-date 2024-01-01 --end-date 2024-12-31
```

#### Custom Parameters

```bash
python run_backtest.py \
  --capital 50000 \
  --commission 0.002 \
  --slippage 0.001 \
  --max-position 0.05 \
  --max-holding-days 15
```

#### Custom Output Directory

```bash
python run_backtest.py --output ./my_backtest_results
```

#### Backtest Multiple Strategies in Sequence

```bash
# Pairs Trading
python run_backtest.py --strategy PairsTradingStrategy --output ./results/pairs

# Sentiment Trading
python run_backtest.py --strategy SentimentTradingStrategy --output ./results/sentiment

# Volatility Trading
python run_backtest.py --strategy VolatilityTradingStrategy --output ./results/volatility
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--strategy` | Strategy name to backtest | All strategies |
| `--start-date` | Start date (YYYY-MM-DD) | Earliest available |
| `--end-date` | End date (YYYY-MM-DD) | Latest available |
| `--capital` | Initial capital | 100,000 |
| `--commission` | Commission rate | 0.001 (0.1%) |
| `--slippage` | Slippage rate | 0.0005 (0.05%) |
| `--max-position` | Max position size (fraction) | 0.1 (10%) |
| `--max-holding-days` | Max holding period | 20 days |
| `--output` | Output directory | ./backtest_results |
| `--db` | Database path | /Volumes/Vault/85_assets_prediction.db |

## Output Files

Each backtest run creates a timestamped directory with:

```
backtest_results/
└── YYYYMMDD_HHMMSS/
    ├── backtest_report.png          # Comprehensive visual report
    ├── performance_metrics.txt      # All performance metrics
    ├── signal_accuracy.txt          # Signal accuracy analysis
    ├── trades_detailed.csv          # Detailed trade log
    └── charts/                      # Individual chart files
        ├── portfolio_value.png
        ├── drawdown.png
        ├── returns_distribution.png
        ├── strategy_performance.png
        └── cumulative_returns.png
```

## Programmatic Usage

You can also use the backtesting components programmatically:

```python
from backtest_engine import BacktestEngine
from performance_metrics import PerformanceMetrics
from signal_accuracy import SignalAccuracyAnalyzer
from visualizations import BacktestVisualizer

# Initialize and run backtest
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

results = engine.run_backtest(
    strategy_name="PairsTradingStrategy",
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Get trades
trades_df = engine.get_trades_df()

# Calculate metrics
metrics = PerformanceMetrics(
    trades_df=trades_df,
    portfolio_values=results['portfolio_value'],
    initial_capital=100000
)

metrics.print_summary()
all_metrics = metrics.calculate_all_metrics()

# Analyze accuracy
analyzer = SignalAccuracyAnalyzer()
accuracy = analyzer.analyze_signal_accuracy(trades_df)
reliability = analyzer.analyze_signal_reliability()

# Create visualizations
viz = BacktestVisualizer(
    trades_df=trades_df,
    portfolio_values=results['portfolio_value'],
    initial_capital=100000
)

viz.create_full_report('my_report.png')
viz.plot_individual_charts('./my_charts')
```

## Interpreting Results

### Key Metrics to Focus On

#### 1. Profitability
- **Total Return %** - Overall strategy performance
- **Annualized Return** - Performance normalized to yearly basis
- **Win Rate** - Percentage of profitable trades (aim for >50%)

#### 2. Risk Management
- **Max Drawdown** - Largest peak-to-trough decline (lower is better)
- **Volatility** - Strategy stability (lower is better for same returns)
- **VaR (95%)** - Worst expected daily loss 95% of the time

#### 3. Risk-Adjusted Performance
- **Sharpe Ratio** - Return per unit of risk
  - \> 1.0 = Good
  - \> 2.0 = Very Good
  - \> 3.0 = Excellent
- **Sortino Ratio** - Similar to Sharpe but focuses on downside risk
- **Calmar Ratio** - Return relative to max drawdown

#### 4. Consistency
- **Profit Factor** - Gross profit / Gross loss (aim for >1.5)
- **Win/Loss Ratio** - Average win / Average loss
- **Consecutive Losses** - Risk of drawdown streaks

#### 5. Signal Quality
- **Overall Accuracy** - Percentage of correct signals
- **Directional Accuracy** - How often signals predict correct direction
- **Strength Correlation** - Do stronger signals perform better?

### Red Flags to Watch For

- Win rate < 40% (unless win/loss ratio is very high)
- Max drawdown > 30%
- Sharpe ratio < 0.5
- Profit factor < 1.0
- Very low number of trades (< 30) - insufficient statistical significance
- High correlation between losses (cluster risk)

### Comparing Strategies

When comparing multiple strategies, consider:

1. **Risk-adjusted returns** (Sharpe/Sortino) not just absolute returns
2. **Consistency** - Lower volatility is preferable
3. **Drawdown characteristics** - How bad can it get?
4. **Strategy correlation** - Combine uncorrelated strategies
5. **Market regime dependency** - Does it work in all conditions?

## Customization

### Adjusting Risk Parameters

Edit parameters in `run_backtest.py` or pass via CLI:

```python
# More conservative
--max-position 0.05    # 5% max per trade
--max-holding-days 10  # Shorter holding period

# More aggressive
--max-position 0.20    # 20% max per trade
--max-holding-days 30  # Longer holding period
```

### Adding Custom Metrics

Extend `PerformanceMetrics` class in [performance_metrics.py](performance_metrics.py):

```python
def _calculate_custom_metric(self) -> Dict:
    """Calculate your custom metric"""
    # Your calculation here
    return {'custom_metric': value}
```

### Creating Custom Visualizations

Extend `BacktestVisualizer` class in [visualizations.py](visualizations.py):

```python
def _plot_custom_chart(self, ax) -> None:
    """Create your custom visualization"""
    # Your plotting code here
    pass
```

## Best Practices

### 1. Use Sufficient Historical Data
- Minimum 1 year of data
- Ideally 3-5 years for robust results
- Include different market regimes (bull, bear, sideways)

### 2. Account for Transaction Costs
- Always include realistic commission and slippage
- Use conservative estimates (0.1-0.2% total cost per trade)

### 3. Avoid Overfitting
- Test on out-of-sample data
- Use walk-forward analysis
- Don't optimize too many parameters

### 4. Consider Liquidity
- Ensure position sizes are realistic for actual trading
- Account for market impact on larger positions

### 5. Multiple Strategy Backtests
- Test each strategy independently
- Compare strategies on same time period
- Consider portfolio allocation across strategies

## Requirements

Install required packages:

```bash
pip install pandas numpy matplotlib scipy scikit-learn
```

## Troubleshooting

### No signals found
- Check that strategies have been run and signals saved to database
- Verify date range includes periods with signals
- Check strategy name spelling

### Empty backtest results
- Ensure price data exists for signal dates
- Check that signals have entry_price, stop_loss, take_profit
- Verify database connection

### Visualization errors
- Ensure matplotlib is installed
- Check that output directory has write permissions
- Verify sufficient data for plotting (minimum 2 data points)

### Performance issues
- Reduce date range for large datasets
- Run backtests for individual strategies rather than all at once
- Consider sampling data for quick tests

## Future Enhancements

Potential improvements:

- [ ] Walk-forward optimization
- [ ] Monte Carlo simulation
- [ ] Strategy portfolio optimization
- [ ] Real-time performance monitoring
- [ ] Automated parameter tuning
- [ ] Machine learning for strategy selection
- [ ] Multi-asset backtesting
- [ ] Advanced order types (limit, stop-limit, etc.)
- [ ] Benchmark comparison (S&P 500, etc.)
- [ ] Tax-aware backtesting

## Support

For issues or questions:
1. Check this README
2. Review example usage in [run_backtest.py](run_backtest.py)
3. Examine test output in `backtest_results/` directory

## License

Part of the Integrated Trading Agent project.
