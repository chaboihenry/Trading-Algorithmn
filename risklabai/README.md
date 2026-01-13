
# RiskLabAI Trading Strategy Integration

This package implements state-of-the-art financial machine learning techniques from Marcos López de Prado's books:
- **"Advances in Financial Machine Learning"**
- **"Machine Learning for Asset Managers"**

## Architecture Overview

```
risklabai/
├── data_structures/    # Information-driven bars (dollar, volume, imbalance)
├── labeling/          # Triple-barrier method & meta-labeling
├── features/          # Fractional differentiation & feature importance
├── sampling/          # CUSUM filter for event-driven sampling
├── cross_validation/  # Purged K-fold to prevent leakage
├── portfolio/         # Hierarchical Risk Parity (HRP)
└── strategy/          # Main strategy orchestrating all components
```

## Key Innovations

### 1. Information-Driven Bars
Instead of time-based sampling, we use:
- **Dollar Bars**: Sample when $X traded
- **Volume Bars**: Sample when X shares traded
- **Imbalance Bars**: Sample when buy/sell imbalance exceeds threshold

**Why it matters**: Better statistical properties (more normal returns, less autocorrelation)

### 2. Triple-Barrier Labeling
Labels based on three barriers:
- **Upper barrier** (take-profit): Price rises X% × volatility
- **Lower barrier** (stop-loss): Price falls Y% × volatility
- **Vertical barrier** (timeout): T periods elapse

**Why it matters**: Labels match real trading mechanics, adapt to volatility

### 3. Meta-Labeling for Bet Sizing
Separates two problems:
1. **Primary model**: Direction (long/short)
2. **Meta model**: Should we trade? How much?

**Why it matters**: Reduces overfitting, better position sizing

### 4. Fractional Differentiation
Finds minimum differentiation order `d` that achieves stationarity while preserving memory.

- `d=0`: Original prices (non-stationary, full memory)
- `d=1`: Returns (stationary, no memory)
- `d≈0.4`: Sweet spot (stationary with memory)

**Why it matters**: Stationary features without information loss

### 5. CUSUM Event Filtering
Only generates labels when cumulative price change exceeds threshold.

**Why it matters**: Focuses on significant events, reduces noise

### 6. Purged K-Fold Cross-Validation
Removes training samples that overlap with test samples in time.

**Why it matters**: Prevents lookahead bias, realistic performance estimates

### 7. Hierarchical Risk Parity (HRP)
Portfolio optimization via:
1. Hierarchical clustering of assets
2. Quasi-diagonalization of covariance matrix
3. Recursive bisection for weight allocation

**Why it matters**: Stable portfolios without matrix inversion

## Usage

### Testing Components

```bash
python test_risklabai.py
```

This will test all components individually.

### Training the Strategy

```python
from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy
import pandas as pd

# Initialize
strategy = RiskLabAIStrategy(
    profit_taking=2.0,
    stop_loss=2.0,
    max_holding=10
)

# Train from tick database (CUSUM → imbalance bars → fractional diff → labels)
results = strategy.train_from_ticks("SPY")

# Generate signals
signal, bet_size = strategy.predict(recent_bars)
```

### Live Trading with Lumibot

```python
from core.risklabai_combined import RiskLabAICombined
from lumibot.brokers import Alpaca
from config.settings import ALPACA_CONFIG

# Setup broker
broker = Alpaca(ALPACA_CONFIG)

# Run strategy
strategy = RiskLabAICombined(
    broker=broker,
    parameters={'symbols': ['SPY', 'QQQ', 'IWM']}
)

strategy.run_all()
```

## Implementation Details

### Data Flow

```
Price Data
    ↓
CUSUM Filter → Events
    ↓
Feature Engineering → Stationary Features
    ↓
Triple-Barrier Labeling → Dynamic Labels
    ↓
Primary Model Training → Direction Predictions
    ↓
Meta-Labeling → Bet Sizing Labels
    ↓
Meta Model Training → Confidence Scores
    ↓
HRP Portfolio Optimization → Position Weights
    ↓
Trade Execution
```

### Model Retraining

Models automatically retrain weekly to adapt to changing market conditions.

### Risk Management

All existing risk management is preserved:
- Stop-loss protection
- Position size limits
- Hedge management

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profit_taking` | 2.0 | Take-profit multiplier vs volatility |
| `stop_loss` | 2.0 | Stop-loss multiplier vs volatility |
| `max_holding` | 10 | Maximum holding period (days) |
| `n_cv_splits` | 5 | Cross-validation folds |

## Performance Metrics

The strategy tracks:
- **Primary model accuracy**: Direction prediction performance
- **Meta model accuracy**: Bet sizing performance
- **Feature importance**: Which features drive predictions
- **Portfolio statistics**: Returns, volatility, Sharpe ratio

## References

1. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
3. RiskLabAI Library: [Documentation](https://github.com/risklabai/RiskLabAI)

## Next Steps

1. ✓ Install RiskLabAI: `pip install RiskLabAI`
2. ✓ Test components: `python test_risklabai.py`
3. ⏳ Paper trade: Update `core/live_trader.py` to use `RiskLabAICombined`
4. ⏳ Monitor performance
5. ⏳ Tune parameters based on results

## Troubleshooting

**Import errors**: Ensure you're in the `trading` conda environment:
```bash
conda activate trading
```

**Insufficient data**: Need at least 500 bars for training. Use daily data.

**Training fails**: Check for missing values in data. Ensure `close`, `open`, `high`, `low`, `volume` columns exist.

## Support

For issues with:
- **RiskLabAI library**: Check [RiskLabAI GitHub](https://github.com/risklabai/RiskLabAI)
- **This integration**: Review logs in `logs/` directory
- **Trading strategy**: See `docs/` for strategy documentation
