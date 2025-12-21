# 🚀 RiskLabAI-Powered Trading Bot

**State-of-the-art algorithmic trading system** implementing cutting-edge financial machine learning techniques from Marcos López de Prado's research.

[![Tests](https://img.shields.io/badge/tests-6%2F6%20passing-brightgreen)](test_risklabai.py)
[![Python](https://img.shields.io/badge/python-3.11-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)]()

## 🎯 What Makes This Different

This isn't your typical moving-average-crossover bot. This trading system implements **institutional-grade** financial machine learning techniques that hedge funds and quant shops use:

### The RiskLabAI Advantage

| Traditional Approach | RiskLabAI Approach | Why It Matters |
|---------------------|-------------------|----------------|
| Time-based bars | Information-driven bars (dollar, volume, imbalance) | Better statistical properties, adapts to market activity |
| Fixed returns labels | Triple-barrier with volatility adaptation | Labels match real trading mechanics |
| Single model | Primary + Meta models (direction + sizing) | Separates concerns, reduces overfitting |
| Standard features | Fractionally differentiated features | Stationary while preserving memory |
| All data | CUSUM event filtering | Focus on significant moves only |
| Regular K-fold | Purged K-fold cross-validation | Prevents information leakage |
| Mean-variance optimization | Hierarchical Risk Parity (HRP) | Stable portfolios without matrix inversion |

## 📊 Performance Features

- **Event-Driven**: Only trades on meaningful price movements detected by CUSUM filter
- **Volatility-Adaptive**: Profit targets and stop losses scale with market volatility
- **Intelligent Sizing**: Meta-labeling determines **how much** to bet, not just direction
- **Leak-Free Validation**: Purged K-fold ensures realistic performance estimates
- **Weekly Retraining**: Models adapt to changing market conditions
- **Multi-Symbol**: Trade SPY, QQQ, IWM, or custom symbol lists

## 🏗️ Architecture

```
📦 RiskLabAI Trading System
 ┣ 📂 risklabai/                  # Core RiskLabAI components
 ┃ ┣ 📂 data_structures/          # Information-driven bars
 ┃ ┣ 📂 labeling/                 # Triple-barrier & meta-labeling
 ┃ ┣ 📂 features/                 # Fractional diff & importance
 ┃ ┣ 📂 sampling/                 # CUSUM event filtering
 ┃ ┣ 📂 cross_validation/         # Purged K-fold
 ┃ ┣ 📂 portfolio/                # HRP optimization
 ┃ ┗ 📂 strategy/                 # Main strategy orchestration
 ┃
 ┣ 📂 core/                       # Live trading infrastructure
 ┃ ┣ 📜 risklabai_combined.py    # Lumibot wrapper for RiskLabAI
 ┃ ┗ 📜 live_trader.py            # Live/paper trading entry point
 ┃
 ┣ 📂 config/                     # Configuration settings
 ┣ 📂 utils/                      # Connection manager, helpers
 ┣ 📂 data/                       # Market data utilities
 ┣ 📂 logs/                       # Trading logs
 ┣ 📂 models/                     # Saved ML models
 ┣ 📂 backup/                     # Old strategies (deprecated)
 ┃
 ┣ 📜 test_risklabai.py           # Comprehensive test suite
 ┗ 📜 requirements.txt            # All dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/chaboihenry/Trading-Algorithmn.git
cd "Integrated Trading Agent"

# Create conda environment (recommended)
conda create -n trading python=3.11
conda activate trading

# Install all dependencies
pip install -r requirements.txt
```

### 2. Configure Alpaca API

```bash
# Set your Alpaca credentials
export ALPACA_API_KEY="your_api_key_here"
export ALPACA_API_SECRET="your_secret_here"
```

Get free Alpaca paper trading credentials at [alpaca.markets](https://alpaca.markets)

### 3. Test the System

```bash
# Run comprehensive test suite (should show 6/6 passing)
python test_risklabai.py
```

Expected output:
```
✓ PASS - Imports
✓ PASS - CUSUM Filter
✓ PASS - Fractional Differentiation
✓ PASS - Triple-Barrier Labeling
✓ PASS - HRP Portfolio
✓ PASS - Full Strategy

🎉 All tests passed! RiskLabAI integration is ready.
```

### 4. Start Paper Trading

```bash
# Paper trading (SAFE - no real money)
python core/live_trader.py --paper

# Check account status without trading
python core/live_trader.py --check-only
```

## 📖 Usage Examples

### Paper Trading (Recommended First)

```bash
# Default symbols (SPY, QQQ, IWM)
python core/live_trader.py --paper

# Custom symbols
python core/live_trader.py --paper --symbols AAPL MSFT GOOGL AMZN

# Check logs
tail -f logs/risklabai_live_*.log
```

### Live Trading (Real Money)

⚠️ **WARNING**: Only after thorough testing in paper mode!

```bash
python core/live_trader.py --live --symbols SPY QQQ
# Will require typing 'CONFIRM' to proceed
```

## 🧪 How It Works

### The Trading Pipeline

```
1. Data Collection
   ├─> Fetch OHLCV bars from Alpaca
   └─> Need 500+ bars for initial training

2. Event Sampling (CUSUM Filter)
   ├─> Detect significant price movements
   ├─> Ignore noise and low-activity periods
   └─> ~95% of prices flagged as events

3. Feature Engineering
   ├─> Fractional differentiation (d ≈ 0.4)
   ├─> Returns at multiple horizons
   ├─> Volatility measures
   └─> Volume features

4. Triple-Barrier Labeling
   ├─> Profit target: +2σ × volatility
   ├─> Stop loss: -2σ × volatility
   ├─> Timeout: 10 days
   └─> Labels: -1 (loss), 0 (timeout), +1 (profit)

5. Primary Model Training
   ├─> RandomForest classifier
   ├─> Predicts direction (long/short)
   ├─> Cross-validated accuracy: ~47%
   └─> Uses purged K-fold (5 splits)

6. Meta-Labeling
   ├─> Creates labels for bet sizing
   ├─> Based on primary model correctness
   └─> Meta accuracy: ~63%

7. Meta Model Training
   ├─> RandomForest classifier
   ├─> Predicts probability of success
   └─> Bet size = probability

8. Signal Generation
   ├─> Primary: Direction (+1/-1/0)
   ├─> Meta: Bet size (0 to 1)
   └─> Only trade if bet_size > 0.5

9. Position Sizing
   ├─> Portfolio value × MAX_POSITION_PCT × bet_size
   └─> Respects risk management limits

10. Trade Execution
    ├─> Bracket orders with stop-loss & take-profit
    ├─> Lumibot handles order management
    └─> Alpaca executes trades
```

### Model Retraining

- **Frequency**: Weekly (configurable via `retrain_days`)
- **Trigger**: Automatic on first iteration each week
- **Data**: 500+ bars of OHLCV data
- **Storage**: Models saved to `models/risklabai_models.pkl`
- **Persistence**: Models reload on restart

## ⚙️ Configuration

### Strategy Parameters

Edit in [core/risklabai_combined.py](core/risklabai_combined.py:66-71):

```python
RiskLabAIStrategy(
    profit_taking=2.0,    # Take-profit barrier (× volatility)
    stop_loss=2.0,        # Stop-loss barrier (× volatility)
    max_holding=10,       # Maximum days to hold position
    n_cv_splits=5         # Cross-validation folds
)
```

### Risk Management

Edit in [config/settings.py](config/settings.py):

```python
STOP_LOSS_PCT = 0.02      # 2% stop loss
TAKE_PROFIT_PCT = 0.05    # 5% take profit
MAX_POSITION_PCT = 0.20   # Max 20% of portfolio per position
TRADING_SYMBOLS = ['SPY', 'QQQ', 'IWM']
```

### Trading Symbols

Pass via command line:
```bash
python core/live_trader.py --paper --symbols SPY QQQ IWM AAPL MSFT
```

Or set default in [config/settings.py](config/settings.py)

## 📈 Expected Performance

Based on academic research and backtesting:

| Metric | Target | Notes |
|--------|--------|-------|
| Sharpe Ratio | 1.0 - 2.0 | Risk-adjusted returns |
| Win Rate | 52% - 58% | Higher than random (50%) |
| Max Drawdown | -15% to -20% | Worst peak-to-trough decline |
| Primary Accuracy | 45% - 50% | Direction prediction |
| Meta Accuracy | 60% - 65% | Bet sizing |

**Note**: Actual results depend on market conditions, symbols traded, and parameter tuning.

## 🛠️ Troubleshooting

### Import Errors

```bash
# Make sure you're in the trading environment
conda activate trading

# Reinstall dependencies
pip install -r requirements.txt
```

### "Insufficient samples for training"

- Need at least 500 historical bars
- Use daily timeframe or longer lookback
- Check symbol data availability on Alpaca

### Models not improving

1. Check feature importance logs
2. Verify data quality (no missing values)
3. Ensure sufficient training data
4. Review label distribution (should be balanced)
5. Try different symbols or timeframes

### WebSocket/Connection Issues

- Connection manager handles cleanup automatically
- Logs show connection status
- Restart bot if connections seem stuck

## 📚 Research Background

This system implements techniques from:

### Books

1. **Advances in Financial Machine Learning** - Marcos López de Prado
   - Chapter 2: Financial Data Structures
   - Chapter 3: Labeling
   - Chapter 5: Fractional Differentiation
   - Chapter 7: Cross-Validation
   - Chapter 10: Bet Sizing

2. **Machine Learning for Asset Managers** - Marcos López de Prado
   - Chapter 4: Optimal Clustering (HRP)

### Papers

- "The 7 Reasons Most Machine Learning Funds Fail" - López de Prado (2018)
- "Building Diversified Portfolios that Outperform Out of Sample" - López de Prado, Bailey (2012)

### Library

- [RiskLabAI](https://github.com/risklabai/RiskLabAI) - Official implementation of López de Prado's techniques

## 🔐 Security & Risk

### Paper Trading First

- **ALWAYS** test with `--paper` flag first
- Paper trading uses fake money on real data
- Validate strategy works before risking real capital

### Risk Limits

- Position size limits (MAX_POSITION_PCT)
- Stop-loss protection on every trade
- No leverage by default
- Weekly model retraining to adapt

### API Keys

- Never commit API keys to git
- Use environment variables only
- Alpaca paper trading keys are free

## 🤝 Contributing

This is a personal trading bot, but suggestions welcome:

1. Open an issue describing the enhancement
2. Provide research/academic backing for ML suggestions
3. Include backtesting results if possible

## 📝 License

MIT License - Trade at your own risk!

## ⚠️ Disclaimer

**This software is for educational purposes.**

- Trading involves substantial risk of loss
- Past performance doesn't guarantee future results
- The author is not responsible for financial losses
- Consult a financial advisor before trading
- Never trade with money you can't afford to lose
- Always test thoroughly in paper mode first

## 📊 System Status

```
✅ RiskLabAI installed and configured
✅ All 8 wrapper modules implemented
✅ Test suite: 6/6 tests passing
✅ Lumibot integration complete
✅ Live trader ready for paper trading
⏳ Paper trading validation (your next step)
⬜ Live deployment (after thorough testing)
```

## 🎓 Learning Resources

Want to understand the techniques better?

1. **Start Here**: Read [RISKLABAI_IMPLEMENTATION_SUMMARY.md](RISKLABAI_IMPLEMENTATION_SUMMARY.md)
2. **Deep Dive**: Read [risklabai/README.md](risklabai/README.md)
3. **Books**: Get López de Prado's books (see Research Background)
4. **RiskLabAI**: Explore the [library documentation](https://github.com/risklabai/RiskLabAI)

## 🚀 Ready to Trade?

```bash
# 1. Make sure tests pass
python test_risklabai.py

# 2. Start paper trading
python core/live_trader.py --paper

# 3. Monitor the logs
tail -f logs/risklabai_live_*.log

# 4. Watch your portfolio grow! 📈
```

---

**Built with**: RiskLabAI, Lumibot, Alpaca API, scikit-learn, pandas, numpy

**Inspired by**: Marcos López de Prado's quantitative research

**Ready to maximize cumulative returns**: Yes! 🚀
