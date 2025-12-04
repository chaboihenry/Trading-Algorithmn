# Automated Trading System - Production Documentation

**Status:** ✅ ACTIVE - Running Daily
**Version:** 2.0 (Stacked Ensemble)
**Last Updated:** 2025-11-24

---

## System Overview

Fully automated trading system that:
1. Collects market data daily (prices, fundamentals, sentiment, options)
2. Generates ML features across multiple timeframes
3. Runs 3 base strategies + 1 ensemble meta-learner
4. Produces **top 5 high-confidence trade recommendations** daily
5. Sends email reports with trade signals and risk assessment

---

## Quick Start

### Daily Automation (Already Configured)

The system runs automatically via macOS LaunchAgents:

- **Daily Runner:** 5:00 PM ET (after market close)
  - Location: `~/Library/LaunchAgents/com.trading.daily.plist`
  - Collects data, runs strategies, sends email report

- **Intraday Runner:** Every 30 minutes during market hours
  - Location: `~/Library/LaunchAgents/com.trading.intraday.plist`
  - Updates positions, monitors risk

### Manual Execution

```bash
# Run all strategies and generate signals
python strategies/run_strategies.py

# Retrain all models (weekly recommended)
python strategies/retrain_all_strategies.py --force-full-retrain

# Run data collection
python master_orchestrator/daily_runner.py
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              DATA COLLECTION (Daily)                    │
│  • Price data (OHLCV)                                  │
│  • Fundamentals (P/E, EPS, revenue)                    │
│  • Sentiment (news, social media)                      │
│  • Economic indicators (VIX, bonds, Fed data)          │
│  • Options data (IV, volume, OI)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────┐
│           ML FEATURES AGGREGATION                       │
│  • 16 multi-timeframe features (5d-100d)               │
│  • 3 market regime features (bull/bear/sideways)       │
│  • Technical indicators (RSI, MACD, BB)                │
│  • Volatility metrics (10d, 20d, 60d)                  │
│  • Volume patterns (surge detection)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────┐
│            BASE STRATEGIES (3)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Pairs Trading│  │  Sentiment   │  │ Volatility   │ │
│  │ (Statistical)│  │  (XGBoost)   │  │  (XGBoost)   │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                  │          │
│         v                 v                  v          │
│      Signals          Signals            Signals        │
└─────────┬───────────────┬──────────────────┬───────────┘
          │               │                  │
          v               v                  v
┌─────────────────────────────────────────────────────────┐
│       ENSEMBLE STRATEGY (Stacked Meta-Learning)         │
│  ┌────────────────────────────────────────────────┐    │
│  │  XGBoost Meta-Learner                         │    │
│  │  • 14 meta-features from base predictions     │    │
│  │  • Learns when each strategy is reliable      │    │
│  │  • Discovers non-linear interactions          │    │
│  │  • 60% confidence threshold                   │    │
│  │  • 58.62% test accuracy, 86.3% avg confidence │    │
│  └────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────┐
│         TOP 5 TRADE RECOMMENDATIONS                     │
│  • BUY/SELL signals with entry/exit prices            │
│  • Stop losses calculated                              │
│  • High confidence only (>80%)                         │
│  • Risk assessment included                            │
│  • Emailed daily report                                │
└─────────────────────────────────────────────────────────┘
```

---

## Trading Strategies

### 1. Pairs Trading Strategy
**Type:** Statistical arbitrage
**Approach:** Cointegration-based mean reversion
**Features:**
- Tests 3,570+ potential pairs
- Uses Johansen cointegration test
- Z-score based entry/exit signals
- Relative valuation filtering

### 2. Sentiment Trading Strategy
**Type:** ML-based (XGBoost)
**Approach:** Multi-class classification (5 labels)
**Features:**
- News sentiment + social media analysis
- 48 features (reduced to top 25 via feature selection)
- Multi-timeframe momentum + regime detection
- Incremental learning (daily updates)

### 3. Volatility Trading Strategy
**Type:** ML-based (XGBoost)
**Approach:** Volatility regime prediction
**Features:**
- GARCH volatility forecasting
- Options-based IV analysis
- 39 features (reduced to top 20 via feature selection)
- Economic indicator integration

### 4. Ensemble Strategy (PRIMARY)
**Type:** Stacked meta-learning
**Approach:** XGBoost combines base predictions
**Performance:**
- **58.62% test accuracy**
- **86.3% average confidence**
- **Top 5 signals daily**

**Why it's better:**
- Learns when each strategy is reliable
- Captures non-linear strategy interactions
- Adapts to changing market conditions
- Conservative 60% confidence threshold

---

## Key Features

### Multi-Timeframe Analysis
Captures market behavior across different time horizons:
- **5-day:** Day trader perspective
- **10-day:** Short-term swing traders
- **20-day:** Monthly traders
- **50-day:** Institutional moves
- **100-day:** Long-term trends

### Market Regime Detection
Classifies current market state:
- **Bullish:** SMA20 > SMA50, price > SMA20
- **Bearish:** SMA20 < SMA50, price < SMA20
- **Sideways:** Neither bullish nor bearish

### Smart Feature Selection
- Reduces overfitting (48 → 25 features for sentiment)
- 70% memory reduction
- 5x faster training
- 1-2% better accuracy

### Incremental Learning
- Daily model updates (no full retrain needed)
- Learns from yesterday's results
- Maintains historical knowledge
- Weekly full retrain for freshness

---

## Performance Metrics

### Current Performance (as of 2025-11-24)

**Ensemble Strategy:**
- Test Accuracy: 58.62%
- Meta-Confidence: 86.3%
- Signals Generated: 9 per day (avg)
- Quality: Top 5 only (>80% confidence)

**Base Strategies:**
- Sentiment: 48.40% test accuracy, 25 features
- Volatility: Test accuracy varies, 20 features
- Pairs: Statistical (no ML model)

### Expected Results
- Win rate: 57-62% (vs 50% random)
- Sharpe ratio: >1.0 (risk-adjusted returns)
- Signal quality: High confidence only
- Drawdown: <15% (conservative threshold)

---

## File Structure

```
Integrated Trading Agent/
├── data_collection/
│   ├── 01_collect_assets.py          # Get stock universe
│   ├── 02_collect_price_data.py      # OHLCV data
│   ├── 03_collect_fundamentals.py    # P/E, EPS, financials
│   ├── 04_collect_economic_indicators.py  # VIX, bonds, Fed
│   ├── 05_collect_sentiment.py       # News + social sentiment
│   ├── 06_collect_earnings.py        # Earnings reports
│   ├── 07_collect_insider_trades.py  # Insider activity
│   ├── 08_collect_analyst_ratings.py # Analyst coverage
│   ├── 09_collect_options_data.py    # Options IV, volume
│   └── 10_collect_news_events.py     # News events
│
├── data_preprocessing/
│   ├── 05_ml_features_aggregator.py  # Main feature engineering
│   └── add_new_features_columns.py   # DB schema updates
│
├── strategies/
│   ├── base_strategy.py              # Base class for all strategies
│   ├── pairs_trading.py              # Pairs strategy
│   ├── sentiment_trading.py          # Sentiment ML strategy
│   ├── volatility_trading.py         # Volatility ML strategy
│   ├── stacked_ensemble.py           # Meta-learning ensemble (800+ lines)
│   ├── ensemble_strategy.py          # Wrapper (imports stacked_ensemble)
│   ├── run_strategies.py             # Run all strategies
│   ├── retrain_all_strategies.py     # Retrain all ML models
│   ├── feature_selector.py           # Smart feature selection
│   └── incremental_trainer.py        # Incremental learning framework
│
├── master_orchestrator/
│   ├── daily_runner.py               # Main daily automation
│   ├── intraday_runner.py            # Intraday updates
│   ├── wait_for_drive.py             # External drive wait logic
│   └── launchd/                      # macOS LaunchAgent configs
│
├── backtesting/
│   ├── backtest_engine.py            # Walk-forward validation
│   ├── generate_daily_report.py      # Email report generator
│   └── results/                      # Daily reports (markdown)
│
└── README_PRODUCTION.md              # This file
```

---

## Database Schema

**Main Database:** `/Volumes/Vault/85_assets_prediction.db`

**Key Tables:**
- `ml_features` - 8,415 rows, 57 columns (all features)
- `trading_signals` - 420 recent signals from all strategies
- `price_data` - Daily OHLCV for 85 assets
- `fundamentals` - P/E, EPS, revenue, margins
- `sentiment_scores` - News + social sentiment
- `options_data` - IV, volume, open interest
- `economic_indicators` - VIX, bonds, Fed data

---

## Daily Workflow

**Automated Daily Sequence (5:00 PM ET):**

1. **Data Collection** (20-30 min)
   - Collect price data for 85 assets
   - Update fundamentals
   - Fetch sentiment scores
   - Get options data
   - Update economic indicators

2. **Feature Engineering** (5-10 min)
   - Aggregate all data sources
   - Calculate 57 features per asset
   - Store in `ml_features` table

3. **Strategy Execution** (2-5 min)
   - Run pairs trading (statistical)
   - Run sentiment trading (XGBoost)
   - Run volatility trading (XGBoost)
   - Run ensemble meta-learner

4. **Signal Generation** (1 min)
   - Ensemble generates predictions
   - Filter for >60% confidence
   - Select top 5 signals
   - Calculate entry/exit prices
   - Set stop losses

5. **Report Generation** (1 min)
   - Create markdown report
   - Include risk assessment
   - Add validation metrics
   - Email to user

**Total Time:** ~30-45 minutes

---

## Configuration

### Email Settings (in LaunchAgent plist)
```xml
<key>TRADING_EMAIL_SENDER</key>
<string>henry.vianna123@gmail.com</string>
<key>TRADING_EMAIL_PASSWORD</key>
<string>bhfy yrsc sbyy efpj</string>
```

### Database Path
```python
DB_PATH = '/Volumes/Vault/85_assets_prediction.db'
```

### Ensemble Confidence Threshold
```python
# In stacked_ensemble.py line 598
confidence_threshold = 0.60  # Only predictions >60% confidence
```

### Retraining Frequency
```python
# In stacked_ensemble.py line 67
retrain_frequency_days = 7  # Full retrain weekly
```

---

## Monitoring & Maintenance

### Daily Checks
- ✅ Email report received?
- ✅ Top 5 signals present?
- ✅ Risk level acceptable (<5/10)?

### Weekly Tasks
- Run full retrain: `python strategies/retrain_all_strategies.py --force-full-retrain`
- Check meta-model accuracy (should be 55-65%)
- Review signal quality (confidence >80%)

### Monthly Reviews
- Win rate tracking (target 57-62%)
- Sharpe ratio calculation (target >1.0)
- Drawdown analysis (should be <15%)
- Feature importance changes

---

## Troubleshooting

### No Email Received
1. Check `/tmp/trading_daily.log`
2. Verify external drive mounted (`/Volumes/Vault`)
3. Check LaunchAgent status: `launchctl list | grep trading`

### Low Signal Count
- Normal: System is conservative (60% threshold)
- Check base strategy performance
- Review market conditions (sideways markets = fewer signals)

### Database Locked
- Close DB Browser for SQLite
- Wait for intraday_runner to complete
- Kill process if necessary: `lsof /Volumes/Vault/85_assets_prediction.db`

### Import Errors
- Verify conda environment: `conda activate trading`
- Check Python path in LaunchAgent plist
- Ensure all dependencies installed

---

## Performance Expectations

### What Success Looks Like

**Short-term (7 days):**
- Daily reports with 5-9 signals
- Average confidence >80%
- Mix of BUY/SELL signals
- Risk level 2-4/10

**Medium-term (30 days):**
- Win rate 57-62%
- Positive returns
- Sharpe ratio >1.0
- Max drawdown <15%

**Long-term (90 days):**
- Sustained win rate improvement
- Meta-learner adapts to market
- Feature importance stabilizes
- Consistent signal quality

---

## Support & Resources

**Logs:**
- Daily: `/tmp/trading_daily.log`
- Errors: `/tmp/trading_daily_error.log`
- Retraining: `strategies/logs/retraining.log`

**Reports:**
- Location: `backtesting/results/daily_report_*.md`
- Format: Markdown with metrics and signals
- Frequency: Daily at 5:00 PM ET

**Documentation:**
- This file: Production overview
- Strategy docs: In each strategy file's docstring
- Code comments: Extensive inline documentation

---

## Version History

**v2.0 (2025-11-24) - CURRENT**
- ✅ Stacked meta-learning ensemble (58.62% test accuracy)
- ✅ Multi-timeframe features (5d-100d)
- ✅ Market regime detection (bull/bear/sideways)
- ✅ Smart feature selection (70% memory reduction)
- ✅ Incremental learning (daily updates)
- ✅ Top 5 high-confidence signals only

**v1.0 (Previous)**
- Basic weighted voting ensemble
- Single timeframe analysis
- Fixed strategy weights
- Lower accuracy (~54% win rate)

---

**System Status:** ✅ PRODUCTION READY - Running Daily

**Next Review:** 2025-12-24 (30-day performance validation)

