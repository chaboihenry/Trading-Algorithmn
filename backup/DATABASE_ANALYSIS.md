# Database Analysis: What Data Powers Your Trading Bot

## Database Overview

Your database at `/Volumes/Vault/85_assets_prediction.db` contains **24 comprehensive tables** with rich financial data for 85 assets from 2020 to present (168,794+ price records).

---

## 📊 Core Data Tables (Currently Used by Bot)

### 1. **raw_price_data** - Historical Price Data
**Used by:** Pairs Strategy, Combined Strategy (Cointegration Analysis)

**Schema:**
- `symbol_ticker` - Stock symbol (e.g., AAPL, MSFT)
- `price_date` - Trading date
- `open`, `high`, `low`, `close` - OHLC prices
- `volume` - Trading volume
- `adj_close` - Adjusted close price
- `dollar_volume` - Volume × Price
- `trades_count` - Number of trades

**How it's used:**
- **Pairs Trading:** Analyzes 120 days of price history to find cointegrated pairs
- **Correlation Analysis:** Identifies pairs with correlation > 0.7
- **Spread Calculation:** Computes hedge ratios and z-scores for mean reversion trading
- **Currently:** 168,794 rows covering 85 stocks

---

### 2. **volatility_metrics** - Volatility Analysis
**Used by:** Combined Strategy (Meta-Learner Features)

**Schema:**
- `symbol_ticker` - Stock symbol
- `vol_date` - Date
- `close_to_close_vol_20d` - 20-day close-to-close volatility ⭐ **ACTIVELY USED**
- `close_to_close_vol_60d` - 60-day volatility
- `parkinson_vol_20d` - Parkinson volatility estimator
- `garman_klass_vol_20d` - Garman-Klass volatility
- `rogers_satchell_vol_20d` - Rogers-Satchell volatility
- `yang_zhang_vol_20d` - Yang-Zhang volatility
- `realized_vol_percentile_1y/3y` - Historical percentile rankings
- `volatility_of_volatility_20d` - Vol of vol (volatility clustering)
- `volatility_trend` - Trend direction (TEXT)
- `volatility_acceleration` - Rate of change
- `volume_weighted_volatility` - Volume-adjusted volatility
- `overnight_vol_ratio` - Overnight vs intraday volatility
- `gap_frequency_60d` - Frequency of price gaps

**How it's used:**
- **Meta-Learner Input:** `close_to_close_vol_20d` used as a feature to assess market conditions
- **Risk Assessment:** Higher volatility = adjust position sizing or confidence
- **Market Regime Detection:** Helps identify high vs low volatility environments

---

### 3. **technical_indicators** - Technical Analysis
**Used by:** Combined Strategy (Meta-Learner Features)

**Schema:**
- `symbol_ticker` - Stock symbol
- `indicator_date` - Date
- `rsi_14` - 14-day Relative Strength Index ⭐ **ACTIVELY USED**
- `rsi_7` - 7-day RSI
- `sma_10/20/50/200` - Simple Moving Averages
- `ema_12/26` - Exponential Moving Averages
- `macd`, `macd_signal`, `macd_histogram` - MACD indicator
- `stochastic_k/d` - Stochastic oscillator
- `bb_upper/middle/lower/width/percent` - Bollinger Bands
- `atr_14/20` - Average True Range (volatility)
- `adx_14` - Average Directional Index (trend strength)
- `plus_di`, `minus_di` - Directional indicators
- `obv` - On-Balance Volume
- `volume_sma_20`, `volume_ratio` - Volume analysis
- `price_distance_to_sma_50/200` - Distance from moving averages
- `price_momentum_5d/20d/60d` - Price momentum

**How it's used:**
- **Meta-Learner Input:** `rsi_14` indicates overbought (>70) or oversold (<30) conditions
- **Trend Identification:** Used by meta-model to weight sentiment vs pairs signals
- **Signal Validation:** Confirms whether market conditions favor the strategy

---

### 4. **trading_signals** - Historical Signal Performance
**Used by:** Combined Strategy (Meta-Learner Training)

**Schema:**
- `signal_date` - When signal was generated
- `symbol_ticker` - Stock symbol
- `signal_type` - BUY, SELL, HOLD
- `strength` - Signal confidence (0-1)
- `metadata` - Additional signal info (JSON)

**How it's used:**
- **Meta-Learner Training:** 348 historical signals used to train XGBoost model
- **Performance Tracking:** Evaluates which signals were profitable
- **Strategy Optimization:** Learns which market conditions favor sentiment vs pairs
- **Minimum Requirement:** Needs 100+ signals to train (you have 348 ✓)

---

## 💎 Additional Rich Data (Available but NOT Currently Used)

### 5. **fundamental_data** - Company Fundamentals
**Potential Use:** Fundamental analysis, value investing strategies

**Likely Contains:**
- P/E ratio, P/B ratio, PEG ratio
- Market cap, enterprise value
- Revenue, earnings, profit margins
- Debt ratios, cash flow
- Dividend yield

### 6. **sentiment_data** - News & Social Sentiment
**Potential Use:** Alternative to current FinBERT news analysis

**Likely Contains:**
- Pre-computed sentiment scores from news
- Social media sentiment
- Sentiment time series
- Source attribution

### 7. **analyst_ratings** - Wall Street Analyst Data
**Potential Use:** Contrarian or consensus trading signals

**Likely Contains:**
- Buy/Sell/Hold ratings
- Price targets
- Rating changes
- Analyst firm names

### 8. **earnings_data** - Earnings Reports
**Potential Use:** Earnings surprise strategies

**Likely Contains:**
- Earnings announcement dates
- Actual vs expected earnings
- Revenue surprises
- Guidance updates

### 9. **insider_trading** - Insider Activity
**Potential Use:** Following smart money

**Likely Contains:**
- Insider buy/sell transactions
- Transaction sizes
- Insider roles (CEO, CFO, etc.)
- Filing dates

### 10. **options_data** - Options Market Data
**Potential Use:** Volatility trading, gamma exposure

**Likely Contains:**
- Implied volatility
- Put/Call ratios
- Options volume
- Open interest
- Greeks (delta, gamma, theta, vega)

### 11. **economic_indicators** - Macro Economic Data
**Potential Use:** Regime-based trading (risk-on vs risk-off)

**Likely Contains:**
- Interest rates (Fed funds rate)
- Inflation (CPI, PCE)
- Employment data (unemployment, payrolls)
- GDP growth
- Consumer confidence

### 12. **vix_term_structure** - Volatility Term Structure
**Potential Use:** VIX trading, market regime detection

**Likely Contains:**
- VIX levels at different maturities
- Contango vs backwardation
- Volatility risk premium

### 13. **correlation_analysis** - Asset Correlations
**Potential Use:** Portfolio diversification, pair discovery

**Likely Contains:**
- Pairwise correlations
- Rolling correlation time series
- Correlation breakdowns

### 14. **ml_features** - Pre-computed ML Features
**Potential Use:** Ready-to-use features for ML models

**Likely Contains:**
- Engineered features
- Normalized values
- Lag features
- Technical combinations

### 15. **model_metadata** - Model Tracking
**Potential Use:** Model versioning, performance tracking

**Likely Contains:**
- Model names and versions
- Training parameters
- Performance metrics
- Last training date

### 16. **performance_metrics** - Strategy Performance
**Potential Use:** Strategy comparison, optimization

**Likely Contains:**
- Sharpe ratio, Sortino ratio
- Max drawdown
- Win rate
- Risk-adjusted returns

### 17. **news_events** - Major News Events
**Potential Use:** Event-driven trading

**Likely Contains:**
- Earnings releases
- M&A announcements
- Product launches
- Regulatory news

### 18. **relative_valuation** - Peer Comparison
**Potential Use:** Mean reversion to sector averages

**Likely Contains:**
- Sector P/E vs individual
- Relative strength rankings
- Valuation percentiles

### 19. **volatility_regimes** - Market Regime Classification
**Potential Use:** Adaptive strategy selection

**Likely Contains:**
- High/medium/low volatility periods
- Regime dates
- Transition probabilities

### 20. **pairs_statistics** - Pair Trading Stats
**Potential Use:** Pre-computed pair quality scores

**Likely Contains:**
- Cointegration test results
- Half-life statistics
- Historical spread metrics

### 21. **signal_quality_monitoring** - Signal Quality
**Potential Use:** Filter out low-quality signals

**Likely Contains:**
- Signal accuracy by type
- Degradation alerts
- Quality scores

### 22. **trades** - Historical Trade Records
**Potential Use:** Performance analysis, debugging

**Likely Contains:**
- Entry/exit dates and prices
- Position sizes
- P&L
- Strategy attribution

### 23. **assets** - Asset Metadata
**Potential Use:** Symbol lookup, filtering

**Likely Contains:**
- Symbol list
- Company names
- Sectors
- Exchange information

### 24. **data_pipeline_status** - Data Health
**Potential Use:** Monitoring data freshness

**Likely Contains:**
- Last update timestamps
- Data quality flags
- Pipeline errors

---

## 🤖 How the Combined Strategy Uses This Data

### **Initialization Phase**
1. **Loads Price Data** (`raw_price_data`)
   - Fetches 120 days of history for 85 stocks
   - Handles duplicate entries by averaging

2. **Finds Cointegrated Pairs**
   - Tests all possible pairs (435 combinations for 30 stocks)
   - Filters by correlation > 0.7
   - Runs Engle-Granger cointegration test (p-value < 0.05)
   - Calculates half-life for mean reversion speed
   - Ranks by quality score (correlation + cointegration + half-life)
   - Keeps top 15 pairs

3. **Trains Meta-Learner** (if needed)
   - Loads 348 historical signals from `trading_signals`
   - Joins with `raw_price_data` for entry prices
   - Joins with `volatility_metrics` for volatility context
   - Joins with `technical_indicators` for RSI context
   - Trains XGBoost classifier to predict profitable signals
   - Saves trained model for reuse

### **Daily Trading Iteration**
1. **Sentiment Analysis** (via Alpaca News API)
   - Fetches news for 14 tech stocks
   - Analyzes with FinBERT (not from database)
   - Generates sentiment score and confidence

2. **Pairs Signal Generation**
   - Calculates current spread for each pair
   - Computes z-score from historical mean
   - Triggers on z-score > 1.5 or < -1.5

3. **Meta-Learner Decision**
   - Combines sentiment + pairs signals
   - Fetches current volatility from `volatility_metrics`
   - Fetches current RSI from `technical_indicators`
   - Creates feature vector with 12 engineered features
   - Predicts probability of profitable trade
   - Only trades if confidence > 60%

4. **Position Sizing**
   - Uses 10% of portfolio per position
   - Scales by meta-learner confidence
   - Max portfolio utilization adaptive

### **Weekly Retraining**
- Re-trains meta-learner every 7 days
- Incorporates new signals and market data
- Adapts to changing market conditions

---

## 🎯 Key Insights: What Makes This Powerful

### 1. **Multi-Timeframe Analysis**
- Short-term: 7-14 day indicators (RSI, short volatility)
- Medium-term: 20-60 day indicators (SMA, volatility regimes)
- Long-term: 120-200 day indicators (cointegration, trend)

### 2. **Multiple Volatility Measures**
- Close-to-close (standard)
- Parkinson (high-low range)
- Garman-Klass (OHLC)
- Rogers-Satchell (drift-independent)
- Yang-Zhang (optimal, handles gaps)

This diversity captures different market microstructures!

### 3. **Historical Signal Database**
- 348 signals = enough for reliable ML training
- Tracks actual profitability, not just predictions
- Enables continuous learning and improvement

### 4. **Cointegration-Based Pairs**
- More reliable than correlation alone
- Mean-reverting spreads = predictable opportunities
- Quality scoring ensures only best pairs traded

---

## 💡 Recommendations for Enhancement

### **Quick Wins (Easy to Add)**

1. **Use Pre-computed ML Features**
   - Query `ml_features` table instead of computing on the fly
   - Faster execution, consistent feature engineering

2. **Add Options Implied Volatility**
   - Query `options_data` for IV levels
   - Compare to realized vol for vol risk premium trades

3. **Filter by Insider Activity**
   - Query `insider_trading` before trades
   - Avoid stocks with heavy insider selling
   - Follow when insiders are buying

4. **Use Analyst Sentiment**
   - Query `analyst_ratings` for consensus
   - Contrarian: fade extreme ratings
   - Momentum: follow rating upgrades

### **Advanced Features**

5. **Regime-Aware Strategy Selection**
   - Query `volatility_regimes` to detect market state
   - Use sentiment in low-vol regimes
   - Use pairs in high-vol regimes

6. **Economic Calendar Integration**
   - Query `economic_indicators` for scheduled releases
   - Reduce position size before major events (FOMC, CPI)
   - Avoid trading during high-impact news

7. **Earnings Avoidance**
   - Query `earnings_data` for announcement dates
   - Close positions before earnings
   - Re-enter after reaction settles

8. **Fundamental Filters**
   - Query `fundamental_data` for value metrics
   - Only trade stocks with P/E < sector average
   - Avoid high debt companies in rising rate environment

---

## 📈 Current Data Quality

```
✓ 168,794 price records across 85 stocks (2020-present)
✓ 348 historical signals (sufficient for ML training)
✓ Daily updates with full OHLCV data
✓ Comprehensive technical indicators (30+ metrics)
✓ Multiple volatility measures (10+ variants)
✓ Duplicate handling implemented (averaging)
```

---

## 🔄 Data Flow Summary

```
Database Tables
    ↓
Combined Strategy Initialization
    ↓
┌─────────────────────────────────────────┐
│ 1. Load raw_price_data (120 days)      │
│ 2. Find cointegrated pairs              │
│ 3. Load trading_signals (historical)   │
│ 4. Train meta-learner                  │
└─────────────────────────────────────────┘
    ↓
Daily Trading Loop
    ↓
┌─────────────────────────────────────────┐
│ For each symbol:                        │
│   1. Get news → FinBERT sentiment      │
│   2. Calculate pairs z-score            │
│   3. Query volatility_metrics (vol)    │
│   4. Query technical_indicators (RSI)  │
│   5. Meta-learner combines signals     │
│   6. Execute trades if confidence high │
└─────────────────────────────────────────┘
    ↓
Position Management
```

---

## 🎓 Bottom Line

Your database is a **treasure trove** of financial data! Currently, you're only using:

**Active:** 4 out of 24 tables (17%)
- `raw_price_data` - Price history
- `volatility_metrics` - Volatility context
- `technical_indicators` - RSI
- `trading_signals` - ML training data

**Unused but Available:** 20 tables with rich data including:
- Fundamentals, earnings, insider trading
- Options, economic indicators, analyst ratings
- Sentiment, news events, regime classification

The bot makes **intelligent multi-signal decisions** by:
1. **Sentiment** from FinBERT news analysis (external)
2. **Pairs trading** from cointegration analysis (database)
3. **Meta-learning** that weighs signals based on volatility + RSI (database)

This is significantly more sophisticated than sentiment-only trading! You're combining:
- Statistical arbitrage (pairs)
- NLP sentiment analysis (FinBERT)
- Machine learning (XGBoost meta-learner)
- Technical indicators (RSI)
- Volatility regimes

**The result:** A multi-strategy system that adapts to market conditions and learns from historical performance. 🚀
