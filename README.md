# 🤖 RiskLabAI Algorithmic Trading System

> **Production-grade quantitative trading bot** implementing institutional-level machine learning techniques from cutting-edge financial research.

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Quality](https://img.shields.io/badge/code%20quality-production-brightgreen)]()
[![ML Framework](https://img.shields.io/badge/ML-RiskLabAI-orange)](https://github.com/risklabai/RiskLabAI)

---

## 📊 Project Overview

A sophisticated algorithmic trading system that combines **institutional-grade financial machine learning** with **real-time market microstructure analysis**. Built on research from Marcos López de Prado's *Advances in Financial Machine Learning*, this system implements techniques used by quantitative hedge funds to generate alpha.

### Key Achievements

- **🎯 Realistic Backtesting**: Achieved 1.83% returns with proper train/test split and zero look-ahead bias
- **📈 Multi-Symbol Trading**: Supports 99+ liquid US equities with per-symbol ML models
- **⚡ Tick-Level Data**: Processes real-time tick data into information-driven bars for optimal signal extraction
- **🔬 Rigorous Validation**: Purged K-fold cross-validation prevents data leakage in time-series forecasting
- **🛡️ Risk Management**: Kelly Criterion position sizing with dynamic stop-loss and take-profit levels

---

## 🎯 What Makes This Different

This isn't your typical moving-average bot. This system implements **institutional-grade techniques** that distinguish professional quantitative trading:

### Technical Differentiation

| Traditional Approach | This Implementation | Impact |
|---------------------|---------------------|--------|
| Manual P&L calculations | **Direct Alpaca API integration** | Eliminates calculation errors, single source of truth |
| Hard-coded local paths | **Fully portable via Docker** | Runs anywhere, reproducible environments |
| Time-based bars (1min, 5min) | **Tick imbalance bars** | Adapts to market activity, better statistical properties |
| Fixed returns labels | **Triple-barrier labeling** with volatility scaling | Labels match real trading mechanics |
| Single model | **Primary + Meta models** (direction + confidence) | Separates prediction from bet sizing |
| Raw price features | **Fractionally differentiated** features | Achieves stationarity while preserving memory |
| All data points | **CUSUM event filtering** | Focuses on statistically significant moves |
| Standard K-fold CV | **Purged K-fold** with embargo | Eliminates look-ahead bias in time series |
| Train on all data | **70/30 train/test split** | Validates on truly unseen data |

### 🔗 Production-Grade Architecture

**Zero Manual Calculations** - All position data comes directly from Alpaca's API:
- `position.unrealized_plpc` → Profit/Loss percentage
- `position.unrealized_pl` → Dollar P&L
- `position.avg_entry_price` → Average entry price
- `position.current_price` → Real-time market price

**Fully Portable** - No hard-coded paths or local dependencies:
- ✅ Works with or without tick database (auto-detects availability)
- ✅ Docker-ready for any environment
- ✅ Environment variables for all configuration
- ✅ Models download automatically from GitHub releases

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LIVE TRADING SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│  │   Alpaca     │───▶│  Tick Data   │───▶│  Imbalance   │    │
│  │   Market     │    │   Storage    │    │   Bars       │    │
│  │   Feed       │    │   (SQLite)   │    │  Generator   │    │
│  └──────────────┘    └──────────────┘    └──────────────┘    │
│         │                                         │            │
│         │                                         ▼            │
│         │                              ┌──────────────────┐   │
│         │                              │   RiskLabAI      │   │
│         │                              │   Strategy       │   │
│         │                              │                  │   │
│         │                              │  • CUSUM Filter  │   │
│         │                              │  • Frac. Diff    │   │
│         │                              │  • Triple Label  │   │
│         │                              │  • Primary Model │   │
│         │                              │  • Meta Model    │   │
│         │                              └──────────────────┘   │
│         │                                         │            │
│         │                                         ▼            │
│         │                              ┌──────────────────┐   │
│         └─────────────────────────────▶│   Position       │   │
│                                        │   Sizer          │   │
│                                        │  (Kelly Criter.) │   │
│                                        └──────────────────┘   │
│                                                 │              │
│                                                 ▼              │
│                                        ┌──────────────────┐   │
│                                        │   Lumibot        │   │
│                                        │   Broker         │   │
│                                        │   (Alpaca API)   │   │
│                                        └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
📦 Integrated Trading Agent/
├── 📂 risklabai/                    # Core ML framework
│   ├── labeling/                    # Triple-barrier & meta-labeling
│   ├── features/                    # Fractional differentiation
│   ├── sampling/                    # CUSUM event filtering
│   ├── cross_validation/            # Purged K-fold
│   └── strategy/                    # Strategy orchestration
│
├── 📂 core/                         # Trading infrastructure
│   └── risklabai_combined.py        # Lumibot integration
│
├── 📂 data/                         # Market data pipeline
│   ├── tick_storage.py              # SQLite tick database
│   ├── tick_to_bars.py              # Imbalance bar generator
│   └── alpaca_tick_client.py        # Real-time data fetching
│
├── 📂 config/                       # Configuration
│   ├── tick_config.py               # Optimal parameters
│   └── all_symbols.py               # Symbol universe (tier_1-5)
│
├── 📂 scripts/                      # Setup & Research Tools
│   ├── setup/                       # Production setup scripts
│   │   ├── master_setup.py          # End-to-end orchestration
│   │   ├── fetch_all_symbols.py     # Symbol universe builder
│   │   ├── backfill_ticks.py        # Historical data downloader
│   │   ├── train_all_symbols.py     # Multi-symbol model training
│   │   └── init_tick_tables.py      # Database initialization
│   └── research/                    # Optimization & calibration
│       ├── find_optimal_d.py        # Fractional differencing calibration
│       ├── calibrate_threshold.py   # Tick bar threshold tuning
│       ├── parameter_sweep_parallel.py  # Grid search optimization
│       └── apply_optimal_params.py  # Best parameter results
│
├── 📂 test_suite/                   # Validation & backtesting
│   ├── backtest_multi_symbol.py     # Comprehensive backtest
│   └── test_prediction_logic.py     # Unit tests
│
├── 📂 models/                       # Trained ML models (99 symbols)
└── 📜 run_live_trading.py           # Main entry point
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker** (recommended) OR Python 3.11+
- Alpaca trading account ([free paper trading](https://alpaca.markets))
- ~2GB disk space for tick data (optional - bot works without it)

### 🐳 Docker Deployment (Recommended)

**Fully portable, reproducible environment - runs anywhere Docker runs.**

```bash
# 1. Clone repository
git clone https://github.com/chaboihenry/Trading-Algorithmn.git
cd "Integrated Trading Agent"

# 2. Configure credentials
cp .env.test .env
# Edit .env with your Alpaca API keys

# 3. Run with Docker Compose
docker-compose up -d

# 4. View logs
docker-compose logs -f trading-bot

# 5. Stop bot
docker-compose down
```

**What Docker gives you:**
- ✅ No Python environment setup
- ✅ Consistent behavior across any machine
- ✅ Isolated from system dependencies
- ✅ Pre-configured with all models
- ✅ Easy scaling and deployment

### 💻 Manual Installation (Alternative)

If you prefer not to use Docker:

```bash
# Clone repository
git clone https://github.com/chaboihenry/Trading-Algorithmn.git
cd "Integrated Trading Agent"

# Create conda environment
conda create -n trading python=3.11
conda activate trading

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your Alpaca credentials:

```env
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here
DATA_PATH=./data  # Optional: for tick data storage
```

### Training Models

Train models on historical tick data:

```bash
# Train all tier_1 symbols (99 models, ~20-30 minutes)
python scripts/setup/train_all_symbols.py --tier tier_1

# Train specific symbols
python scripts/setup/train_all_symbols.py --symbols AAPL MSFT GOOGL
```

**Training Output:**
```
[AAPL] ✓ 2,218 bars generated from 2.4M ticks
[AAPL] Train/Test Split: 1,552 train bars, 666 test bars
[AAPL] Primary model CV accuracy: 0.514 ± 0.028
[AAPL] Meta model CV accuracy: 0.507 ± 0.037
[AAPL] ✓ Model saved to models/risklabai_AAPL_models.pkl
```

### Backtesting

Validate strategy performance on unseen test data:

```bash
# Backtest tier_1 symbols (uses held-out 30% test data)
python test_suite/backtest_multi_symbol.py --tier tier_1

# Custom parameters
python test_suite/backtest_multi_symbol.py --tier tier_1 \
    --capital 100000 \
    --bars 1000 \
    --kelly 0.1
```

**Backtest Results:**
```
================================================================================
BACKTEST RESULTS (70/30 Split - Unseen Test Data)
================================================================================

PORTFOLIO PERFORMANCE:
  Starting Capital:    $100,000.00
  Final Value:         $101,832.47
  Total P&L:           $1,832.47
  Total Return:        1.83%
  Sharpe Ratio:        0.54
  Max Drawdown:        -1.98%

TRADE STATISTICS:
  Total Trades:        68
  Win Rate:            55.9%
  Average Win:         $184.21
  Average Loss:        $157.86
  Profit Factor:       1.48
  Avg Hold Time:       654.5 hours

TOP PERFORMERS:
  GOOGL: +$1,191 (2 trades)
  AMAT:  +$1,054 (5 trades)
  LLY:   +$506 (2 trades)
```

### Live Trading

```bash
# Paper trading (RECOMMENDED - no real money)
python run_live_trading.py

# Monitor logs
tail -f logs/live_trading_*.log
```

---

## 🔬 Technical Deep Dive

### 1. Tick-Based Market Microstructure

Traditional time-based bars (1min, 5min) **miss important market information**. This system uses **tick imbalance bars** that form when buy-sell imbalance exceeds a threshold:

```python
# Adaptive bar formation based on order flow
if abs(cumulative_imbalance) >= threshold:
    # Create new bar - market has shown directional conviction
    bars.append(current_bar)
    cumulative_imbalance = 0
```

**Benefits:**
- Bars form more frequently during high activity (earnings, news)
- Fewer bars during quiet periods (overnight, holidays)
- Better statistical properties (closer to IID assumption)

### 2. Triple-Barrier Labeling

Labels match how traders actually think:

```
Price Path:
    │     ┌─── Hit profit target → Label: +1 (winner)
    │    ╱
    ├───●
    │    ╲
    │     └─── Hit stop loss → Label: -1 (loser)
    │
    └─────────► Timeout (20 bars) → Label: 0 (neutral)
```

**Parameters** (from parameter sweep):
- Profit target: **4.0%**
- Stop loss: **2.0%**
- Max holding: **20 bars**

**Result**: Realistic labels that reflect actual trade outcomes.

### 3. Fractional Differentiation

Achieves **stationarity** (required for ML) while **preserving memory**:

```python
# d = 0.30 preserves 70% of memory
stationary_returns = fractional_diff(prices, d=0.30)
```

Traditional differencing (`prices[t] - prices[t-1]`) loses all memory. Fractional differentiation finds the **minimum differencing** needed for stationarity.

### 4. Meta-Labeling (Bet Sizing)

Two-stage ML approach:

**Stage 1 - Primary Model**: Predicts direction (long/short)
```
Accuracy: 51.4% (slightly better than random)
```

**Stage 2 - Meta Model**: Predicts "Will primary model be correct?"
```
Accuracy: 50.7%
Bet Size: probability_of_correctness
```

**Why this works**: Even 51% accuracy × proper sizing = positive expectancy

### 5. Purged K-Fold Cross-Validation

Standard K-fold **leaks information** in time series:

```
Standard K-Fold (❌ WRONG):
Train: [████──────]  Test: [──████────]  ← Test data influenced by train
                                          (overlapping time periods)

Purged K-Fold (✅ CORRECT):
Train: [████──────]  Embargo: [──]  Test: [────████──]
                      ↑ 1% gap prevents leakage
```

### 6. Look-Ahead Bias Prevention

**The Problem**: Traditional backtests execute trades at prices **you already know**:

```python
# ❌ WRONG - Look-ahead bias
signal = model.predict(data[:current_bar+1])  # Includes current close
price = data['close'][current_bar]  # Already know this!
execute_trade(price)  # Unrealistic
```

**The Solution**: Execute on **next bar's open**:

```python
# ✅ CORRECT - Realistic execution
signal = model.predict(data[:current_bar])  # Don't peek
pending_orders[symbol] = signal  # Store signal
# ... next iteration ...
price = data['open'][current_bar+1]  # Next bar's open (realistic)
execute_trade(price)  # Can actually get this price
```

### 7. Train/Test Split

**70/30 chronological split**:
- First 70% → Training (with purged K-fold CV)
- Last 30% → **Never seen by models** (held-out test set)

```
Timeline: [════════════════════════════════════════════]
          [████████████████ TRAIN ████][═══ TEST ═══]
                   70%                        30%
```

This ensures backtest results reflect **true out-of-sample performance**.

---

## 📈 Performance Metrics

### Backtest Results (Tier 1 - 99 Symbols)

| Metric | Value | Industry Standard | Status |
|--------|-------|-------------------|--------|
| **Total Return** | 1.83% | 5-15% annual | ⚠️ Needs improvement |
| **Sharpe Ratio** | 0.54 | >1.0 target | ⚠️ Risk-adjusted returns low |
| **Max Drawdown** | -1.98% | <-10% acceptable | ✅ Excellent risk control |
| **Win Rate** | 55.9% | >50% target | ✅ Above random |
| **Profit Factor** | 1.48 | >1.5 target | ⚠️ Close to target |
| **Avg Hold Time** | 27 days | Varies | ℹ️ Medium-term strategy |

### Model Performance (Cross-Validation)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Primary (Direction) | 51.4% ± 2.8% | 0.52 | 0.51 | 0.51 |
| Meta (Confidence) | 50.7% ± 3.7% | 0.51 | 0.50 | 0.50 |

**Interpretation**: Models show **slight edge over random** (50%), which combined with proper risk management and position sizing, creates positive expectancy.

### Strategy Analysis

✅ **Strengths:**
- Excellent risk control (low drawdown)
- Consistent win rate >50%
- Positive profit factor
- Zero look-ahead bias
- Properly validated on unseen data

⚠️ **Areas for Improvement:**
- Sharpe ratio needs improvement (target: >1.0)
- Returns could be higher
- Some symbols underperform (ORCL, PANW, APH)

### Next Steps for Optimization:

1. **Remove losing symbols** (ORCL: -$1,498, PANW: -$677)
2. **Adjust margin threshold** (3% → 2% for more trades)
3. **Tune barrier parameters** (profit target, stop loss)
4. **Focus on top performers** (GOOGL, AMAT, LLY)

---

## 🛠️ Technology Stack

### Core Frameworks
- **Python 3.11** - Modern async/await support
- **RiskLabAI** - Financial ML implementations
- **Lumibot** - Trading framework & broker integration
- **scikit-learn** - Machine learning models
- **pandas/numpy** - Data manipulation

### Data & Storage
- **Alpaca API** - Market data & trade execution
- **SQLite** - Tick data storage (~1M ticks/day/symbol)
- **polars** - High-performance data processing

### Deployment & Monitoring
- **APScheduler** - Scheduled strategy execution
- **logging** - Comprehensive error tracking
- **pytest** - Unit & integration testing

---

## 🧪 Testing & Validation

### Test Suite

```bash
# Run all tests
python test_suite/test_prediction_logic.py
```

**Tests:**
- ✅ Probability margin filtering (3% threshold)
- ✅ Model loading & initialization
- ✅ Feature generation pipeline
- ✅ Signal mapping (2-class & 3-class models)
- ✅ Position sizing calculations

### Comprehensive Backtest

```bash
python test_suite/backtest_multi_symbol.py --tier tier_1
```

**Validates:**
- Multi-symbol portfolio simulation
- Realistic order execution (next bar's open)
- Kelly Criterion position sizing
- Stop-loss & take-profit mechanics
- Train/test split integrity

---

## 🔐 Risk Management

### Position Sizing
- **Kelly Criterion**: Optimal bet size based on win probability and odds
- **Kelly Fraction**: 0.1 (10% of suggested Kelly - conservative)
- **Max Position**: 10% of portfolio per symbol

### Trade Protection
- **Stop-Loss**: 2.0% automatic exit on losses
- **Take-Profit**: 4.0% automatic profit capture
- **Max Holding**: 20 bars timeout (prevents dead capital)

### Portfolio Limits
- **Daily Loss Limit**: 3% max per day
- **Max Drawdown**: 10% hard stop
- **Consecutive Losses**: Pause after 3 losses
- **Max Trades/Day**: 15 (prevents overtrading)

---

## 📚 Research Background

This implementation is based on cutting-edge quantitative finance research:

### Primary Source
**Advances in Financial Machine Learning** (2018)
*Marcos López de Prado*

Chapters implemented:
- Ch 2: Financial Data Structures (tick bars)
- Ch 3: Labeling (triple-barrier method)
- Ch 5: Fractional Differentiation
- Ch 7: Cross-Validation (purged K-fold)
- Ch 10: Bet Sizing (meta-labeling)

### Supporting Research
- "The 7 Reasons Most Machine Learning Funds Fail" - López de Prado
- "Building Diversified Portfolios that Outperform Out of Sample" - López de Prado & Bailey
- Machine Learning for Asset Managers - López de Prado

### Why This Matters

Most ML trading bots fail because they:
1. ❌ Use time-based bars (poor statistical properties)
2. ❌ Have look-ahead bias (unrealistic backtests)
3. ❌ Overfit on training data (no proper CV)
4. ❌ Don't consider bet sizing (only direction)
5. ❌ Ignore market microstructure

This implementation addresses **all** these failure modes.

---

## 📊 Project Highlights

### For Recruiters

This project demonstrates:

**Machine Learning Engineering:**
- ✅ Production ML pipeline (data → features → training → prediction)
- ✅ Cross-validation with time-series data
- ✅ Model persistence & versioning
- ✅ Batch training infrastructure (99 models)

**Software Engineering:**
- ✅ Clean architecture (separation of concerns)
- ✅ Error handling & logging
- ✅ Database design (tick storage)
- ✅ API integration (Alpaca)
- ✅ Async/event-driven programming

**Financial Domain Knowledge:**
- ✅ Market microstructure understanding
- ✅ Risk management implementation
- ✅ Backtesting methodology
- ✅ Position sizing algorithms

**Data Engineering:**
- ✅ Real-time data pipelines
- ✅ Large dataset handling (2M+ ticks/day)
- ✅ Feature engineering
- ✅ Data validation & cleaning

---

## ⚠️ Disclaimer

**This software is for educational and demonstration purposes.**

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- The author is not responsible for financial losses
- Always test thoroughly in paper trading before risking capital
- Consult a financial advisor before making investment decisions

---

## 📫 Contact

**Built by:** Henry (Portfolio Project)
**LinkedIn:** https://www.linkedin.com/in/henry-vianna-258266230/
**GitHub:** [@chaboihenry](https://github.com/chaboihenry)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

**⭐ If you found this project interesting, please star the repository!**

*Built with RiskLabAI • Lumibot • Alpaca API • Python 3.11*
