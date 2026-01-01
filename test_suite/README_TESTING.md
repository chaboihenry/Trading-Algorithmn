# Test Suite for RiskLabAI Trading Bot

This directory contains comprehensive tests for the trading bot that can be run **anytime, outside of market hours**.

## Why Test Outside Market Hours?

Testing during market hours is risky and time-consuming:
- Bugs can cost you money
- Market time is precious and limited
- Hard to reproduce issues

With this test suite, you can:
- **Test the entire pipeline** with historical data
- **Predict actual profitability** before going live
- **Catch bugs before live trading**
- **Validate changes** without waiting for market open
- **Simulate realistic trading scenarios**

---

## Test Files

### ⭐ 1. `backtest_multi_symbol.py` - **COMPREHENSIVE PROFIT BACKTEST** (RECOMMENDED)

**This is the primary tool to determine if your improvements will lead to real profit.**

Simulates actual portfolio trading with all your trained models using real historical data.

**What it tests:**
- Multi-symbol portfolio trading (all models working together)
- Real tick data → bar generation → signals → trade execution
- Realistic position sizing (Kelly criterion)
- Actual profit/loss calculations
- Risk metrics (Sharpe ratio, max drawdown)
- Per-symbol performance breakdown

**When to run:**
- ✅ **Before going live** - To validate strategy will be profitable
- ✅ After training new models - To see performance
- ✅ After strategy changes - To verify improvements work
- ✅ To compare different parameter sets

**How to run:**
```bash
# Test all tier_1 models (RECOMMENDED starting point)
python test_suite/backtest_multi_symbol.py --tier tier_1

# Test specific symbols
python test_suite/backtest_multi_symbol.py --symbols AAPL MSFT GOOGL NVDA

# Test ALL trained models
python test_suite/backtest_multi_symbol.py

# Custom parameters (more aggressive)
python test_suite/backtest_multi_symbol.py --tier tier_1 --capital 100000 --bars 1000 --kelly 0.15
```

**Expected output:**
```
================================================================================
BACKTEST RESULTS
================================================================================

PORTFOLIO PERFORMANCE:
  Starting Capital:    $100,000.00
  Final Value:         $108,450.23
  Total P&L:           $8,450.23
  Total Return:        8.45%
  Sharpe Ratio:        1.85
  Max Drawdown:        -3.45%

TRADE STATISTICS:
  Total Trades:        127
  Win Rate:            58.3%
  Average Win:         $245.30
  Average Loss:        $123.40
  Profit Factor:       2.34
  Avg Hold Time:       12.5 hours

PER-SYMBOL PERFORMANCE:
  AAPL  :  15 trades | Total P&L: $2,145.30 | Avg: $143.02
  MSFT  :  12 trades | Total P&L: $1,834.50 | Avg: $152.88
  GOOGL :  10 trades | Total P&L: $1,523.40 | Avg: $152.34
  NVDA  :  18 trades | Total P&L: $1,245.20 | Avg: $69.18
  META  :   8 trades | Total P&L:   $892.10 | Avg: $111.51
  ...

================================================================================
✅ STRONG POSITIVE RESULTS - Strategy shows promise!
================================================================================
```

**Interpretation:**
- ✅ **Positive return + Sharpe > 1.0** = Strategy is profitable, ready for paper trading
- ⚠️  **Positive return but Sharpe < 1.0** = Profitable but risky, needs refinement
- ❌ **Negative return** = Strategy needs major improvements before live trading

**Arguments:**
- `--tier tier_1`: Test all tier_1 symbols (99 models)
- `--symbols AAPL MSFT`: Test specific symbols only
- `--capital 100000`: Starting capital (default: $100k)
- `--bars 500`: Number of bars to simulate per symbol (default: 500)
- `--kelly 0.1`: Kelly fraction for position sizing (default: 0.1 = 10% of portfolio per trade, conservative)

---

### 2. `test_prediction_logic.py` - Unit Tests

**What it tests:**
- Probability margin filter (3% threshold)
- Model loading
- Feature generation
- Correct signal mapping

**When to run:**
- After changing prediction logic
- After retraining models
- Quick validation

**How to run:**
```bash
python test_suite/test_prediction_logic.py
```

**Expected output:**
```
================================================================================
RISKLABAI PREDICTION LOGIC UNIT TESTS
================================================================================

TEST: Probability Margin Filter
✓ PASS: Long with 5% margin
✓ PASS: Neutral - only 1% margin
✓ PASS: Short with 10% margin
...

================================================================================
TEST SUITE SUMMARY
================================================================================
✓ PASS: margin_filter
✓ PASS: model_loading
✓ PASS: feature_generation

✓ ALL TESTS PASSED
```

---

### 3. `test_live_trading_simulation.py` - Single Symbol Simulation

**What it tests:**
- Complete trading pipeline for one symbol
- Signal generation patterns
- Order execution flow

**When to run:**
- Quick single-symbol validation
- Testing signal distribution

**How to run:**
```bash
python test_suite/test_live_trading_simulation.py
```

---

## Pre-Live Trading Checklist

**Run these tests BEFORE starting live/paper trading:**

### 1. ✅ Run Comprehensive Backtest
```bash
python test_suite/backtest_multi_symbol.py --tier tier_1
```

**Look for:**
- ✅ **Total Return > 5%** (over 500 bars of historical data)
- ✅ **Sharpe Ratio > 1.0** (risk-adjusted returns are good)
- ✅ **Max Drawdown < -10%** (risk is controlled)
- ✅ **Win Rate > 50%** (more winners than losers)
- ✅ **Profit Factor > 1.5** (winners are bigger than losers)

**Decision:**
- If all metrics are green → ✅ Strategy is ready for paper trading
- If metrics are mixed → ⚠️ Review per-symbol performance, may need parameter tuning
- If metrics are red → ❌ Strategy needs significant improvement

### 2. ✅ Run Unit Tests
```bash
python test_suite/test_prediction_logic.py
```
All tests should PASS.

### 3. ✅ Review Results
- Check signal distribution (should be balanced, not 97% neutral)
- Review per-symbol performance (which symbols perform best?)
- Verify trades are executed properly

---

## What the Backtest Tells You

The backtest simulates **exactly what would happen** if you ran your bot with real money:

### Portfolio Performance
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good, >2.0 is excellent)
- **Max Drawdown**: Worst peak-to-trough decline (how much you could lose)

### Trade Quality
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Total wins ÷ total losses (>1.5 means winners dominate)
- **Avg Win vs Avg Loss**: Are winners bigger than losers?
- **Avg Hold Time**: How long positions are held

### Per-Symbol Insights
- Which symbols are most profitable?
- Which symbols have the best win rate?
- Are any symbols consistently losing? (remove them!)

---

## Interpreting Results

### ✅ Excellent Strategy (Ready for Live)
```
Total Return:     >10%
Sharpe Ratio:     >1.5
Max Drawdown:     <-8%
Win Rate:         >55%
Profit Factor:    >2.0
```

### ⚠️ Decent Strategy (Needs Refinement)
```
Total Return:     3-10%
Sharpe Ratio:     0.8-1.5
Max Drawdown:     -8% to -15%
Win Rate:         50-55%
Profit Factor:    1.2-2.0
```

### ❌ Weak Strategy (Major Work Needed)
```
Total Return:     <3% or negative
Sharpe Ratio:     <0.8
Max Drawdown:     >-15%
Win Rate:         <50%
Profit Factor:    <1.2
```

---

## Example Workflow

**1. Train models for tier_1:**
```bash
python scripts/train_all_symbols.py --tier tier_1
```

**2. Run comprehensive backtest:**
```bash
python test_suite/backtest_multi_symbol.py --tier tier_1 --capital 100000 --bars 1000
```

**3. Review results:**
- ✅ Total Return: 8.5% → Good!
- ✅ Sharpe: 1.8 → Excellent risk-adjusted returns
- ✅ Max DD: -5.2% → Low risk
- ✅ Win Rate: 58% → More winners than losers
- ✅ Profit Factor: 2.3 → Winners are 2.3x bigger than losers

**4. Start paper trading:**
```bash
python run_live_trading.py
```

**5. Monitor for 30 days, then go live** ✅

---

## Troubleshooting

### Backtest shows negative returns

**Possible causes:**
1. **Models need more data** - Train with more historical data (--days 730)
2. **Kelly fraction too high** - Reduce from 0.1 to 0.05
3. **Strategy parameters need tuning** - Adjust profit target / stop loss
4. **Some symbols are bad** - Check per-symbol performance, remove losers

### Win rate is low (<45%)

**Possible causes:**
1. **Profit target too aggressive** - Reduce from 4% to 3%
2. **Stop loss too tight** - Increase from 2% to 2.5%
3. **Meta threshold too high** - Lower from 0.5 to 0.4

### Max drawdown is too high (>-15%)

**Possible causes:**
1. **Position sizing too large** - Reduce Kelly fraction
2. **Not enough diversification** - Add more symbols
3. **Stop losses not working** - Check trade logs for stop execution

---

## Next Steps After Testing

If backtest results are **positive**:

1. ✅ **Start paper trading** for 30 days
   ```bash
   python run_live_trading.py
   ```

2. ✅ **Monitor daily performance**
   - Compare paper results to backtest
   - Look for similar Sharpe / drawdown / win rate

3. ✅ **Adjust if needed**
   - If paper < backtest: Market conditions changed
   - If paper > backtest: You're in a good market

4. ✅ **Go live with small capital**
   - Start with 10-20% of target capital
   - Scale up as results prove out

---

## Questions?

**Q: How much data do I need for reliable backtest?**
A: Minimum 500 bars per symbol (default). More is better - use `--bars 1000` for higher confidence.

**Q: What's a good Sharpe ratio?**
A: >1.0 is good, >1.5 is great, >2.0 is exceptional. Anything <0.8 means too much risk for the return.

**Q: My backtest shows 50% returns, is this real?**
A: Probably overfitting. If it seems too good to be true, it probably is. Expect 10-30% annually in real trading.

**Q: Can I trust these results?**
A: The backtest uses real historical data and realistic position sizing. However, past performance doesn't guarantee future results. Always start with paper trading.

Good luck with testing! 🚀
