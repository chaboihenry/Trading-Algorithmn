# Test Suite for RiskLabAI Trading Bot

This directory contains comprehensive tests for the trading bot that can be run **anytime, outside of market hours**.

## Why Test Outside Market Hours?

Testing during market hours is risky and time-consuming:
- Bugs can cost you money
- Market time is precious and limited
- Hard to reproduce issues

With this test suite, you can:
- **Test the entire pipeline** with historical data
- **Catch bugs before live trading**
- **Validate changes** without waiting for market open
- **Simulate realistic trading scenarios**

---

## Test Files

### 1. `test_prediction_logic.py` - Unit Tests

**What it tests:**
- Probability margin filter (3% threshold)
- Model loading
- Feature generation

**When to run:**
- After changing prediction logic
- After retraining models
- Before starting live trading

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
...

TEST: Model Loading
✓ Models loaded successfully
...

TEST: Feature Generation
✓ Features generated successfully
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

### 2. `test_live_trading_simulation.py` - Full Pipeline Simulation

**What it tests:**
- Complete trading pipeline
- Data loading from tick database
- Bar generation
- Signal generation
- Order execution (mocked)
- Position tracking

**When to run:**
- After major changes to the bot
- Before deploying to live trading
- To validate signal generation patterns
- To check signal distribution (should be ~5% short, 41% neutral, 54% long)

**How to run:**
```bash
python test_suite/test_live_trading_simulation.py
```

**Expected output:**
```
================================================================================
STARTING LIVE TRADING SIMULATION
================================================================================
Symbol: SPY
Simulation period: Last 5 days
Starting portfolio value: $100,000.00

LOADING HISTORICAL DATA
✓ Loaded 9,024,939 ticks
✓ Generated 7286 bars
✓ Using 500 bars for simulation

RUNNING SIMULATION
--- Iteration 1 @ 2025-12-20 14:31:36 ---
Primary model probabilities - Short: 0.4514, Neutral: 0.0542, Long: 0.4945
✓ Signal accepted: margin=4.31% (>3.0%), prob=49.45% (>1.50%)
Meta model trade probability: 0.2292
✅ SIGNAL=1, Bet size=0.23
Mock order: BUY 28 SPY @ $400 (order_id=order_1000)
LONG SPY: 28 shares @ $400.00
...

SIMULATION RESULTS
================================================================================
Starting portfolio value: $100,000.00
Final portfolio value: $101,250.00
Total return: 1.25%

Signal distribution (50 total):
  Short (-1):    2 ( 4.0%)
  Neutral (0):  20 (40.0%)
  Long (1):     28 (56.0%)

Trades executed: 15
  Buy orders: 10
  Sell orders: 5

✓ Simulation completed successfully
```

---

## What the Tests Verify

### ✅ Fixed Issues Validation

The tests ensure our fixes are working:

1. **97% Neutral Training Labels → 42% Short / 58% Long**
   - Test validates balanced label distribution
   - Checks force_directional is active

2. **100% Long Predictions → 5% Short / 41% Neutral / 54% Long**
   - Test validates 3% margin filter is working
   - Checks that low-confidence trades are filtered out

3. **Signal Quality**
   - Verifies probabilities are balanced
   - Confirms margin calculation is correct
   - Validates threshold logic

### ✅ Bot Stability

The tests catch common issues:
- Model loading failures
- Feature generation errors
- Prediction crashes
- Order execution bugs
- Position tracking errors

---

## Pre-Live Trading Checklist

Run these tests **before starting live trading** after any changes:

1. **Run unit tests:**
   ```bash
   python test_suite/test_prediction_logic.py
   ```
   ✅ All tests should PASS

2. **Run simulation:**
   ```bash
   python test_suite/test_live_trading_simulation.py
   ```
   ✅ Should complete without errors
   ✅ Signal distribution should be balanced (~5% short, ~40% neutral, ~55% long)
   ✅ Trades should execute successfully

3. **Check signal logs:**
   - Look for "✓ Signal accepted" messages
   - Look for "✗ Signal filtered" messages
   - Verify margin calculations are correct

4. **Verify model accuracy:**
   - Primary model accuracy should be ~50% (balanced classes)
   - Meta model accuracy should be ~51%

---

## Troubleshooting

### Test fails: "Model not found"

**Problem:** Models haven't been trained yet

**Solution:**
```bash
python scripts/retrain_aggressive.py
```

This will create `models/risklabai_tick_models_aggressive.pkl`

---

### Test fails: "No tick data found"

**Problem:** Tick database is empty

**Solution:**
```bash
python scripts/backfill_ticks.py
```

This will download historical tick data to `/Volumes/Vault/trading_data/tick-data-storage.db`

---

### Simulation predicts 100% neutral

**Problem:** Margin threshold is too high or model isn't loaded

**Check:**
1. Model is loaded correctly
2. MARGIN_THRESHOLD = 0.03 (3%) in [risklabai_strategy.py](../risklabai/strategy/risklabai_strategy.py:424)
3. force_directional=True in model training

---

### Simulation crashes during prediction

**Problem:** Usually a feature generation issue

**Debug:**
1. Run test_prediction_logic.py first (simpler)
2. Check for NaN values in features
3. Verify bar data has all required columns (open, high, low, close, volume)

---

## Adding New Tests

To add new tests, follow this pattern:

```python
def test_your_feature():
    """Test description."""
    logger.info("=" * 80)
    logger.info("TEST: Your Feature Name")
    logger.info("=" * 80)

    try:
        # Your test logic here
        # ...

        logger.info("✓ Test passed")
        return True

    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


# Add to main()
results['your_feature'] = test_your_feature()
```

---

## Test Coverage

Current test coverage:

| Component | Tested | File |
|-----------|--------|------|
| Probability margin filter | ✅ | test_prediction_logic.py |
| Model loading | ✅ | test_prediction_logic.py |
| Feature generation | ✅ | test_prediction_logic.py |
| Full trading pipeline | ✅ | test_live_trading_simulation.py |
| Order execution | ✅ | test_live_trading_simulation.py |
| Position tracking | ✅ | test_live_trading_simulation.py |
| Signal distribution | ✅ | test_live_trading_simulation.py |

---

## Next Steps

After all tests pass:

1. **Review test results** - Check signal distribution, trade counts, portfolio value
2. **Run live bot in paper trading mode:**
   ```bash
   python run_live_trading.py
   ```
3. **Monitor logs** - Watch for signal quality and execution
4. **Validate profitability metrics** - Let it run for 30 days before live trading

---

## Questions?

If tests fail unexpectedly:
1. Check the error message carefully
2. Verify models are trained (`scripts/retrain_aggressive.py`)
3. Verify tick data exists (`scripts/backfill_ticks.py`)
4. Check that Vault drive is mounted (`/Volumes/Vault`)
5. Review recent code changes

Good luck with testing! 🚀
