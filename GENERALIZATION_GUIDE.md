# Model Generalization & Validation Guide

## The Problem: Overfitting

Your backtest revealed a critical issue: **models completely fail on unseen data** despite working well in training.

**Backtest Results (4 years):**
- Total trades: 3
- Win rate: 0%
- All 3 trades hit stop loss
- Return: -2.25%

**Live Performance (recent):**
- 22 open positions
- ~$100k → $100.7k
- This is likely just market beta, not alpha

**Root Cause:** Models memorized training patterns but can't predict new data (classic overfitting).

---

## Solution: 3-Step Validation Framework

### Step 1: Diagnose Overfitting

**Tool:** `scripts/research/diagnose_overfitting.py`

This measures:
- **Performance Gap**: Train vs test accuracy difference (key metric)
- **Feature Stability**: Whether important features change across folds
- **Label Distribution**: Train/test distribution mismatch
- **Sample Independence**: Effective sample size after accounting for autocorrelation

**Usage:**

```bash
# Diagnose specific symbols
python scripts/research/diagnose_overfitting.py --symbols AAPL MSFT GOOGL

# Diagnose entire tier
python scripts/research/diagnose_overfitting.py --tier tier_1
```

**What to Look For:**

```
Primary Model:
   Train Accuracy: 73.1%
   Test Accuracy:  51.2%
   GAP:            21.9% ❌ OVERFITTING!
```

- Gap > 10% = **Severe overfitting**
- Gap 5-10% = **Moderate overfitting**
- Gap < 5% = **Good generalization**

**Example Output:**

```
RECOMMENDATIONS:
🔴 PRIMARY MODEL OVERFITTING - Gap > 10%
   → Increase regularization (C=0.1 or C=0.01 for LogisticRegression)
   → Reduce max_depth for RandomForest (try max_depth=5)
   → Add dropout if using neural networks

🔴 UNSTABLE FEATURE IMPORTANCE
   → Features changing importance across folds indicates overfitting
   → Remove low-importance features
   → Increase regularization to enforce feature selection

🔴 INSUFFICIENT EFFECTIVE SAMPLES
   → Only 300 independent samples (need >500)
   → High autocorrelation means bars aren't independent
   → Collect more historical data OR use simpler models
```

---

### Step 2: Retrain with Regularization

**Tool:** `scripts/research/retrain_with_regularization.py`

This tests 5 regularization strategies and selects the one with **best validation performance** (not training performance).

**Regularization Configs:**

1. **Baseline** (current):
   - RandomForest: max_depth=None, min_samples_split=2
   - Problem: Too complex, overfits

2. **Light Regularization**:
   - RandomForest: max_depth=10, min_samples_split=5
   - LogisticRegression for meta (C=1.0)

3. **Medium Regularization** (RECOMMENDED):
   - RandomForest: max_depth=5, min_samples_split=10, n_estimators=50
   - LogisticRegression for meta (C=0.1)
   - **Best balance of bias/variance**

4. **Heavy Regularization**:
   - LogisticRegression for both (C=0.01)
   - Very simple, may underfit

5. **Simple Tree**:
   - DecisionTree: max_depth=3
   - Simplest option

**Usage:**

```bash
# Test all configurations and auto-select best
python scripts/research/retrain_with_regularization.py --symbol AAPL --test-all

# Retrain tier_1 with recommended config
python scripts/research/retrain_with_regularization.py --tier tier_1 --config medium_regularization

# Test all configs for tier_1 (will take time)
python scripts/research/retrain_with_regularization.py --tier tier_1 --test-all
```

**Expected Output:**

```
AAPL: CONFIGURATION COMPARISON
================================================================================
Configuration                  Train      Val        Test       Val Gap    Test Gap
--------------------------------------------------------------------------------
Medium Regularization          58.3%      56.1%      54.8%      2.2%       3.5%     ✅
Light Regularization           62.5%      54.3%      53.1%      8.2%       9.4%
Heavy Regularization           53.2%      52.8%      52.1%      0.4%       1.1%     ✅ (but low acc)
Baseline (Current)             73.1%      51.2%      50.8%      21.9%      22.3%    ❌
Simple Tree                    51.9%      51.1%      50.5%      0.8%       1.4%     ✅ (but low acc)
================================================================================

BEST CONFIGURATION: Medium Regularization
   Validation Accuracy: 56.1%
   Test Accuracy:       54.8%
   Validation Gap:      2.2%    ✅ Good generalization!
```

**Key Principle:** Choose model with **smallest validation gap**, not highest training accuracy.

---

### Step 3: Walk-Forward Validation

**Tool:** `scripts/research/walk_forward_validation.py`

This is the **GOLD STANDARD** test - simulates realistic retraining schedule:

1. Train on rolling 6-month window
2. Test on next 1-month period
3. Retrain monthly
4. Repeat for 12+ months

This answers: **"What would my returns be over the last year with monthly retraining?"**

**Usage:**

```bash
# Test single symbol for 12 months
python scripts/research/walk_forward_validation.py --symbol AAPL --months 12

# Test tier_1 (first 10 symbols)
python scripts/research/walk_forward_validation.py --tier tier_1 --months 12

# Custom window sizes
python scripts/research/walk_forward_validation.py --symbol AAPL --train-months 6 --test-months 1
```

**Expected Output:**

```
WALK-FORWARD VALIDATION: AAPL
Training window: 6 months
Test window: 1 month(s)
================================================================================

Window 1/12: 2024-01-01 to 2024-01-31
  Primary Acc: 54.2%
  Meta Acc:    52.1%
  Trades:      15
  Win Rate:    53.3%
  Return:      +1.2%
  Sharpe:      0.85

Window 2/12: 2024-02-01 to 2024-02-29
  Primary Acc: 56.8%
  Meta Acc:    54.9%
  Trades:      18
  Win Rate:    55.6%
  Return:      +2.1%
  Sharpe:      1.23

...

WALK-FORWARD SUMMARY: AAPL
================================================================================
Periods tested:        12
Average accuracy:      55.3% ± 2.1%

TRADING PERFORMANCE:
Total trades:          187
Average win rate:      54.2%
Cumulative return:     +14.3%
Annualized return:     +14.3%  ✅ VIABLE!
Average Sharpe:        1.05

STABILITY:
Positive periods:      9/12 (75.0%)  ✅ Consistent!
```

**Interpretation:**

- **Annualized > 10%** = Promising strategy
- **Annualized 5-10%** = Marginal
- **Annualized < 5%** = Not viable
- **Consistency > 70%** = Robust
- **Consistency < 50%** = Unreliable

---

## Complete Workflow

### 1. Initial Diagnosis

```bash
# Check current overfitting levels
python scripts/research/diagnose_overfitting.py --tier tier_1
```

This will tell you HOW BAD the overfitting is.

### 2. Retrain with Better Regularization

```bash
# Test all configurations, auto-select best
python scripts/research/retrain_with_regularization.py --tier tier_1 --test-all
```

This creates new models in `models/risklabai_{symbol}_models.pkl` with better generalization.

### 3. Validate with Walk-Forward

```bash
# Test realistic performance over 1 year
python scripts/research/walk_forward_validation.py --tier tier_1 --months 12
```

This tells you the TRUTH about expected returns.

### 4. Re-Run Realistic Backtest

```bash
# Full 1-year backtest with new models
python test_suite/realistic_backtest.py --tier tier_1 --days 252 --capital 100000
```

Compare to baseline:
- **Before**: 3 trades, 0% win rate, -2.25% return
- **After**: ??? trades, ??% win rate, ??% return

### 5. Re-Diagnose

```bash
# Verify overfitting is reduced
python scripts/research/diagnose_overfitting.py --tier tier_1
```

Check that performance gaps are now < 10%.

---

## Measuring Improvement

### Before Regularization
```
Primary Model Gap:     21.9%  ❌
Test Accuracy:         50.8%  (random!)
Backtest Win Rate:     0.0%   ❌
Backtest Return:       -2.25% ❌
```

### After Regularization (Target)
```
Primary Model Gap:     3.5%   ✅ (< 10%)
Test Accuracy:         54.8%  ✅ (> 50%)
Walk-Forward Return:   +14.3% ✅
Consistency:           75%    ✅ (> 70%)
```

---

## Understanding the Results

### What "Good Generalization" Looks Like

**Diagnostic Output:**
```
Primary Model:
   Train Accuracy: 58.3%
   Test Accuracy:  54.8%
   GAP:            3.5%   ✅ Small gap = generalizes well
```

**Walk-Forward Output:**
```
Annualized return:     +14.3%
Positive periods:      9/12 (75%)
Average Sharpe:        1.05
```

**Realistic Backtest:**
```
Total Trades:          187
Win Rate:              54.2%
Total Return:          +14.3%
Max Drawdown:          -8.5%
```

### What Means "Still Overfitted"

**Diagnostic Output:**
```
Primary Model:
   Train Accuracy: 68.2%
   Test Accuracy:  52.1%
   GAP:            16.1%   ❌ Large gap = still overfitting
```

**Walk-Forward Output:**
```
Annualized return:     +2.1%
Positive periods:      6/12 (50%)  ❌ Coin flip
Average Sharpe:        0.15        ❌ Poor risk-adjusted return
```

### What Means "No Predictive Power"

```
Test Accuracy:  50.2%   ❌ No better than random
Win Rate:       49.8%   ❌ Essentially 50/50
Return:         -0.5%   ❌ Negative after costs
```

---

## Expected Realistic Returns

With **good generalization**, expect:

- **Annual Return**: 10-20% (not 100%+)
- **Win Rate**: 52-58% (not 70%+)
- **Sharpe Ratio**: 0.8-1.5 (not 3.5+)
- **Max Drawdown**: 10-20%
- **Consistency**: 60-80% of months profitable

Your **previous reported metrics** (73.1% win rate, 3.53 Sharpe) were from **overfitted training data**. Real performance will be MUCH more modest.

---

## Next Steps

1. **Run diagnosis** to see current overfitting levels
2. **Retrain with regularization** to improve generalization
3. **Walk-forward validate** to get realistic return estimates
4. **Re-run realistic backtest** to verify improvement
5. **Only deploy to live** if walk-forward returns are consistently positive

## Critical Rule

**NEVER deploy a model to live trading based on training performance alone.**

Always validate with:
1. ✅ Held-out test set (30% of data)
2. ✅ Walk-forward validation (simulated retraining)
3. ✅ Out-of-sample backtest (full year)

All three must show positive results before going live.

---

## File Reference

- `scripts/research/diagnose_overfitting.py` - Measure overfitting severity
- `scripts/research/retrain_with_regularization.py` - Retrain with better generalization
- `scripts/research/walk_forward_validation.py` - Realistic validation with retraining
- `test_suite/realistic_backtest.py` - Full backtest simulation

---

**Remember:** The backtest showing 0% win rate is **telling you the truth**. Your live performance (~0.7% gain) is likely just market noise. Fix the generalization problem before deploying more capital.
