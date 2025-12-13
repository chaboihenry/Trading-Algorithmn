# Monitoring Guide

Complete guide to monitoring your trading bot's performance, health, and risk status.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dashboard](#dashboard)
3. [Performance Tracker](#performance-tracker)
4. [Bug Fixes Verification](#bug-fixes-verification)
5. [Real-Time Position Check](#real-time-position-check)

---

## Quick Start

### Setup Environment

1. **Create `.env` file** (if not already done):
   ```bash
   cp .env.template .env
   ```

2. **Add your Alpaca API credentials to `.env`**:
   ```bash
   ALPACA_API_KEY=your_actual_key_here
   ALPACA_API_SECRET=your_actual_secret_here
   ```

3. **Verify setup**:
   ```bash
   python monitoring/dashboard.py
   ```

---

## Dashboard

**Purpose:** Real-time snapshot of your trading bot's current status.

### Usage

```bash
python monitoring/dashboard.py
```

### What It Shows

#### 1. Account Overview
- Total portfolio value
- Cash available
- Amount invested in positions
- Buying power (cash + margin)
- Account status (active/restricted)
- Pattern Day Trader status

**Example Output:**
```
================================================================================
  ACCOUNT OVERVIEW
================================================================================

  Portfolio Value:  $99,583.36
  Cash Available:   $20,974.00 (21.1%)
  In Positions:     $78,609.36 (78.9%)
  Buying Power:     $100,000.00

  ✅ Account Status: ACTIVE
```

#### 2. Current Positions
- All open positions
- Entry price vs current price
- Unrealized profit/loss ($ and %)
- Position value
- Portfolio allocation %

**Example Output:**
```
================================================================================
  CURRENT POSITIONS
================================================================================

  Total Positions: 10

  Symbol   Qty        Entry      Current    P/L ($)      P/L (%)    Value        Alloc
  ------- --------- --------- --------- ----------- --------- ----------- -------
  AAPL     50.00    $175.20   $182.50   $365.00     +4.17%    $9,125.00   9.2%
  MSFT     30.00    $380.00   $372.10   -$237.00    -2.08%    $11,163.00  11.2%
  ...
  ------- --------- --------- --------- ----------- --------- ----------- -------
  TOTAL                                 +$1,245.00  +1.61%    $78,609.36
```

⚠️ Positions exceeding `MAX_POSITION_PCT` (15%) will be flagged!

#### 3. Risk Protection Status
- How many positions have stop-loss orders
- How many have take-profit orders
- Which positions are unprotected (critical!)

**Example Output:**
```
================================================================================
  RISK PROTECTION STATUS
================================================================================

  Protection Summary:
    ✅ Fully Protected:      8/10
    ⚠️  Stop-loss only:       1/10
    ⚠️  Take-profit only:     0/10
    🚫 No protection:        1/10

  ⚠️  WARNING: 1 position(s) lack protection!
     Run: python tests/test_bug_fixes.py to create missing orders
```

**What This Means:**
- **Fully Protected:** Position has BOTH stop-loss AND take-profit orders
- **Partial:** Missing one of the two protective orders
- **No Protection:** ⚠️ CRITICAL - position has no safety net!

#### 4. Hedge Allocation
- Current market sentiment (bearish/bullish)
- Inverse ETF positions (if any)
- Hedge allocation vs maximum allowed
- Recommendation (add hedge, exit hedge, or maintain)

**Example Output:**
```
================================================================================
  HEDGE ALLOCATION
================================================================================

  Market Sentiment:
    Overbought:     3/5 major stocks
    Bearish Ratio:  60.0%
    Market Status:  🔴 BEARISH
    Overbought:     AAPL, NVDA, TSLA

  Hedge Positions:
    Current Value:  $1,960.00
    Allocation:     2.0% / 20.0% max

  Active Hedges:
    SH: 50.00 shares @ $39.20 = $1,960.00 (P/L: +$50.00 / +2.56%)

  💡 Recommendation: Consider adding hedge (market is bearish)
```

**Hedging Explained:**
- When market is bearish (many stocks overbought), the bot buys inverse ETFs
- Inverse ETFs PROFIT when the market drops
- This protects your portfolio during crashes
- Example: If market drops -10%, SH (inverse S&P 500 ETF) goes up ~10%

#### 5. Recent Activity
- Last 10 orders (all statuses)
- Shows what the bot has been doing
- Helps identify issues (e.g., orders getting rejected)

**Example Output:**
```
================================================================================
  RECENT ACTIVITY (Last 10 Orders)
================================================================================

  Date                 Symbol   Side   Qty      Price      Status       Type
  ------------------- ------- ----- ------- --------- ----------- -----------
  2024-12-12 14:23:15  AAPL    BUY    50.00  $175.20   ✅filled    market
  2024-12-12 14:23:30  AAPL    SELL   50.00  $165.00   ⏳new       stop_limit
  ...
```

### When to Use Dashboard

- **Daily:** Check at start of trading day to see overnight changes
- **Before trading:** Verify positions have protection
- **After bot runs:** Confirm bot actions were correct
- **When suspicious:** Investigate unexpected behavior

---

## Performance Tracker

**Purpose:** Long-term performance analysis and real money readiness assessment.

### Usage

```bash
# Analyze last 90 days (default)
python monitoring/performance_tracker.py

# Analyze specific period
python monitoring/performance_tracker.py --days 30
python monitoring/performance_tracker.py --days 180
```

### What It Shows

#### 1. Portfolio Performance
- Total return over period
- Average daily return
- Volatility (risk measure)

**Example Output:**
```
================================================================================
PORTFOLIO PERFORMANCE
================================================================================
  Period:           2024-09-14 to 2024-12-12 (90 days)
  Starting Equity:  $100,000.00
  Ending Equity:    $112,450.00
  Total Return:     +12.45%
  Avg Daily Return: +0.138%
  Volatility:       1.234%
```

#### 2. Risk Metrics

**Sharpe Ratio:**
- Measures risk-adjusted returns
- Higher is better
- Formula: (Return - Risk Free Rate) / Volatility
- Interpretation:
  - **> 2.0:** Excellent
  - **> 1.0:** Good
  - **> 0:** Positive but below average
  - **< 0:** Poor (losing money)

**Maximum Drawdown:**
- Worst peak-to-trough decline
- Example: Portfolio was $110K, dropped to $99K = -10% drawdown
- Lower is better
- Interpretation:
  - **< 10%:** Excellent
  - **< 20%:** Moderate
  - **> 20%:** High risk

**Example Output:**
```
================================================================================
RISK METRICS
================================================================================
  Sharpe Ratio:     1.85
                    ✅ Good (> 1.0)

  Max Drawdown:     -8.50%
                    ✅ Excellent (< 10%)
```

#### 3. Trade Statistics
- Total number of trades
- Win rate (% of profitable trades)
- Average win vs average loss
- Profit factor (total wins / total losses)

**Example Output:**
```
================================================================================
TRADE STATISTICS
================================================================================
  Total Trades:     127
  Total Volume:     $254,680.00
  Avg Order Size:   $2,005.35

  Closed Positions: 48
  Winners:          29 (60.4%)
  Losers:           19
  Average Win:      $245.50
  Average Loss:     $98.20
  Profit Factor:    3.75
  Net P/L:          $5,233.50
                    ✅ Win rate >= 50%
```

**Profit Factor Explained:**
- Ratio of total wins to total losses
- **> 2.0:** Excellent (wins are 2x bigger than losses)
- **> 1.5:** Good
- **> 1.0:** Profitable (but barely)
- **< 1.0:** Losing money overall

#### 4. Real Money Trading Readiness

Shows if bot meets criteria for live trading with real money:

**Example Output:**
```
================================================================================
REAL MONEY TRADING READINESS
================================================================================

  Criteria:
    ✅ Min Trading Days:    90/90 days
    ✅ Min Sharpe Ratio:   1.85/1.0
    ✅ Max Drawdown:       8.5%/10.0%
    ✅ Min Return:         12.5%/0.0%
    ✅ Stop-Loss Compliance: Required

  Status: 5/5 criteria met

  ✅ BOT IS READY FOR REAL MONEY TRADING!
     All performance criteria have been met.
```

**Criteria (defined in `config/settings.py`):**
```python
REAL_MONEY_CRITERIA = {
    'min_days': 90,          # Must trade for 90 days
    'min_sharpe': 1.0,       # Sharpe ratio >= 1.0
    'max_drawdown': 0.10,    # Max drawdown <= 10%
    'min_return': 0.0,       # Must be profitable
    'stop_loss_compliance': 1.0  # All positions must have protection
}
```

### When to Use Performance Tracker

- **Weekly:** Check progress toward real money criteria
- **Monthly:** Analyze performance trends
- **Before going live:** Verify all criteria are met
- **After strategy changes:** Ensure changes improved performance

---

## Bug Fixes Verification

**Purpose:** Verify critical bugs have been fixed and create missing protection orders.

### Usage

```bash
python tests/test_bug_fixes.py
```

### What It Tests

1. **Cash Detection Bug**
   - Verifies `get_cash()` returns float (not None)
   - Displays current cash, buying power, portfolio value

2. **Stop-Loss Verification Bug**
   - Actually checks Alpaca for stop orders (bot used to just assume they existed)
   - Identifies unprotected positions
   - Creates missing stop-loss and take-profit orders

3. **Hedge Logic Bug**
   - Verifies hedge manager is working
   - Checks market sentiment
   - Shows current hedge allocation

**Example Output:**
```
================================================================================
BUG FIX VERIFICATION TEST SUITE
================================================================================

TEST 1: Cash Detection (Bug Fix)
================================================================================
✅ Cash: $20,974.00
✅ Buying power: $100,000.00
✅ Portfolio value: $99,583.36
✅ PASS: Cash detection working correctly!

TEST 2: Stop-Loss Verification (Bug Fix)
================================================================================
Found 10 position(s)
🚫 AMC: NO PROTECTION - CRITICAL!
🚫 BAC: NO PROTECTION - CRITICAL!

⚠️  WARNING: 2 position(s) have NO protection!
   The bot will create stop-loss orders for them.

Creating stop-loss for AMC:
  Quantity: 1500.00 shares
  Entry: $5.02, Current: $4.93
  Stop: $4.77 (-5.0%)
✅ Stop-loss created for AMC
✅ Protected 10/10 positions
✅ PASS: Bracket order verification working correctly!

TEST 3: Hedge Logic (Bug Fix)
================================================================================
Market Sentiment Analysis:
  Overbought: 3/5
  Bearish ratio: 60.0%
  Is bearish: YES
✅ PASS: Hedge logic working correctly!

================================================================================
✅ ALL TESTS PASSED (3/3)

All bugs fixed! Safe to run live bot.
================================================================================
```

### When to Use

- **After setup:** Verify everything works
- **When positions lack protection:** The test will create missing orders
- **Before live trading:** Ensure all bugs are fixed
- **After code changes:** Verify nothing broke

---

## Real-Time Position Check

**Purpose:** Quick check of current positions without full dashboard.

### Usage

```bash
python tests/check_positions_now.py
```

### What It Shows

- Current positions
- Protection status
- Quick summary

Useful for:
- Quick checks during trading hours
- Verifying orders were filled
- CI/CD integration

---

## Interpreting the Data

### Position Allocation

**Safe:** Each position <= 15% of portfolio
```
AAPL: 10.5% ✅
MSFT: 12.3% ✅
```

**Risky:** Any position > 15%
```
NVDA: 18.7% ⚠️  (TOO LARGE!)
```

**Why it matters:** Concentration risk. If one stock crashes, you lose a lot.

### Protection Status

**Fully Protected:**
```
AAPL:
  ✅ Stop-loss at $165 (-5%)
  ✅ Take-profit at $201 (+15%)
```

**Unprotected:**
```
AMC:
  🚫 NO stop-loss (DANGER!)
  🚫 NO take-profit
```

**Why it matters:** Without stop-loss, a position can lose 100%. With stop-loss, max loss is 5%.

### Hedge Status

**Healthy Market (No Hedge Needed):**
```
Market sentiment: 20% overbought
Hedge allocation: 0%
✅ No action needed
```

**Bearish Market (Hedge Active):**
```
Market sentiment: 80% overbought
Hedge allocation: 15% (in SH)
🛡️  Protected if market crashes
```

**Why it matters:** During crashes, inverse ETFs profit and offset losses from long positions.

---

## Automation

### Daily Dashboard Email

You can automate the dashboard to run daily and email results:

```bash
# Add to crontab (runs at 9:30 AM Eastern every weekday)
30 9 * * 1-5 cd /path/to/project && python monitoring/dashboard.py > daily_report.txt 2>&1 && mail -s "Daily Trading Report" you@email.com < daily_report.txt
```

### Weekly Performance Report

```bash
# Add to crontab (runs at 5 PM Eastern every Friday)
0 17 * * 5 cd /path/to/project && python monitoring/performance_tracker.py --days 7 > weekly_report.txt 2>&1 && mail -s "Weekly Performance Report" you@email.com < weekly_report.txt
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'alpaca'"

**Solution:** Activate the trading environment first:
```bash
# If using conda
conda activate trading

# If using venv
source venv/bin/activate
```

### "API credentials not found!"

**Solution:** Create `.env` file with your Alpaca API keys:
```bash
cp .env.template .env
# Edit .env and add your actual keys
```

### "No portfolio history available"

**Reason:** Bot hasn't been running long enough to generate history.

**Solution:** Run bot for at least 2-3 days, then try again.

### Dashboard shows 0 positions but I know I have positions

**Possible causes:**
1. Wrong API keys (paper vs live)
2. API connection issue
3. Account restricted

**Solution:**
1. Verify `.env` has correct keys
2. Check Alpaca dashboard directly
3. Run: `python tests/test_bug_fixes.py` to diagnose

---

## Best Practices

### Daily Routine

1. **Morning (before market open):**
   ```bash
   python monitoring/dashboard.py
   ```
   - Check overnight changes
   - Verify all positions have protection
   - Review hedge status

2. **Evening (after market close):**
   ```bash
   python monitoring/dashboard.py
   ```
   - Review daily performance
   - Check if bot made expected trades
   - Verify no issues

### Weekly Routine

1. **Performance review:**
   ```bash
   python monitoring/performance_tracker.py --days 7
   ```
   - Analyze weekly performance
   - Check if moving toward real money criteria

2. **Monthly review:**
   ```bash
   python monitoring/performance_tracker.py --days 30
   ```
   - Longer-term trend analysis
   - Decide if strategy adjustments needed

### Before Going Live

**Checklist:**

1. ✅ Run performance tracker for 90+ days:
   ```bash
   python monitoring/performance_tracker.py --days 90
   ```

2. ✅ Verify all criteria met:
   - Sharpe ratio >= 1.0
   - Max drawdown <= 10%
   - Win rate >= 50%
   - Profitable overall

3. ✅ Run bug verification:
   ```bash
   python tests/test_bug_fixes.py
   ```

4. ✅ Verify protection:
   ```bash
   python monitoring/dashboard.py
   ```
   - 100% of positions must have stop-loss
   - Ideally 100% have take-profit too

5. ✅ Test with small amount first:
   - Start with $500-1000
   - Run for 2 weeks
   - Scale up if successful

---

## Support

For questions or issues:
1. Check this guide first
2. Review code comments in `monitoring/` directory
3. Run test suite: `python tests/test_bug_fixes.py`
4. Check logs in `logs/` directory

---

## Summary

**Quick Reference:**

```bash
# Real-time status
python monitoring/dashboard.py

# Performance analysis
python monitoring/performance_tracker.py --days 90

# Verify bugs fixed
python tests/test_bug_fixes.py

# Quick position check
python tests/check_positions_now.py
```

**Key Metrics to Watch:**

- **Portfolio allocation:** No position > 15%
- **Protection coverage:** 100% of positions should have stop-loss
- **Sharpe ratio:** Target >= 1.0
- **Max drawdown:** Target <= 10%
- **Win rate:** Target >= 50%

**When to Act:**

- 🚫 **Unprotected positions:** Run `test_bug_fixes.py` immediately
- ⚠️ **Position > 15%:** Reduce position size
- 🔴 **Bearish market + no hedge:** Consider adding hedge (bot should do this automatically)
- ❌ **Sharpe < 0.5 for 30+ days:** Review strategy

---

🤖 This guide was generated to help you monitor and optimize your trading bot safely and effectively.
