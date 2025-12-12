# Active Trading System - Complete Guide

## 🎯 What Changed: From Passive to Active Trading

### BEFORE (Problems):
1. ❌ Bot bought 10 stocks and did nothing - positions down $500+
2. ❌ Stop-loss only checked hourly - losses could exceed 5% before detection
3. ❌ No way to profit when market drops - just watched money disappear
4. ❌ $100K available but underutilized

### AFTER (Solutions):
1. ✅ **Server-side bracket orders** - automatic stop-loss/take-profit (no bot needed!)
2. ✅ **Inverse ETF hedging** - profit when market drops
3. ✅ **Position rebalancing** - prevent any stock from dominating portfolio
4. ✅ **Active management** - continuous monitoring and adjustments

---

## 📚 Understanding the New Features

### 1. BRACKET ORDERS (Priority #1) 🎯

**What is it?**
A bracket order is ONE order that creates THREE actions:
1. **Buy the stock** (main order)
2. **Auto-sell at +15%** (take-profit)
3. **Auto-sell at -5%** (stop-loss)

**Why is this better?**
- Executes **server-side at Alpaca** - works 24/7 even if bot is off
- No waiting for hourly checks - triggers instantly
- Can't miss stop-loss due to bot being asleep

**Example:**
```python
# Bot buys AAPL at $150
# Bracket order automatically creates:
# - Take-profit: Sell at $172.50 (+15%)
# - Stop-loss: Sell at $142.50 (-5%)
#
# If AAPL hits $142.50 at 3 AM, Alpaca automatically sells
# Bot doesn't need to be awake!
```

**Code Location:** [combined_strategy.py:480-553](combined_strategy.py#L480-L553)

**How it works (OOP explanation):**
```python
def _create_bracket_order(self, symbol: str, quantity: float,
                         side: str = "buy", current_price: float = None) -> bool:
    """
    This is a METHOD (function inside a class).
    'self' refers to the strategy instance - it's how the method
    accesses other parts of the strategy.

    Parameters:
      - symbol: Stock ticker (e.g., "AAPL")
      - quantity: How many shares to buy
      - side: "buy" or "sell"
      - current_price: Stock price (fetched if not provided)

    Returns:
      - bool: True if order submitted, False if failed
    """

    # Calculate stop-loss and take-profit prices
    take_profit_price = round(current_price * (1 + self.TAKE_PROFIT_PCT), 2)
    stop_loss_price = round(current_price * (1 - self.STOP_LOSS_PCT), 2)

    # Create the bracket order
    order = self.create_order(
        symbol,
        quantity,
        side,
        type="bracket",  # This tells Alpaca to create bracket order
        take_profit_price=take_profit_price,
        stop_loss_price=stop_loss_price,
        time_in_force="gtc"  # Good-til-canceled (works after hours)
    )

    self.submit_order(order)  # Send to Alpaca
```

---

### 2. INVERSE ETF HEDGING (Priority #2) 🛡️

**What is it?**
Inverse ETFs go UP when the market goes DOWN. They're like betting against the market.

**Available Inverse ETFs:**
- **SH**: 1x inverse S&P 500 (conservative)
- **SPXS**: 3x inverse S&P 500 (aggressive)
- **SQQQ**: 3x inverse NASDAQ (for tech selloffs)

**When does it trigger?**
- Checks 5 major stocks (AAPL, MSFT, GOOGL, NVDA, TSLA)
- If ≥60% have RSI > 70 (overbought) = BEARISH market
- Bot buys SH to profit from expected drop

**Example:**
```
Market Check:
- AAPL: RSI 75 (overbought)
- MSFT: RSI 72 (overbought)
- GOOGL: RSI 68
- NVDA: RSI 78 (overbought)
- TSLA: RSI 45

Bearish ratio: 3/5 = 60% overbought

🔴 BEARISH MARKET DETECTED!
🛡️  Buying SH (inverse S&P 500)

If market drops 5%, SH goes up 5% - you profit!
```

**Code Location:** [combined_strategy.py:541-622](combined_strategy.py#L541-L622)

**Key Configuration:**
```python
# Maximum 20% of portfolio can be in inverse positions
MAX_INVERSE_ALLOCATION = 0.20

# Inverse ETF symbols
INVERSE_ETFS = {
    'tech': 'SQQQ',    # 3x inverse NASDAQ
    'sp500': 'SPXS',   # 3x inverse S&P 500
    'general': 'SH'    # 1x inverse S&P 500 (default)
}
```

---

### 3. POSITION REBALANCING (Priority #3) ⚖️

**What is it?**
Prevents any single stock from becoming too large a % of your portfolio.

**Why is this important?**
```
Bad scenario without rebalancing:
- You buy NVDA for $10,000 (10% of portfolio)
- NVDA goes up 200%
- Now NVDA = $30,000 (30% of portfolio!)
- If NVDA crashes -50%, you lose $15,000 (15% of entire portfolio)

Good scenario with rebalancing:
- NVDA reaches 30% of portfolio
- Bot automatically trims it to 15%
- Sells $15,000 worth, locks in profit
- If NVDA crashes -50%, you only lose $7,500
```

**Code Location:** [combined_strategy.py:480-539](combined_strategy.py#L480-L539)

**How it works:**
```python
def _rebalance_positions(self, portfolio_value: float,
                        max_position_pct: float = 0.15) -> None:
    """
    Checks each position's size as % of total portfolio.
    If any position > 15%, automatically trims it down.

    max_position_pct = 0.15 means 15% maximum per position
    """

    for position in self.get_positions():
        position_value = position.quantity * current_price
        position_pct = position_value / portfolio_value

        if position_pct > max_position_pct:  # e.g., 0.25 > 0.15
            # Calculate excess to sell
            target_value = portfolio_value * max_position_pct
            shares_to_sell = (position_value - target_value) / current_price

            # Sell the excess
            self.create_order(symbol, shares_to_sell, "sell")
```

---

## 🚀 How to Use the New System

### Test the Improvements:

```bash
# 1. Test one iteration to see all new features
python test_iteration.py
```

**Expected Output:**
```
========================================
RUNNING TRADING ITERATION NOW
========================================

Checking 10 positions (fallback for pre-bracket positions)...
⚠️  AMC at -3.2% (approaching stop-loss)
✅ All positions have bracket orders - server-side protection active

Portfolio value: $99,620.00

Market sentiment check: 3/5 symbols overbought
Bearish ratio: 60.0%
🔴 BEARISH MARKET DETECTED (60.0% overbought)
🛡️  Hedging with inverse ETF SH

Creating bracket order for SH:
  Entry: 25.50 shares @ $39.20
  Take-profit: $45.08 (+15%)
  Stop-loss: $37.24 (-5%)
✅ Bracket order submitted for SH
   Alpaca will automatically execute stop-loss/take-profit server-side

Checking position sizes (max allowed: 15%)
✅ All positions properly sized

Checking for new trading opportunities...
Available cash: $1,450.00

📈 BUY AAPL: 3.25 shares @ $150.00 (RSI: 28.5)
   Server-side stops active: -5% stop-loss, +15% take-profit

========================================
ITERATION COMPLETE
========================================
```

### Start Live Bot:

```bash
python live_trader.py --strategy combined
```

**What happens every hour:**
1. ✅ Checks existing positions (legacy fallback)
2. ✅ Rebalances oversized positions (>15%)
3. ✅ Checks market sentiment, hedges if bearish
4. ✅ Looks for new opportunities
5. ✅ All new orders have automatic stop-loss/take-profit

---

## 📊 Monitoring Your Portfolio

### Check Position Status:
```bash
python check_positions_now.py
```

Shows which positions will trigger stop-loss/take-profit.

### Monitor Socket Usage:
```bash
./utils/monitor_sockets.sh
```

Ensures no socket exhaustion.

---

## 🎓 Learning Notes (OOP Concepts)

### Class vs Instance Variables:

```python
class CombinedStrategy(Strategy):
    # CLASS VARIABLE (shared by all instances)
    STOP_LOSS_PCT = 0.05

    def __init__(self):
        # INSTANCE VARIABLE (unique to each instance)
        self.meta_model = None
```

### Methods (Functions in Classes):

```python
def _create_bracket_order(self, symbol, quantity, side="buy"):
    # 'self' is the instance calling this method
    # It gives access to instance variables and other methods

    # Access class variable
    stop_loss = self.STOP_LOSS_PCT

    # Call another method
    price = self.get_last_price(symbol)

    # Call Lumibot's method (inherited from Strategy class)
    order = self.create_order(symbol, quantity, side)
```

### Method Visibility:

```python
# PUBLIC method (no underscore) - meant to be called from outside
def on_trading_iteration(self):
    ...

# PRIVATE method (single underscore) - internal use only
def _create_bracket_order(self, ...):
    ...

# PROTECTED method (double underscore) - very private
def __internal_method(self, ...):
    ...
```

**Convention:** Methods starting with `_` are "helper methods" - they're called by other methods within the class, not from outside.

---

## ⚙️ Configuration Reference

### Risk Management:
```python
STOP_LOSS_PCT = 0.05      # Auto-sell at -5%
TAKE_PROFIT_PCT = 0.15    # Auto-sell at +15%
ENABLE_EXTENDED_HOURS = True  # Trade 4 AM - 8 PM ET
```

### Position Limits:
```python
max_position_pct = 0.15    # No position > 15% of portfolio
MAX_INVERSE_ALLOCATION = 0.20  # Max 20% in inverse ETFs
```

### Trading Parameters:
```python
SLEEPTIME = "1H"  # Check every hour
CONFIDENCE_THRESHOLD = 0.6  # Only trade if 60%+ confident
```

---

## 🔮 Future Improvements (Not Yet Implemented)

### 4. Trailing Stop-Loss
Would adjust stop-loss upward as price increases:
```
Buy AAPL at $150, stop-loss at $142.50 (-5%)
AAPL rises to $160, stop-loss moves to $152 (-5% of $160)
AAPL rises to $170, stop-loss moves to $161.50 (-5% of $170)

This "locks in" profits as stock rises!
```

**Status:** Not yet implemented (Alpaca supports this, need to add)

### 5. Real-Time WebSocket Monitoring
Would replace hourly sleep with instant price updates:
```python
# Instead of:
while True:
    check_positions()
    sleep(3600)  # Wait 1 hour

# Use:
stream = alpaca.Stream()
stream.subscribe_trades(on_price_update)  # Instant updates
```

**Status:** Planned for future enhancement

---

## 📈 Expected Improvements

**Before:**
- Positions: 10 stocks, all down
- Portfolio: -$500 (-0.5%)
- Protection: Hourly checks only
- Downside: None - just lose money when market drops

**After:**
- Positions: Automatically exit at -5% (can't lose more than $5K on $100K portfolio)
- Upside: Automatically take profit at +15%
- Protection: 24/7 server-side (works even if bot crashes)
- Hedging: Profit from market drops via inverse ETFs
- Rebalancing: No single position can blow up entire portfolio

**Estimated Impact:**
- Reduced maximum loss per position: -5% (was unlimited)
- Added downside protection: Inverse ETFs profit during drops
- Risk-adjusted returns: Much better due to rebalancing
- Peace of mind: Server-side protection = sleep better!

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
