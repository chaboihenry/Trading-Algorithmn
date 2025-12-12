# Changelog - Active Risk Management Update

## December 12, 2025

### Major Features Added

#### 1. **Hourly Active Risk Management**
- **Stop-Loss**: Automatically sells positions that drop 5% or more below entry price
- **Take-Profit**: Automatically sells positions that gain 15% or more above entry price
- **Frequency**: Checks every hour (changed from daily)
- **Extended Hours**: Works 4 AM - 8 PM ET (not just market hours)
- **Order Types**:
  - Stop-loss uses market orders (immediate exit to prevent further losses)
  - Take-profit uses limit orders (better execution price)

#### 2. **Socket Exhaustion Fix**
- **Problem**: Bot was exhausting available ports due to 24-hour sleep cycle + continuous WebSocket reconnections
- **Solution**:
  - Reduced SLEEPTIME from "24H" to "1H"
  - Added connection timeout settings (300 seconds)
  - Disabled auto-reconnect for daily strategy
  - Created system tuning scripts to reduce TIME_WAIT duration

#### 3. **Testing & Monitoring Tools**
- **`test_iteration.py`**: Manually trigger a trading iteration without waiting
- **`check_positions_now.py`**: Check which positions will trigger at next iteration
- **`utils/monitor_sockets.sh`**: Real-time socket usage monitoring
- **`utils/fix_socket_exhaustion.sh`**: System-level socket configuration

### Technical Changes

#### Combined Strategy ([combined_strategy.py](combined_strategy.py))
- **Fixed Position Entry Price Access**:
  - Now uses Alpaca TradingClient API directly instead of Lumibot wrapper
  - Alpaca Position objects have `avg_entry_price`, `current_price`, `qty` attributes
  - Created `_check_risk_management_alpaca()` and `_submit_risk_exit_order_alpaca()` methods

- **Simplified Trading Signals**:
  - Removed dependencies on `sentiment_strategy` and `pairs_strategy`
  - Uses RSI-based signals from database:
    - RSI < 30 = oversold (potential buy)
    - RSI > 70 = overbought (potential sell)
  - Maximum position size: $500 or 5% of cash per trade
  - Only trades top 5 symbols to reduce API calls

- **Enhanced Logging**:
  - Logs positions approaching thresholds (-3% warning, +10% gaining)
  - Shows entry price and current price in risk warnings
  - Clear emoji indicators: 🛑 (stop-loss), 💰 (take-profit), ⚠️ (warning), 📈 (gaining)

#### Live Trader ([live_trader.py](live_trader.py))
- Logs now save to `logs/` directory with daily rotation
- Added connection timeout settings to prevent socket exhaustion
- Better error handling and logging

#### File Organization
- Created `utils/` folder for utility scripts
- Moved `backfill_historical_data.py` and `requirements.txt` to utils
- Added monitoring and system tuning utilities

### Configuration Changes

**Risk Thresholds** ([combined_strategy.py](combined_strategy.py#L58-L61)):
```python
STOP_LOSS_PCT = 0.05      # Exit at -5% loss
TAKE_PROFIT_PCT = 0.15    # Exit at +15% profit
ENABLE_EXTENDED_HOURS = True  # Trade after hours
```

**Trading Schedule** ([combined_strategy.py](combined_strategy.py#L53)):
```python
SLEEPTIME = "1H"  # Check every hour (was "24H")
```

### How to Use

#### Run Manual Test (No Waiting):
```bash
python test_iteration.py
```

#### Check Current Positions Against Thresholds:
```bash
python check_positions_now.py
```

#### Monitor Socket Usage:
```bash
./utils/monitor_sockets.sh
```

#### Apply Socket Exhaustion Fix (macOS):
```bash
sudo ./utils/fix_socket_exhaustion.sh
```

#### Start Live Bot with Hourly Risk Management:
```bash
python live_trader.py --strategy combined
```

### What Happens Now

1. **Every Hour (Top of Hour)**:
   - Bot checks all 10 positions
   - Calculates P&L percentage for each
   - Triggers stop-loss for positions below -5%
   - Triggers take-profit for positions above +15%
   - Logs warnings for positions approaching thresholds

2. **New Trading**:
   - Only buys if RSI < 30 (oversold)
   - Maximum $500 per position
   - Only trades top 5 symbols (AAPL, MSFT, GOOGL, META, NVDA)
   - Skips high volatility stocks (volatility > 0.4)

3. **Protection Mode**:
   - Bot prioritizes protecting existing positions
   - New trades are conservative and limited
   - Risk management runs BEFORE looking for new opportunities

### Current Portfolio Status

- **10 positions** being monitored
- **Portfolio value**: ~$99,620
- **Positions losing money**: Multiple positions down -2% to -3%
- **Next check**: Top of the hour
- **Protection active**: Hourly risk management

### Known Limitations

1. **Trading signals simplified**: Not using sentiment analysis or pairs trading yet
2. **Meta-model**: Loaded but not actively used for signal generation
3. **Limited to top 5 symbols**: To reduce API call overhead

### Future Improvements

1. Re-integrate sentiment analysis (FinBERT) for better entry signals
2. Re-enable pairs trading for mean reversion opportunities
3. Use meta-model for dynamic weight adjustment
4. Increase trading frequency for faster responses
5. Add dynamic position sizing based on volatility

---

**Git Commit**: `1ba2411`
**Branch**: `main`
**Status**: ✅ Pushed to GitHub

🤖 Generated with [Claude Code](https://claude.com/claude-code)
