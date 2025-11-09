# Database Update Automation

Automated system to keep your trading algorithm database up-to-date with live market data.

## Quick Start

### Run Update Once (Right Now)

```bash
cd "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"
/Users/henry/miniconda3/envs/trading/bin/python update_database.py
```

This will:
- ✅ Check if database is accessible
- ✅ Run all 10 data collectors
- ✅ Run all 4 preprocessing scripts
- ✅ Update database with latest data
- ✅ Log everything to `logs/database_update.log`

**If external drive is disconnected**, it will fail immediately.

### Run Update with Drive Wait

```bash
/Users/henry/miniconda3/envs/trading/bin/python update_database.py --wait-for-drive
```

This will **wait up to 60 minutes** for the external drive to be connected before starting.

---

## Set Up Daily Automation

Run the setup script to automate daily updates at 9:30 AM:

```bash
cd "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"
./setup_data_automation.sh
```

Choose option **1** (macOS LaunchAgent) for best results.

### What Gets Automated

**Every day at 9:30 AM, the system will:**

1. **Check for External Drive** (wait up to 60 minutes if disconnected)
2. **Collect Fresh Data:**
   - Assets list
   - Price data (OHLCV)
   - Fundamentals
   - Economic indicators
   - Sentiment analysis
   - Earnings reports
   - Insider trades
   - Analyst ratings
   - Options data
   - News events

3. **Preprocess Data:**
   - Calculate correlations
   - Generate technical indicators
   - Compute volatility metrics
   - Analyze pairs for trading signals

4. **Log Results** to `logs/database_update.log`

---

## Data Collection Scripts

All scripts run in this order:

### Phase 1: Data Collection (10 scripts)

| Script | Purpose | Data Source |
|--------|---------|-------------|
| `01_collect_assets.py` | Load 85 assets list | Static list |
| `02_collect_price_data.py` | OHLCV price data | yfinance |
| `03_collect_fundamentals.py` | Current fundamentals | yfinance |
| `04_collect_economic_indicators.py` | Economic data | Alpha Vantage |
| `05_collect_sentiment.py` | Market sentiment | Alpha Vantage |
| `06_collect_earnings.py` | Earnings reports | yfinance |
| `07_collect_insider_trades.py` | Insider trading | yfinance |
| `08_collect_analyst_ratings.py` | Analyst ratings | yfinance |
| `09_collect_options_data.py` | Options chain | Polygon |
| `10_collect_news_events.py` | News articles | NewsAPI |

### Phase 2: Preprocessing (4 scripts)

| Script | Purpose | Output |
|--------|---------|--------|
| `01_calculate_correlations.py` | Find correlated pairs | correlation_analysis table |
| `02_calculate_technical_indicators.py` | RSI, MACD, Bollinger | technical_indicators table |
| `03_calculate_volatility.py` | Historical volatility | volatility_metrics table |
| `04_analyze_pairs.py` | Cointegration & signals | pairs_statistics table |

---

## External Drive Handling

### How It Works

The script automatically detects if the database drive (`/Volumes/Vault/`) is connected:

**Drive Connected:**
- ✅ Starts immediately
- Updates all data
- Completes normally

**Drive Disconnected (with `--wait-for-drive`):**
- ⏳ Waits up to 60 minutes
- 🔍 Checks every 10 seconds
- ✅ Starts when drive connects
- ❌ Fails after 60 minutes if still disconnected

**Drive Disconnected (without `--wait-for-drive`):**
- ❌ Fails immediately
- 📝 Logs error message

### Recommended Setup

For automation (LaunchAgent/cron), **always use** `--wait-for-drive`:

```bash
python update_database.py --wait-for-drive
```

This ensures the script waits for you to connect the drive if needed.

---

## Monitoring

### View Live Progress

```bash
tail -f logs/database_update.log
```

### Check Last Run Status

```bash
grep "DATABASE UPDATE COMPLETED" logs/database_update.log | tail -1
```

### View Error Log (if automated via LaunchAgent)

```bash
tail -f logs/data_update_error.log
```

---

## Managing Automation

### Check if Automation is Running

```bash
launchctl list | grep tradingalgo
```

You should see: `com.tradingalgo.dataupdate`

### Stop Automation

```bash
launchctl unload ~/Library/LaunchAgents/com.tradingalgo.dataupdate.plist
```

### Start Automation

```bash
launchctl load ~/Library/LaunchAgents/com.tradingalgo.dataupdate.plist
```

### Remove Automation Completely

```bash
launchctl unload ~/Library/LaunchAgents/com.tradingalgo.dataupdate.plist
rm ~/Library/LaunchAgents/com.tradingalgo.dataupdate.plist
```

---

## Testing

### Test Database Connection

```bash
python -c "from pathlib import Path; print('✓ Connected' if Path('/Volumes/Vault/85_assets_prediction.db').exists() else '✗ Not found')"
```

### Test Single Data Collector

```bash
cd "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"
/Users/henry/miniconda3/envs/trading/bin/python data_collectors/02_collect_price_data.py
```

### Dry Run (Check what will execute)

```bash
python update_database.py --help
```

---

## Troubleshooting

### "Database not found"

**Problem:** External drive not connected
**Solution:** Connect the "Vault" drive or use `--wait-for-drive`

### "Script timed out"

**Problem:** Data collector took > 10 minutes
**Solution:** Check internet connection, API rate limits

### "Failed with exit code 1"

**Problem:** Script encountered an error
**Solution:** Check `logs/database_update.log` for details

### No data in tables

**Problem:** Scripts ran but didn't collect data
**Solution:**
1. Check API keys in individual scripts
2. Verify internet connection
3. Check API rate limits

### Automation not running

**Problem:** LaunchAgent not executing
**Solution:**
```bash
# Check if loaded
launchctl list | grep tradingalgo

# Reload
launchctl unload ~/Library/LaunchAgents/com.tradingalgo.dataupdate.plist
launchctl load ~/Library/LaunchAgents/com.tradingalgo.dataupdate.plist

# Check system logs
log show --predicate 'process == "update_database"' --last 1h
```

---

## Files Created

| File | Purpose |
|------|---------|
| `update_database.py` | Main update script |
| `setup_data_automation.sh` | Automation setup wizard |
| `logs/database_update.log` | Execution logs |
| `logs/data_update_error.log` | Error logs (LaunchAgent only) |
| `~/Library/LaunchAgents/com.tradingalgo.dataupdate.plist` | LaunchAgent config |

---

## What's Next

After database is updated daily, your **pairs paper trading system** will have fresh data to:
- Identify new trading signals
- Execute virtual trades
- Track performance
- Optimize strategy

**Both systems run independently at 9:30 AM:**
1. `update_database.py` - Updates market data
2. `run_daily_trading.py` - Executes paper trades

Check both dashboards:
```bash
# Data update status
tail logs/database_update.log

# Trading performance
/Users/henry/miniconda3/envs/trading/bin/python strategies/portfolio_dashboard.py
```