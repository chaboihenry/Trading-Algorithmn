# Automated Pairs Paper Trading

Automated daily execution system for the pairs trading strategy with $1,000 virtual capital.

## Quick Start

### Option 1: Automated Setup (Recommended)

Run the setup script to configure automation:

```bash
cd strategies
chmod +x setup_automation.sh
./setup_automation.sh
```

Choose your automation method:
- **macOS LaunchAgent** - Best for macOS, runs automatically at specified time daily
- **cron** - Traditional Unix scheduling
- **Manual** - Run yourself when needed

### Option 2: Manual Execution

#### Run Once (Immediate)
```bash
cd strategies
/Users/henry/miniconda3/envs/trading/bin/python run_daily_trading.py --run-once
```

#### Run Continuously (Scheduled)
```bash
cd strategies
/Users/henry/miniconda3/envs/trading/bin/python run_daily_trading.py --time 09:30
```

This will check every minute and execute at 9:30 AM daily.

## Command Line Options

```bash
# Run at specific time (default: 09:30 AM)
python run_daily_trading.py --time 16:00

# Run once and exit (for cron jobs)
python run_daily_trading.py --run-once

# Custom database path
python run_daily_trading.py --db-path /path/to/database.db

# Show help
python run_daily_trading.py --help
```

## What It Does

The automated system:

1. **Checks Exit Conditions** for open positions
   - Mean reversion (z-score < 0.5)
   - Stop loss (-10%)
   - Take profit (+20%)

2. **Looks for Entry Signals**
   - Z-score > 1.5 (diverged pairs)
   - Cointegration p-value < 0.20
   - Half-life between 3-100 days

3. **Updates Performance Metrics**
   - Total capital
   - Cumulative P&L
   - Win rate
   - Daily P&L

4. **Logs Everything**
   - `daily_trading.log` - Main log file
   - `daily_trading_error.log` - Error log (if using LaunchAgent)

## Monitoring

### View Dashboard
```bash
cd strategies
/Users/henry/miniconda3/envs/trading/bin/python portfolio_dashboard.py
```

### Check Logs
```bash
cd strategies
tail -f daily_trading.log
```

### View Database
```bash
sqlite3 /Volumes/Vault/85_assets_prediction.db
```

```sql
-- Check open positions
SELECT * FROM paper_trading_positions WHERE status = 'OPEN';

-- Check performance history
SELECT * FROM paper_trading_performance ORDER BY performance_date DESC LIMIT 10;

-- Check closed positions
SELECT * FROM paper_trading_positions WHERE status = 'CLOSED' ORDER BY exit_date DESC;
```

## Managing Automation

### macOS LaunchAgent

```bash
# Stop
launchctl unload ~/Library/LaunchAgents/com.pairstrading.daily.plist

# Start
launchctl load ~/Library/LaunchAgents/com.pairstrading.daily.plist

# Remove completely
launchctl unload ~/Library/LaunchAgents/com.pairstrading.daily.plist
rm ~/Library/LaunchAgents/com.pairstrading.daily.plist
```

### cron

```bash
# View crontab
crontab -l

# Edit crontab
crontab -e

# Remove all cron jobs
crontab -r
```

## Trading Parameters

Current settings (in `pairs_paper_trading.py`):

- **Initial Capital**: $1,000
- **Position Size**: 15% per trade
- **Max Positions**: 5 concurrent
- **Entry Threshold**: |Z-score| > 1.5
- **Exit Threshold**: |Z-score| < 0.5
- **Stop Loss**: 10%
- **Take Profit**: 20%
- **Cointegration P-value**: < 0.20
- **Half-life Range**: 3-100 days

## Files

- `run_daily_trading.py` - Automated scheduler script
- `setup_automation.sh` - Setup automation (LaunchAgent/cron)
- `pairs_paper_trading.py` - Core trading logic
- `portfolio_dashboard.py` - Performance dashboard
- `daily_trading.log` - Execution logs

## Troubleshooting

### Database Not Found

If you see "unable to open database file":
- Check that `/Volumes/Vault/` is mounted
- Verify database path in scripts
- Use `--db-path` option to specify custom path

### Python Module Errors

Always use the conda environment:
```bash
/Users/henry/miniconda3/envs/trading/bin/python
```

Not:
```bash
python  # System Python
python3  # May be system Python
```

### No Trading Signals

If no trades execute:
- Verify `pairs_statistics` table has recent data
- Run `04_analyze_pairs.py` to refresh pair statistics
- Check z-score thresholds in `pairs_paper_trading.py`

### Logs Not Updating

For LaunchAgent:
```bash
# Check if it's loaded
launchctl list | grep pairstrading

# View system logs
log show --predicate 'process == "run_daily_trading"' --last 1h
```

## Next Steps

1. **Let it run for a week** to collect performance data
2. **Monitor the dashboard** daily to track progress
3. **Analyze results** to optimize parameters
4. **Backtest** different thresholds if needed
5. **Scale up capital** if performance is good

## Support

For issues or questions, check:
- Log files in `strategies/logs/`
- Database queries for position status
- Trading parameters in `pairs_paper_trading.py`