# macOS LaunchAgent Setup for Automated Trading

This directory contains LaunchAgent configuration files that enable the trading system to run automatically every weekday at market open (9:30 AM ET).

## Features

✅ **Automatic Scheduling** - Runs every day at 9:30 AM ET (including weekends for 24/7 crypto)
✅ **External Drive Support** - Waits up to 2 hours for drive to be connected
✅ **Dual Pipeline** - Runs both daily (Tiers 0-5) and intraday (continuous) tasks
✅ **Persistent Logging** - All output saved to `/tmp/trading_*.log`
✅ **Error Handling** - Automatic retries with exponential backoff

## Files

| File | Purpose |
|------|---------|
| `com.trading.daily.plist` | LaunchAgent for daily pipeline (runs once at 9:30 AM) |
| `com.trading.intraday.plist` | LaunchAgent for intraday pipeline (runs continuously 9:30 AM - 4 PM) |
| `wait_and_run_intraday.sh` | Wrapper script for intraday continuous mode |
| `install.sh` | Installation script |
| `README.md` | This file |

## Quick Install

```bash
# From the project root
cd master_orchestrator/launchd
./install.sh
```

This will:
1. Copy `.plist` files to `~/Library/LaunchAgents/`
2. Load the LaunchAgents into macOS
3. Enable automatic execution

## Manual Install

If you prefer to install manually:

```bash
# Copy plist files
cp com.trading.*.plist ~/Library/LaunchAgents/

# Load the agents
launchctl load ~/Library/LaunchAgents/com.trading.daily.plist
launchctl load ~/Library/LaunchAgents/com.trading.intraday.plist
```

## How It Works

### Schedule

**Every day at 9:30 AM ET, two LaunchAgents trigger:**

**Daily Pipeline (runs once):**
1. `wait_for_drive.py --daily` starts
2. Checks if `/Volumes/Vault` is mounted
3. Waits up to 2 hours (checking every 30 seconds)
4. Once drive is available, runs daily tasks (Tiers 0-5)
5. Completes and exits

**Intraday Pipeline (runs continuously):**
1. `wait_and_run_intraday.sh` starts
2. Checks if `/Volumes/Vault` is mounted
3. Waits up to 2 hours (checking every 30 seconds)
4. Once drive is available, starts continuous loop
5. Runs price/volume collection every 1 minute
6. Runs technical indicators/ML features every 5 minutes
7. Continues until 4:00 PM ET market close
8. Exits automatically

Note: Runs 7 days a week to capture 24/7 crypto trading data

### External Drive Handling

The system handles the scenario where your Mac wakes/starts at 9:30 AM but the external drive isn't connected yet:

```
9:30:00 AM - LaunchAgent triggers
9:30:01 AM - wait_for_drive.py starts checking for /Volumes/Vault
9:30:30 AM - Check #1: Drive not mounted, waiting...
9:31:00 AM - Check #2: Drive not mounted, waiting...
...
10:15:00 AM - Check #90: Drive mounted! Starting orchestrator...
```

Maximum wait time: **2 hours** (until 11:30 AM)

If the drive isn't connected after 2 hours, the job fails and you'll see an error in the logs.

## Management

### Check Status

```bash
# List all trading LaunchAgents
launchctl list | grep trading

# Should show:
# com.trading.daily
# com.trading.intraday
```

### View Logs

```bash
# Real-time daily logs
tail -f /tmp/trading_daily.log

# Real-time intraday logs
tail -f /tmp/trading_intraday.log

# View errors
tail -f /tmp/trading_daily_error.log
tail -f /tmp/trading_intraday_error.log
```

### Manual Trigger (Testing)

```bash
# Trigger daily pipeline manually (runs once and exits)
launchctl start com.trading.daily

# Trigger intraday pipeline manually (runs continuously until 4 PM)
launchctl start com.trading.intraday

# Or run directly:
cd "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"

# Daily (runs once)
./master_orchestrator/wait_for_drive.py --daily

# Intraday (runs continuously - press Ctrl+C to stop)
./master_orchestrator/launchd/wait_and_run_intraday.sh
```

### Disable

```bash
# Temporarily disable (until next reboot)
launchctl unload ~/Library/LaunchAgents/com.trading.daily.plist
launchctl unload ~/Library/LaunchAgents/com.trading.intraday.plist
```

### Re-enable

```bash
# Re-enable after unloading
launchctl load ~/Library/LaunchAgents/com.trading.daily.plist
launchctl load ~/Library/LaunchAgents/com.trading.intraday.plist
```

### Permanently Remove

```bash
# Unload
launchctl unload ~/Library/LaunchAgents/com.trading.daily.plist
launchctl unload ~/Library/LaunchAgents/com.trading.intraday.plist

# Delete files
rm ~/Library/LaunchAgents/com.trading.daily.plist
rm ~/Library/LaunchAgents/com.trading.intraday.plist
```

## Troubleshooting

### LaunchAgent Not Running

**Check if loaded:**
```bash
launchctl list | grep trading
```

**If not listed, reload:**
```bash
launchctl load ~/Library/LaunchAgents/com.trading.daily.plist
```

### Drive Never Detected

**Check mount point:**
```bash
ls /Volumes/
# Should show "Vault" when connected

diskutil list
# Shows all mounted drives
```

**Test drive detection:**
```bash
./master_orchestrator/wait_for_drive.py --daily --max-wait-hours 0.1
# Waits max 6 minutes for testing
```

### Job Failed

**Check logs:**
```bash
# Look for errors
cat /tmp/trading_daily_error.log
cat /tmp/trading_daily.log

# Check system logs
log show --predicate 'subsystem == "com.apple.launchd"' --last 1h | grep trading
```

### Wrong Time Zone

The plist files use **local time** (not ET). If your Mac is in a different timezone:

**Example: Pacific Time (PT)**
- Market opens 9:30 AM ET = 6:30 AM PT
- Change `<key>Hour</key><integer>9</integer>` to `<integer>6</integer>`

**Edit plist files:**
```bash
nano ~/Library/LaunchAgents/com.trading.daily.plist
# Change Hour to match your timezone
# Reload after editing:
launchctl unload ~/Library/LaunchAgents/com.trading.daily.plist
launchctl load ~/Library/LaunchAgents/com.trading.daily.plist
```

### Permissions Issues

**If LaunchAgent can't access files:**
```bash
# Grant Full Disk Access to Python
# System Preferences → Security & Privacy → Privacy → Full Disk Access
# Add: /Users/henry/miniconda3/envs/trading/bin/python
```

## Execution Timeline

**Typical day:**

```
9:30:00 AM - Both LaunchAgents trigger simultaneously

DAILY PIPELINE (runs once):
9:30:01 AM - wait_for_drive.py checks for external drive
9:30:01 AM - Drive found! Starting daily tasks
9:30:05 AM - Tier 0: Asset Collection
9:30:30 AM - Tier 1: Fundamentals, Economic Indicators
9:35:00 AM - Tier 2: Insider Trading, Analyst Ratings
9:40:00 AM - Tier 3: News, Sentiment, Earnings, Options
9:50:00 AM - Tier 4: ML Features Aggregation
9:55:00 AM - Tier 5: Generate Trading Signals
9:58:00 AM - Daily pipeline completes and exits ✅

INTRADAY PIPELINE (runs continuously):
9:30:01 AM - wait_and_run_intraday.sh checks for external drive
9:30:01 AM - Drive found! Starting continuous loop
9:30:05 AM - Cycle 1: Price Data, Volume Data
9:35:00 AM - Cycle 2: Price Data, Technical Indicators
9:40:00 AM - Cycle 3: Price Data, ML Features Update
... (continues every 30 seconds, checking task intervals)
4:00:00 PM - Market closes, intraday pipeline stops automatically ✅
```

## Security Notes

- LaunchAgents run as your user (not root)
- Have same permissions as your account
- Can access external drives if you can
- Logs are world-readable in `/tmp/` (contains no secrets)
- Database on external drive is only accessible when mounted

## Best Practices

1. **Test First**: Run `install.sh` on a Friday evening, verify Saturday morning
2. **Monitor Logs**: Check logs daily for first week
3. **External Drive**: Use a powered hub if USB drive disconnects
4. **Backup Database**: Daily backups of `/Volumes/Vault/85_assets_prediction.db`
5. **Update Conda**: Keep `trading` environment updated

---

**Created:** 2025-11-15
**System:** macOS LaunchAgent
**Python Environment:** `/Users/henry/miniconda3/envs/trading`
**Database:** `/Volumes/Vault/85_assets_prediction.db`
