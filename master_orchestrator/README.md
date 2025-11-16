# Master Orchestrator System

Automated two-tier scheduling system for the trading data pipeline.

## Overview

The Master Orchestrator provides comprehensive automation for all data collection, processing, and signal generation tasks. It implements a two-tier architecture:

1. **Daily Runner** - Executes at 9:30 AM for slow-changing data (fundamentals, sentiment, news, etc.)
2. **Intraday Runner** - Executes every 1-5 minutes during market hours for live market data

## Architecture

```
orchestrator.py (Master Controller)
├── daily_runner.py (Daily Tasks - 9:30 AM)
│   ├── Asset Collection
│   ├── Fundamentals, Insider, Analyst, Economic Data
│   ├── News & Sentiment
│   ├── Options & Earnings
│   ├── ML Features Aggregation
│   └── Signal Generation
│
├── intraday_runner.py (Live Market Data - 1-5 min)
│   ├── Price Data (every 1 min)
│   ├── ML Features Refresh (every 5 min)
│   └── Intraday Signals (every 5 min)
│
├── health_monitor.py (Data Freshness Tracking)
│   ├── Green/Yellow/Red Status
│   ├── Task Success/Failure Tracking
│   └── Pipeline Health Dashboard
│
└── dependency_graph.yaml (Execution Order Definition)
    ├── Task Dependencies
    ├── Timing Configuration
    └── Freshness Thresholds
```

## Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Master controller - unified interface for all operations |
| `daily_runner.py` | Executes daily scheduled tasks with dependency management |
| `intraday_runner.py` | Executes high-frequency tasks during market hours |
| `health_monitor.py` | Tracks data freshness and task status |
| `dependency_graph.yaml` | Defines execution order, timing, and thresholds |

## Usage

### Quick Start

```bash
# Check health of all data sources
python3 master_orchestrator/orchestrator.py --health

# Update stale data
python3 master_orchestrator/orchestrator.py --update-stale

# Run daily pipeline once
python3 master_orchestrator/orchestrator.py --daily

# Run intraday pipeline continuously
python3 master_orchestrator/orchestrator.py --intraday

# Run full system (daily + continuous intraday)
python3 master_orchestrator/orchestrator.py --full
```

### Common Operations

#### 1. Health Check
```bash
# Check if data is fresh
python3 master_orchestrator/orchestrator.py --health

# Or use health monitor directly
python3 master_orchestrator/health_monitor.py
```

**Output:**
```
📊 DATA PIPELINE HEALTH DASHBOARD
================================================================================

🔴 CRITICAL - Data Critically Stale:
   • price_data         - Critically stale (updated 2d 5h ago)

🟡 WARNING - Data Stale:
   • sentiment          - Stale (updated 3h 15m ago)

🟢 HEALTHY - Data Fresh:
   • fundamentals       - Fresh (updated 45m ago)
   • ml_features        - Fresh (updated 5m ago)
   • news               - Fresh (updated 20m ago)

📈 SUMMARY: 8 healthy, 1 stale, 1 critical, 0 unknown (Total: 10)
```

#### 2. Daily Pipeline
```bash
# Run all daily tasks (9:30 AM schedule)
python3 master_orchestrator/orchestrator.py --daily

# Dry run (test without executing)
python3 master_orchestrator/orchestrator.py --daily --dry-run

# Direct usage
python3 master_orchestrator/daily_runner.py
```

**Daily Schedule:**
- `09:30:00` - Asset Collection (Tier 0)
- `09:30:30` - Fundamentals, Insider Trading (Tier 1)
- `09:31:00` - Analyst Ratings (Tier 1)
- `09:31:30` - Economic Indicators (Tier 1)
- `09:32:00` - News Events (Tier 2)
- `09:32:30` - Sentiment Analysis (Tier 2)
- `09:33:00` - Options Data (Tier 3)
- `09:33:30` - Earnings Data (Tier 3)
- `09:34:00` - ML Features Aggregation (Tier 4)
- `09:35:00` - Generate Trading Signals (Tier 5)

#### 3. Intraday Pipeline
```bash
# Run continuously (checks every 30 seconds)
python3 master_orchestrator/orchestrator.py --intraday

# Run once and exit
python3 master_orchestrator/orchestrator.py --intraday --once

# Custom check interval (every 60 seconds)
python3 master_orchestrator/orchestrator.py --intraday --interval 60

# Direct usage
python3 master_orchestrator/intraday_runner.py
```

**Intraday Schedule:**
- **Every 1 minute** (during market hours):
  - Price data collection
- **Every 5 minutes** (during market hours):
  - ML features refresh (incremental)
  - Trading signal generation (incremental)

#### 4. Full System
```bash
# Run complete system (production mode)
python3 master_orchestrator/orchestrator.py --full

# This will:
# 1. Run daily pipeline (blocking)
# 2. Start intraday pipeline (continuous)
# 3. Monitor and restart if intraday crashes
# 4. Run until interrupted (Ctrl+C)
```

#### 5. Update Stale Data
```bash
# Update all yellow/red data sources
python3 master_orchestrator/orchestrator.py --update-stale

# Update only critically stale (red) sources
python3 master_orchestrator/orchestrator.py --update-stale --min-staleness red
```

## Key Features

### 1. Dependency Management
Tasks execute in the correct order based on their dependencies:

```yaml
# Example from dependency_graph.yaml
- name: "ML Features Aggregation"
  tier: 4
  dependencies: ["Fundamental Data", "News Events", "Sentiment Analysis"]
```

If a dependency fails, dependent tasks are automatically skipped.

### 2. Failure Isolation
- Each task runs in isolation with try/except blocks
- One script failing doesn't crash the entire pipeline
- Retry logic with exponential backoff (3 attempts: 10s, 30s, 60s)
- Critical vs non-critical task designation

### 3. Idempotency
All tasks use `INSERT OR REPLACE` to ensure:
- Safe to run multiple times
- No duplicate data
- Incremental updates only process new data

### 4. Data Freshness Tracking
Three-tier status system:

| Status | Meaning | Price Data | Fundamentals | Sentiment |
|--------|---------|-----------|--------------|-----------|
| 🟢 Green | Fresh | < 5 min | < 24 hours | < 1 hour |
| 🟡 Yellow | Stale | 5-15 min | 24-48 hours | 1-2 hours |
| 🔴 Red | Critical | > 15 min | > 48 hours | > 2 hours |

### 5. Health Monitoring
Tracks for each task:
- Last run timestamp
- Success/failure status
- Consecutive failures
- Success rate (%)
- Average runtime
- Records processed
- Error messages

### 6. Market Hours Awareness
Intraday tasks automatically:
- Only run during market hours (9:30 AM - 4:00 PM ET)
- Skip execution when market is closed
- Support pre-market and post-market hours (configurable)

### 7. Smart Scheduling
- **Daily tasks**: Run once per day at scheduled times
- **Intraday tasks**: Run at specified intervals (1-5 min)
- **Incremental mode**: Only process new data (via `--incremental` flag)
- **Interval-based**: Tasks track last execution and skip if too soon

## Configuration

### dependency_graph.yaml

Edit this file to customize:

```yaml
# Daily schedule timing
daily_schedule:
  - name: "Asset Collection"
    script: "data_collectors/01_collect_assets.py"
    time: "09:30:00"
    tier: 0
    dependencies: []
    max_runtime_seconds: 120
    critical: true

# Intraday intervals
intraday_schedule:
  - name: "Price Data"
    script: "data_collectors/02_collect_price_data.py"
    interval_minutes: 1
    tier: 0
    max_runtime_seconds: 45
    critical: true
    market_hours_only: true

# Freshness thresholds
freshness_thresholds:
  green:
    price_data: 300      # 5 minutes
    fundamentals: 86400  # 24 hours
  yellow:
    price_data: 900      # 15 minutes
    fundamentals: 172800 # 48 hours

# Retry policy
retry_policy:
  max_attempts: 3
  backoff_seconds: [10, 30, 60]
```

## macOS LaunchAgent Setup

To run automatically using macOS `launchd`, you'll create launch agents later. The scripts are designed to work seamlessly with launchd:

### Daily Runner (runs at 9:30 AM)
```xml
<!-- Future: com.trading.daily.plist -->
<dict>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/path/to/orchestrator.py</string>
        <string>--daily</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
</dict>
```

### Intraday Runner (runs continuously)
```xml
<!-- Future: com.trading.intraday.plist -->
<dict>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>/path/to/orchestrator.py</string>
        <string>--intraday</string>
    </array>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
</dict>
```

## Logging

All components log to:
- **Console**: Real-time output with colored status indicators
- **Log files**: `/tmp/{component_name}.log`

### Log Locations
- `/tmp/orchestrator.log` - Master orchestrator
- `/tmp/daily_runner.log` - Daily pipeline
- `/tmp/intraday_runner.log` - Intraday pipeline

### Log Format
```
2025-11-15 09:30:15 - INFO - ▶️  Running: Asset Collection
2025-11-15 09:30:17 - INFO -    ✅ Asset Collection completed in 2.1s
```

## Database Schema

The system creates and maintains a status tracking table:

```sql
CREATE TABLE data_pipeline_status (
    task_name TEXT PRIMARY KEY,
    last_run_start TIMESTAMP,
    last_run_end TIMESTAMP,
    last_success TIMESTAMP,
    last_failure TIMESTAMP,
    consecutive_failures INTEGER DEFAULT 0,
    total_runs INTEGER DEFAULT 0,
    total_successes INTEGER DEFAULT 0,
    total_failures INTEGER DEFAULT 0,
    last_error_message TEXT,
    last_records_processed INTEGER,
    average_runtime_seconds REAL,
    status TEXT DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

## Troubleshooting

### Task Keeps Failing
```bash
# Check task status
python3 master_orchestrator/health_monitor.py

# Look for the task in "PIPELINE TASK STATUS"
# Check consecutive_failures and last_error_message

# Run just that task manually
python3 data_collectors/XX_failing_script.py
```

### Data Is Stale
```bash
# Check what's stale
python3 master_orchestrator/orchestrator.py --health

# Update all stale data
python3 master_orchestrator/orchestrator.py --update-stale
```

### Intraday Pipeline Not Running
```bash
# Check if it's market hours
python3 -c "from intraday_runner import IntradayRunner; r = IntradayRunner(); print(r.is_market_hours())"

# Run once to test
python3 master_orchestrator/orchestrator.py --intraday --once

# Check logs
tail -f /tmp/intraday_runner.log
```

### Dependencies Not Met
Tasks may skip if dependencies failed. Check:
1. Which task failed (health monitor)
2. Why it failed (error message in status)
3. Fix the failing task
4. Re-run the pipeline

## Performance Optimizations

Built-in optimizations:
- ✅ **Batch API calls** - Collect multiple assets in single request
- ✅ **Smart scheduling** - Only run when needed (interval-based)
- ✅ **Incremental updates** - Process only new data
- ✅ **Parallel execution** - Tasks in same tier can run concurrently
- ✅ **Caching** - Database acts as persistent cache
- ✅ **Early exit** - Stop on critical failures
- ✅ **Timeouts** - Prevent hung tasks from blocking pipeline

## Testing

### Dry Run
```bash
# Test daily pipeline without executing
python3 master_orchestrator/orchestrator.py --daily --dry-run

# Test intraday pipeline without executing
python3 master_orchestrator/orchestrator.py --intraday --dry-run --once
```

**Dry run output:**
```
🧪 DRY RUN MODE - No scripts will actually execute
▶️  Running: Asset Collection
   [DRY RUN] Would execute: python3 data_collectors/01_collect_assets.py
   ✅ Asset Collection completed in 0.0s
```

### Health Check
```bash
# Quick health check
python3 master_orchestrator/orchestrator.py --health
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success - all tasks completed successfully |
| 1 | Failure - one or more tasks failed |

## Future Enhancements

Potential improvements:
- [ ] Email/Slack notifications on critical failures
- [ ] Web dashboard for real-time monitoring
- [ ] Automatic retry of failed tasks
- [ ] Machine learning for optimal scheduling
- [ ] Database connection pooling
- [ ] Distributed execution across multiple machines
- [ ] API endpoints for external control

## Support

For issues or questions:
1. Check logs in `/tmp/*.log`
2. Run health check: `python3 master_orchestrator/orchestrator.py --health`
3. Test individual scripts: `python3 data_collectors/XX_script.py`
4. Review `dependency_graph.yaml` for configuration

---

**Last Updated:** 2025-11-15
**Version:** 1.0.0
**Python Version:** 3.9+
