#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Health Monitor for Trading Data Pipeline

Tracks data freshness, task status, and pipeline health.
Provides green/yellow/red status indicators for all data sources.
"""

import sqlite3
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sys


class HealthStatus(Enum):
    """Health status levels"""
    GREEN = "GREEN"    # Data is fresh
    YELLOW = "YELLOW"  # Data is stale but acceptable
    RED = "RED"        # Data is critically stale
    UNKNOWN = "UNKNOWN"  # No data or status available


@dataclass
class DataSourceHealth:
    """Health information for a data source"""
    source: str
    status: HealthStatus
    last_updated: Optional[datetime]
    age_seconds: Optional[int]
    threshold_green: int
    threshold_yellow: int
    message: str


class HealthMonitor:
    """Monitor and track health of all data sources"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 config_path: Optional[str] = None):
        """
        Initialize health monitor

        Args:
            db_path: Path to SQLite database
            config_path: Path to dependency_graph.yaml (auto-detected if None)
        """
        self.db_path = db_path

        # Auto-detect config path
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "dependency_graph.yaml"

        self.config_path = config_path
        self.config = self._load_config()
        self.thresholds = self.config.get('freshness_thresholds', {})

        self._ensure_status_table()

    def _load_config(self) -> dict:
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load config from {self.config_path}: {e}")
            return {}

    def _ensure_status_table(self):
        """Create pipeline status table if it doesn't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_pipeline_status (
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
            """)
            conn.commit()

    def check_table_freshness(self, table_name: str,
                             timestamp_column: str = 'last_updated') -> Tuple[Optional[datetime], Optional[int]]:
        """
        Check freshness of a specific table

        Returns:
            (last_updated, age_in_seconds) or (None, None) if no data
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(f"""
                    SELECT MAX({timestamp_column})
                    FROM {table_name}
                    WHERE {timestamp_column} IS NOT NULL
                """)
                result = cursor.fetchone()

                if result and result[0]:
                    last_updated = datetime.fromisoformat(result[0])
                    age_seconds = int((datetime.now() - last_updated).total_seconds())
                    return last_updated, age_seconds
                else:
                    return None, None

        except sqlite3.OperationalError:
            # Table doesn't exist
            return None, None
        except Exception as e:
            print(f"⚠️  Error checking {table_name}: {e}")
            return None, None

    def get_task_status(self, task_name: str) -> Optional[Dict]:
        """Get status information for a specific task"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM data_pipeline_status
                    WHERE task_name = ?
                """, (task_name,))

                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None

        except Exception as e:
            print(f"⚠️  Error getting task status for {task_name}: {e}")
            return None

    def update_task_status(self, task_name: str, status: str,
                          success: bool = True, error_message: Optional[str] = None,
                          records_processed: int = 0, runtime_seconds: float = 0):
        """
        Update status for a pipeline task

        Args:
            task_name: Name of the task
            status: Current status (RUNNING, SUCCESS, FAILED)
            success: Whether the task succeeded
            error_message: Error message if failed
            records_processed: Number of records processed
            runtime_seconds: How long the task took
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()

                # Get current stats
                current = self.get_task_status(task_name) or {}

                total_runs = current.get('total_runs', 0) + 1
                consecutive_failures = 0 if success else current.get('consecutive_failures', 0) + 1

                if success:
                    total_successes = current.get('total_successes', 0) + 1
                    total_failures = current.get('total_failures', 0)
                    last_success = now
                    last_failure = current.get('last_failure')
                else:
                    total_successes = current.get('total_successes', 0)
                    total_failures = current.get('total_failures', 0) + 1
                    last_success = current.get('last_success')
                    last_failure = now

                # Calculate rolling average runtime
                avg_runtime = current.get('average_runtime_seconds') or 0
                if avg_runtime == 0 or avg_runtime is None:
                    new_avg_runtime = runtime_seconds
                else:
                    # Exponential moving average (weight recent runs more)
                    alpha = 0.3
                    new_avg_runtime = alpha * runtime_seconds + (1 - alpha) * avg_runtime

                # Check if task exists
                if current:
                    # Update existing record
                    conn.execute("""
                        UPDATE data_pipeline_status
                        SET last_run_end = ?, last_success = ?, last_failure = ?,
                            consecutive_failures = ?, total_runs = ?, total_successes = ?,
                            total_failures = ?, last_error_message = ?,
                            last_records_processed = ?, average_runtime_seconds = ?,
                            status = ?, updated_at = ?
                        WHERE task_name = ?
                    """, (
                        now, last_success, last_failure,
                        consecutive_failures, total_runs, total_successes, total_failures,
                        error_message, records_processed, new_avg_runtime,
                        status, now, task_name
                    ))
                else:
                    # Insert new record
                    conn.execute("""
                        INSERT INTO data_pipeline_status (
                            task_name, last_run_end, last_success, last_failure,
                            consecutive_failures, total_runs, total_successes, total_failures,
                            last_error_message, last_records_processed, average_runtime_seconds,
                            status, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_name, now, last_success, last_failure,
                        consecutive_failures, total_runs, total_successes, total_failures,
                        error_message, records_processed, new_avg_runtime,
                        status, now
                    ))
                conn.commit()

        except Exception as e:
            print(f"⚠️  Error updating task status for {task_name}: {e}")

    def mark_task_started(self, task_name: str):
        """Mark that a task has started running"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                now = datetime.now().isoformat()

                # Check if task exists
                cursor = conn.execute(
                    "SELECT task_name FROM data_pipeline_status WHERE task_name = ?",
                    (task_name,)
                )
                exists = cursor.fetchone() is not None

                if exists:
                    # Update existing record
                    conn.execute("""
                        UPDATE data_pipeline_status
                        SET last_run_start = ?, status = 'RUNNING', updated_at = ?
                        WHERE task_name = ?
                    """, (now, now, task_name))
                else:
                    # Insert new record
                    conn.execute("""
                        INSERT INTO data_pipeline_status (
                            task_name, last_run_start, status, updated_at
                        ) VALUES (?, ?, 'RUNNING', ?)
                    """, (task_name, now, now))

                conn.commit()
        except Exception as e:
            print(f"⚠️  Error marking task started for {task_name}: {e}")

    def check_data_source_health(self, source_name: str, table_name: str,
                                 timestamp_column: str = 'last_updated') -> DataSourceHealth:
        """
        Check health of a specific data source

        Args:
            source_name: Logical name (e.g., 'price_data', 'fundamentals')
            table_name: Database table name
            timestamp_column: Column containing timestamp

        Returns:
            DataSourceHealth object with status
        """
        # Get thresholds for this source
        green_threshold = self.thresholds.get('green', {}).get(source_name, 3600)
        yellow_threshold = self.thresholds.get('yellow', {}).get(source_name, 7200)

        # Check table freshness
        last_updated, age_seconds = self.check_table_freshness(table_name, timestamp_column)

        # Determine status
        if last_updated is None:
            status = HealthStatus.UNKNOWN
            message = f"No data in {table_name}"
        elif age_seconds <= green_threshold:
            status = HealthStatus.GREEN
            message = f"Fresh (updated {self._format_age(age_seconds)} ago)"
        elif age_seconds <= yellow_threshold:
            status = HealthStatus.YELLOW
            message = f"Stale (updated {self._format_age(age_seconds)} ago)"
        else:
            status = HealthStatus.RED
            message = f"Critically stale (updated {self._format_age(age_seconds)} ago)"

        return DataSourceHealth(
            source=source_name,
            status=status,
            last_updated=last_updated,
            age_seconds=age_seconds,
            threshold_green=green_threshold,
            threshold_yellow=yellow_threshold,
            message=message
        )

    def get_overall_health(self) -> Dict[str, DataSourceHealth]:
        """
        Get health status for all data sources

        Returns:
            Dictionary mapping source names to DataSourceHealth objects
        """
        sources = {
            'price_data': ('price_data', 'timestamp'),
            'fundamentals': ('fundamentals', 'last_updated'),
            'sentiment': ('sentiment_scores', 'scored_at'),
            'news': ('news', 'published_date'),
            'economic': ('economic_indicators', 'date'),
            'options': ('options_data', 'last_updated'),
            'earnings': ('earnings', 'report_date'),
            'insider': ('insider_trades', 'transaction_date'),
            'analyst': ('analyst_ratings', 'rating_date'),
            'ml_features': ('ml_features', 'feature_date'),
        }

        health_report = {}
        for source_name, (table_name, timestamp_col) in sources.items():
            health_report[source_name] = self.check_data_source_health(
                source_name, table_name, timestamp_col
            )

        return health_report

    def print_health_dashboard(self):
        """Print a colored health dashboard to console"""
        print("\n" + "="*80)
        print("📊 DATA PIPELINE HEALTH DASHBOARD")
        print("="*80)

        health_report = self.get_overall_health()

        # Group by status
        green = []
        yellow = []
        red = []
        unknown = []

        for source, health in health_report.items():
            if health.status == HealthStatus.GREEN:
                green.append((source, health))
            elif health.status == HealthStatus.YELLOW:
                yellow.append((source, health))
            elif health.status == HealthStatus.RED:
                red.append((source, health))
            else:
                unknown.append((source, health))

        # Print by status (red first, then yellow, then green)
        if red:
            print("\n🔴 CRITICAL - Data Critically Stale:")
            for source, health in sorted(red):
                print(f"   • {source:20s} - {health.message}")

        if yellow:
            print("\n🟡 WARNING - Data Stale:")
            for source, health in sorted(yellow):
                print(f"   • {source:20s} - {health.message}")

        if green:
            print("\n🟢 HEALTHY - Data Fresh:")
            for source, health in sorted(green):
                print(f"   • {source:20s} - {health.message}")

        if unknown:
            print("\n⚪ UNKNOWN - No Data:")
            for source, health in sorted(unknown):
                print(f"   • {source:20s} - {health.message}")

        # Overall summary
        print("\n" + "-"*80)
        total = len(health_report)
        print(f"📈 SUMMARY: {len(green)} healthy, {len(yellow)} stale, "
              f"{len(red)} critical, {len(unknown)} unknown (Total: {total})")
        print("="*80 + "\n")

    def print_task_status_dashboard(self):
        """Print status of all pipeline tasks"""
        print("\n" + "="*80)
        print("⚙️  PIPELINE TASK STATUS")
        print("="*80)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT task_name, status, last_run_end, last_success,
                           consecutive_failures, total_runs, total_successes,
                           average_runtime_seconds, last_error_message
                    FROM data_pipeline_status
                    ORDER BY task_name
                """)

                rows = cursor.fetchall()

                if not rows:
                    print("\n   No task history available yet.\n")
                    return

                for row in rows:
                    task = dict(row)

                    # Status indicator
                    if task['status'] == 'SUCCESS':
                        indicator = "✅"
                    elif task['status'] == 'FAILED':
                        indicator = "❌"
                    elif task['status'] == 'RUNNING':
                        indicator = "🔄"
                    else:
                        indicator = "⏸️ "

                    # Success rate
                    success_rate = 0
                    if task['total_runs'] > 0:
                        success_rate = (task['total_successes'] / task['total_runs']) * 100

                    print(f"\n{indicator} {task['task_name']}")
                    print(f"   Status: {task['status']} | "
                          f"Success Rate: {success_rate:.1f}% ({task['total_successes']}/{task['total_runs']})")

                    if task['last_success']:
                        last_success_dt = datetime.fromisoformat(task['last_success'])
                        age = self._format_age(int((datetime.now() - last_success_dt).total_seconds()))
                        print(f"   Last Success: {age} ago")

                    if task['consecutive_failures'] > 0:
                        print(f"   ⚠️  Consecutive Failures: {task['consecutive_failures']}")

                    if task['average_runtime_seconds']:
                        print(f"   Avg Runtime: {task['average_runtime_seconds']:.1f}s")

                    if task['last_error_message']:
                        print(f"   Last Error: {task['last_error_message'][:60]}...")

        except Exception as e:
            print(f"⚠️  Error fetching task status: {e}")

        print("\n" + "="*80 + "\n")

    def _format_age(self, seconds: int) -> str:
        """Format age in seconds to human-readable string"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days}d {hours}h"

    def get_stale_sources(self, min_status: HealthStatus = HealthStatus.YELLOW) -> List[str]:
        """
        Get list of data sources that are at least as stale as min_status

        Args:
            min_status: Minimum staleness level (YELLOW or RED)

        Returns:
            List of source names that need updating
        """
        health_report = self.get_overall_health()

        stale = []
        for source, health in health_report.items():
            if min_status == HealthStatus.YELLOW:
                if health.status in [HealthStatus.YELLOW, HealthStatus.RED]:
                    stale.append(source)
            elif min_status == HealthStatus.RED:
                if health.status == HealthStatus.RED:
                    stale.append(source)

        return stale

    def is_healthy(self) -> bool:
        """Check if overall pipeline is healthy (no red or yellow statuses)"""
        health_report = self.get_overall_health()

        for health in health_report.values():
            if health.status in [HealthStatus.YELLOW, HealthStatus.RED]:
                return False

        return True


def main():
    """Main entry point for health monitoring CLI"""
    monitor = HealthMonitor()

    # Print both dashboards
    monitor.print_health_dashboard()
    monitor.print_task_status_dashboard()

    # Exit with error code if unhealthy
    if not monitor.is_healthy():
        print("⚠️  WARNING: Pipeline has stale data sources!")
        sys.exit(1)
    else:
        print("✅ All data sources are healthy!")
        sys.exit(0)


if __name__ == "__main__":
    main()
