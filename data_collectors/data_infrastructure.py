"""
Data Infrastructure Module
===========================
Centralized configuration and utilities for data quality, rate limiting, and consistency
"""

import time
import hashlib
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataConfig:
    """
    Centralized data configuration for consistent lookback windows across all scripts

    This ensures all data collectors use the same time windows, preventing
    inconsistencies that can compromise ML model training.
    """

    # Training data windows
    HISTORICAL_DAYS = 730  # 2 years for training (minimum for robust ML)
    FEATURE_LOOKBACK = 252  # 1 year for technical features
    SENTIMENT_LOOKBACK = 90  # 3 months for sentiment (news/analyst data)
    EARNINGS_LOOKBACK = 365  # 1 year for earnings history
    INSIDER_LOOKBACK = 90  # 3 months for insider trading patterns

    # Minimum data requirements
    MINIMUM_DATA_POINTS = 60  # Minimum data points for any calculation
    MINIMUM_TRADING_DAYS = 252  # 1 year of trading for ML training

    # Forward-fill limits (prevent artificial patterns)
    MAX_FFILL_DAYS = {
        'sentiment_score': 7,  # Max 1 week for sentiment
        'analyst_rating': 30,  # Max 1 month for analyst ratings
        'fundamental_data': 90,  # Max 1 quarter for fundamentals
        'technical_indicators': 2,  # Max 2 days for technical indicators
        'volume': 0,  # Never forward-fill volume
        'price': 0,  # Never forward-fill price
    }

    @staticmethod
    def get_date_range(lookback_type: str = 'historical') -> Tuple[datetime, datetime]:
        """
        Get consistent date range for data collection

        Args:
            lookback_type: One of 'historical', 'feature', 'sentiment', 'earnings', 'insider'

        Returns:
            Tuple of (start_date, end_date)
        """
        end = datetime.now()

        lookback_map = {
            'historical': DataConfig.HISTORICAL_DAYS,
            'feature': DataConfig.FEATURE_LOOKBACK,
            'sentiment': DataConfig.SENTIMENT_LOOKBACK,
            'earnings': DataConfig.EARNINGS_LOOKBACK,
            'insider': DataConfig.INSIDER_LOOKBACK,
        }

        days = lookback_map.get(lookback_type, DataConfig.HISTORICAL_DAYS)
        start = end - timedelta(days=days)

        return start, end

    @staticmethod
    def get_max_ffill_limit(column_type: str) -> int:
        """Get maximum forward-fill limit for a column type"""
        return DataConfig.MAX_FFILL_DAYS.get(column_type, 3)


class RateLimitManager:
    """
    Intelligent rate limiting with exponential backoff

    Prevents API rate limit violations that cause data gaps.
    Supports multiple APIs with different rate limits.
    """

    def __init__(self):
        """Initialize rate limit trackers for different APIs"""
        # Polygon.io: 5 calls per minute (free tier)
        self.polygon_calls = deque(maxlen=5)

        # Yahoo Finance: 60 calls per minute (unofficial limit)
        self.yfinance_calls = deque(maxlen=60)

        # News API: 100 calls per day (free tier)
        self.newsapi_calls = deque(maxlen=100)
        self.newsapi_daily_reset = datetime.now().date()

        # Alpha Vantage: 5 calls per minute, 500 per day
        self.alphavantage_calls = deque(maxlen=5)
        self.alphavantage_daily = deque(maxlen=500)
        self.alphavantage_daily_reset = datetime.now().date()

    def wait_if_needed(self, api_type: str = 'yfinance') -> None:
        """
        Smart rate limiting with exponential backoff

        Args:
            api_type: One of 'polygon', 'yfinance', 'newsapi', 'alphavantage'
        """
        current_time = time.time()
        current_date = datetime.now().date()

        if api_type == 'polygon':
            self._rate_limit_polygon(current_time)
        elif api_type == 'yfinance':
            self._rate_limit_yfinance(current_time)
        elif api_type == 'newsapi':
            self._rate_limit_newsapi(current_time, current_date)
        elif api_type == 'alphavantage':
            self._rate_limit_alphavantage(current_time, current_date)
        else:
            logger.warning(f"Unknown API type: {api_type}")

    def _rate_limit_polygon(self, current_time: float) -> None:
        """Rate limit for Polygon.io (5 calls/minute)"""
        if len(self.polygon_calls) == 5:
            time_elapsed = current_time - self.polygon_calls[0]
            if time_elapsed < 60:
                wait_time = 60 - time_elapsed + 1
                logger.info(f"Polygon rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        self.polygon_calls.append(current_time)

    def _rate_limit_yfinance(self, current_time: float) -> None:
        """Rate limit for Yahoo Finance (60 calls/minute, conservative)"""
        if len(self.yfinance_calls) == 60:
            time_elapsed = current_time - self.yfinance_calls[0]
            if time_elapsed < 60:
                wait_time = 60 - time_elapsed + 1
                logger.info(f"Yahoo Finance rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        self.yfinance_calls.append(current_time)

    def _rate_limit_newsapi(self, current_time: float, current_date) -> None:
        """Rate limit for News API (100 calls/day)"""
        # Reset daily counter if new day
        if current_date > self.newsapi_daily_reset:
            self.newsapi_calls.clear()
            self.newsapi_daily_reset = current_date
            logger.info("News API daily limit reset")

        if len(self.newsapi_calls) >= 100:
            logger.error("News API daily limit reached (100 calls)")
            raise Exception("News API daily limit exceeded")

        self.newsapi_calls.append(current_time)

    def _rate_limit_alphavantage(self, current_time: float, current_date) -> None:
        """Rate limit for Alpha Vantage (5 calls/minute, 500/day)"""
        # Reset daily counter if new day
        if current_date > self.alphavantage_daily_reset:
            self.alphavantage_daily.clear()
            self.alphavantage_daily_reset = current_date
            logger.info("Alpha Vantage daily limit reset")

        # Check daily limit
        if len(self.alphavantage_daily) >= 500:
            logger.error("Alpha Vantage daily limit reached (500 calls)")
            raise Exception("Alpha Vantage daily limit exceeded")

        # Check per-minute limit
        if len(self.alphavantage_calls) == 5:
            time_elapsed = current_time - self.alphavantage_calls[0]
            if time_elapsed < 60:
                wait_time = 60 - time_elapsed + 1
                logger.info(f"Alpha Vantage rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        self.alphavantage_calls.append(current_time)
        self.alphavantage_daily.append(current_time)


class DataQualityManager:
    """
    Intelligent data filling and quality control

    Prevents blind forward-filling that creates artificial patterns.
    """

    @staticmethod
    def smart_fill(df: pd.DataFrame, col: str, method: str = 'ffill', limit: Optional[int] = None) -> pd.Series:
        """
        Intelligent filling with limits based on data type

        Args:
            df: DataFrame containing the column
            col: Column name to fill
            method: Fill method ('ffill', 'interpolate', or 'zero')
            limit: Maximum number of consecutive NaNs to fill (auto-detected if None)

        Returns:
            Filled Series
        """
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            return pd.Series()

        # Auto-detect limit based on column type
        if limit is None:
            if col in ['sentiment_score', 'analyst_rating', 'rating_numeric']:
                limit = DataConfig.MAX_FFILL_DAYS['sentiment_score']  # 7 days
            elif col in ['pe_ratio', 'price_to_book', 'debt_to_equity', 'profit_margin']:
                limit = DataConfig.MAX_FFILL_DAYS['fundamental_data']  # 90 days
            elif col in ['rsi_14', 'macd', 'macd_histogram', 'bb_position']:
                limit = DataConfig.MAX_FFILL_DAYS['technical_indicators']  # 2 days
            elif col in ['volume', 'trades_count', 'shares_traded']:
                limit = 0  # Never forward-fill volume
            elif col in ['close', 'open', 'high', 'low']:
                limit = 0  # Never forward-fill price
            else:
                limit = 3  # Conservative default

        # Apply filling method
        if method == 'ffill':
            filled = df[col].ffill(limit=limit)
        elif method == 'interpolate':
            filled = df[col].interpolate(method='linear', limit=limit)
        elif method == 'zero':
            filled = df[col].fillna(0)
        else:
            logger.warning(f"Unknown fill method: {method}, using ffill")
            filled = df[col].ffill(limit=limit)

        # Log filling statistics
        original_nulls = df[col].isna().sum()
        remaining_nulls = filled.isna().sum()
        filled_count = original_nulls - remaining_nulls

        if filled_count > 0:
            logger.debug(f"Filled {filled_count}/{original_nulls} nulls in {col} (limit={limit})")

        return filled

    @staticmethod
    def validate_data_quality(df: pd.DataFrame, required_cols: List[str], min_rows: int = None) -> Tuple[bool, str]:
        """
        Validate data quality before processing

        Args:
            df: DataFrame to validate
            required_cols: List of required columns
            min_rows: Minimum number of rows required

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if DataFrame is empty
        if df.empty:
            return False, "DataFrame is empty"

        # Check minimum rows
        if min_rows and len(df) < min_rows:
            return False, f"Insufficient rows: {len(df)} < {min_rows}"

        # Check required columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check for excessive nulls (>80% null = bad data)
        for col in required_cols:
            null_pct = df[col].isna().sum() / len(df)
            if null_pct > 0.8:
                return False, f"Column {col} is {null_pct:.1%} null (>80% threshold)"

        return True, "Data quality validated"


class DataValidator:
    """
    Validate data ranges and remove outliers

    Prevents garbage data from breaking ML models:
    - RSI of 500 → capped to 100
    - PE ratio of -1000 → capped to -100
    - Negative volume → capped to 0
    """

    # Valid ranges for all metrics (min, max)
    VALID_RANGES = {
        # Technical indicators
        'rsi_14': (0, 100),
        'rsi_7': (0, 100),
        'rsi_21': (0, 100),
        'macd': (-100, 100),
        'macd_signal': (-100, 100),
        'macd_histogram': (-50, 50),
        'bb_position': (0, 1),
        'stoch_k': (0, 100),
        'stoch_d': (0, 100),

        # Fundamental ratios
        'pe_ratio': (-100, 500),
        'price_to_book': (0, 100),
        'price_to_sales': (0, 100),
        'ev_ebitda': (-50, 200),
        'profit_margin': (-5, 5),  # -500% to 500%
        'return_on_equity': (-5, 5),
        'return_on_assets': (-5, 5),
        'debt_to_equity': (0, 50),
        'current_ratio': (0, 20),
        'quick_ratio': (0, 20),

        # Growth metrics
        'revenue_growth': (-0.99, 10),  # -99% to 1000%
        'earnings_growth': (-0.99, 10),
        'eps_growth': (-0.99, 10),

        # Market data
        'volume': (0, 1e12),  # Max 1 trillion shares
        'market_cap': (0, 1e13),  # Max 10 trillion USD
        'price': (0.01, 100000),  # Min $0.01, max $100k
        'close': (0.01, 100000),
        'open': (0.01, 100000),
        'high': (0.01, 100000),
        'low': (0.01, 100000),

        # Returns (daily)
        'return_1d': (-0.5, 0.5),  # ±50% daily max
        'return_5d': (-0.7, 0.7),  # ±70% weekly max
        'return_20d': (-0.9, 2.0),  # -90% to 200% monthly

        # Volatility
        'volatility_10d': (0, 3),  # 0% to 300% annualized
        'volatility_20d': (0, 3),
        'volatility_60d': (0, 3),
        'implied_volatility_30d': (0, 5),  # 0% to 500%
        'iv_percentile_1y': (0, 100),

        # Sentiment
        'sentiment_score': (-1, 1),
        'sentiment_ma_7d': (-1, 1),
        'sentiment_ma_30d': (-1, 1),

        # Options
        'put_call_ratio': (0, 10),
        'open_interest': (0, 1e9),

        # Analyst ratings
        'rating_numeric': (1, 5),
        'upside_to_target': (-0.9, 5),  # -90% to 500%

        # Insider trading
        'insider_buy_count': (0, 1000),
        'insider_sell_count': (0, 1000),
        'insider_net_sentiment': (-1000, 1000),
        'total_shares_traded': (0, 1e10),

        # Economic indicators
        'vix': (5, 100),
        'treasury_10y': (0, 20),  # 0% to 20%
        'treasury_2y': (0, 20),
        'yield_curve_10y_2y': (-5, 5),  # -5% to 5%
        'gdp_growth': (-0.5, 0.5),  # -50% to 50% (quarterly)
        'inflation_rate': (-0.2, 0.5),  # -20% to 50% (annual)
        'unemployment_rate': (0, 50),  # 0% to 50%

        # Beta and correlation
        'beta': (-5, 5),
        'correlation': (-1, 1),
    }

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, log_outliers: bool = True) -> pd.DataFrame:
        """
        Validate and clean DataFrame by capping outliers

        Args:
            df: DataFrame to validate
            log_outliers: Whether to log outlier detection

        Returns:
            Cleaned DataFrame with outliers capped
        """
        if df.empty:
            return df

        df_clean = df.copy()
        total_outliers = 0

        for col, (min_val, max_val) in cls.VALID_RANGES.items():
            if col not in df_clean.columns:
                continue

            # Find outliers (before capping)
            outlier_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
            outlier_count = outlier_mask.sum()

            if outlier_count > 0:
                total_outliers += outlier_count

                if log_outliers:
                    # Log examples of outliers
                    outlier_values = df_clean.loc[outlier_mask, col].values
                    sample_values = outlier_values[:3]  # Show first 3
                    logger.warning(
                        f"Found {outlier_count} outliers in '{col}' "
                        f"(valid range: [{min_val}, {max_val}]). "
                        f"Examples: {sample_values}"
                    )

                # Cap values instead of dropping (preserves data)
                df_clean[col] = df_clean[col].clip(lower=min_val, upper=max_val)

        if total_outliers > 0:
            logger.info(f"Capped {total_outliers} total outliers across {len(df_clean.columns)} columns")

        return df_clean

    @classmethod
    def validate_column(cls, series: pd.Series, column_name: str) -> pd.Series:
        """
        Validate a single column/series

        Args:
            series: Series to validate
            column_name: Column name (used for lookup in VALID_RANGES)

        Returns:
            Cleaned series
        """
        if column_name not in cls.VALID_RANGES:
            logger.debug(f"No validation range defined for '{column_name}'")
            return series

        min_val, max_val = cls.VALID_RANGES[column_name]

        # Find outliers
        outlier_mask = (series < min_val) | (series > max_val)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            logger.warning(f"Capping {outlier_count} outliers in '{column_name}'")
            return series.clip(lower=min_val, upper=max_val)

        return series

    @classmethod
    def add_validation_range(cls, column_name: str, min_val: float, max_val: float):
        """
        Add a custom validation range for a column

        Args:
            column_name: Column name
            min_val: Minimum valid value
            max_val: Maximum valid value
        """
        cls.VALID_RANGES[column_name] = (min_val, max_val)
        logger.info(f"Added validation range for '{column_name}': [{min_val}, {max_val}]")


class MarketCalendar:
    """
    Market calendar for NYSE trading days

    Prevents filling data on weekends/holidays when markets are closed.
    Only fills missing data on valid trading days.
    """

    # NYSE holidays (major U.S. market holidays)
    # Update annually or fetch from API
    NYSE_HOLIDAYS_2024 = [
        '2024-01-01',  # New Year's Day
        '2024-01-15',  # Martin Luther King Jr. Day
        '2024-02-19',  # Presidents' Day
        '2024-03-29',  # Good Friday
        '2024-05-27',  # Memorial Day
        '2024-06-19',  # Juneteenth
        '2024-07-04',  # Independence Day
        '2024-09-02',  # Labor Day
        '2024-11-28',  # Thanksgiving
        '2024-12-25',  # Christmas
    ]

    NYSE_HOLIDAYS_2025 = [
        '2025-01-01',  # New Year's Day
        '2025-01-20',  # Martin Luther King Jr. Day
        '2025-02-17',  # Presidents' Day
        '2025-04-18',  # Good Friday
        '2025-05-26',  # Memorial Day
        '2025-06-19',  # Juneteenth
        '2025-07-04',  # Independence Day
        '2025-09-01',  # Labor Day
        '2025-11-27',  # Thanksgiving
        '2025-12-25',  # Christmas
    ]

    @classmethod
    def get_nyse_holidays(cls, year: int = None) -> List[str]:
        """
        Get NYSE holidays for a specific year

        Args:
            year: Year (defaults to current year)

        Returns:
            List of holiday dates in 'YYYY-MM-DD' format
        """
        if year is None:
            year = datetime.now().year

        if year == 2024:
            return cls.NYSE_HOLIDAYS_2024
        elif year == 2025:
            return cls.NYSE_HOLIDAYS_2025
        else:
            # For other years, return common holidays (approximate)
            logger.warning(f"Using approximate holidays for year {year}")
            return []

    @classmethod
    def is_trading_day(cls, date: datetime) -> bool:
        """
        Check if a date is a valid NYSE trading day

        Args:
            date: Date to check

        Returns:
            True if trading day, False if weekend/holiday
        """
        # Check if weekend (Saturday=5, Sunday=6)
        if date.weekday() >= 5:
            return False

        # Check if holiday
        date_str = date.strftime('%Y-%m-%d')
        holidays = cls.get_nyse_holidays(date.year)

        if date_str in holidays:
            return False

        return True

    @classmethod
    def get_trading_days(cls, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get all trading days in a date range

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of trading days
        """
        trading_days = []
        current_date = start_date

        while current_date <= end_date:
            if cls.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    @classmethod
    def get_previous_trading_day(cls, date: datetime) -> datetime:
        """
        Get the previous trading day before a given date

        Args:
            date: Reference date

        Returns:
            Previous trading day
        """
        prev_date = date - timedelta(days=1)

        while not cls.is_trading_day(prev_date):
            prev_date -= timedelta(days=1)

        return prev_date

    @classmethod
    def get_next_trading_day(cls, date: datetime) -> datetime:
        """
        Get the next trading day after a given date

        Args:
            date: Reference date

        Returns:
            Next trading day
        """
        next_date = date + timedelta(days=1)

        while not cls.is_trading_day(next_date):
            next_date += timedelta(days=1)

        return next_date

    @classmethod
    def fill_missing_trading_days(cls, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Fill missing data ONLY for trading days (not weekends/holidays)

        This prevents artificial filling on non-trading days.

        Args:
            df: DataFrame with date column
            date_col: Name of date column

        Returns:
            DataFrame with filled trading days
        """
        if df.empty or date_col not in df.columns:
            return df

        # Convert to datetime if needed
        df[date_col] = pd.to_datetime(df[date_col])

        # Get date range
        start_date = df[date_col].min()
        end_date = df[date_col].max()

        # Get all trading days in range
        trading_days = cls.get_trading_days(start_date, end_date)

        # Create complete index of trading days
        trading_days_df = pd.DataFrame({date_col: trading_days})

        # Merge to identify missing trading days
        df_filled = trading_days_df.merge(df, on=date_col, how='left')

        missing_count = df_filled.isna().any(axis=1).sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing trading days to fill")

        return df_filled


class DatabaseWriter:
    """
    Database writer with retry logic for handling write conflicts

    Prevents "database is locked" errors with exponential backoff.
    """

    def __init__(self, db_path: str, max_retries: int = 5):
        """
        Initialize database writer

        Args:
            db_path: Path to SQLite database
            max_retries: Maximum retry attempts for locked database
        """
        self.db_path = db_path
        self.max_retries = max_retries

    def write_with_retry(self, df: pd.DataFrame, table_name: str,
                        if_exists: str = 'append') -> bool:
        """
        Write DataFrame to database with retry logic

        Args:
            df: DataFrame to write
            table_name: Target table name
            if_exists: 'append', 'replace', or 'fail'

        Returns:
            True if successful, False otherwise
        """
        import sqlite3

        for attempt in range(self.max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                conn.close()
                logger.debug(f"Wrote {len(df)} rows to {table_name}")
                return True

            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff: 0.5s, 1s, 2s, 4s, 8s
                    logger.warning(
                        f"Database locked (attempt {attempt+1}/{self.max_retries}). "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Database error: {e}")
                    return False

            except Exception as e:
                logger.error(f"Unexpected error writing to {table_name}: {e}")
                return False

        logger.error(f"Failed to write to {table_name} after {self.max_retries} attempts")
        return False

    def execute_with_retry(self, query: str, params: tuple = None) -> bool:
        """
        Execute SQL query with retry logic

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            True if successful, False otherwise
        """
        import sqlite3

        for attempt in range(self.max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                cursor = conn.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                conn.commit()
                conn.close()
                return True

            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    wait_time = (2 ** attempt) * 0.5
                    logger.warning(f"Database locked. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Database error: {e}")
                    return False

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return False

        logger.error(f"Failed to execute query after {self.max_retries} attempts")
        return False


class DataFreshness:
    """
    Track data freshness and staleness in metadata table

    Helps identify when data hasn't been updated recently.
    """

    def __init__(self, db_path: str):
        """
        Initialize data freshness tracker

        Args:
            db_path: Path to database
        """
        self.db_path = db_path
        self.db_writer = DatabaseWriter(db_path)
        self._ensure_metadata_table()

    def _ensure_metadata_table(self):
        """Create data_metadata table if it doesn't exist"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_metadata (
                table_name TEXT PRIMARY KEY,
                last_updated TIMESTAMP,
                row_count INTEGER,
                date_range_start DATE,
                date_range_end DATE,
                update_status TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def update_metadata(self, table_name: str, row_count: int = None,
                       date_range: Tuple[datetime, datetime] = None,
                       status: str = 'success', error_msg: str = None):
        """
        Update metadata for a table

        Args:
            table_name: Name of table
            row_count: Number of rows in table
            date_range: Tuple of (start_date, end_date)
            status: Update status ('success', 'partial', 'failed')
            error_msg: Error message if failed
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        # Get actual row count if not provided
        if row_count is None:
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                row_count = result[0] if result else 0
            except:
                row_count = 0

        # Get actual date range if not provided
        start_date_str = None
        end_date_str = None
        if date_range:
            start_date_str = date_range[0].strftime('%Y-%m-%d')
            end_date_str = date_range[1].strftime('%Y-%m-%d')

        # Update metadata
        conn.execute("""
            INSERT OR REPLACE INTO data_metadata
            (table_name, last_updated, row_count, date_range_start, date_range_end, update_status, error_message)
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?)
        """, (table_name, row_count, start_date_str, end_date_str, status, error_msg))

        conn.commit()
        conn.close()

        logger.info(f"Updated metadata for {table_name}: {row_count} rows, status={status}")

    def get_stale_tables(self, max_age_hours: int = 24) -> List[str]:
        """
        Get list of tables that haven't been updated recently

        Args:
            max_age_hours: Maximum age in hours before considered stale

        Returns:
            List of stale table names
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        query = f"""
            SELECT table_name, last_updated,
                   (julianday('now') - julianday(last_updated)) * 24 as hours_old
            FROM data_metadata
            WHERE hours_old > ?
            OR update_status != 'success'
            ORDER BY hours_old DESC
        """

        result = pd.read_sql(query, conn, params=(max_age_hours,))
        conn.close()

        if not result.empty:
            logger.warning(f"Found {len(result)} stale tables:")
            for _, row in result.iterrows():
                logger.warning(f"  - {row['table_name']}: {row['hours_old']:.1f} hours old")

        return result['table_name'].tolist()

    def get_metadata_summary(self) -> pd.DataFrame:
        """
        Get summary of all table metadata

        Returns:
            DataFrame with metadata for all tables
        """
        import sqlite3

        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                table_name,
                last_updated,
                row_count,
                date_range_start,
                date_range_end,
                update_status,
                (julianday('now') - julianday(last_updated)) * 24 as hours_since_update
            FROM data_metadata
            ORDER BY last_updated DESC
        """

        result = pd.read_sql(query, conn)
        conn.close()

        return result


class NewsFilterManager:
    """
    News article deduplication and relevance filtering

    Prevents duplicate articles and irrelevant news from polluting sentiment analysis.
    """

    @staticmethod
    def calculate_relevance(article: Dict, ticker: str, company_name: str = None) -> float:
        """
        Calculate relevance score for a news article

        Args:
            article: News article dictionary with 'title' and 'description'
            ticker: Stock ticker symbol
            company_name: Company name (optional, for better matching)

        Returns:
            Relevance score 0-1 (higher = more relevant)
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        content = article.get('content', '').lower()

        ticker_lower = ticker.lower()
        score = 0.0

        # Check ticker in title (strongest signal)
        if ticker_lower in title or f"${ticker_lower}" in title:
            score += 0.5

        # Check ticker in description
        if ticker_lower in description or f"${ticker_lower}" in description:
            score += 0.3

        # Check company name if provided
        if company_name:
            company_lower = company_name.lower()
            if company_lower in title:
                score += 0.4
            if company_lower in description:
                score += 0.2

        # Check for financial keywords (relevance indicators)
        financial_keywords = ['earnings', 'revenue', 'profit', 'stock', 'shares', 'trading',
                              'quarterly', 'analyst', 'upgrade', 'downgrade', 'buy', 'sell']
        keyword_count = sum(1 for kw in financial_keywords if kw in title or kw in description)
        score += min(keyword_count * 0.05, 0.2)  # Max 0.2 from keywords

        # Penalize generic market news
        generic_terms = ['market', 'dow', 'sp500', 's&p 500', 'nasdaq', 'index']
        if any(term in title for term in generic_terms) and ticker_lower not in title:
            score *= 0.5  # Reduce score for generic market news

        return min(score, 1.0)

    @staticmethod
    def filter_news_articles(articles: List[Dict], ticker: str, company_name: str = None,
                            min_relevance: float = 0.3) -> List[Dict]:
        """
        Remove duplicates and irrelevant articles

        Args:
            articles: List of news article dictionaries
            ticker: Stock ticker symbol
            company_name: Company name (optional)
            min_relevance: Minimum relevance score threshold

        Returns:
            Filtered list of unique, relevant articles
        """
        seen_titles = set()
        filtered = []

        for article in articles:
            # Skip if missing required fields
            if not article.get('title'):
                continue

            # Create hash of title for deduplication
            title_normalized = article['title'].lower().strip()
            title_hash = hashlib.md5(title_normalized.encode()).hexdigest()

            # Skip duplicates
            if title_hash in seen_titles:
                logger.debug(f"Duplicate article: {article['title'][:50]}...")
                continue

            # Check relevance
            relevance = NewsFilterManager.calculate_relevance(article, ticker, company_name)
            if relevance < min_relevance:
                logger.debug(f"Low relevance ({relevance:.2f}): {article['title'][:50]}...")
                continue

            # Add to filtered list
            seen_titles.add(title_hash)
            article['relevance_score'] = relevance
            filtered.append(article)

        logger.info(f"Filtered {len(articles)} articles → {len(filtered)} unique, relevant articles")
        return filtered

    @staticmethod
    def deduplicate_by_content(articles: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Advanced deduplication using content similarity

        Args:
            articles: List of articles
            similarity_threshold: Jaccard similarity threshold for duplicates

        Returns:
            Deduplicated list
        """
        if len(articles) <= 1:
            return articles

        unique_articles = []
        seen_content = []

        for article in articles:
            content = article.get('description', '') or article.get('title', '')
            if not content:
                continue

            # Tokenize content
            tokens = set(content.lower().split())

            # Check similarity with existing articles
            is_duplicate = False
            for seen_tokens in seen_content:
                # Jaccard similarity
                intersection = len(tokens & seen_tokens)
                union = len(tokens | seen_tokens)
                similarity = intersection / union if union > 0 else 0

                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Content duplicate ({similarity:.2f}): {article.get('title', '')[:50]}...")
                    break

            if not is_duplicate:
                unique_articles.append(article)
                seen_content.append(tokens)

        logger.info(f"Content deduplication: {len(articles)} → {len(unique_articles)}")
        return unique_articles


# ============================================================================
# Utility Functions
# ============================================================================

def decay_sentiment(df: pd.DataFrame, sentiment_col: str = 'sentiment_score',
                   date_col: str = 'date', half_life_days: int = 7) -> pd.DataFrame:
    """
    Apply exponential decay to sentiment scores based on age

    Old sentiment data loses relevance over time. A 30-day old news article
    should have less impact than yesterday's news.

    Formula: decayed_sentiment = sentiment * e^(-λ * days_old)
    where λ = ln(2) / half_life

    Args:
        df: DataFrame with sentiment scores and dates
        sentiment_col: Name of sentiment column
        date_col: Name of date column
        half_life_days: Days for sentiment to decay to 50% (default 7 days)

    Returns:
        DataFrame with decayed_sentiment column added
    """
    if df.empty or sentiment_col not in df.columns or date_col not in df.columns:
        logger.warning(f"Cannot apply sentiment decay: missing columns")
        return df

    df = df.copy()

    # Convert to datetime if needed
    df[date_col] = pd.to_datetime(df[date_col])

    # Calculate days since most recent date
    most_recent_date = df[date_col].max()
    df['days_old'] = (most_recent_date - df[date_col]).dt.days

    # Exponential decay: λ = ln(2) / half_life
    decay_lambda = np.log(2) / half_life_days
    df['decay_factor'] = np.exp(-decay_lambda * df['days_old'])

    # Apply decay
    df['decayed_sentiment'] = df[sentiment_col] * df['decay_factor']

    logger.info(
        f"Applied sentiment decay (half-life={half_life_days} days). "
        f"Avg decay factor: {df['decay_factor'].mean():.2f}"
    )

    return df


def calculate_synthetic_iv(historical_returns: pd.Series, window: int = 20) -> float:
    """
    Calculate synthetic implied volatility from historical volatility

    Used as fallback when options data is unavailable.

    Args:
        historical_returns: Series of historical returns
        window: Lookback window for volatility calculation

    Returns:
        Annualized implied volatility (as decimal, e.g., 0.25 = 25%)
    """
    if len(historical_returns) < window:
        logger.warning(f"Insufficient data for IV calculation: {len(historical_returns)} < {window}")
        return 0.20  # Default 20% IV

    # Calculate historical volatility (annualized)
    recent_returns = historical_returns.tail(window)
    std_dev = recent_returns.std()
    annualized_vol = std_dev * np.sqrt(252)  # 252 trading days

    # Adjust for options market premium (IV typically 10-20% higher than HV)
    synthetic_iv = annualized_vol * 1.15  # 15% premium

    return float(synthetic_iv)


def get_options_data_with_fallback(symbol: str, db_path: str,
                                   max_attempts: int = 3) -> Dict:
    """
    Get options data with fallback to synthetic IV

    Attempts to fetch real options data, falls back to calculated IV if unavailable.

    Args:
        symbol: Stock ticker
        db_path: Path to database
        max_attempts: Maximum API retry attempts

    Returns:
        Dictionary with options metrics (implied_volatility_30d, put_call_ratio, etc.)
    """
    import sqlite3

    conn = sqlite3.connect(db_path)

    # Try to get real options data
    try:
        query = """
            SELECT implied_volatility_30d, iv_percentile_1y, put_call_ratio
            FROM options_data
            WHERE symbol_ticker = ?
            ORDER BY options_date DESC
            LIMIT 1
        """
        result = pd.read_sql(query, conn, params=(symbol,))

        if not result.empty:
            logger.debug(f"Found real options data for {symbol}")
            conn.close()
            return {
                'implied_volatility_30d': result['implied_volatility_30d'].iloc[0],
                'iv_percentile_1y': result['iv_percentile_1y'].iloc[0],
                'put_call_ratio': result['put_call_ratio'].iloc[0],
                'data_source': 'real'
            }
    except Exception as e:
        logger.debug(f"Could not fetch options data for {symbol}: {e}")

    # Fallback: Calculate synthetic IV from historical prices
    try:
        query = """
            SELECT close, price_date
            FROM raw_price_data
            WHERE symbol_ticker = ?
            ORDER BY price_date DESC
            LIMIT 60
        """
        prices = pd.read_sql(query, conn, params=(symbol,))
        conn.close()

        if len(prices) >= 20:
            # Calculate returns
            prices = prices.sort_values('price_date')
            returns = prices['close'].pct_change().dropna()

            # Calculate synthetic IV
            synthetic_iv = calculate_synthetic_iv(returns, window=20)

            logger.info(f"Using synthetic IV for {symbol}: {synthetic_iv:.1%}")

            return {
                'implied_volatility_30d': synthetic_iv,
                'iv_percentile_1y': 50.0,  # Neutral percentile
                'put_call_ratio': 1.0,  # Neutral P/C ratio
                'data_source': 'synthetic'
            }
    except Exception as e:
        logger.warning(f"Could not calculate synthetic IV for {symbol}: {e}")
        conn.close()

    # Final fallback: Default values
    logger.warning(f"Using default IV for {symbol}")
    return {
        'implied_volatility_30d': 0.25,  # Market average
        'iv_percentile_1y': 50.0,
        'put_call_ratio': 1.0,
        'data_source': 'default'
    }


# Singleton instances for reuse
_rate_limit_manager = None
_data_quality_manager = None
_data_validator = None
_market_calendar = None
_news_filter_manager = None


def get_rate_limit_manager() -> RateLimitManager:
    """Get singleton RateLimitManager instance"""
    global _rate_limit_manager
    if _rate_limit_manager is None:
        _rate_limit_manager = RateLimitManager()
    return _rate_limit_manager


def get_data_quality_manager() -> DataQualityManager:
    """Get singleton DataQualityManager instance"""
    global _data_quality_manager
    if _data_quality_manager is None:
        _data_quality_manager = DataQualityManager()
    return _data_quality_manager


def get_data_validator() -> DataValidator:
    """Get singleton DataValidator instance"""
    global _data_validator
    if _data_validator is None:
        _data_validator = DataValidator()
    return _data_validator


def get_market_calendar() -> MarketCalendar:
    """Get singleton MarketCalendar instance"""
    global _market_calendar
    if _market_calendar is None:
        _market_calendar = MarketCalendar()
    return _market_calendar


def get_news_filter_manager() -> NewsFilterManager:
    """Get singleton NewsFilterManager instance"""
    global _news_filter_manager
    if _news_filter_manager is None:
        _news_filter_manager = NewsFilterManager()
    return _news_filter_manager


if __name__ == "__main__":
    # Test data infrastructure
    logging.basicConfig(level=logging.INFO)

    print("=== DataConfig Test ===")
    start, end = DataConfig.get_date_range('historical')
    print(f"Historical range: {start.date()} to {end.date()} ({DataConfig.HISTORICAL_DAYS} days)")

    print("\n=== RateLimitManager Test ===")
    rlm = get_rate_limit_manager()
    print("Testing Yahoo Finance rate limit (10 calls)...")
    for i in range(10):
        rlm.wait_if_needed('yfinance')
        print(f"  Call {i+1} completed")

    print("\n=== DataQualityManager Test ===")
    dqm = get_data_quality_manager()
    test_df = pd.DataFrame({
        'price': [100, 101, np.nan, 103, 104],
        'volume': [1000, np.nan, np.nan, 1200, 1300],
        'sentiment': [0.5, np.nan, np.nan, np.nan, 0.7]
    })
    print("Test DataFrame:")
    print(test_df)
    print("\nAfter smart fill:")
    test_df['price_filled'] = dqm.smart_fill(test_df, 'price', method='ffill')
    test_df['volume_filled'] = dqm.smart_fill(test_df, 'volume', method='zero')
    test_df['sentiment_filled'] = dqm.smart_fill(test_df, 'sentiment', method='ffill')
    print(test_df)

    print("\n=== NewsFilterManager Test ===")
    nfm = get_news_filter_manager()
    test_articles = [
        {'title': 'AAPL reports strong earnings', 'description': 'Apple Inc quarterly earnings beat expectations'},
        {'title': 'Apple reports strong earnings', 'description': 'AAPL quarterly earnings beat expectations'},  # Duplicate
        {'title': 'Market rallies on Fed news', 'description': 'Dow Jones and S&P 500 rally'},  # Irrelevant
        {'title': 'AAPL stock upgraded to buy', 'description': 'Analyst upgrades Apple to strong buy'},
    ]
    filtered = nfm.filter_news_articles(test_articles, 'AAPL', 'Apple Inc', min_relevance=0.3)
    print(f"Filtered {len(test_articles)} → {len(filtered)} articles:")
    for article in filtered:
        print(f"  - {article['title']} (relevance: {article['relevance_score']:.2f})")
