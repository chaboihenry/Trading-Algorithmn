"""
ML Features Aggregator (ENHANCED)
==================================

CRITICAL: Aggregates all features into ml_features table
This is the missing link between raw data and trading signals

Features Aggregated:
- Price returns (daily, 5-day, 20-day)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Sentiment scores and sentiment-price divergence
- Volatility regime indicators
- Lag features (t-1, t-5, t-20)
- Normalized features using StandardScaler

ENHANCEMENTS:
- Intelligent data filling (no blind forward-fill)
- Consistent lookback windows via DataConfig
- Data quality validation

Output: ml_features table ready for signal generation
"""

import sqlite3
import pandas as pd
import numpy as np
# datetime, timedelta not used - only pd.to_datetime() is used
from sklearn.preprocessing import StandardScaler
import logging
import sys
from pathlib import Path

# Add data_collectors to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'data_collectors'))
from data_infrastructure import DataConfig, get_data_quality_manager, get_data_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLFeaturesAggregator:
    """Aggregate all features into unified ML features table"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """Initialize aggregator with data quality manager and validator"""
        self.db_path = db_path
        self.data_quality = get_data_quality_manager()
        self.data_validator = get_data_validator()
        logger.info(f"Initialized MLFeaturesAggregator (ENHANCED)")
        logger.info(f"Database: {db_path}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def _create_ml_features_table(self) -> None:
        """Create ml_features table if not exists"""
        conn = self._get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_features (
                feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_ticker TEXT NOT NULL,
                feature_date DATE NOT NULL,

                -- Price returns
                return_1d REAL,
                return_5d REAL,
                return_20d REAL,

                -- Technical indicators
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                macd_histogram REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                bb_width REAL,
                bb_position REAL,

                -- Volatility features
                volatility_10d REAL,
                volatility_20d REAL,
                volatility_30d REAL,
                atr_14 REAL,
                volatility_regime TEXT,

                -- Sentiment features
                sentiment_score REAL,
                sentiment_ma_7d REAL,
                sentiment_price_divergence REAL,

                -- Lag features
                return_1d_lag1 REAL,
                return_1d_lag5 REAL,
                return_1d_lag20 REAL,
                rsi_14_lag1 REAL,
                volatility_20d_lag1 REAL,

                -- Normalized features (for ML)
                return_1d_norm REAL,
                rsi_14_norm REAL,
                macd_norm REAL,
                volatility_20d_norm REAL,
                sentiment_score_norm REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol_ticker, feature_date)
            )
        """)

        conn.commit()
        conn.close()
        logger.info("ml_features table created/verified")

    def _get_price_returns(self, ticker: str, lookback_days: int = 100) -> pd.DataFrame:
        """Calculate price returns for a ticker"""
        conn = self._get_db_connection()

        query = """
            SELECT price_date, close
            FROM raw_price_data
            WHERE symbol_ticker = ?
            ORDER BY price_date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(ticker, lookback_days))
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values('price_date')
        df['return_1d'] = df['close'].pct_change()
        df['return_5d'] = df['close'].pct_change(periods=5)
        df['return_20d'] = df['close'].pct_change(periods=20)

        return df[['price_date', 'return_1d', 'return_5d', 'return_20d']]

    def _get_technical_indicators(self, ticker: str, lookback_days: int = 100) -> pd.DataFrame:
        """Get technical indicators for a ticker"""
        conn = self._get_db_connection()

        query = """
            SELECT
                indicator_date,
                rsi_14,
                macd,
                macd_signal,
                macd_histogram,
                bb_upper,
                bb_middle,
                bb_lower,
                bb_width,
                atr_14
            FROM technical_indicators
            WHERE symbol_ticker = ?
            ORDER BY indicator_date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(ticker, lookback_days))
        conn.close()

        if df.empty:
            return pd.DataFrame()

        # Calculate Bollinger Band position (where price is relative to bands)
        # 1.0 = at upper band, 0.0 = at lower band, 0.5 = at middle
        df['bb_position'] = (df['bb_middle'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def _get_volatility_metrics(self, ticker: str, lookback_days: int = 100) -> pd.DataFrame:
        """Get volatility metrics for a ticker"""
        conn = self._get_db_connection()

        query = """
            SELECT
                vol_date,
                close_to_close_vol_10d,
                close_to_close_vol_20d,
                close_to_close_vol_60d,
                yang_zhang_vol_10d,
                yang_zhang_vol_20d
            FROM volatility_metrics
            WHERE symbol_ticker = ?
            ORDER BY vol_date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(ticker, lookback_days))
        conn.close()

        if df.empty:
            return pd.DataFrame()

        # Rename columns to match expected names
        df.rename(columns={
            'vol_date': 'metric_date',
            'close_to_close_vol_10d': 'volatility_10d',
            'close_to_close_vol_20d': 'volatility_20d',
            'close_to_close_vol_60d': 'volatility_30d'
        }, inplace=True)

        # Determine volatility regime
        # High volatility: > 75th percentile
        # Low volatility: < 25th percentile
        # Normal: between
        vol_75 = df['volatility_20d'].quantile(0.75)
        vol_25 = df['volatility_20d'].quantile(0.25)

        df['volatility_regime'] = df['volatility_20d'].apply(
            lambda x: 'high' if x > vol_75 else ('low' if x < vol_25 else 'normal')
        )

        return df

    def _get_sentiment_data(self, ticker: str, lookback_days: int = 100) -> pd.DataFrame:
        """Get sentiment data for a ticker"""
        conn = self._get_db_connection()

        query = """
            SELECT
                sentiment_date,
                sentiment_score
            FROM sentiment_data
            WHERE symbol_ticker = ?
            ORDER BY sentiment_date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(ticker, lookback_days))
        conn.close()

        if df.empty:
            return pd.DataFrame()

        df = df.sort_values('sentiment_date')

        # Calculate 7-day moving average of sentiment
        df['sentiment_ma_7d'] = df['sentiment_score'].rolling(window=7, min_periods=1).mean()

        return df

    def _calculate_sentiment_price_divergence(self, ticker: str,
                                              returns_df: pd.DataFrame,
                                              sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate divergence between sentiment and price returns"""
        if returns_df.empty or sentiment_df.empty:
            return pd.DataFrame()

        # Merge on date
        returns_df['date'] = pd.to_datetime(returns_df['price_date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['sentiment_date'])

        merged = pd.merge(
            returns_df[['date', 'return_5d']],
            sentiment_df[['date', 'sentiment_score', 'sentiment_ma_7d']],
            on='date',
            how='inner'
        )

        if merged.empty:
            return pd.DataFrame()

        # Normalize both to [-1, 1] range for comparison
        merged['return_5d_norm'] = merged['return_5d'] / (merged['return_5d'].abs().max() + 1e-8)
        merged['sentiment_norm'] = merged['sentiment_score'] / (merged['sentiment_score'].abs().max() + 1e-8)

        # Divergence = sentiment - price returns
        # Positive divergence: sentiment bullish but price down (potential buy)
        # Negative divergence: sentiment bearish but price up (potential sell)
        merged['sentiment_price_divergence'] = merged['sentiment_norm'] - merged['return_5d_norm']

        return merged[['date', 'sentiment_score', 'sentiment_ma_7d', 'sentiment_price_divergence']]

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series prediction"""
        if df.empty:
            return df

        # Create lag features
        df['return_1d_lag1'] = df['return_1d'].shift(1)
        df['return_1d_lag5'] = df['return_1d'].shift(5)
        df['return_1d_lag20'] = df['return_1d'].shift(20)
        df['rsi_14_lag1'] = df['rsi_14'].shift(1)
        df['volatility_20d_lag1'] = df['volatility_20d'].shift(1)

        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using StandardScaler"""
        if df.empty:
            return df

        features_to_normalize = [
            'return_1d', 'rsi_14', 'macd', 'volatility_20d', 'sentiment_score'
        ]

        for feature in features_to_normalize:
            if feature in df.columns:
                scaler = StandardScaler()
                valid_data = df[feature].dropna()

                if len(valid_data) > 0:
                    df[f'{feature}_norm'] = scaler.fit_transform(df[[feature]])
                else:
                    df[f'{feature}_norm'] = np.nan

        return df

    def aggregate_features_for_ticker(self, ticker: str) -> int:
        """
        Aggregate all features for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Number of records created
        """
        logger.info(f"Aggregating features for {ticker}")

        # Get all data sources
        returns_df = self._get_price_returns(ticker, lookback_days=100)
        technical_df = self._get_technical_indicators(ticker, lookback_days=100)
        volatility_df = self._get_volatility_metrics(ticker, lookback_days=100)
        sentiment_df = self._get_sentiment_data(ticker, lookback_days=100)

        if returns_df.empty:
            logger.warning(f"No price data for {ticker}, skipping")
            return 0

        # Start with returns as base
        returns_df['date'] = pd.to_datetime(returns_df['price_date'])
        merged_df = returns_df[['date', 'return_1d', 'return_5d', 'return_20d']].copy()

        # Merge technical indicators
        if not technical_df.empty:
            technical_df['date'] = pd.to_datetime(technical_df['indicator_date'])
            merged_df = pd.merge(merged_df, technical_df, on='date', how='left')

        # Merge volatility metrics
        if not volatility_df.empty:
            volatility_df['date'] = pd.to_datetime(volatility_df['metric_date'])
            merged_df = pd.merge(merged_df, volatility_df, on='date', how='left')

        # Calculate and merge sentiment divergence
        if not sentiment_df.empty:
            divergence_df = self._calculate_sentiment_price_divergence(
                ticker, returns_df, sentiment_df
            )
            if not divergence_df.empty:
                merged_df = pd.merge(merged_df, divergence_df, on='date', how='left')

        # Intelligent data filling (prevents artificial patterns)
        # Apply column-specific filling strategies
        for col in merged_df.columns:
            if col == 'date':
                continue  # Skip date column

            # Determine fill strategy based on column type
            if col == 'sentiment_score' or col.startswith('sentiment_'):
                # Sentiment data: max 7 days forward-fill
                merged_df[col] = self.data_quality.smart_fill(merged_df, col, method='ffill', limit=7)
            elif col in ['rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'bb_position']:
                # Technical indicators: max 2 days interpolation
                merged_df[col] = self.data_quality.smart_fill(merged_df, col, method='interpolate', limit=2)
            elif col in ['volume']:
                # Volume: never forward-fill (use zero)
                merged_df[col] = self.data_quality.smart_fill(merged_df, col, method='zero', limit=0)
            elif col.startswith('return_'):
                # Returns: never forward-fill (interpolate very small gaps only)
                merged_df[col] = self.data_quality.smart_fill(merged_df, col, method='interpolate', limit=1)
            else:
                # Default: conservative 3-day forward-fill
                merged_df[col] = self.data_quality.smart_fill(merged_df, col, method='ffill', limit=3)

        # Create lag features
        merged_df = self._create_lag_features(merged_df)

        # Normalize features
        merged_df = self._normalize_features(merged_df)

        # Validate data and cap outliers (prevents garbage data from breaking models)
        merged_df = self.data_validator.validate_dataframe(merged_df, log_outliers=True)

        # Drop rows with NaN in critical features
        merged_df = merged_df.dropna(subset=['return_1d'])

        if merged_df.empty:
            logger.warning(f"No valid features after processing for {ticker}")
            return 0

        # Insert into database
        conn = self._get_db_connection()
        cursor = conn.cursor()

        records_inserted = 0
        for _, row in merged_df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO ml_features (
                        symbol_ticker, feature_date,
                        return_1d, return_5d, return_20d,
                        rsi_14, macd, macd_signal, macd_histogram,
                        bb_upper, bb_middle, bb_lower, bb_width, bb_position,
                        volatility_10d, volatility_20d, volatility_30d, atr_14,
                        volatility_regime,
                        sentiment_score, sentiment_ma_7d, sentiment_price_divergence,
                        return_1d_lag1, return_1d_lag5, return_1d_lag20,
                        rsi_14_lag1, volatility_20d_lag1,
                        return_1d_norm, rsi_14_norm, macd_norm,
                        volatility_20d_norm, sentiment_score_norm
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                             ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker, row['date'].strftime('%Y-%m-%d'),
                    row.get('return_1d'), row.get('return_5d'), row.get('return_20d'),
                    row.get('rsi_14'), row.get('macd'), row.get('macd_signal'), row.get('macd_histogram'),
                    row.get('bb_upper'), row.get('bb_middle'), row.get('bb_lower'),
                    row.get('bb_width'), row.get('bb_position'),
                    row.get('volatility_10d'), row.get('volatility_20d'), row.get('volatility_30d'),
                    row.get('atr_14'), row.get('volatility_regime'),
                    row.get('sentiment_score'), row.get('sentiment_ma_7d'),
                    row.get('sentiment_price_divergence'),
                    row.get('return_1d_lag1'), row.get('return_1d_lag5'), row.get('return_1d_lag20'),
                    row.get('rsi_14_lag1'), row.get('volatility_20d_lag1'),
                    row.get('return_1d_norm'), row.get('rsi_14_norm'), row.get('macd_norm'),
                    row.get('volatility_20d_norm'), row.get('sentiment_score_norm')
                ))
                records_inserted += 1
            except Exception as e:
                logger.error(f"Error inserting record for {ticker} on {row['date']}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"✓ Inserted {records_inserted} feature records for {ticker}")
        return records_inserted

    def run(self) -> None:
        """Run feature aggregation for all tickers"""
        logger.info("="*60)
        logger.info("ML FEATURES AGGREGATION STARTED")
        logger.info("="*60)

        # Create table
        self._create_ml_features_table()

        # Get all tickers
        conn = self._get_db_connection()
        tickers_df = pd.read_sql("SELECT DISTINCT symbol_ticker FROM assets ORDER BY symbol_ticker", conn)
        conn.close()

        total_tickers = len(tickers_df)
        logger.info(f"Processing {total_tickers} tickers")

        total_records = 0
        successful_tickers = 0
        failed_tickers = 0

        for idx, row in tickers_df.iterrows():
            ticker = row['symbol_ticker']
            try:
                records = self.aggregate_features_for_ticker(ticker)
                total_records += records
                if records > 0:
                    successful_tickers += 1
                else:
                    failed_tickers += 1
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                failed_tickers += 1

        logger.info("="*60)
        logger.info("ML FEATURES AGGREGATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Total tickers processed: {total_tickers}")
        logger.info(f"Successful: {successful_tickers}")
        logger.info(f"Failed: {failed_tickers}")
        logger.info(f"Total feature records: {total_records}")
        logger.info("="*60)


if __name__ == "__main__":
    aggregator = MLFeaturesAggregator()
    aggregator.run()