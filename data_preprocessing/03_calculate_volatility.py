import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VolatilityCalculator:
    """
    Calculates volatility metrics for all assets and populates volatility_metrics table

    Database: /Volumes/Vault/85_assets_prediction.db
    Source Table: raw_price_data
    Target Table: volatility_metrics

    Volatility Measures Calculated:
        - Close-to-Close (10d, 20d, 60d)
        - Parkinson (10d, 20d) - uses high/low
        - Garman-Klass (10d, 20d) - uses OHLC
        - Rogers-Satchell (10d, 20d) - drift-independent
        - Yang-Zhang (10d, 20d) - combines overnight and intraday
        - Realized volatility percentiles
        - Volatility of volatility
        - Volume-weighted volatility
        - Gap analysis and overnight/intraday ratios
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the calculator with database path"""
        self.db_path = db_path

        # Annualization factor (252 trading days)
        self.annualization_factor = np.sqrt(252)

        logger.info(f"Initialized VolatilityCalculator")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_all_tickers(self) -> List[str]:
        """Get all tickers from assets table"""
        try:
            conn = self._get_db_connection()
            query = "SELECT symbol_ticker FROM assets ORDER BY symbol_ticker"
            df = pd.read_sql(query, conn)
            conn.close()
            tickers = df['symbol_ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} tickers from assets table")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving tickers: {str(e)}")
            raise

    def _load_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Load OHLCV price data for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with price data
        """
        try:
            conn = self._get_db_connection()
            query = """
                SELECT price_date, open, high, low, close, volume
                FROM raw_price_data
                WHERE symbol_ticker = ?
                ORDER BY price_date
            """
            df = pd.read_sql(query, conn, params=(ticker,))
            conn.close()

            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()

            # Convert date
            df['price_date'] = pd.to_datetime(df['price_date'])

            # Calculate returns for various calculations
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            return df

        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_close_to_close_vol(self, returns: pd.Series, window: int) -> pd.Series:
        """
        Calculate traditional close-to-close volatility (annualized)

        Args:
            returns: Return series
            window: Rolling window size

        Returns:
            Annualized volatility
        """
        return returns.rolling(window=window).std() * self.annualization_factor

    def _calculate_parkinson_vol(self, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """
        Calculate Parkinson volatility estimator
        More efficient than close-to-close, uses high-low range

        Formula: sqrt((1/(4*ln(2))) * (ln(H/L))^2)

        Args:
            high: High prices
            low: Low prices
            window: Rolling window size

        Returns:
            Annualized Parkinson volatility
        """
        hl_ratio = np.log(high / low)
        parkinson = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2))
        return parkinson.rolling(window=window).mean() * self.annualization_factor

    def _calculate_garman_klass_vol(self, open_p: pd.Series, high: pd.Series,
                                    low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """
        Calculate Garman-Klass volatility estimator
        Uses OHLC data for better estimate than close-to-close

        Args:
            open_p: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size

        Returns:
            Annualized Garman-Klass volatility
        """
        hl = np.log(high / low)
        co = np.log(close / open_p)

        gk = 0.5 * (hl ** 2) - (2 * np.log(2) - 1) * (co ** 2)
        return np.sqrt(gk.rolling(window=window).mean()) * self.annualization_factor

    def _calculate_rogers_satchell_vol(self, open_p: pd.Series, high: pd.Series,
                                       low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """
        Calculate Rogers-Satchell volatility estimator
        Drift-independent estimator

        Args:
            open_p: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size

        Returns:
            Annualized Rogers-Satchell volatility
        """
        rs = np.log(high / close) * np.log(high / open_p) + \
             np.log(low / close) * np.log(low / open_p)

        return np.sqrt(rs.rolling(window=window).mean()) * self.annualization_factor

    def _calculate_yang_zhang_vol(self, open_p: pd.Series, high: pd.Series,
                                  low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        """
        Calculate Yang-Zhang volatility estimator
        Combines overnight and intraday volatility, independent of drift

        Args:
            open_p: Open prices
            high: High prices
            low: Low prices
            close: Close prices
            window: Rolling window size

        Returns:
            Annualized Yang-Zhang volatility
        """
        # Overnight volatility
        overnight = np.log(open_p / close.shift(1))
        overnight_vol = overnight.rolling(window=window).var()

        # Open-to-close volatility
        open_close = np.log(close / open_p)
        oc_vol = open_close.rolling(window=window).var()

        # Rogers-Satchell component
        rs = np.log(high / close) * np.log(high / open_p) + \
             np.log(low / close) * np.log(low / open_p)
        rs_vol = rs.rolling(window=window).mean()

        # Combine components with weighting factor k
        k = 0.34 / (1.34 + (window + 1) / (window - 1))

        yang_zhang = overnight_vol + k * oc_vol + (1 - k) * rs_vol

        return np.sqrt(yang_zhang) * self.annualization_factor

    def _calculate_vol_percentile(self, volatility: pd.Series, window: int) -> pd.Series:
        """
        Calculate percentile rank of current volatility vs historical

        Args:
            volatility: Volatility series
            window: Lookback window for percentile calculation

        Returns:
            Percentile rank (0-100)
        """
        return volatility.rolling(window=window).apply(
            lambda x: (x <= x.iloc[-1]).sum() / len(x) * 100 if len(x) > 0 else np.nan,
            raw=False
        )

    def _calculate_vol_of_vol(self, volatility: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate volatility of volatility (vol changes)

        Args:
            volatility: Volatility series
            window: Rolling window

        Returns:
            Volatility of volatility
        """
        vol_returns = volatility.pct_change()
        return vol_returns.rolling(window=window).std()

    def _calculate_vol_clustering_index(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate volatility clustering index
        Measures autocorrelation in squared returns

        Args:
            returns: Return series
            window: Rolling window

        Returns:
            Clustering index
        """
        squared_returns = returns ** 2
        clustering = squared_returns.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan,
            raw=False
        )
        return clustering

    def _classify_volatility_trend(self, vol_20d: pd.Series, vol_60d: pd.Series) -> pd.Series:
        """
        Classify volatility trend

        Args:
            vol_20d: 20-day volatility
            vol_60d: 60-day volatility

        Returns:
            Trend classification: Increasing, Decreasing, Stable
        """
        ratio = vol_20d / vol_60d

        trend = pd.Series(index=ratio.index, dtype=str)
        trend[ratio > 1.1] = 'Increasing'
        trend[ratio < 0.9] = 'Decreasing'
        trend[(ratio >= 0.9) & (ratio <= 1.1)] = 'Stable'

        return trend

    def _calculate_vol_acceleration(self, vol_series: pd.Series) -> pd.Series:
        """
        Calculate rate of change in volatility

        Args:
            vol_series: Volatility time series

        Returns:
            Volatility acceleration (% change)
        """
        return vol_series.pct_change(periods=5)

    def _calculate_volume_weighted_vol(self, returns: pd.Series, volume: pd.Series,
                                       window: int = 20) -> pd.Series:
        """
        Calculate volume-weighted volatility

        Args:
            returns: Return series
            volume: Volume series
            window: Rolling window

        Returns:
            Volume-weighted volatility
        """
        volume_normalized = volume / volume.rolling(window=window).mean()
        weighted_returns = returns * volume_normalized
        return weighted_returns.rolling(window=window).std() * self.annualization_factor

    def _count_abnormal_volume(self, volume: pd.Series, window: int = 20,
                               threshold: float = 2.0) -> pd.Series:
        """
        Count days with abnormal volume

        Args:
            volume: Volume series
            window: Rolling window
            threshold: Std devs above mean to count as abnormal

        Returns:
            Count of abnormal volume days
        """
        vol_mean = volume.rolling(window=window).mean()
        vol_std = volume.rolling(window=window).std()

        abnormal = (volume > (vol_mean + threshold * vol_std)).astype(int)
        return abnormal.rolling(window=window).sum()

    def _calculate_gap_metrics(self, open_p: pd.Series, close: pd.Series,
                               window: int = 60) -> Dict[str, pd.Series]:
        """
        Calculate gap frequency and size

        Args:
            open_p: Open prices
            close: Close prices (previous day)
            window: Rolling window

        Returns:
            Dictionary with gap metrics
        """
        # Gap as percentage
        gap = (open_p - close.shift(1)) / close.shift(1)

        # Gap frequency (% of days with gap > 1%)
        has_gap = (np.abs(gap) > 0.01).astype(int)
        gap_freq = has_gap.rolling(window=window).mean()

        # Average gap size
        avg_gap_size = np.abs(gap).rolling(window=window).mean()

        # Gap contribution to volatility
        gap_vol = gap.rolling(window=window).std()
        total_vol = (close.pct_change()).rolling(window=window).std()
        gap_contribution = gap_vol / (total_vol + 1e-10)  # Avoid division by zero

        return {
            'frequency': gap_freq,
            'avg_size': avg_gap_size,
            'contribution': gap_contribution
        }

    def _calculate_overnight_intraday_ratios(self, open_p: pd.Series, close: pd.Series,
                                            window: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate overnight vs intraday volatility ratios

        Args:
            open_p: Open prices
            close: Close prices
            window: Rolling window

        Returns:
            Dictionary with overnight and intraday vol ratios
        """
        # Overnight returns
        overnight_ret = np.log(open_p / close.shift(1))
        overnight_vol = overnight_ret.rolling(window=window).std()

        # Intraday returns
        intraday_ret = np.log(close / open_p)
        intraday_vol = intraday_ret.rolling(window=window).std()

        # Total volatility
        total_vol = np.log(close / close.shift(1)).rolling(window=window).std()

        # Calculate ratios
        overnight_ratio = overnight_vol / (total_vol + 1e-10)
        intraday_ratio = intraday_vol / (total_vol + 1e-10)

        return {
            'overnight_ratio': overnight_ratio,
            'intraday_ratio': intraday_ratio
        }

    def calculate_volatility_metrics(self, ticker: str) -> pd.DataFrame:
        """
        Calculate all volatility metrics for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with all volatility metrics
        """
        logger.info(f"Calculating volatility metrics for {ticker}")

        # Load price data
        df = self._load_price_data(ticker)

        if df.empty:
            logger.warning(f"No data available for {ticker}")
            return pd.DataFrame()

        # Initialize result dataframe
        result = pd.DataFrame({
            'symbol_ticker': ticker,
            'vol_date': df['price_date']
        })

        # Close-to-close volatility
        result['close_to_close_vol_10d'] = self._calculate_close_to_close_vol(df['returns'], 10)
        result['close_to_close_vol_20d'] = self._calculate_close_to_close_vol(df['returns'], 20)
        result['close_to_close_vol_60d'] = self._calculate_close_to_close_vol(df['returns'], 60)

        # Parkinson volatility
        result['parkinson_vol_10d'] = self._calculate_parkinson_vol(df['high'], df['low'], 10)
        result['parkinson_vol_20d'] = self._calculate_parkinson_vol(df['high'], df['low'], 20)

        # Garman-Klass volatility
        result['garman_klass_vol_10d'] = self._calculate_garman_klass_vol(
            df['open'], df['high'], df['low'], df['close'], 10
        )
        result['garman_klass_vol_20d'] = self._calculate_garman_klass_vol(
            df['open'], df['high'], df['low'], df['close'], 20
        )

        # Rogers-Satchell volatility
        result['rogers_satchell_vol_10d'] = self._calculate_rogers_satchell_vol(
            df['open'], df['high'], df['low'], df['close'], 10
        )
        result['rogers_satchell_vol_20d'] = self._calculate_rogers_satchell_vol(
            df['open'], df['high'], df['low'], df['close'], 20
        )

        # Yang-Zhang volatility
        result['yang_zhang_vol_10d'] = self._calculate_yang_zhang_vol(
            df['open'], df['high'], df['low'], df['close'], 10
        )
        result['yang_zhang_vol_20d'] = self._calculate_yang_zhang_vol(
            df['open'], df['high'], df['low'], df['close'], 20
        )

        # Realized volatility percentiles
        result['realized_vol_percentile_1y'] = self._calculate_vol_percentile(
            result['close_to_close_vol_20d'], 252
        )
        result['realized_vol_percentile_3y'] = self._calculate_vol_percentile(
            result['close_to_close_vol_20d'], 756
        )

        # Volatility of volatility
        result['volatility_of_volatility_20d'] = self._calculate_vol_of_vol(
            result['close_to_close_vol_20d'], 20
        )

        # Volatility clustering
        result['vol_clustering_index'] = self._calculate_vol_clustering_index(df['returns'], 20)

        # Volatility trend
        result['volatility_trend'] = self._classify_volatility_trend(
            result['close_to_close_vol_20d'],
            result['close_to_close_vol_60d']
        )

        # Volatility acceleration
        result['volatility_acceleration'] = self._calculate_vol_acceleration(
            result['close_to_close_vol_20d']
        )

        # Volume-weighted volatility
        result['volume_weighted_volatility'] = self._calculate_volume_weighted_vol(
            df['returns'], df['volume'], 20
        )

        # Abnormal volume count
        result['abnormal_volume_count_20d'] = self._count_abnormal_volume(df['volume'], 20)

        # Gap metrics
        gap_metrics = self._calculate_gap_metrics(df['open'], df['close'], 60)
        result['gap_frequency_60d'] = gap_metrics['frequency']
        result['avg_gap_size_60d'] = gap_metrics['avg_size']
        result['gap_volatility_contribution'] = gap_metrics['contribution']

        # Overnight/Intraday ratios
        day_ratios = self._calculate_overnight_intraday_ratios(df['open'], df['close'], 20)
        result['overnight_vol_ratio'] = day_ratios['overnight_ratio']
        result['intraday_vol_ratio'] = day_ratios['intraday_ratio']

        logger.info(f"Calculated {len(result)} days of volatility metrics for {ticker}")
        return result

    def calculate_all_volatility_metrics(self) -> pd.DataFrame:
        """
        Calculate volatility metrics for all assets

        Returns:
            DataFrame with volatility metrics for all assets
        """
        logger.info("Starting volatility calculation for all assets")

        tickers = self._get_all_tickers()
        all_metrics = []

        for idx, ticker in enumerate(tickers, 1):
            try:
                metrics = self.calculate_volatility_metrics(ticker)

                if not metrics.empty:
                    all_metrics.append(metrics)

                # Log progress every 10 tickers
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} tickers processed")

            except Exception as e:
                logger.error(f"Error calculating volatility for {ticker}: {str(e)}")

        if all_metrics:
            df = pd.concat(all_metrics, ignore_index=True)
            logger.info(f"Calculated volatility metrics for {len(df)} total records")
            return df
        else:
            logger.warning("No volatility metrics calculated")
            return pd.DataFrame()

    def populate_volatility_table(self, replace: bool = False) -> None:
        """
        Populate the volatility_metrics table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting volatility_metrics table population")

        try:
            # Calculate all volatility metrics
            vol_df = self.calculate_all_volatility_metrics()

            if vol_df.empty:
                logger.error("No volatility data calculated. Aborting database insertion.")
                return

            # Convert date to string format for database
            vol_df['vol_date'] = vol_df['vol_date'].dt.strftime('%Y-%m-%d')

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM volatility_metrics")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, vol_date FROM volatility_metrics",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['vol_date'])
                )

                # Filter to only new records
                new_records_mask = ~vol_df.apply(
                    lambda row: (row['symbol_ticker'], row['vol_date']) in existing_keys,
                    axis=1
                )
                new_vol_df = vol_df[new_records_mask]

                if len(new_vol_df) > 0:
                    new_vol_df.to_sql('volatility_metrics', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_vol_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                vol_df.to_sql('volatility_metrics', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(vol_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM volatility_metrics")
            final_count = cursor.fetchone()[0]
            logger.info(f"volatility_metrics table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as record_count,
                    MIN(vol_date) as earliest_date,
                    MAX(vol_date) as latest_date,
                    ROUND(AVG(close_to_close_vol_20d), 4) as avg_vol_20d
                FROM volatility_metrics
                GROUP BY symbol_ticker
                ORDER BY symbol_ticker
                LIMIT 10
            """, conn)

            logger.info(f"\nVolatility Metrics Summary (first 10 tickers):")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Show recent volatility for a sample ticker
            sample_df = pd.read_sql("""
                SELECT
                    vol_date,
                    ROUND(close_to_close_vol_20d, 4) as cc_vol,
                    ROUND(parkinson_vol_20d, 4) as park_vol,
                    ROUND(yang_zhang_vol_20d, 4) as yz_vol,
                    volatility_trend,
                    ROUND(realized_vol_percentile_1y, 1) as vol_pct
                FROM volatility_metrics
                WHERE symbol_ticker = 'AAPL'
                ORDER BY vol_date DESC
                LIMIT 5
            """, conn)

            logger.info(f"\nRecent Volatility for AAPL (sample):")
            logger.info(f"\n{sample_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated volatility_metrics table")

        except Exception as e:
            logger.error(f"Error populating volatility_metrics table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize calculator
    calculator = VolatilityCalculator()

    print(f"\n{'='*60}")
    print(f"Volatility Metrics Calculation Script")
    print(f"{'='*60}")
    print(f"Database: {calculator.db_path}")

    # Populate database
    print(f"\n{'='*60}")
    print("Calculating volatility metrics for all assets...")
    print(f"{'='*60}\n")

    calculator.populate_volatility_table(replace=True)

    print(f"\n{'='*60}")
    print("Volatility metrics calculation complete!")
    print(f"{'='*60}\n")