import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairsAnalyzer:
    """
    Analyzes pairs for cointegration and trading statistics, populates pairs_statistics table

    Database: /Volumes/Vault/85_assets_prediction.db
    Source Tables: raw_price_data, correlation_analysis
    Target Table: pairs_statistics

    Analysis:
        - Cointegration testing (Engle-Granger)
        - Hedge ratio calculation (OLS regression)
        - Spread calculation and statistics
        - Z-score analysis
        - Mean reversion half-life
        - Trading signal generation
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the analyzer with database path"""
        self.db_path = db_path

        # Minimum correlation threshold for pair consideration
        self.min_correlation = 0.7

        # Cointegration p-value threshold
        self.coint_pvalue_threshold = 0.05

        # Z-score thresholds for trading signals
        self.zscore_entry_threshold = 2.0
        self.zscore_extreme_threshold = 3.0

        # Minimum data points for analysis
        self.min_data_points = 252  # 1 year

        logger.info(f"Initialized PairsAnalyzer")
        logger.info(f"Min correlation: {self.min_correlation}")
        logger.info(f"Cointegration p-value threshold: {self.coint_pvalue_threshold}")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_top_correlated_pairs(self, top_n: int = 100) -> pd.DataFrame:
        """
        Get top correlated pairs from correlation_analysis table

        Args:
            top_n: Number of top pairs to retrieve

        Returns:
            DataFrame with top correlated pairs
        """
        try:
            conn = self._get_db_connection()
            query = f"""
                SELECT
                    asset_1,
                    asset_2,
                    correlation_90d,
                    correlation_stability
                FROM correlation_analysis
                WHERE correlation_90d IS NOT NULL
                  AND ABS(correlation_90d) >= {self.min_correlation}
                ORDER BY ABS(correlation_90d) DESC
                LIMIT {top_n}
            """
            df = pd.read_sql(query, conn)
            conn.close()
            logger.info(f"Retrieved {len(df)} top correlated pairs")
            return df
        except Exception as e:
            logger.error(f"Error retrieving correlated pairs: {str(e)}")
            return pd.DataFrame()

    def _load_price_series(self, ticker: str, start_date: str = None) -> pd.Series:
        """
        Load close price series for a ticker

        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date filter

        Returns:
            Series indexed by date with close prices
        """
        try:
            conn = self._get_db_connection()

            if start_date:
                query = """
                    SELECT price_date, close
                    FROM raw_price_data
                    WHERE symbol_ticker = ? AND price_date >= ?
                    ORDER BY price_date
                """
                df = pd.read_sql(query, conn, params=(ticker, start_date))
            else:
                query = """
                    SELECT price_date, close
                    FROM raw_price_data
                    WHERE symbol_ticker = ?
                    ORDER BY price_date
                """
                df = pd.read_sql(query, conn, params=(ticker,))

            conn.close()

            if df.empty:
                return pd.Series(dtype=float)

            df['price_date'] = pd.to_datetime(df['price_date'])
            series = df.set_index('price_date')['close']
            return series

        except Exception as e:
            logger.error(f"Error loading price series for {ticker}: {str(e)}")
            return pd.Series(dtype=float)

    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict:
        """
        Test for cointegration using Engle-Granger test

        Args:
            series1: First price series
            series2: Second price series

        Returns:
            Dictionary with cointegration score, p-value, and hedge ratio
        """
        try:
            # Align series on common dates
            combined = pd.DataFrame({
                's1': series1,
                's2': series2
            }).dropna()

            if len(combined) < self.min_data_points:
                return {'score': None, 'pvalue': None, 'hedge_ratio': None}

            # Perform cointegration test
            score, pvalue, _ = coint(combined['s1'], combined['s2'])

            # Calculate hedge ratio (beta from OLS regression)
            # y = alpha + beta*x + epsilon
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            X = combined['s2'].values.reshape(-1, 1)
            y = combined['s1'].values
            model.fit(X, y)
            hedge_ratio = model.coef_[0]

            return {
                'score': float(score),
                'pvalue': float(pvalue),
                'hedge_ratio': float(hedge_ratio)
            }

        except Exception as e:
            logger.warning(f"Cointegration test failed: {str(e)}")
            return {'score': None, 'pvalue': None, 'hedge_ratio': None}

    def _calculate_spread(self, series1: pd.Series, series2: pd.Series,
                         hedge_ratio: float) -> pd.Series:
        """
        Calculate spread between two assets

        Spread = Series1 - hedge_ratio * Series2

        Args:
            series1: First price series
            series2: Second price series
            hedge_ratio: Hedge ratio from cointegration test

        Returns:
            Spread series
        """
        # Align series
        combined = pd.DataFrame({
            's1': series1,
            's2': series2
        }).dropna()

        spread = combined['s1'] - hedge_ratio * combined['s2']
        return spread

    def _calculate_zscore(self, spread: pd.Series, window: int = 60) -> pd.Series:
        """
        Calculate rolling Z-score of spread

        Args:
            spread: Spread series
            window: Rolling window for mean/std calculation

        Returns:
            Z-score series
        """
        spread_mean = spread.rolling(window=window, min_periods=window).mean()
        spread_std = spread.rolling(window=window, min_periods=window).std()

        zscore = (spread - spread_mean) / spread_std
        return zscore

    def _calculate_half_life(self, spread: pd.Series) -> Optional[float]:
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process

        Formula: Half-life = -ln(2) / lambda
        where lambda comes from AR(1): spread(t) = lambda * spread(t-1) + epsilon

        Args:
            spread: Spread series

        Returns:
            Half-life in days (or None if calculation fails)
        """
        try:
            # Remove NaN values
            spread_clean = spread.dropna()

            if len(spread_clean) < 50:
                return None

            # Calculate lagged spread
            spread_lag = spread_clean.shift(1).dropna()
            spread_diff = spread_clean.diff().dropna()

            # Align series
            df = pd.DataFrame({
                'lag': spread_lag,
                'diff': spread_diff
            }).dropna()

            if len(df) < 50:
                return None

            # OLS regression: spread_diff = lambda * spread_lag + c
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            X = df['lag'].values.reshape(-1, 1)
            y = df['diff'].values
            model.fit(X, y)

            lambda_param = model.coef_[0]

            # Calculate half-life
            if lambda_param >= 0:
                # No mean reversion
                return None

            half_life = -np.log(2) / lambda_param

            # Sanity check: half-life should be positive and reasonable
            if half_life > 0 and half_life < 500:
                return float(half_life)
            else:
                return None

        except Exception as e:
            logger.warning(f"Half-life calculation failed: {str(e)}")
            return None

    def _count_reversions(self, zscore: pd.Series, window: int = 60,
                         threshold: float = 2.0) -> int:
        """
        Count mean reversions (crosses back through zero from extreme)

        Args:
            zscore: Z-score series
            window: Lookback window
            threshold: Z-score threshold for "extreme"

        Returns:
            Count of reversions
        """
        recent = zscore.tail(window)

        # Find extreme points (abs(zscore) > threshold)
        extreme_points = np.abs(recent) > threshold

        # Count zero crossings after extreme points
        reversions = 0
        was_extreme = False
        last_sign = 0

        for z in recent:
            if np.abs(z) > threshold:
                was_extreme = True
                last_sign = np.sign(z)
            elif was_extreme and np.sign(z) != last_sign:
                # Crossed zero after being extreme
                reversions += 1
                was_extreme = False

        return reversions

    def _classify_spread_direction(self, zscore_current: float) -> str:
        """
        Classify current spread direction

        Args:
            zscore_current: Current z-score

        Returns:
            Direction: 'Wide', 'Narrow', 'Neutral'
        """
        if zscore_current > 0.5:
            return 'Wide'
        elif zscore_current < -0.5:
            return 'Narrow'
        else:
            return 'Neutral'

    def _generate_entry_signal(self, zscore_current: float, zscore_previous: float) -> str:
        """
        Generate trading entry signal based on z-score

        Args:
            zscore_current: Current z-score
            zscore_previous: Previous z-score

        Returns:
            Signal: 'Long_Spread', 'Short_Spread', 'None'
        """
        # Long spread = buy asset1, sell asset2 (when spread is narrow)
        # Short spread = sell asset1, buy asset2 (when spread is wide)

        if zscore_current < -self.zscore_entry_threshold and zscore_previous >= -self.zscore_entry_threshold:
            return 'Long_Spread'
        elif zscore_current > self.zscore_entry_threshold and zscore_previous <= self.zscore_entry_threshold:
            return 'Short_Spread'
        else:
            return 'None'

    def analyze_pair(self, ticker1: str, ticker2: str) -> pd.DataFrame:
        """
        Analyze a single pair and calculate all statistics

        Args:
            ticker1: First ticker
            ticker2: Second ticker

        Returns:
            DataFrame with daily pair statistics
        """
        logger.info(f"Analyzing pair: {ticker1} - {ticker2}")

        # Load price data
        series1 = self._load_price_series(ticker1)
        series2 = self._load_price_series(ticker2)

        if series1.empty or series2.empty:
            logger.warning(f"Insufficient data for {ticker1} or {ticker2}")
            return pd.DataFrame()

        # Test cointegration
        coint_result = self._test_cointegration(series1, series2)

        if coint_result['hedge_ratio'] is None:
            logger.warning(f"Cointegration test failed for {ticker1} - {ticker2}")
            return pd.DataFrame()

        # Calculate spread
        spread = self._calculate_spread(series1, series2, coint_result['hedge_ratio'])

        if len(spread) < 60:
            logger.warning(f"Insufficient spread data for {ticker1} - {ticker2}")
            return pd.DataFrame()

        # Calculate spread statistics
        spread_mean_60d = spread.rolling(window=60).mean()
        spread_std_60d = spread.rolling(window=60).std()
        zscore = self._calculate_zscore(spread, window=60)

        # Calculate spread volatility
        spread_returns = spread.pct_change()
        spread_vol = spread_returns.rolling(window=20).std()

        # Calculate spread momentum and acceleration
        spread_momentum = spread.pct_change(periods=5)
        spread_accel = spread_momentum.diff(periods=5)

        # Calculate half-life
        half_life = self._calculate_half_life(spread)

        # Calculate spread percentile
        spread_percentile = spread.rolling(window=252).apply(
            lambda x: (x <= x.iloc[-1]).sum() / len(x) * 100 if len(x) > 0 else np.nan,
            raw=False
        )

        # Build result DataFrame
        result = pd.DataFrame({
            'stat_date': spread.index,
            'symbol_ticker_1': ticker1,
            'symbol_ticker_2': ticker2,
            'cointegration_score': coint_result['score'],
            'cointegration_pvalue': coint_result['pvalue'],
            'hedge_ratio': coint_result['hedge_ratio'],
            'spread': spread.values,
            'spread_mean_60d': spread_mean_60d.values,
            'spread_std_60d': spread_std_60d.values,
            'spread_percentile': spread_percentile.values,
            'spread_zscore': zscore.values,
            'spread_volatility_20d': spread_vol.values,
            'spread_momentum_5d': spread_momentum.values,
            'spread_acceleration': spread_accel.values,
            'half_life_mean_reversion': half_life
        })

        # Calculate reversion count for each row
        result['reversion_count_60d'] = [
            self._count_reversions(zscore.iloc[:i+1], window=60, threshold=2.0)
            if i >= 60 else 0
            for i in range(len(result))
        ]

        # Days since last reversion (placeholder - complex to calculate)
        result['days_since_last_reversion'] = None

        # Classify spread direction
        result['spread_direction'] = result['spread_zscore'].apply(self._classify_spread_direction)

        # Flag extreme spreads
        result['spread_extreme_flag'] = (
            np.abs(result['spread_zscore']) > self.zscore_extreme_threshold
        ).astype(int)

        # Generate entry signals
        result['entry_signal'] = 'None'
        for i in range(1, len(result)):
            result.loc[result.index[i], 'entry_signal'] = self._generate_entry_signal(
                result.iloc[i]['spread_zscore'],
                result.iloc[i-1]['spread_zscore']
            )

        logger.info(f"Analyzed {len(result)} days for pair {ticker1} - {ticker2}")
        return result

    def analyze_top_pairs(self, top_n: int = 50) -> pd.DataFrame:
        """
        Analyze top N correlated pairs

        Args:
            top_n: Number of top pairs to analyze

        Returns:
            DataFrame with pair statistics for all analyzed pairs
        """
        logger.info(f"Starting analysis of top {top_n} correlated pairs")

        # Get top correlated pairs
        top_pairs = self._get_top_correlated_pairs(top_n)

        if top_pairs.empty:
            logger.warning("No correlated pairs found")
            return pd.DataFrame()

        all_pair_stats = []

        for idx, row in top_pairs.iterrows():
            ticker1 = row['asset_1']
            ticker2 = row['asset_2']

            try:
                pair_stats = self.analyze_pair(ticker1, ticker2)

                if not pair_stats.empty:
                    all_pair_stats.append(pair_stats)

                # Log progress every 5 pairs
                if (idx + 1) % 5 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(top_pairs)} pairs analyzed")

            except Exception as e:
                logger.error(f"Error analyzing pair {ticker1}-{ticker2}: {str(e)}")

        if all_pair_stats:
            df = pd.concat(all_pair_stats, ignore_index=True)
            logger.info(f"Analyzed {len(df)} total records across all pairs")
            return df
        else:
            logger.warning("No pair statistics calculated")
            return pd.DataFrame()

    def populate_pairs_table(self, top_n: int = 50, replace: bool = False) -> None:
        """
        Populate the pairs_statistics table in the database

        Args:
            top_n: Number of top pairs to analyze
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting pairs_statistics table population")

        try:
            # Analyze top pairs
            pairs_df = self.analyze_top_pairs(top_n)

            if pairs_df.empty:
                logger.error("No pair statistics calculated. Aborting database insertion.")
                return

            # Convert date to string format for database
            pairs_df['stat_date'] = pd.to_datetime(pairs_df['stat_date']).dt.strftime('%Y-%m-%d')

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM pairs_statistics")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker_1, symbol_ticker_2, stat_date FROM pairs_statistics",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker_1'], existing_df['symbol_ticker_2'], existing_df['stat_date'])
                )

                # Filter to only new records
                new_records_mask = ~pairs_df.apply(
                    lambda row: (row['symbol_ticker_1'], row['symbol_ticker_2'], row['stat_date']) in existing_keys,
                    axis=1
                )
                new_pairs_df = pairs_df[new_records_mask]

                if len(new_pairs_df) > 0:
                    new_pairs_df.to_sql('pairs_statistics', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_pairs_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                pairs_df.to_sql('pairs_statistics', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(pairs_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM pairs_statistics")
            final_count = cursor.fetchone()[0]
            logger.info(f"pairs_statistics table now contains {final_count} total records")

            # Show cointegration summary
            coint_summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker_1,
                    symbol_ticker_2,
                    ROUND(AVG(cointegration_score), 4) as avg_coint_score,
                    ROUND(AVG(cointegration_pvalue), 4) as avg_pvalue,
                    ROUND(AVG(hedge_ratio), 4) as avg_hedge_ratio,
                    ROUND(AVG(half_life_mean_reversion), 2) as avg_half_life
                FROM pairs_statistics
                GROUP BY symbol_ticker_1, symbol_ticker_2
                ORDER BY avg_pvalue
                LIMIT 10
            """, conn)

            logger.info(f"\nTop Cointegrated Pairs (by p-value):")
            logger.info(f"\n{coint_summary_df.to_string(index=False)}")

            # Show recent signals
            signals_df = pd.read_sql("""
                SELECT
                    stat_date,
                    symbol_ticker_1,
                    symbol_ticker_2,
                    ROUND(spread_zscore, 2) as zscore,
                    spread_direction,
                    entry_signal
                FROM pairs_statistics
                WHERE entry_signal != 'None'
                ORDER BY stat_date DESC
                LIMIT 10
            """, conn)

            logger.info(f"\nRecent Trading Signals:")
            logger.info(f"\n{signals_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated pairs_statistics table")

        except Exception as e:
            logger.error(f"Error populating pairs_statistics table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = PairsAnalyzer()

    print(f"\n{'='*60}")
    print(f"Pairs Trading Analysis Script")
    print(f"{'='*60}")
    print(f"Database: {analyzer.db_path}")
    print(f"Min correlation: {analyzer.min_correlation}")
    print(f"Cointegration threshold: {analyzer.coint_pvalue_threshold}")

    # Populate database (analyze top 50 pairs)
    print(f"\n{'='*60}")
    print("Analyzing top correlated pairs for trading opportunities...")
    print(f"{'='*60}\n")

    analyzer.populate_pairs_table(top_n=50, replace=True)

    print(f"\n{'='*60}")
    print("Pairs analysis complete!")
    print(f"{'='*60}\n")