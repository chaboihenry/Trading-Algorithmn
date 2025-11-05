import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging
from scipy.stats import pearsonr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Calculates correlation metrics between asset pairs for correlation_analysis table

    Database: /Volumes/Vault/85_assets_prediction.db
    Source Table: raw_price_data
    Target Table: correlation_analysis

    Calculates:
        - Rolling correlations (30d, 90d, 180d)
        - Correlation stability metrics
        - Directional correlation (upside/downside)
        - Volatility correlation
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the analyzer with database path"""
        self.db_path = db_path

        # Correlation windows (in trading days)
        self.windows = {
            'short': 30,    # 1 month
            'medium': 90,   # 3 months
            'long': 180     # 6 months
        }

        # Minimum data points required for correlation calculation
        self.min_data_points = 30

        logger.info(f"Initialized CorrelationAnalyzer")
        logger.info(f"Correlation windows: {self.windows}")

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

    def _load_price_data(self, ticker: str, start_date: str = None) -> pd.DataFrame:
        """
        Load price data for a single ticker

        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date (YYYY-MM-DD), defaults to all available data

        Returns:
            DataFrame with price_date, close, returns, volatility
        """
        try:
            conn = self._get_db_connection()

            if start_date:
                query = """
                    SELECT price_date, close
                    FROM raw_price_data
                    WHERE symbol_ticker = ?
                      AND price_date >= ?
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
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()

            # Convert date and set as index
            df['price_date'] = pd.to_datetime(df['price_date'])
            df = df.set_index('price_date')

            # Calculate returns
            df['returns'] = df['close'].pct_change()

            # Calculate rolling volatility (20-day)
            df['volatility'] = df['returns'].rolling(window=20).std()

            return df

        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_rolling_correlation(self, returns1: pd.Series, returns2: pd.Series,
                                      window: int) -> pd.Series:
        """
        Calculate rolling correlation between two return series

        Args:
            returns1: First return series
            returns2: Second return series
            window: Rolling window size

        Returns:
            Rolling correlation series
        """
        return returns1.rolling(window=window).corr(returns2)

    def _calculate_correlation_stability(self, rolling_corr: pd.Series) -> float:
        """
        Calculate correlation stability (1 - std of rolling correlations)
        Higher values indicate more stable correlation

        Args:
            rolling_corr: Rolling correlation series

        Returns:
            Stability score (0-1)
        """
        if rolling_corr.empty or rolling_corr.isna().all():
            return None

        std = rolling_corr.std()
        if pd.isna(std):
            return None

        stability = max(0, 1 - std)
        return round(stability, 4)

    def _calculate_directional_correlation(self, returns1: pd.Series, returns2: pd.Series,
                                          direction: str = 'upside') -> float:
        """
        Calculate correlation during upside or downside moves

        Args:
            returns1: First return series
            returns2: Second return series
            direction: 'upside' or 'downside'

        Returns:
            Directional correlation coefficient
        """
        # Combine series
        combined = pd.DataFrame({
            'ret1': returns1,
            'ret2': returns2
        }).dropna()

        if len(combined) < self.min_data_points:
            return None

        # Filter by direction
        if direction == 'upside':
            filtered = combined[combined['ret1'] > 0]
        else:  # downside
            filtered = combined[combined['ret1'] < 0]

        if len(filtered) < self.min_data_points:
            return None

        try:
            corr, _ = pearsonr(filtered['ret1'], filtered['ret2'])
            return round(corr, 4) if not np.isnan(corr) else None
        except:
            return None

    def _calculate_volatility_correlation(self, vol1: pd.Series, vol2: pd.Series) -> float:
        """
        Calculate correlation between volatilities

        Args:
            vol1: First volatility series
            vol2: Second volatility series

        Returns:
            Volatility correlation coefficient
        """
        # Combine and drop NaN
        combined = pd.DataFrame({
            'vol1': vol1,
            'vol2': vol2
        }).dropna()

        if len(combined) < self.min_data_points:
            return None

        try:
            corr, _ = pearsonr(combined['vol1'], combined['vol2'])
            return round(corr, 4) if not np.isnan(corr) else None
        except:
            return None

    def calculate_pair_correlation(self, ticker1: str, ticker2: str,
                                   calculation_date: str = None) -> Dict:
        """
        Calculate all correlation metrics for a pair of assets

        Args:
            ticker1: First ticker
            ticker2: Second ticker
            calculation_date: Date for calculation (defaults to latest available)

        Returns:
            Dictionary with correlation metrics
        """
        logger.info(f"Calculating correlations for {ticker1} - {ticker2}")

        # Determine lookback period (need at least 180 days for long window)
        if calculation_date is None:
            calculation_date = datetime.now().strftime('%Y-%m-%d')

        lookback_date = (datetime.strptime(calculation_date, '%Y-%m-%d') -
                        timedelta(days=365)).strftime('%Y-%m-%d')

        # Load data for both tickers
        df1 = self._load_price_data(ticker1, start_date=lookback_date)
        df2 = self._load_price_data(ticker2, start_date=lookback_date)

        if df1.empty or df2.empty:
            logger.warning(f"Insufficient data for {ticker1} or {ticker2}")
            return None

        # Align data on common dates
        combined = pd.DataFrame({
            'returns1': df1['returns'],
            'returns2': df2['returns'],
            'vol1': df1['volatility'],
            'vol2': df2['volatility']
        }).dropna()

        if len(combined) < self.min_data_points:
            logger.warning(f"Insufficient overlapping data for {ticker1} - {ticker2}")
            return None

        # Calculate correlation for each window
        correlations = {}

        for window_name, window_size in self.windows.items():
            if len(combined) >= window_size:
                # Get most recent window
                recent_data = combined.tail(window_size)

                try:
                    corr, _ = pearsonr(recent_data['returns1'], recent_data['returns2'])
                    correlations[f'corr_{window_name}'] = round(corr, 4) if not np.isnan(corr) else None
                except:
                    correlations[f'corr_{window_name}'] = None
            else:
                correlations[f'corr_{window_name}'] = None

        # Calculate rolling correlation for stability (use 90-day window)
        if len(combined) >= self.windows['medium']:
            rolling_corr = self._calculate_rolling_correlation(
                combined['returns1'],
                combined['returns2'],
                window=self.windows['medium']
            )
            stability = self._calculate_correlation_stability(rolling_corr)
        else:
            stability = None

        # Calculate directional correlations
        upside_corr = self._calculate_directional_correlation(
            combined['returns1'], combined['returns2'], direction='upside'
        )
        downside_corr = self._calculate_directional_correlation(
            combined['returns1'], combined['returns2'], direction='downside'
        )

        # Calculate volatility correlation
        vol_corr = self._calculate_volatility_correlation(
            combined['vol1'], combined['vol2']
        )

        # Build result dictionary
        result = {
            'asset_1': ticker1,
            'asset_2': ticker2,
            'calculation_date': calculation_date,
            'correlation_30d': correlations.get('corr_short'),
            'correlation_90d': correlations.get('corr_medium'),
            'correlation_180d': correlations.get('corr_long'),
            'correlation_stability': stability,
            'directional_correlation_upside': upside_corr,
            'directional_correlation_downside': downside_corr,
            'volatility_correlation': vol_corr,
            'sample_size': len(combined)
        }

        return result

    def calculate_all_correlations(self, calculation_date: str = None) -> pd.DataFrame:
        """
        Calculate correlations for all asset pairs

        Args:
            calculation_date: Date for calculation (defaults to latest)

        Returns:
            DataFrame with correlation metrics for all pairs
        """
        logger.info("Starting correlation calculation for all asset pairs")

        tickers = self._get_all_tickers()
        n_tickers = len(tickers)

        logger.info(f"Calculating correlations for {n_tickers} assets")
        logger.info(f"Total pairs: {n_tickers * (n_tickers - 1) // 2}")

        all_correlations = []
        pair_count = 0
        total_pairs = n_tickers * (n_tickers - 1) // 2

        # Calculate for all unique pairs
        for i in range(n_tickers):
            for j in range(i + 1, n_tickers):
                ticker1 = tickers[i]
                ticker2 = tickers[j]

                corr_data = self.calculate_pair_correlation(ticker1, ticker2, calculation_date)

                if corr_data is not None:
                    all_correlations.append(corr_data)

                pair_count += 1

                # Log progress every 100 pairs
                if pair_count % 100 == 0:
                    logger.info(f"Progress: {pair_count}/{total_pairs} pairs calculated")

        if all_correlations:
            df = pd.DataFrame(all_correlations)
            logger.info(f"Calculated correlations for {len(df)} asset pairs")
            return df
        else:
            logger.warning("No correlations calculated")
            return pd.DataFrame()

    def populate_correlation_table(self, replace: bool = False) -> None:
        """
        Populate the correlation_analysis table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting correlation_analysis table population")

        try:
            # Calculate all correlations
            corr_df = self.calculate_all_correlations()

            if corr_df.empty:
                logger.error("No correlation data calculated. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM correlation_analysis")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing pairs for the same calculation date
                calc_date = corr_df['calculation_date'].iloc[0]
                existing_df = pd.read_sql(
                    "SELECT asset_1, asset_2, calculation_date FROM correlation_analysis WHERE calculation_date = ?",
                    conn,
                    params=(calc_date,)
                )
                existing_keys = set(
                    zip(existing_df['asset_1'], existing_df['asset_2'], existing_df['calculation_date'])
                )

                # Filter to only new records
                new_records_mask = ~corr_df.apply(
                    lambda row: (row['asset_1'], row['asset_2'], row['calculation_date']) in existing_keys,
                    axis=1
                )
                new_corr_df = corr_df[new_records_mask]

                if len(new_corr_df) > 0:
                    new_corr_df.to_sql('correlation_analysis', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_corr_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                corr_df.to_sql('correlation_analysis', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(corr_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM correlation_analysis")
            final_count = cursor.fetchone()[0]
            logger.info(f"correlation_analysis table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    COUNT(*) as total_pairs,
                    ROUND(AVG(correlation_90d), 3) as avg_corr_90d,
                    ROUND(AVG(correlation_stability), 3) as avg_stability,
                    ROUND(AVG(volatility_correlation), 3) as avg_vol_corr,
                    MIN(sample_size) as min_sample_size,
                    MAX(sample_size) as max_sample_size
                FROM correlation_analysis
            """, conn)

            logger.info(f"\nCorrelation Analysis Summary:")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Top correlated pairs
            top_corr_df = pd.read_sql("""
                SELECT
                    asset_1,
                    asset_2,
                    ROUND(correlation_90d, 3) as corr_90d,
                    ROUND(correlation_stability, 3) as stability
                FROM correlation_analysis
                WHERE correlation_90d IS NOT NULL
                ORDER BY ABS(correlation_90d) DESC
                LIMIT 10
            """, conn)

            logger.info(f"\nTop 10 Correlated Pairs:")
            logger.info(f"\n{top_corr_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated correlation_analysis table")

        except Exception as e:
            logger.error(f"Error populating correlation_analysis table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = CorrelationAnalyzer()

    print(f"\n{'='*60}")
    print(f"Correlation Analysis Script")
    print(f"{'='*60}")
    print(f"Database: {analyzer.db_path}")
    print(f"Correlation windows: {analyzer.windows}")

    # Populate database
    print(f"\n{'='*60}")
    print("Calculating correlations for all asset pairs...")
    print(f"{'='*60}\n")

    analyzer.populate_correlation_table(replace=True)

    print(f"\n{'='*60}")
    print("Correlation analysis complete!")
    print(f"{'='*60}\n")