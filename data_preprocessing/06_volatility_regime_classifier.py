"""
Volatility Regime Classifier
=============================

Classifies market into volatility regimes using clustering on VIX, realized volatility,
and volatility of volatility.

Features:
- KMeans clustering (n=3: Low/Medium/High volatility regimes)
- Uses VIX, 20-day realized volatility, and volatility of volatility
- Detects regime transitions as trading signals
- Populates volatility_regimes table

Output: volatility_regimes table with regime classifications and transitions
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolatilityRegimeClassifier:
    """Classify market volatility into Low/Medium/High regimes"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """Initialize classifier"""
        self.db_path = db_path
        self.n_regimes = 3  # Low, Medium, High
        self.vix_ticker = "^VIX"  # VIX index ticker

        logger.info(f"Initialized VolatilityRegimeClassifier")
        logger.info(f"Database: {db_path}")
        logger.info(f"Regimes: {self.n_regimes} (Low/Medium/High)")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def _create_regimes_table(self) -> None:
        """Create volatility_regimes table if not exists"""
        conn = self._get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volatility_regimes (
                regime_id INTEGER PRIMARY KEY AUTOINCREMENT,
                regime_date DATE NOT NULL UNIQUE,

                -- Features used for classification
                vix_value REAL,
                realized_vol_20d REAL,
                vol_of_vol REAL,

                -- Regime classification
                regime_cluster INTEGER NOT NULL,
                regime_label TEXT NOT NULL,

                -- Regime transitions
                previous_regime TEXT,
                regime_changed INTEGER DEFAULT 0,
                days_in_regime INTEGER DEFAULT 1,

                -- Risk metrics
                risk_level TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("volatility_regimes table created/verified")

    def _get_vix_data(self, lookback_days: int = 500) -> pd.DataFrame:
        """Get VIX data from price_data table"""
        conn = self._get_db_connection()

        query = """
            SELECT price_date, close_price as vix_close
            FROM price_data
            WHERE symbol_ticker = ?
            ORDER BY price_date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(self.vix_ticker, lookback_days))
        conn.close()

        if df.empty:
            logger.warning(f"No VIX data found for {self.vix_ticker}")
            return pd.DataFrame()

        df = df.sort_values('price_date')
        df['price_date'] = pd.to_datetime(df['price_date'])

        return df

    def _calculate_market_realized_volatility(self, lookback_days: int = 500) -> pd.DataFrame:
        """
        Calculate market-wide realized volatility
        Uses SPY as market proxy
        """
        conn = self._get_db_connection()

        query = """
            SELECT price_date, close_price
            FROM price_data
            WHERE symbol_ticker = 'SPY'
            ORDER BY price_date DESC
            LIMIT ?
        """

        df = pd.read_sql(query, conn, params=(lookback_days,))
        conn.close()

        if df.empty:
            logger.warning("No SPY data found for realized volatility calculation")
            return pd.DataFrame()

        df = df.sort_values('price_date')
        df['price_date'] = pd.to_datetime(df['price_date'])

        # Calculate returns
        df['returns'] = df['close_price'].pct_change()

        # Calculate 20-day realized volatility (annualized)
        df['realized_vol_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252) * 100

        # Calculate volatility of volatility (20-day rolling std of realized vol)
        df['vol_of_vol'] = df['realized_vol_20d'].rolling(window=20).std()

        return df[['price_date', 'realized_vol_20d', 'vol_of_vol']].dropna()

    def _prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for clustering

        Returns:
            DataFrame with date, vix, realized_vol, vol_of_vol
        """
        logger.info("Preparing features for volatility regime classification...")

        # Get VIX data
        vix_df = self._get_vix_data(lookback_days=500)
        if vix_df.empty:
            logger.error("Cannot classify regimes without VIX data")
            return pd.DataFrame()

        # Get realized volatility
        realized_vol_df = self._calculate_market_realized_volatility(lookback_days=500)
        if realized_vol_df.empty:
            logger.error("Cannot classify regimes without realized volatility data")
            return pd.DataFrame()

        # Merge on date
        features_df = pd.merge(
            vix_df,
            realized_vol_df,
            on='price_date',
            how='inner'
        )

        # Drop any remaining NaN values
        features_df = features_df.dropna()

        logger.info(f"Prepared {len(features_df)} samples for clustering")
        logger.info(f"Date range: {features_df['price_date'].min()} to {features_df['price_date'].max()}")

        return features_df

    def _classify_regimes(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify volatility regimes using KMeans clustering

        Args:
            features_df: DataFrame with features

        Returns:
            DataFrame with regime classifications
        """
        logger.info("Running KMeans clustering for regime classification...")

        # Prepare feature matrix
        X = features_df[['vix_close', 'realized_vol_20d', 'vol_of_vol']].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Add cluster assignments
        features_df['regime_cluster'] = clusters

        # Determine regime labels based on average VIX in each cluster
        cluster_stats = features_df.groupby('regime_cluster')['vix_close'].mean().sort_values()

        regime_mapping = {
            cluster_stats.index[0]: 'Low',      # Lowest avg VIX
            cluster_stats.index[1]: 'Medium',   # Middle avg VIX
            cluster_stats.index[2]: 'High'      # Highest avg VIX
        }

        features_df['regime_label'] = features_df['regime_cluster'].map(regime_mapping)

        # Calculate risk level
        risk_mapping = {'Low': 'Low', 'Medium': 'Moderate', 'High': 'High'}
        features_df['risk_level'] = features_df['regime_label'].map(risk_mapping)

        # Log cluster statistics
        logger.info("Cluster Statistics:")
        for cluster_id in sorted(cluster_stats.index):
            label = regime_mapping[cluster_id]
            count = len(features_df[features_df['regime_cluster'] == cluster_id])
            avg_vix = cluster_stats[cluster_id]
            logger.info(f"  {label} Regime: {count} samples, Avg VIX: {avg_vix:.2f}")

        return features_df

    def _detect_regime_transitions(self, regimes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect regime transitions and calculate days in regime

        Args:
            regimes_df: DataFrame with regime classifications

        Returns:
            DataFrame with transition information
        """
        logger.info("Detecting regime transitions...")

        regimes_df = regimes_df.sort_values('price_date')

        # Detect previous regime
        regimes_df['previous_regime'] = regimes_df['regime_label'].shift(1)

        # Flag regime changes
        regimes_df['regime_changed'] = (
            regimes_df['regime_label'] != regimes_df['previous_regime']
        ).astype(int)

        # Calculate days in current regime
        regime_groups = (regimes_df['regime_changed'] == 1).cumsum()
        regimes_df['days_in_regime'] = regimes_df.groupby(regime_groups).cumcount() + 1

        # Count transitions
        transitions = regimes_df[regimes_df['regime_changed'] == 1]
        logger.info(f"Detected {len(transitions)} regime transitions")

        # Log transition summary
        if not transitions.empty:
            transition_summary = transitions.groupby(
                ['previous_regime', 'regime_label']
            ).size().reset_index(name='count')

            logger.info("Regime Transition Summary:")
            for _, row in transition_summary.iterrows():
                if pd.notna(row['previous_regime']):
                    logger.info(f"  {row['previous_regime']} → {row['regime_label']}: {row['count']} times")

        return regimes_df

    def _save_regimes(self, regimes_df: pd.DataFrame) -> int:
        """
        Save regime classifications to database

        Args:
            regimes_df: DataFrame with regime data

        Returns:
            Number of records saved
        """
        logger.info("Saving regime classifications to database...")

        conn = self._get_db_connection()
        cursor = conn.cursor()

        saved_count = 0

        for _, row in regimes_df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO volatility_regimes (
                        regime_date, vix_value, realized_vol_20d, vol_of_vol,
                        regime_cluster, regime_label, previous_regime,
                        regime_changed, days_in_regime, risk_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['price_date'].strftime('%Y-%m-%d'),
                    row['vix_close'],
                    row['realized_vol_20d'],
                    row['vol_of_vol'],
                    int(row['regime_cluster']),
                    row['regime_label'],
                    row['previous_regime'] if pd.notna(row['previous_regime']) else None,
                    int(row['regime_changed']),
                    int(row['days_in_regime']),
                    row['risk_level']
                ))
                saved_count += 1

            except Exception as e:
                logger.error(f"Error saving regime for {row['price_date']}: {e}")

        conn.commit()
        conn.close()

        logger.info(f"✓ Saved {saved_count} regime classifications")
        return saved_count

    def run(self) -> None:
        """Run volatility regime classification"""
        logger.info("="*60)
        logger.info("VOLATILITY REGIME CLASSIFICATION STARTED")
        logger.info("="*60)

        # Create table
        self._create_regimes_table()

        # Prepare features
        features_df = self._prepare_features()

        if features_df.empty:
            logger.error("No features available for classification")
            return

        # Classify regimes
        regimes_df = self._classify_regimes(features_df)

        # Detect transitions
        regimes_df = self._detect_regime_transitions(regimes_df)

        # Save to database
        saved = self._save_regimes(regimes_df)

        # Summary statistics
        logger.info("="*60)
        logger.info("VOLATILITY REGIME CLASSIFICATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Total samples classified: {len(regimes_df)}")
        logger.info(f"Records saved: {saved}")
        logger.info("="*60)

        # Current regime
        if not regimes_df.empty:
            latest = regimes_df.iloc[-1]
            logger.info(f"\nCurrent Volatility Regime: {latest['regime_label']}")
            logger.info(f"VIX: {latest['vix_close']:.2f}")
            logger.info(f"Realized Vol (20d): {latest['realized_vol_20d']:.2f}%")
            logger.info(f"Days in regime: {latest['days_in_regime']}")
            logger.info(f"Risk Level: {latest['risk_level']}")


if __name__ == "__main__":
    classifier = VolatilityRegimeClassifier()
    classifier.run()