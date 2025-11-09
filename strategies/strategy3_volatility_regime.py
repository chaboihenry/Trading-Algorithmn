"""
Volatility Regime Trading Strategy
===================================
Trades volatility regime changes and IV/HV spreads
Target: 25-50% annual returns (higher risk)

Strategy Logic:
- Monitors implied volatility (IV) vs historical volatility (HV)
- Identifies regime changes (low→high, high→low)
- Trades volatility mispricing opportunities

Signal Types:
- BUY_VOL: IV > HV in low vol regime (vol likely to increase)
- SELL_VOL: IV < HV in high vol regime (vol likely to decrease)
- BUY_HEDGES: Regime shift from low to high (protection needed)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolatilityRegimeStrategy:
    """Volatility regime trading strategy"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        """
        Initialize volatility regime strategy

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.strategy_name = "volatility_regime"
        logger.info(f"Initialized {self.strategy_name} strategy")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        return sqlite3.connect(self.db_path)

    def _calculate_iv_hv_spread(self) -> pd.DataFrame:
        """
        Calculate IV/HV spread for all tickers

        Returns:
            DataFrame with IV/HV spreads
        """
        conn = self._get_db_connection()

        # Get latest options data (for IV)
        query_iv = """
            SELECT
                symbol_ticker,
                implied_volatility,
                options_date
            FROM options_data
            WHERE options_date = (
                SELECT MAX(options_date)
                FROM options_data
            )
            GROUP BY symbol_ticker
        """

        # Get latest historical volatility
        query_hv = """
            SELECT
                symbol_ticker,
                volatility_20d as hist_volatility,
                metric_date
            FROM volatility_metrics
            WHERE metric_date = (
                SELECT MAX(metric_date)
                FROM volatility_metrics
            )
        """

        iv_data = pd.read_sql(query_iv, conn)
        hv_data = pd.read_sql(query_hv, conn)
        conn.close()

        if iv_data.empty or hv_data.empty:
            logger.warning("Insufficient IV or HV data")
            return pd.DataFrame()

        # Merge IV and HV data
        merged = pd.merge(
            iv_data,
            hv_data,
            on='symbol_ticker',
            how='inner'
        )

        # Calculate spread
        merged['iv_hv_spread'] = merged['implied_volatility'] - merged['hist_volatility']

        return merged

    def generate_signals(self,
                        max_signals: int = 10,
                        min_spread: float = 0.05) -> pd.DataFrame:
        """
        Generate trading signals based on volatility regime

        Args:
            max_signals: Maximum number of signals to return
            min_spread: Minimum IV/HV spread threshold

        Returns:
            DataFrame with trading signals
        """
        logger.info(f"Generating signals with min_spread={min_spread}")

        # Calculate IV/HV spread
        spread_data = self._calculate_iv_hv_spread()

        if spread_data.empty:
            logger.warning("No spread data available")
            return pd.DataFrame()

        conn = self._get_db_connection()

        # Get volatility regime and other features
        query = """
            SELECT
                m.symbol_ticker,
                m.feature_date,
                m.volatility_regime,
                m.volatility_20d,
                m.volatility_30d,
                m.atr_14
            FROM ml_features m
            WHERE m.feature_date = (
                SELECT MAX(feature_date)
                FROM ml_features
            )
        """

        regime_data = pd.read_sql(query, conn)

        # Get previous regime for regime change detection
        query_prev = """
            WITH ranked AS (
                SELECT
                    symbol_ticker,
                    feature_date,
                    volatility_regime,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol_ticker
                        ORDER BY feature_date DESC
                    ) as rn
                FROM ml_features
                WHERE volatility_regime IS NOT NULL
            )
            SELECT
                symbol_ticker,
                volatility_regime as prev_volatility_regime
            FROM ranked
            WHERE rn = 2
        """

        prev_regime = pd.read_sql(query_prev, conn)
        conn.close()

        # Merge all data
        signals = pd.merge(regime_data, spread_data, on='symbol_ticker', how='inner')
        signals = pd.merge(signals, prev_regime, on='symbol_ticker', how='left')

        # Detect regime changes
        signals['vol_regime_change'] = signals.apply(
            lambda row: self._detect_regime_change(
                row.get('prev_volatility_regime'),
                row['volatility_regime']
            ),
            axis=1
        )

        # Generate signals based on conditions
        def generate_signal(row):
            iv_hv = row['iv_hv_spread']
            regime = row['volatility_regime']
            regime_change = row['vol_regime_change']

            # BUY_VOL: High IV/HV spread in low vol regime
            if iv_hv > min_spread and regime == 'low':
                return 'BUY_VOL'

            # SELL_VOL: Negative IV/HV spread in high vol regime
            elif iv_hv < -min_spread and regime == 'high':
                return 'SELL_VOL'

            # BUY_HEDGES: Regime change from low to high
            elif regime_change == 'low_to_high':
                return 'BUY_HEDGES'

            else:
                return 'NO_SIGNAL'

        signals['signal'] = signals.apply(generate_signal, axis=1)
        signals['confidence'] = signals['iv_hv_spread'].abs()

        # Filter valid signals
        signals = signals[
            (signals['signal'] != 'NO_SIGNAL') &
            (signals['iv_hv_spread'].abs() > min_spread)
        ].copy()

        if signals.empty:
            logger.warning("No valid signals generated")
            return pd.DataFrame()

        # Sort and limit
        signals = signals.sort_values('confidence', ascending=False).head(max_signals)

        # Add strategy metadata
        signals['strategy'] = self.strategy_name
        signals['signal_date'] = datetime.now().strftime('%Y-%m-%d')

        # Normalize confidence
        if len(signals) > 0:
            max_conf = signals['confidence'].max()
            if max_conf > 0:
                signals['confidence'] = signals['confidence'] / max_conf

        logger.info(f"Generated {len(signals)} signals")
        logger.info(f"BUY_VOL signals: {len(signals[signals['signal'] == 'BUY_VOL'])}")
        logger.info(f"SELL_VOL signals: {len(signals[signals['signal'] == 'SELL_VOL'])}")
        logger.info(f"BUY_HEDGES signals: {len(signals[signals['signal'] == 'BUY_HEDGES'])}")

        return signals

    def _detect_regime_change(self, prev_regime: str, current_regime: str) -> str:
        """Detect volatility regime changes"""
        if pd.isna(prev_regime) or pd.isna(current_regime):
            return None

        if prev_regime == 'low' and current_regime == 'high':
            return 'low_to_high'
        elif prev_regime == 'high' and current_regime == 'low':
            return 'high_to_low'
        else:
            return None

    def get_volatility_history(self, ticker: str, days: int = 60) -> pd.DataFrame:
        """
        Get volatility history for a ticker

        Args:
            ticker: Stock ticker symbol
            days: Number of days to retrieve

        Returns:
            DataFrame with volatility history
        """
        conn = self._get_db_connection()

        query = """
            SELECT
                metric_date,
                volatility_10d,
                volatility_20d,
                volatility_30d,
                atr_14
            FROM volatility_metrics
            WHERE symbol_ticker = ?
            ORDER BY metric_date DESC
            LIMIT ?
        """

        history = pd.read_sql(query, conn, params=(ticker, days))
        conn.close()

        return history

    def get_regime_analysis(self, ticker: str) -> dict:
        """
        Get detailed regime analysis for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with regime analysis
        """
        conn = self._get_db_connection()

        # Get current regime
        query_current = """
            SELECT
                volatility_regime,
                volatility_20d,
                feature_date
            FROM ml_features
            WHERE symbol_ticker = ?
            ORDER BY feature_date DESC
            LIMIT 1
        """

        current = pd.read_sql(query_current, conn, params=(ticker,))

        # Get regime distribution
        query_dist = """
            SELECT
                volatility_regime,
                COUNT(*) as count
            FROM ml_features
            WHERE symbol_ticker = ?
                AND volatility_regime IS NOT NULL
            GROUP BY volatility_regime
        """

        distribution = pd.read_sql(query_dist, conn, params=(ticker,))
        conn.close()

        if current.empty:
            return {"error": "No regime data available"}

        return {
            "current_regime": current.iloc[0]['volatility_regime'],
            "current_vol": current.iloc[0]['volatility_20d'],
            "as_of_date": current.iloc[0]['feature_date'],
            "regime_distribution": distribution.to_dict('records')
        }


if __name__ == "__main__":
    # Example usage
    strategy = VolatilityRegimeStrategy()

    # Generate signals
    signals = strategy.generate_signals(max_signals=10, min_spread=0.05)

    if not signals.empty:
        print("\n" + "="*80)
        print("VOLATILITY REGIME SIGNALS")
        print("="*80)
        print(signals[['symbol_ticker', 'signal', 'volatility_regime',
                      'iv_hv_spread', 'vol_regime_change', 'confidence']].to_string(index=False))
        print("="*80)

        # Show detailed info for top signal
        if len(signals) > 0:
            top_signal = signals.iloc[0]
            print(f"\nTop Signal Details:")
            print(f"Ticker: {top_signal['symbol_ticker']}")
            print(f"Signal: {top_signal['signal']}")
            print(f"Current Regime: {top_signal['volatility_regime']}")
            print(f"IV/HV Spread: {top_signal['iv_hv_spread']:.2%}")
            if pd.notna(top_signal['vol_regime_change']):
                print(f"Regime Change: {top_signal['vol_regime_change']}")
            print(f"Confidence: {top_signal['confidence']:.2%}")
    else:
        print("\nNo signals generated with current thresholds")
