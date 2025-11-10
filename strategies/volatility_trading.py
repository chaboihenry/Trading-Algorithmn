"""
Enhanced Strategy 3: Volatility Regime Trading with GARCH & Random Forest
==========================================================================
Uses GARCH models for volatility forecasting and Random Forest for regime prediction
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from base_strategy import BaseStrategy
import logging

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class VolatilityTradingStrategy(BaseStrategy):
    """Trade based on GARCH volatility forecasts and Random Forest regime prediction"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 lookback_days: int = 90,
                 min_samples: int = 30):
        super().__init__(db_path)
        self.lookback_days = lookback_days
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.name = "VolatilityTradingStrategy"

    def _get_volatility_features(self) -> pd.DataFrame:
        """Get comprehensive volatility features for ML model"""
        conn = self._conn()
        query = f"""
            SELECT
                vm.symbol_ticker,
                vm.vol_date,
                -- Multiple volatility estimators
                vm.close_to_close_vol_10d,
                vm.close_to_close_vol_20d,
                vm.close_to_close_vol_60d,
                vm.parkinson_vol_10d,
                vm.parkinson_vol_20d,
                vm.garman_klass_vol_10d,
                vm.garman_klass_vol_20d,
                vm.yang_zhang_vol_10d,
                vm.yang_zhang_vol_20d,
                -- Volatility dynamics
                vm.realized_vol_percentile_1y,
                vm.realized_vol_percentile_3y,
                vm.volatility_of_volatility_20d,
                vm.vol_clustering_index,
                vm.volatility_trend,
                vm.volatility_acceleration,
                -- Volume features
                vm.volume_weighted_volatility,
                vm.abnormal_volume_count_20d,
                -- Gap features
                vm.gap_frequency_60d,
                vm.avg_gap_size_60d,
                vm.gap_volatility_contribution,
                -- Intraday features
                vm.overnight_vol_ratio,
                vm.intraday_vol_ratio,
                -- Technical indicators
                ti.rsi_14,
                ti.atr_14,
                ti.atr_20,
                ti.bb_width,
                ti.adx_14,
                -- Price features
                mf.return_1d,
                mf.return_5d,
                mf.return_20d,
                rpd.close as current_price
            FROM volatility_metrics vm
            LEFT JOIN technical_indicators ti
                ON vm.symbol_ticker = ti.symbol_ticker
                AND vm.vol_date = ti.indicator_date
            LEFT JOIN ml_features mf
                ON vm.symbol_ticker = mf.symbol_ticker
                AND vm.vol_date = mf.feature_date
            LEFT JOIN raw_price_data rpd
                ON vm.symbol_ticker = rpd.symbol_ticker
                AND vm.vol_date = rpd.price_date
            WHERE vm.vol_date >= date('now', '-{self.lookback_days} days')
            ORDER BY vm.symbol_ticker, vm.vol_date
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df

    def _calculate_garch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate GARCH-inspired features (simplified version)"""
        for ticker in df['symbol_ticker'].unique():
            mask = df['symbol_ticker'] == ticker
            ticker_data = df[mask].sort_values('vol_date')

            # GARCH(1,1) approximation using rolling statistics
            # σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}

            # Calculate squared returns (proxy for ε²)
            returns = ticker_data['return_1d'].fillna(0)
            squared_returns = returns ** 2

            # Estimate GARCH components using exponential weighting
            alpha = 0.1  # Weight on past squared return
            beta = 0.85   # Weight on past variance
            omega = 0.05  # Long-term variance

            # Initialize conditional variance
            cond_var = [squared_returns.iloc[0] if len(squared_returns) > 0 else omega]

            # Iterate to calculate GARCH conditional variance
            for i in range(1, len(squared_returns)):
                next_var = omega + alpha * squared_returns.iloc[i-1] + beta * cond_var[-1]
                cond_var.append(next_var)

            df.loc[mask, 'garch_cond_vol'] = np.sqrt(cond_var)

            # Volatility forecast (1-day ahead)
            df.loc[mask, 'garch_forecast'] = df.loc[mask, 'garch_cond_vol'].shift(1)

            # Volatility surprise (actual vs expected)
            df.loc[mask, 'vol_surprise'] = (
                df.loc[mask, 'close_to_close_vol_10d'] - df.loc[mask, 'garch_forecast']
            )

        return df

    def _create_regime_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility regime labels for classification"""
        df = df.sort_values(['symbol_ticker', 'vol_date'])

        # Calculate future volatility change (for labeling)
        df['future_vol_change'] = df.groupby('symbol_ticker')['close_to_close_vol_20d'].shift(-5)
        df['vol_change_pct'] = (
            (df['future_vol_change'] - df['close_to_close_vol_20d']) /
            df['close_to_close_vol_20d']
        )

        # Create regime transition labels
        # 0 = vol decreasing, 1 = vol stable, 2 = vol increasing
        conditions = [
            df['vol_change_pct'] < -0.15,  # Significant decrease
            df['vol_change_pct'] > 0.15,    # Significant increase
        ]
        choices = [0, 2]
        df['regime_label'] = np.select(conditions, choices, default=1)

        return df

    def _prepare_ml_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix for Random Forest"""
        feature_cols = [
            'close_to_close_vol_10d', 'close_to_close_vol_20d', 'close_to_close_vol_60d',
            'parkinson_vol_10d', 'parkinson_vol_20d',
            'garman_klass_vol_10d', 'garman_klass_vol_20d',
            'yang_zhang_vol_10d', 'yang_zhang_vol_20d',
            'realized_vol_percentile_1y', 'realized_vol_percentile_3y',
            'volatility_of_volatility_20d', 'vol_clustering_index',
            'volatility_acceleration', 'volume_weighted_volatility',
            'abnormal_volume_count_20d', 'gap_frequency_60d',
            'rsi_14', 'atr_14', 'bb_width', 'adx_14',
            'return_1d', 'return_5d', 'return_20d',
            'garch_cond_vol', 'garch_forecast', 'vol_surprise'
        ]

        # Convert volatility_trend to numeric
        trend_map = {'decreasing': -1, 'stable': 0, 'increasing': 1}
        df['volatility_trend_num'] = df['volatility_trend'].map(trend_map).fillna(0)
        feature_cols.append('volatility_trend_num')

        # Clean data
        df_clean = df.dropna(subset=feature_cols + ['regime_label'])

        X = df_clean[feature_cols].values
        y = df_clean['regime_label'].values

        return X, y, df_clean

    def _train_random_forest(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Random Forest for regime prediction"""
        if len(X) < self.min_samples:
            logger.warning(f"Not enough samples ({len(X)}) to train model")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        logger.info(f"Random Forest trained: Train accuracy={train_score:.3f}, Test accuracy={test_score:.3f}")

    def _get_feature_importance(self) -> dict:
        """Get feature importance from Random Forest"""
        if self.model is None:
            return {}

        feature_names = [
            'close_to_close_vol_10d', 'close_to_close_vol_20d', 'close_to_close_vol_60d',
            'parkinson_vol_10d', 'parkinson_vol_20d',
            'garman_klass_vol_10d', 'garman_klass_vol_20d',
            'yang_zhang_vol_10d', 'yang_zhang_vol_20d',
            'realized_vol_percentile_1y', 'realized_vol_percentile_3y',
            'volatility_of_volatility_20d', 'vol_clustering_index',
            'volatility_acceleration', 'volume_weighted_volatility',
            'abnormal_volume_count_20d', 'gap_frequency_60d',
            'rsi_14', 'atr_14', 'bb_width', 'adx_14',
            'return_1d', 'return_5d', 'return_20d',
            'garch_cond_vol', 'garch_forecast', 'vol_surprise',
            'volatility_trend_num'
        ]

        importance = dict(zip(feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def generate_signals(self) -> pd.DataFrame:
        """Generate ML-enhanced volatility trading signals"""
        # Get volatility features
        df = self._get_volatility_features()

        if df.empty:
            logger.warning("No volatility data available")
            return pd.DataFrame()

        # Calculate GARCH features
        df = self._calculate_garch_features(df)

        # Create regime labels
        df = self._create_regime_labels(df)

        # Prepare ML features
        X, y, df_clean = self._prepare_ml_features(df)

        if len(X) == 0:
            logger.warning("No valid features after cleaning")
            return pd.DataFrame()

        # Train Random Forest
        self._train_random_forest(X, y)

        if self.model is None:
            logger.warning("Model training failed")
            return pd.DataFrame()

        # Log feature importance
        importance = self._get_feature_importance()
        logger.info("Top 5 volatility features:")
        for feat, imp in list(importance.items())[:5]:
            logger.info(f"  {feat}: {imp:.3f}")

        # Get latest data for prediction
        latest_data = df_clean.groupby('symbol_ticker').tail(1)

        # Prepare features
        feature_cols = [
            'close_to_close_vol_10d', 'close_to_close_vol_20d', 'close_to_close_vol_60d',
            'parkinson_vol_10d', 'parkinson_vol_20d',
            'garman_klass_vol_10d', 'garman_klass_vol_20d',
            'yang_zhang_vol_10d', 'yang_zhang_vol_20d',
            'realized_vol_percentile_1y', 'realized_vol_percentile_3y',
            'volatility_of_volatility_20d', 'vol_clustering_index',
            'volatility_acceleration', 'volume_weighted_volatility',
            'abnormal_volume_count_20d', 'gap_frequency_60d',
            'rsi_14', 'atr_14', 'bb_width', 'adx_14',
            'return_1d', 'return_5d', 'return_20d',
            'garch_cond_vol', 'garch_forecast', 'vol_surprise',
            'volatility_trend_num'
        ]

        X_latest = latest_data[feature_cols].values
        X_latest_scaled = self.scaler.transform(X_latest)

        # Predict regime
        regime_pred = self.model.predict(X_latest_scaled)
        regime_proba = self.model.predict_proba(X_latest_scaled)

        # Generate signals
        signals = []

        for idx, (_, row) in enumerate(latest_data.iterrows()):
            predicted_regime = regime_pred[idx]
            confidence = np.max(regime_proba[idx])

            # Only trade high-confidence predictions
            if confidence < 0.6:
                continue

            price = row['current_price']
            atr = row['atr_14']

            # Trading logic based on predicted regime
            if predicted_regime == 2:  # Vol increasing
                signal_type = 'BUY'  # Buy before vol expansion
                strength = confidence
            elif predicted_regime == 0:  # Vol decreasing
                signal_type = 'SELL'  # Sell before vol contraction
                strength = confidence
            else:
                continue  # No signal for stable regime

            signals.append({
                'symbol_ticker': row['symbol_ticker'],
                'signal_date': row['vol_date'],
                'signal_type': signal_type,
                'strength': strength,
                'entry_price': price,
                'stop_loss': price * (1 - 2*atr/price) if signal_type == 'BUY' else price * (1 + 2*atr/price),
                'take_profit': price * (1 + 3*atr/price) if signal_type == 'BUY' else price * (1 - 3*atr/price),
                'metadata': f'{{"predicted_regime": {predicted_regime}, "confidence": {confidence:.3f}, "garch_forecast": {row["garch_forecast"]:.4f}, "model": "RandomForest"}}'
            })

        return pd.DataFrame(signals)


if __name__ == "__main__":
    strategy = VolatilityTradingStrategy()
    signals = strategy.run()
    print(f"Generated {len(signals)} signals")