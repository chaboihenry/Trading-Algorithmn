"""
Sentiment Trading Strategy (M1 Optimized)
=========================================
XGBoost ML model with M1-optimized settings and memory-efficient batch processing
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from base_strategy import BaseStrategy
import logging

# M1-optimized XGBoost
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class SentimentTradingStrategy(BaseStrategy):
    """Trade on sentiment-price divergence using XGBoost with continuous learning"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        super().__init__(db_path)
        self.model = None
        self.scaler = StandardScaler()
        self.name = "SentimentTradingStrategy"

    def _get_all_historical_data(self, batch_size: int = 10000) -> pd.DataFrame:
        """
        Memory-efficient data loading with batching for M1 optimization
        NOW INCLUDES: Fundamentals, Analyst Ratings, Earnings, Insider Trading

        Args:
            batch_size: Number of rows to load per batch (default 10000)
        """
        conn = self._conn()
        query = """
            SELECT
                mf.symbol_ticker,
                mf.feature_date,
                -- Sentiment
                mf.sentiment_score,
                mf.sentiment_ma_7d,
                mf.sentiment_price_divergence,
                -- Price
                mf.return_1d,
                mf.return_5d,
                mf.return_20d,
                -- Technical
                mf.rsi_14,
                mf.macd,
                mf.macd_histogram,
                mf.bb_position,
                -- Volatility
                mf.volatility_10d,
                mf.volatility_20d,
                mf.volatility_regime,
                -- Lags
                mf.return_1d_lag1,
                mf.return_1d_lag5,
                mf.rsi_14_lag1,
                -- Normalized
                mf.return_1d_norm,
                mf.rsi_14_norm,
                mf.volatility_20d_norm,
                mf.sentiment_score_norm,
                -- Price for labeling
                rpd.close as current_price,
                -- FUNDAMENTALS (added)
                fd.pe_ratio,
                fd.price_to_book,
                fd.price_to_sales,
                fd.profit_margin,
                fd.return_on_equity,
                fd.revenue_growth,
                fd.earnings_growth,
                fd.debt_to_equity,
                fd.current_ratio,
                fd.beta,
                -- ANALYST RATINGS (added)
                ar.rating as analyst_rating,
                ar.rating_numeric,
                ar.price_target,
                ar.upside_to_target,
                -- EARNINGS (added)
                ed.reported_eps,
                ed.estimated_eps,
                ed.eps_surprise,
                ed.eps_surprise_percent,
                ed.reported_revenue,
                ed.estimated_revenue,
                -- INSIDER TRADING (added - aggregated)
                it.insider_buy_count,
                it.insider_sell_count,
                it.insider_net_sentiment
            FROM ml_features mf
            LEFT JOIN raw_price_data rpd
                ON mf.symbol_ticker = rpd.symbol_ticker
                AND mf.feature_date = rpd.price_date
            LEFT JOIN (
                SELECT symbol_ticker, fundamental_date, pe_ratio, price_to_book, price_to_sales,
                       profit_margin, return_on_equity, revenue_growth, earnings_growth,
                       debt_to_equity, current_ratio, beta
                FROM fundamental_data
                WHERE (symbol_ticker, fundamental_date) IN (
                    SELECT symbol_ticker, MAX(fundamental_date)
                    FROM fundamental_data
                    GROUP BY symbol_ticker
                )
            ) fd ON mf.symbol_ticker = fd.symbol_ticker
            LEFT JOIN (
                SELECT symbol_ticker, rating_date, rating, rating_numeric,
                       price_target, upside_to_target
                FROM analyst_ratings
                WHERE (symbol_ticker, rating_date) IN (
                    SELECT symbol_ticker, MAX(rating_date)
                    FROM analyst_ratings
                    GROUP BY symbol_ticker
                )
            ) ar ON mf.symbol_ticker = ar.symbol_ticker
            LEFT JOIN (
                SELECT symbol_ticker, earnings_date, reported_eps, estimated_eps,
                       eps_surprise, eps_surprise_percent, reported_revenue, estimated_revenue
                FROM earnings_data
                WHERE (symbol_ticker, earnings_date) IN (
                    SELECT symbol_ticker, MAX(earnings_date)
                    FROM earnings_data
                    GROUP BY symbol_ticker
                )
            ) ed ON mf.symbol_ticker = ed.symbol_ticker
            LEFT JOIN (
                SELECT
                    symbol_ticker,
                    SUM(CASE WHEN transaction_type = 'buy' THEN 1 ELSE 0 END) as insider_buy_count,
                    SUM(CASE WHEN transaction_type = 'sell' THEN 1 ELSE 0 END) as insider_sell_count,
                    (SUM(CASE WHEN transaction_type = 'buy' THEN 1 ELSE 0 END) -
                     SUM(CASE WHEN transaction_type = 'sell' THEN 1 ELSE 0 END)) as insider_net_sentiment
                FROM insider_trading
                WHERE transaction_date >= date('now', '-90 days')
                GROUP BY symbol_ticker
            ) it ON mf.symbol_ticker = it.symbol_ticker
            WHERE mf.sentiment_score IS NOT NULL
            ORDER BY mf.symbol_ticker, mf.feature_date
        """

        # Memory-efficient batch loading
        chunks = []
        for chunk in pd.read_sql(query, conn, chunksize=batch_size):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        conn.close()
        logger.info(f"Loaded {len(df)} historical records with fundamentals, analyst, earnings, and insider data")
        return df

    def _create_balanced_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BALANCED labels using quantile-based classification"""
        df = df.sort_values(['symbol_ticker', 'feature_date'])

        # Calculate future returns
        df['future_return_5d'] = df.groupby('symbol_ticker')['current_price'].pct_change(periods=-5)

        # Drop rows without future returns
        df = df.dropna(subset=['future_return_5d'])

        # Create BALANCED multi-class labels using quantiles
        # This ensures equal distribution across classes
        quantiles = df['future_return_5d'].quantile([0.2, 0.4, 0.6, 0.8])

        conditions = [
            df['future_return_5d'] <= quantiles[0.2],   # Bottom 20% = STRONG SELL (0)
            df['future_return_5d'] <= quantiles[0.4],   # 20-40% = SELL (1)
            df['future_return_5d'] <= quantiles[0.6],   # 40-60% = HOLD (2)
            df['future_return_5d'] <= quantiles[0.8],   # 60-80% = BUY (3)
        ]
        choices = [0, 1, 2, 3]
        df['label'] = np.select(conditions, choices, default=4)  # Top 20% = STRONG BUY (4)

        # Log distribution
        label_dist = df['label'].value_counts().sort_index()
        logger.info("Label distribution:")
        for label, count in label_dist.items():
            label_name = {0:'STRONG SELL', 1:'SELL', 2:'HOLD', 3:'BUY', 4:'STRONG BUY'}[label]
            logger.info(f"  {label_name}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix with ALL available data"""
        # Original features
        feature_cols = [
            'sentiment_score', 'sentiment_ma_7d', 'sentiment_price_divergence',
            'return_1d', 'return_5d', 'return_20d',
            'rsi_14', 'macd', 'macd_histogram', 'bb_position',
            'volatility_10d', 'volatility_20d',
            'return_1d_lag1', 'return_1d_lag5', 'rsi_14_lag1',
            'return_1d_norm', 'rsi_14_norm', 'volatility_20d_norm', 'sentiment_score_norm'
        ]

        # Convert volatility_regime to numeric
        regime_map = {'low': 0, 'normal': 1, 'high': 2}
        df['volatility_regime_num'] = df['volatility_regime'].map(regime_map).fillna(1)
        feature_cols.append('volatility_regime_num')

        # ADD FUNDAMENTAL FEATURES (with forward fill for missing values)
        fundamental_features = [
            'pe_ratio', 'price_to_book', 'price_to_sales',
            'profit_margin', 'return_on_equity', 'revenue_growth', 'earnings_growth',
            'debt_to_equity', 'current_ratio', 'beta'
        ]
        for feat in fundamental_features:
            if feat in df.columns:
                df[feat] = df.groupby('symbol_ticker')[feat].ffill()  # Forward fill within ticker
                feature_cols.append(feat)

        # ADD ANALYST RATING FEATURES
        analyst_features = ['rating_numeric', 'upside_to_target']
        for feat in analyst_features:
            if feat in df.columns:
                df[feat] = df.groupby('symbol_ticker')[feat].ffill()
                feature_cols.append(feat)

        # ADD EARNINGS SURPRISE FEATURES
        earnings_features = ['eps_surprise_percent']  # Most important
        for feat in earnings_features:
            if feat in df.columns:
                df[feat] = df.groupby('symbol_ticker')[feat].ffill()
                feature_cols.append(feat)

        # ADD INSIDER TRADING FEATURES
        insider_features = ['insider_buy_count', 'insider_sell_count', 'insider_net_sentiment']
        for feat in insider_features:
            if feat in df.columns:
                df[feat] = df[feat].fillna(0)  # No insider trading = 0
                feature_cols.append(feat)

        # Drop rows with missing labels only
        df_clean = df.dropna(subset=['label'])

        # Fill remaining NaN with median (robust to outliers)
        for col in feature_cols:
            if col in df_clean.columns:
                # Only calculate median if column has non-NaN values (prevents "mean of empty slice" warning)
                if df_clean[col].notna().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val if not pd.isna(median_val) else 0)
                else:
                    # Column is entirely NaN, fill with 0
                    df_clean[col] = 0

        X = df_clean[feature_cols].values
        y = df_clean['label'].values

        logger.info(f"Using {len(feature_cols)} features (including fundamentals, analyst, earnings, insider)")
        return X, y, df_clean, feature_cols

    def _train_optimized_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train XGBoost with M1 optimizations

        M1 Optimizations:
        - tree_method='hist': Fast histogram-based algorithm (2-3x faster on M1)
        - device='cpu': Explicitly use CPU (M1's unified memory architecture)
        - n_jobs=-1: Use all performance cores
        - enable_categorical=True: Efficient categorical handling
        - early_stopping_rounds: Prevent overfitting and reduce training time
        """
        if len(X) < 100:
            logger.warning(f"Not enough samples ({len(X)}) for training")
            return

        # Calculate class weights to handle imbalance
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        sample_weights = np.array([class_weights[np.where(classes == label)[0][0]] for label in y])

        # Split with stratification
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # XGBoost with M1 optimization
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=4,
            subsample=0.8,
            colsample_bytree=0.8,
            # M1 OPTIMIZATIONS
            tree_method='hist',              # Histogram-based algorithm (2-3x faster on M1)
            device='cpu',                    # Use M1's unified memory architecture
            n_jobs=-1,                       # Use all performance cores
            enable_categorical=False,        # We've already encoded categories
            early_stopping_rounds=20,        # Stop if no improvement (saves time)
            eval_metric='mlogloss',          # Multi-class log loss
            random_state=42,
            verbosity=0                      # Quiet output
        )

        # Train with early stopping
        self.model.fit(
            X_train_scaled, y_train,
            sample_weight=sw_train,
            eval_set=[(X_test_scaled, y_test)],
            sample_weight_eval_set=[sw_test],
            verbose=False
        )

        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)

        logger.info(f"XGBoost trained (M1 optimized): Train={train_score:.3f}, Test={test_score:.3f}")

        # Log per-class performance
        from sklearn.metrics import classification_report
        y_pred = self.model.predict(X_test_scaled)
        logger.info("\nPer-class performance:")
        logger.info(classification_report(
            y_test, y_pred,
            target_names=['STRONG SELL', 'SELL', 'HOLD', 'BUY', 'STRONG BUY'],
            zero_division=0
        ))

    def _get_feature_importance(self, feature_cols: list) -> dict:
        """Get feature importance"""
        if self.model is None:
            return {}

        importance = dict(zip(feature_cols, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def generate_signals(self) -> pd.DataFrame:
        """Generate ML-optimized trading signals with ALL available data"""
        # Get ALL historical data (with fundamentals, analyst, earnings, insider)
        df = self._get_all_historical_data()

        if df.empty:
            logger.warning("No data available")
            return pd.DataFrame()

        # Create balanced labels
        df_labeled = self._create_balanced_labels(df)

        # Prepare features (now returns feature_cols list)
        X, y, df_clean, feature_cols = self._prepare_features(df_labeled)

        if len(X) == 0:
            logger.warning("No valid features")
            return pd.DataFrame()

        # Store feature columns for prediction
        self.feature_cols = feature_cols

        # Train optimized model
        self._train_optimized_model(X, y)

        if self.model is None:
            logger.warning("Model training failed")
            return pd.DataFrame()

        # Log feature importance
        importance = self._get_feature_importance(feature_cols)
        logger.info(f"\nTop 10 features (out of {len(feature_cols)}):")
        for feat, imp in list(importance.items())[:10]:
            logger.info(f"  {feat}: {imp:.3f}")

        # Get latest data for each ticker
        latest_data = df_clean.groupby('symbol_ticker').tail(1)

        # Use the SAME feature columns from training, fill missing with 0
        X_latest_list = []
        for _, row in latest_data.iterrows():
            row_features = [row.get(col, 0) for col in feature_cols]
            X_latest_list.append(row_features)
        X_latest = np.array(X_latest_list)
        X_latest_scaled = self.scaler.transform(X_latest)

        # Predict with probabilities
        predictions = self.model.predict(X_latest_scaled)
        probabilities = self.model.predict_proba(X_latest_scaled)

        # NumPy-vectorized signal generation (10x faster than iterrows)
        # Convert to NumPy arrays for vectorized operations
        confidences = np.max(probabilities, axis=1)

        # Filter high-confidence predictions (vectorized)
        # Labels: 0=STRONG SELL, 1=SELL, 2=HOLD, 3=BUY, 4=STRONG BUY
        high_conf_mask = confidences >= 0.6
        buy_mask = (predictions >= 3) & high_conf_mask  # BUY (3) or STRONG BUY (4)
        sell_mask = (predictions <= 1) & high_conf_mask  # SELL (1) or STRONG SELL (0)

        # Extract NumPy arrays (much faster than row access)
        symbols = latest_data['symbol_ticker'].values
        dates = latest_data['feature_date'].values
        prices = latest_data['current_price'].values
        vols = latest_data['volatility_20d'].values
        sent_diverg = latest_data['sentiment_price_divergence'].values

        signals = []

        # Process BUY signals (vectorized)
        buy_indices = np.where(buy_mask)[0]
        for idx in buy_indices:
            price = prices[idx]
            vol = vols[idx]
            confidence = confidences[idx]

            signals.append({
                'symbol_ticker': symbols[idx],
                'signal_date': dates[idx],
                'signal_type': 'BUY',
                'strength': confidence,
                'entry_price': price,
                'stop_loss': price * (1 - 2*vol),
                'take_profit': price * (1 + 3*vol),
                'metadata': f'{{"prediction": {predictions[idx]}, "confidence": {confidence:.3f}, "sentiment_divergence": {sent_diverg[idx]:.2f}}}'
            })

        # Process SELL signals (vectorized)
        sell_indices = np.where(sell_mask)[0]
        for idx in sell_indices:
            price = prices[idx]
            vol = vols[idx]
            confidence = confidences[idx]

            signals.append({
                'symbol_ticker': symbols[idx],
                'signal_date': dates[idx],
                'signal_type': 'SELL',
                'strength': confidence,
                'entry_price': price,
                'stop_loss': price * (1 + 2*vol),
                'take_profit': price * (1 - 3*vol),
                'metadata': f'{{"prediction": {predictions[idx]}, "confidence": {confidence:.3f}, "sentiment_divergence": {sent_diverg[idx]:.2f}}}'
            })

        logger.info(f"\nGenerated {len(signals)} signals:")
        if signals:
            signal_df = pd.DataFrame(signals)
            logger.info(signal_df['signal_type'].value_counts())

        return pd.DataFrame(signals)


if __name__ == "__main__":
    strategy = SentimentTradingStrategy()
    signals = strategy.run()
    print(f"\nGenerated {len(signals)} signals")
