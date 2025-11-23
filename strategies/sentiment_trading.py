"""
Sentiment Trading Strategy with Incremental Learning
====================================================
Automatically loads previous model or trains from scratch
Uses incremental updates to avoid full retraining
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from base_strategy import BaseStrategy
from incremental_trainer import IncrementalTrainer
import logging

# Suppress FutureWarning for fillna downcasting
pd.set_option('future.no_silent_downcasting', True)

# M1-optimized XGBoost
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


class SentimentTradingStrategy(BaseStrategy):
    """Trade on sentiment-price divergence using XGBoost with incremental learning"""

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db"):
        super().__init__(db_path)
        self.model = None
        self.scaler = StandardScaler()
        self.name = "SentimentTradingStrategy"
        self.trainer = IncrementalTrainer(db_path=db_path)
        self.feature_cols = None

    def _get_historical_data_query(self) -> str:
        """Return the SQL query for loading training data"""
        return """
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
                -- FUNDAMENTALS
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
                -- ANALYST RATINGS
                ar.rating as analyst_rating,
                ar.rating_numeric,
                ar.price_target,
                ar.upside_to_target,
                -- EARNINGS
                ed.reported_eps,
                ed.estimated_eps,
                ed.eps_surprise,
                ed.eps_surprise_percent,
                ed.reported_revenue,
                ed.estimated_revenue,
                -- Revenue surprise
                CASE
                    WHEN ed.estimated_revenue > 0 THEN
                        ((ed.reported_revenue - ed.estimated_revenue) / ed.estimated_revenue) * 100
                    ELSE NULL
                END as revenue_surprise_percent,
                -- INSIDER TRADING
                it.insider_buy_count,
                it.insider_sell_count,
                it.insider_net_sentiment,
                it.total_shares_traded,
                it.insider_ownership_change,
                -- OPTIONS DATA
                od.implied_volatility_30d,
                od.iv_percentile_1y,
                od.put_call_ratio,
                -- ECONOMIC INDICATORS
                ei.vix as market_vix,
                ei.treasury_10y,
                ei.yield_curve_10y_2y
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
                     SUM(CASE WHEN transaction_type = 'sell' THEN 1 ELSE 0 END)) as insider_net_sentiment,
                    SUM(ABS(shares_traded)) as total_shares_traded,
                    SUM(CASE WHEN transaction_type = 'buy' THEN shares_traded
                             WHEN transaction_type = 'sell' THEN -shares_traded
                             ELSE 0 END) as insider_ownership_change
                FROM insider_trading
                WHERE transaction_date >= date('now', '-90 days')
                GROUP BY symbol_ticker
            ) it ON mf.symbol_ticker = it.symbol_ticker
            LEFT JOIN (
                SELECT symbol_ticker, options_date,
                       implied_volatility_30d, iv_percentile_1y, put_call_ratio
                FROM options_data
                WHERE (symbol_ticker, options_date) IN (
                    SELECT symbol_ticker, MAX(options_date)
                    FROM options_data
                    GROUP BY symbol_ticker
                )
            ) od ON mf.symbol_ticker = od.symbol_ticker
            LEFT JOIN (
                SELECT indicator_date, vix, treasury_10y, yield_curve_10y_2y
                FROM economic_indicators
                WHERE indicator_date = (SELECT MAX(indicator_date) FROM economic_indicators)
            ) ei ON 1=1
            WHERE mf.sentiment_score IS NOT NULL
            ORDER BY mf.symbol_ticker, mf.feature_date
        """

    def _get_all_historical_data(self, batch_size: int = 10000) -> pd.DataFrame:
        """Memory-efficient data loading with batching"""
        conn = self._conn()
        query = self._get_historical_data_query()

        chunks = []
        for chunk in pd.read_sql(query, conn, chunksize=batch_size):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        conn.close()
        logger.info(f"Loaded {len(df)} historical records")
        return df

    def _create_balanced_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BALANCED labels using quantile-based classification"""
        df = df.sort_values(['symbol_ticker', 'feature_date'])
        df['future_return_5d'] = df.groupby('symbol_ticker')['current_price'].pct_change(periods=-5)
        df = df.dropna(subset=['future_return_5d'])

        quantiles = df['future_return_5d'].quantile([0.2, 0.4, 0.6, 0.8])
        conditions = [
            df['future_return_5d'] <= quantiles[0.2],
            df['future_return_5d'] <= quantiles[0.4],
            df['future_return_5d'] <= quantiles[0.6],
            df['future_return_5d'] <= quantiles[0.8],
        ]
        choices = [0, 1, 2, 3]
        df['label'] = np.select(conditions, choices, default=4)

        label_dist = df['label'].value_counts().sort_index()
        logger.info("Label distribution:")
        for label, count in label_dist.items():
            label_name = {0:'STRONG SELL', 1:'SELL', 2:'HOLD', 3:'BUY', 4:'STRONG BUY'}[label]
            logger.info(f"  {label_name}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare comprehensive feature matrix (returns feature_cols for consistency)"""
        feature_cols = [
            'sentiment_score', 'sentiment_ma_7d', 'sentiment_price_divergence',
            'return_1d', 'return_5d', 'return_20d',
            'rsi_14', 'macd', 'macd_histogram', 'bb_position',
            'volatility_10d', 'volatility_20d',
            'return_1d_lag1', 'return_1d_lag5', 'rsi_14_lag1',
            'return_1d_norm', 'rsi_14_norm', 'volatility_20d_norm', 'sentiment_score_norm'
        ]

        # Volatility regime
        regime_map = {'low': 0, 'normal': 1, 'high': 2}
        df['volatility_regime_num'] = df['volatility_regime'].map(regime_map).fillna(1)
        feature_cols.append('volatility_regime_num')

        # Fundamentals
        fundamental_features = [
            'pe_ratio', 'price_to_book', 'price_to_sales',
            'profit_margin', 'return_on_equity', 'revenue_growth', 'earnings_growth',
            'debt_to_equity', 'current_ratio', 'beta'
        ]
        for feat in fundamental_features:
            if feat in df.columns:
                df[feat] = df.groupby('symbol_ticker')[feat].ffill()
                feature_cols.append(feat)

        # Analyst ratings
        analyst_features = ['rating_numeric', 'upside_to_target']
        for feat in analyst_features:
            if feat in df.columns:
                df[feat] = df.groupby('symbol_ticker')[feat].ffill()
                feature_cols.append(feat)

        # Earnings
        earnings_features = ['eps_surprise_percent', 'revenue_surprise_percent']
        for feat in earnings_features:
            if feat in df.columns:
                df[feat] = df.groupby('symbol_ticker')[feat].ffill()
                feature_cols.append(feat)

        # Insider trading
        insider_features = [
            'insider_buy_count', 'insider_sell_count', 'insider_net_sentiment',
            'total_shares_traded', 'insider_ownership_change'
        ]
        for feat in insider_features:
            if feat in df.columns:
                df[feat] = df[feat].fillna(0)
                feature_cols.append(feat)

        # Options
        options_features = ['implied_volatility_30d', 'iv_percentile_1y', 'put_call_ratio']
        for feat in options_features:
            if feat in df.columns:
                df[feat] = df.groupby('symbol_ticker')[feat].ffill()
                feature_cols.append(feat)

        # Economic
        economic_features = ['market_vix', 'treasury_10y', 'yield_curve_10y_2y']
        for feat in economic_features:
            if feat in df.columns:
                df[feat] = df[feat].ffill()
                feature_cols.append(feat)

        # Advanced derived features
        if 'eps_surprise_percent' in df.columns and 'revenue_surprise_percent' in df.columns:
            eps_filled = df['eps_surprise_percent'].fillna(0).infer_objects(copy=False)
            rev_filled = df['revenue_surprise_percent'].fillna(0).infer_objects(copy=False)
            df['earnings_momentum'] = eps_filled * 0.6 + rev_filled * 0.4
            feature_cols.append('earnings_momentum')

        if 'insider_net_sentiment' in df.columns and 'total_shares_traded' in df.columns:
            df['insider_conviction'] = (
                df['insider_net_sentiment'] * np.log1p(df['total_shares_traded'].fillna(0))
            )
            feature_cols.append('insider_conviction')

        if 'upside_to_target' in df.columns and 'pe_ratio' in df.columns:
            df['value_opportunity'] = df['upside_to_target'].fillna(0) / (df['pe_ratio'].fillna(15) + 1)
            feature_cols.append('value_opportunity')

        if 'sentiment_score' in df.columns and 'volatility_20d' in df.columns:
            df['risk_adj_sentiment'] = (
                df['sentiment_score'] / (df['volatility_20d'].fillna(0.2) + 0.01)
            )
            feature_cols.append('risk_adj_sentiment')

        if 'put_call_ratio' in df.columns and 'iv_percentile_1y' in df.columns:
            df['options_fear_index'] = (
                df['put_call_ratio'].fillna(1.0) * df['iv_percentile_1y'].fillna(50) / 100
            )
            feature_cols.append('options_fear_index')

        if 'market_vix' in df.columns and 'yield_curve_10y_2y' in df.columns:
            df['macro_risk'] = (
                (df['market_vix'].fillna(20) / 100) *
                (1 - df['yield_curve_10y_2y'].fillna(0.5))
            )
            feature_cols.append('macro_risk')

        if all(feat in df.columns for feat in ['profit_margin', 'revenue_growth', 'debt_to_equity']):
            profit_filled = df['profit_margin'].fillna(0).infer_objects(copy=False)
            growth_filled = df['revenue_growth'].fillna(0).infer_objects(copy=False)
            debt_filled = df['debt_to_equity'].fillna(1).infer_objects(copy=False)
            df['quality_score'] = (
                profit_filled * 0.4 +
                growth_filled * 0.3 -
                (debt_filled / 10) * 0.3
            )
            feature_cols.append('quality_score')

        if 'return_20d' in df.columns and 'quality_score' in df.columns:
            df['momentum_quality'] = df['return_20d'] * df['quality_score']
            feature_cols.append('momentum_quality')

        if 'upside_to_target' in df.columns and 'insider_net_sentiment' in df.columns:
            df['analyst_insider_agree'] = (
                np.sign(df['upside_to_target'].fillna(0)) *
                np.sign(df['insider_net_sentiment'].fillna(0))
            )
            feature_cols.append('analyst_insider_agree')

        if 'implied_volatility_30d' in df.columns and 'market_vix' in df.columns:
            df['relative_iv'] = (
                df['implied_volatility_30d'].fillna(20) / (df['market_vix'].fillna(20) + 1)
            )
            feature_cols.append('relative_iv')

        # Clean
        df_clean = df.dropna(subset=['label'])

        for col in feature_cols:
            if col in df_clean.columns:
                if df_clean[col].notna().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val if not pd.isna(median_val) else 0)
                else:
                    df_clean[col] = 0

        X = df_clean[feature_cols].values
        y = df_clean['label'].values

        logger.info(f"Using {len(feature_cols)} features")
        return X, y, df_clean, feature_cols

    def train_model(self, force_full_retrain: bool = False) -> bool:
        """
        Train model with incremental learning

        Args:
            force_full_retrain: Force complete retrain instead of incremental

        Returns:
            True if training successful
        """
        # Check if full retrain needed
        if force_full_retrain or self.trainer.should_full_retrain(self.name, days_threshold=90):
            logger.info("=" * 80)
            logger.info("FULL RETRAIN: Loading all historical data")
            logger.info("=" * 80)

            # Full training on all data
            df = self._get_all_historical_data()

            if df.empty:
                logger.warning("No data available")
                return False

            df_labeled = self._create_balanced_labels(df)
            X, y, df_clean, feature_cols = self._prepare_features(df_labeled)

            if len(X) == 0:
                logger.warning("No valid features")
                return False

            # Store feature columns
            self.feature_cols = feature_cols

            # Train new model from scratch
            if len(X) < 100:
                logger.warning(f"Not enough samples ({len(X)}) for training")
                return False

            # Calculate class weights
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            sample_weights = np.array([class_weights[np.where(classes == label)[0][0]] for label in y])

            # Split
            X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
            )

            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train XGBoost
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method='hist',
                device='cpu',
                n_jobs=-1,
                enable_categorical=False,
                early_stopping_rounds=20,
                eval_metric='mlogloss',
                random_state=42,
                verbosity=0
            )

            self.model.fit(
                X_train_scaled, y_train,
                sample_weight=sw_train,
                eval_set=[(X_test_scaled, y_test)],
                sample_weight_eval_set=[sw_test],
                verbose=False
            )

            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)

            logger.info(f"XGBoost trained: Train={train_score:.3f}, Test={test_score:.3f}")

            # Save model
            training_start = df_clean['feature_date'].min()
            training_end = df_clean['feature_date'].max()

            self.trainer.save_model(
                strategy_name=self.name,
                model=self.model,
                scaler=self.scaler,
                training_start_date=str(training_start),
                training_end_date=str(training_end),
                num_training_samples=len(X),
                num_new_samples=len(X),
                is_full_retrain=True,
                train_accuracy=train_score,
                test_accuracy=test_score,
                feature_names=feature_cols,
                hyperparameters=self.model.get_params(),
                notes="Full retrain with all historical data"
            )

            return True

        else:
            logger.info("=" * 80)
            logger.info("INCREMENTAL UPDATE: Loading only new data")
            logger.info("=" * 80)

            # Try incremental learning
            old_model, old_scaler, model_info = self.trainer.load_model(self.name)

            if old_model is None:
                logger.warning("No existing model - performing full retrain")
                return self.train_model(force_full_retrain=True)

            # Get new data only
            query = self._get_historical_data_query()
            new_df, should_retrain = self.trainer.get_new_training_data(
                strategy_name=self.name,
                query=query,
                min_samples=100
            )

            if not should_retrain:
                logger.info("Using existing model (insufficient new data)")
                self.model = old_model
                self.scaler = old_scaler
                self.feature_cols = model_info['feature_names']
                return True

            # Prepare new data
            df_labeled = self._create_balanced_labels(new_df)
            X_new, y_new, df_clean, feature_cols = self._prepare_features(df_labeled)

            if len(X_new) == 0:
                logger.warning("No valid new features - using existing model")
                self.model = old_model
                self.scaler = old_scaler
                self.feature_cols = model_info['feature_names']
                return True

            # Store feature columns
            self.feature_cols = feature_cols

            # Scale new data with OLD scaler (important!)
            X_new_scaled = old_scaler.transform(X_new)

            # Incremental training
            updated_model = self.trainer.incremental_train_xgboost(
                old_model=old_model,
                X_new=X_new_scaled,
                y_new=y_new,
                n_estimators_new=50
            )

            # Evaluate on new data
            train_score = updated_model.score(X_new_scaled, y_new)

            logger.info(f"Incremental update complete: Accuracy on new data={train_score:.3f}")

            # Save updated model
            training_start = model_info['training_end_date']  # Continue from last
            training_end = df_clean['feature_date'].max()

            self.trainer.save_model(
                strategy_name=self.name,
                model=updated_model,
                scaler=old_scaler,  # Keep old scaler
                training_start_date=str(training_start),
                training_end_date=str(training_end),
                num_training_samples=model_info['num_training_samples'] + len(X_new),
                num_new_samples=len(X_new),
                is_full_retrain=False,
                train_accuracy=train_score,
                test_accuracy=train_score,  # No separate test set for incremental
                feature_names=feature_cols,
                hyperparameters=updated_model.get_params(),
                notes=f"Incremental update with {len(X_new)} new samples"
            )

            self.model = updated_model
            self.scaler = old_scaler

            return True

    def generate_signals(self) -> pd.DataFrame:
        """Generate signals using trained/loaded model"""
        # Try to load existing model first
        if self.model is None:
            logger.info("No model loaded - attempting to load from disk")
            old_model, old_scaler, model_info = self.trainer.load_model(self.name)

            if old_model is not None:
                self.model = old_model
                self.scaler = old_scaler
                self.feature_cols = model_info['feature_names']
            else:
                logger.info("No existing model found - training new model")
                success = self.train_model(force_full_retrain=True)
                if not success:
                    return pd.DataFrame()

        if self.model is None:
            logger.warning("Model training failed")
            return pd.DataFrame()

        # Get latest data
        df = self._get_all_historical_data()
        if df.empty:
            return pd.DataFrame()

        df_labeled = self._create_balanced_labels(df)
        _, _, df_clean, _ = self._prepare_features(df_labeled)

        # Get latest for each ticker
        latest_data = df_clean.groupby('symbol_ticker').tail(1)

        # Use SAME features from training
        X_latest_list = []
        for _, row in latest_data.iterrows():
            row_features = [row.get(col, 0) for col in self.feature_cols]
            X_latest_list.append(row_features)
        X_latest = np.array(X_latest_list)
        X_latest_scaled = self.scaler.transform(X_latest)

        # Predict
        predictions = self.model.predict(X_latest_scaled)
        probabilities = self.model.predict_proba(X_latest_scaled)
        confidences = np.max(probabilities, axis=1)

        # Generate signals
        high_conf_mask = confidences >= 0.6
        buy_mask = (predictions >= 3) & high_conf_mask
        sell_mask = (predictions <= 1) & high_conf_mask

        symbols = latest_data['symbol_ticker'].values
        dates = latest_data['feature_date'].values
        prices = latest_data['current_price'].values
        vols = latest_data['volatility_20d'].values
        sent_diverg = latest_data['sentiment_price_divergence'].values

        signals = []

        # BUY signals
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

        # SELL signals
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

        logger.info(f"Generated {len(signals)} signals")
        return pd.DataFrame(signals)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    strategy = SentimentTradingStrategy()

    # Train with incremental learning
    strategy.train_model(force_full_retrain=False)

    # Generate signals
    signals = strategy.run()
    print(f"\nGenerated {len(signals)} signals")
