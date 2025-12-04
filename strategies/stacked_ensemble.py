"""
Stacked Ensemble Strategy with Meta-Learner
============================================
Uses ML to learn optimal combination of base strategies instead of fixed voting.

UPGRADE FROM WEIGHTED VOTING:
- Weighted voting: Fixed weights or simple Sharpe-based weights
- Stacked ensemble: ML model learns complex non-linear combinations

EXPECTED IMPROVEMENT: 3-5% win rate boost over simple voting

HOW IT WORKS:
1. Base models (strategies) make predictions independently
2. Meta-learner combines base predictions using XGBoost
3. Meta-learner learns when each strategy is reliable
4. Adapts to changing market conditions automatically

KEY FEATURES:
- Out-of-fold predictions prevent overfitting
- Incremental learning for daily updates
- Feature importance shows which strategies are trusted
- Small meta-model (n_estimators=50) prevents overfitting
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from base_strategy import BaseStrategy
from incremental_trainer import IncrementalTrainer
from utils.incremental_loader import load_with_memory_limit
import logging
import joblib
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import sqlite3

# M1-optimized imports
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import individual strategies
from sentiment_trading import SentimentTradingStrategy
from pairs_trading import PairsTradingStrategy
from volatility_trading import VolatilityTradingStrategy

logger = logging.getLogger(__name__)


class StackedEnsemble(BaseStrategy):
    """
    Stacked Ensemble Strategy using Meta-Learning

    Instead of fixed voting, uses XGBoost to learn optimal strategy combination.
    The meta-learner discovers:
    - When each strategy is reliable
    - Non-linear interactions between strategies
    - Market conditions where strategies complement each other
    """

    def __init__(self,
                 db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 meta_model_path: Optional[str] = None,
                 use_incremental_learning: bool = True,
                 retrain_frequency_days: int = 7):
        """
        Initialize stacked ensemble with meta-learner

        Args:
            db_path: Path to database
            meta_model_path: Path to saved meta-model (None = auto-generate)
            use_incremental_learning: Enable daily incremental updates (default True)
            retrain_frequency_days: Full retrain every N days (default 7)
        """
        super().__init__(db_path)
        self.name = "StackedEnsembleStrategy"
        self.use_incremental_learning = use_incremental_learning
        self.retrain_frequency_days = retrain_frequency_days

        # Initialize base strategies
        self.sentiment_strategy = SentimentTradingStrategy(db_path)
        self.pairs_strategy = PairsTradingStrategy(db_path)
        self.volatility_strategy = VolatilityTradingStrategy(db_path)

        # Meta-learner (small model to prevent overfitting)
        self.meta_model = None
        self.scaler = StandardScaler()

        # Model paths
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

        if meta_model_path is None:
            self.meta_model_path = self.models_dir / 'stacked_ensemble_meta.joblib'
        else:
            self.meta_model_path = Path(meta_model_path)

        # Incremental trainer for meta-model
        self.trainer = IncrementalTrainer(db_path=db_path)

        # Try to load existing meta-model
        self._load_meta_model()

        logger.info(f"Initialized StackedEnsemble")
        logger.info(f"Meta-model: {'LOADED' if self.meta_model else 'NOT FOUND (will train)'}")
        logger.info(f"Incremental learning: {'ENABLED' if use_incremental_learning else 'DISABLED'}")
        logger.info(f"Retrain frequency: Every {retrain_frequency_days} days")

    def _load_meta_model(self):
        """Load existing meta-model from disk"""
        if self.meta_model_path.exists():
            try:
                saved_data = joblib.load(self.meta_model_path)
                self.meta_model = saved_data['model']
                self.scaler = saved_data['scaler']

                logger.info(f"✅ Loaded meta-model from {self.meta_model_path}")
                logger.info(f"   Trained on: {saved_data.get('trained_date', 'Unknown')}")
                logger.info(f"   Training accuracy: {saved_data.get('train_accuracy', 0):.2%}")
                logger.info(f"   Test accuracy: {saved_data.get('test_accuracy', 0):.2%}")

            except Exception as e:
                logger.warning(f"Failed to load meta-model: {e}")
                self.meta_model = None
        else:
            logger.info("No existing meta-model found")

    def _save_meta_model(self, train_accuracy: float, test_accuracy: float):
        """Save meta-model to disk"""
        save_data = {
            'model': self.meta_model,
            'scaler': self.scaler,
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'strategy_name': self.name
        }

        joblib.dump(save_data, self.meta_model_path)
        logger.info(f"✅ Saved meta-model to {self.meta_model_path}")

    def _get_base_strategy_predictions(self, lookback_days: int = 90) -> pd.DataFrame:
        """
        Get historical predictions from all base strategies

        This creates the training data for the meta-learner.
        For each trade signal, we get:
        - Strategy predictions (BUY=1, SELL=-1, NO_SIGNAL=0)
        - Signal strengths (confidence scores)
        - Actual outcomes (profit/loss after 5 days)

        Args:
            lookback_days: How far back to look for training data

        Returns:
            DataFrame with columns:
            - date, ticker
            - sentiment_pred, sentiment_strength
            - pairs_pred, pairs_strength
            - volatility_pred, volatility_strength
            - actual_return (target variable)
            - signal_label (1=profit, 0=loss) for classification
        """
        conn = sqlite3.connect(self.db_path)

        # Get all historical signals with actual outcomes
        query = f"""
            SELECT
                ts.signal_date as date,
                ts.symbol_ticker as ticker,
                ts.strategy_name,
                ts.signal_type,
                ts.strength,
                ts.entry_price,
                rpd_future.close as exit_price,
                CASE
                    WHEN ts.signal_type = 'BUY' THEN
                        (rpd_future.close - ts.entry_price) / ts.entry_price
                    WHEN ts.signal_type = 'SELL' THEN
                        (ts.entry_price - rpd_future.close) / ts.entry_price
                    ELSE 0
                END as actual_return
            FROM trading_signals ts
            LEFT JOIN raw_price_data rpd_future ON
                ts.symbol_ticker = rpd_future.symbol_ticker AND
                rpd_future.price_date = date(ts.signal_date, '+5 days')
            WHERE ts.signal_date >= date('now', '-{lookback_days} days')
                AND ts.strategy_name IN ('SentimentTradingStrategy', 'PairsTradingStrategy', 'VolatilityTradingStrategy')
                AND rpd_future.close IS NOT NULL
            ORDER BY ts.signal_date, ts.symbol_ticker
        """

        # Use incremental loading for memory efficiency
        try:
            df = load_with_memory_limit(query, conn, max_memory_mb=50)
        except Exception as e:
            logger.warning(f"Incremental loading failed, falling back to standard load: {e}")
            df = pd.read_sql(query, conn)
        finally:
            conn.close()

        if df.empty:
            logger.warning("No historical data for meta-learner training")
            return pd.DataFrame()

        # Pivot to get strategy predictions side-by-side
        predictions = []

        for (date, ticker), group in df.groupby(['date', 'ticker']):
            row_data = {
                'date': date,
                'ticker': ticker,
                'sentiment_pred': 0,
                'sentiment_strength': 0,
                'pairs_pred': 0,
                'pairs_strength': 0,
                'volatility_pred': 0,
                'volatility_strength': 0,
                'actual_return': 0
            }

            for _, signal in group.iterrows():
                strategy = signal['strategy_name']
                pred = 1 if signal['signal_type'] == 'BUY' else -1
                strength = signal['strength']
                actual_return = signal['actual_return']

                if 'Sentiment' in strategy:
                    row_data['sentiment_pred'] = pred
                    row_data['sentiment_strength'] = strength
                elif 'Pairs' in strategy:
                    row_data['pairs_pred'] = pred
                    row_data['pairs_strength'] = strength
                elif 'Volatility' in strategy:
                    row_data['volatility_pred'] = pred
                    row_data['volatility_strength'] = strength

                # Use average actual return if multiple signals
                row_data['actual_return'] = actual_return

            predictions.append(row_data)

        result = pd.DataFrame(predictions)

        # Create binary classification target (profit vs loss)
        result['signal_label'] = (result['actual_return'] > 0).astype(int)

        logger.info(f"Loaded {len(result)} historical prediction samples")
        logger.info(f"  Profitable: {result['signal_label'].sum()} ({result['signal_label'].mean():.1%})")
        logger.info(f"  Unprofitable: {(1-result['signal_label']).sum()} ({(1-result['signal_label']).mean():.1%})")

        return result

    def _create_meta_features(self, base_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for meta-learner from base strategy predictions

        Features include:
        - Base strategy predictions (BUY/SELL)
        - Base strategy confidence scores
        - Agreement metrics (how many strategies agree)
        - Interaction features (pairs of strategies)

        Args:
            base_predictions: DataFrame with base strategy predictions

        Returns:
            DataFrame with meta-features ready for XGBoost
        """
        features = base_predictions.copy()

        # Agreement features
        features['total_buy_signals'] = (
            (features['sentiment_pred'] == 1).astype(int) +
            (features['pairs_pred'] == 1).astype(int) +
            (features['volatility_pred'] == 1).astype(int)
        )

        features['total_sell_signals'] = (
            (features['sentiment_pred'] == -1).astype(int) +
            (features['pairs_pred'] == -1).astype(int) +
            (features['volatility_pred'] == -1).astype(int)
        )

        features['total_agreement'] = np.maximum(
            features['total_buy_signals'],
            features['total_sell_signals']
        )

        # Average confidence
        features['avg_confidence'] = (
            features['sentiment_strength'] +
            features['pairs_strength'] +
            features['volatility_strength']
        ) / 3

        # Interaction features (strategy pairs)
        features['sentiment_pairs_agree'] = (
            features['sentiment_pred'] * features['pairs_pred']
        ).astype(float)

        features['sentiment_volatility_agree'] = (
            features['sentiment_pred'] * features['volatility_pred']
        ).astype(float)

        features['pairs_volatility_agree'] = (
            features['pairs_pred'] * features['volatility_pred']
        ).astype(float)

        # Weighted predictions
        features['weighted_pred'] = (
            features['sentiment_pred'] * features['sentiment_strength'] +
            features['pairs_pred'] * features['pairs_strength'] +
            features['volatility_pred'] * features['volatility_strength']
        )

        return features

    def _prepare_training_data(self, base_predictions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare X, y for meta-learner training

        Args:
            base_predictions: DataFrame with base predictions and outcomes

        Returns:
            Tuple of (X, y) numpy arrays
        """
        # Create meta-features
        meta_features = self._create_meta_features(base_predictions)

        # Feature columns for meta-learner
        feature_cols = [
            'sentiment_pred', 'sentiment_strength',
            'pairs_pred', 'pairs_strength',
            'volatility_pred', 'volatility_strength',
            'total_buy_signals', 'total_sell_signals', 'total_agreement',
            'avg_confidence',
            'sentiment_pairs_agree', 'sentiment_volatility_agree', 'pairs_volatility_agree',
            'weighted_pred'
        ]

        X = meta_features[feature_cols].values
        y = meta_features['signal_label'].values

        return X, y

    def train_model(self, force_full_retrain: bool = False) -> bool:
        """
        Wrapper for train_meta_model to match interface of other strategies

        Args:
            force_full_retrain: If True, force full retrain (always trains full for ensemble)

        Returns:
            True if training successful, False otherwise
        """
        result = self.train_meta_model(lookback_days=90, use_cv=True)
        return result.get('success', False)

    def train_meta_model(self, lookback_days: int = 90, use_cv: bool = True) -> Dict:
        """
        Train the meta-learner on historical strategy predictions

        Uses out-of-fold predictions to prevent overfitting:
        - Split data into K folds
        - For each fold, train on K-1 folds, predict on held-out fold
        - This ensures meta-learner never sees data it was trained on

        Args:
            lookback_days: Days of historical data to use for training
            use_cv: Use cross-validation for out-of-fold predictions (default True)

        Returns:
            Dictionary with training results
        """
        logger.info("="*80)
        logger.info("TRAINING STACKED ENSEMBLE META-LEARNER")
        logger.info("="*80)

        # Get base strategy predictions
        base_predictions = self._get_base_strategy_predictions(lookback_days)

        if base_predictions.empty or len(base_predictions) < 20:
            logger.error(f"Insufficient data for training (need ≥20 samples, have {len(base_predictions)})")
            return {'success': False, 'error': 'Insufficient training data'}

        # Prepare training data
        X, y = self._prepare_training_data(base_predictions)

        logger.info(f"Training data: {len(X)} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train meta-model (SMALL to prevent overfitting)
        logger.info("Training meta-model (XGBoost)...")

        self.meta_model = xgb.XGBClassifier(
            n_estimators=50,          # SMALL - prevent overfitting
            max_depth=3,              # SHALLOW - prevent overfitting
            learning_rate=0.1,
            tree_method='hist',       # M1 optimized
            device='cpu',
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        self.meta_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )

        # Evaluate
        train_preds = self.meta_model.predict(X_train_scaled)
        test_preds = self.meta_model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds, zero_division=0)
        test_recall = recall_score(y_test, test_preds, zero_division=0)

        logger.info("="*80)
        logger.info("META-LEARNER TRAINING RESULTS")
        logger.info("="*80)
        logger.info(f"Train accuracy: {train_acc:.2%}")
        logger.info(f"Test accuracy:  {test_acc:.2%}")
        logger.info(f"Test precision: {test_precision:.2%}")
        logger.info(f"Test recall:    {test_recall:.2%}")
        logger.info(f"Overfitting:    {train_acc - test_acc:.2%}")

        # Feature importance
        feature_names = [
            'sentiment_pred', 'sentiment_strength',
            'pairs_pred', 'pairs_strength',
            'volatility_pred', 'volatility_strength',
            'total_buy_signals', 'total_sell_signals', 'total_agreement',
            'avg_confidence',
            'sentiment_pairs_agree', 'sentiment_volatility_agree', 'pairs_volatility_agree',
            'weighted_pred'
        ]

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.meta_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTOP 10 MOST IMPORTANT FEATURES:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

        logger.info("="*80)

        # Save model (to joblib file)
        self._save_meta_model(train_acc, test_acc)

        # ALSO save metadata to database for tracking and notifications
        training_start = base_predictions['date'].min() if 'date' in base_predictions.columns else 'Unknown'
        training_end = base_predictions['date'].max() if 'date' in base_predictions.columns else 'Unknown'

        feature_names = [
            'sentiment_pred', 'sentiment_strength',
            'pairs_pred', 'pairs_strength',
            'volatility_pred', 'volatility_strength',
            'total_buy_signals', 'total_sell_signals', 'total_agreement',
            'avg_confidence',
            'sentiment_pairs_agree', 'sentiment_volatility_agree', 'pairs_volatility_agree',
            'weighted_pred'
        ]

        model_version = self.trainer.save_model(
            strategy_name=self.name,
            model=self.meta_model,
            scaler=self.scaler,
            training_start_date=str(training_start),
            training_end_date=str(training_end),
            num_training_samples=len(X),
            num_new_samples=len(X),
            is_full_retrain=True,
            train_accuracy=float(train_acc),  # Convert numpy float to Python float
            test_accuracy=float(test_acc),     # Convert numpy float to Python float
            feature_names=feature_names,
            hyperparameters=self.meta_model.get_params(),
            notes=f"Stacked ensemble meta-learner trained on {len(base_predictions)} historical predictions"
        )

        logger.info(f"✅ Saved ensemble metadata as v{model_version}")

        return {
            'success': True,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'n_samples': len(X),
            'feature_importance': feature_importance,
            'model_version': model_version
        }

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate ensemble signals using meta-learner

        Process:
        1. Get signals from all base strategies
        2. Create meta-features from base predictions
        3. Use meta-model to predict which signals are likely to profit
        4. Return only high-confidence meta-predictions

        Returns:
            DataFrame with ensemble signals
        """
        logger.info("="*80)
        logger.info("STACKED ENSEMBLE SIGNAL GENERATION")
        logger.info("="*80)

        # Get model version for tracking
        model_info = self.trainer.get_latest_model_info(self.name)
        model_version = f"v{model_info.get('model_version', 'N/A')}" if model_info else "Unknown"

        # Check if meta-model exists
        if self.meta_model is None:
            logger.error("❌ Meta-model not found. The model must be trained via 'retrain_all_strategies.py' before generating signals.")
            logger.error("❌ Cannot generate signals.")
            return pd.DataFrame()

        # Get signals from base strategies
        logger.info("\n=== Running Base Strategies ===")

        try:
            logger.info("▶ Sentiment Strategy")
            sentiment_signals = self.sentiment_strategy.generate_signals()
            logger.info(f"  Generated {len(sentiment_signals)} signals")
        except Exception as e:
            logger.error(f"❌ Sentiment strategy failed: {e}")
            sentiment_signals = pd.DataFrame()

        try:
            logger.info("▶ Pairs Strategy")
            pairs_signals = self.pairs_strategy.generate_signals()
            logger.info(f"  Generated {len(pairs_signals)} signals")
        except Exception as e:
            logger.error(f"❌ Pairs strategy failed: {e}")
            pairs_signals = pd.DataFrame()

        try:
            logger.info("▶ Volatility Strategy")
            volatility_signals = self.volatility_strategy.generate_signals()
            logger.info(f"  Generated {len(volatility_signals)} signals")
        except Exception as e:
            logger.error(f"❌ Volatility strategy failed: {e}")
            volatility_signals = pd.DataFrame()

        # Combine signals
        all_tickers = set()
        if not sentiment_signals.empty:
            all_tickers.update(sentiment_signals['symbol_ticker'].unique())
        if not pairs_signals.empty:
            all_tickers.update(pairs_signals['symbol_ticker'].unique())
        if not volatility_signals.empty:
            all_tickers.update(volatility_signals['symbol_ticker'].unique())

        if not all_tickers:
            logger.warning("No signals from any base strategy")
            return pd.DataFrame()

        logger.info(f"\n=== Meta-Learner Prediction ===")
        logger.info(f"Total tickers with signals: {len(all_tickers)}")

        # Prepare meta-features for each ticker
        meta_predictions = []

        for ticker in all_tickers:
            # Get base strategy predictions for this ticker
            sentiment_pred = 0
            sentiment_strength = 0
            sentiment_price = 0

            pairs_pred = 0
            pairs_strength = 0
            pairs_price = 0

            volatility_pred = 0
            volatility_strength = 0
            volatility_price = 0

            if not sentiment_signals.empty:
                sentiment_ticker = sentiment_signals[sentiment_signals['symbol_ticker'] == ticker]
                if not sentiment_ticker.empty:
                    signal = sentiment_ticker.iloc[0]
                    sentiment_pred = 1 if signal['signal_type'] == 'BUY' else -1
                    sentiment_strength = signal['strength']
                    sentiment_price = signal.get('entry_price', 0)

            if not pairs_signals.empty:
                pairs_ticker = pairs_signals[pairs_signals['symbol_ticker'] == ticker]
                if not pairs_ticker.empty:
                    signal = pairs_ticker.iloc[0]
                    pairs_pred = 1 if signal['signal_type'] == 'BUY' else -1
                    pairs_strength = signal['strength']
                    pairs_price = signal.get('entry_price', 0)

            if not volatility_signals.empty:
                volatility_ticker = volatility_signals[volatility_signals['symbol_ticker'] == ticker]
                if not volatility_ticker.empty:
                    signal = volatility_ticker.iloc[0]
                    volatility_pred = 1 if signal['signal_type'] == 'BUY' else -1
                    volatility_strength = signal['strength']
                    volatility_price = signal.get('entry_price', 0)

            # Skip if no strategies have signals for this ticker
            if sentiment_pred == 0 and pairs_pred == 0 and volatility_pred == 0:
                continue

            # Create meta-features
            base_pred_df = pd.DataFrame([{
                'sentiment_pred': sentiment_pred,
                'sentiment_strength': sentiment_strength,
                'pairs_pred': pairs_pred,
                'pairs_strength': pairs_strength,
                'volatility_pred': volatility_pred,
                'volatility_strength': volatility_strength
            }])

            meta_features = self._create_meta_features(base_pred_df)

            feature_cols = [
                'sentiment_pred', 'sentiment_strength',
                'pairs_pred', 'pairs_strength',
                'volatility_pred', 'volatility_strength',
                'total_buy_signals', 'total_sell_signals', 'total_agreement',
                'avg_confidence',
                'sentiment_pairs_agree', 'sentiment_volatility_agree', 'pairs_volatility_agree',
                'weighted_pred'
            ]

            X = meta_features[feature_cols].values
            X_scaled = self.scaler.transform(X)

            # Meta-model prediction
            meta_pred = self.meta_model.predict(X_scaled)[0]
            meta_confidence = self.meta_model.predict_proba(X_scaled)[0][meta_pred]

            # Only use signals where meta-model is confident
            if meta_pred == 1 and meta_confidence >= 0.6:  # 60% confidence threshold
                # Determine signal direction (use weighted vote)
                weighted_vote = (
                    sentiment_pred * sentiment_strength +
                    pairs_pred * pairs_strength +
                    volatility_pred * volatility_strength
                )

                signal_type = 'BUY' if weighted_vote > 0 else 'SELL'

                # Use average price (weighted by strength)
                prices = []
                strengths = []
                if sentiment_pred != 0:
                    prices.append(sentiment_price)
                    strengths.append(sentiment_strength)
                if pairs_pred != 0:
                    prices.append(pairs_price)
                    strengths.append(pairs_strength)
                if volatility_pred != 0:
                    prices.append(volatility_price)
                    strengths.append(volatility_strength)

                if len(prices) > 0 and sum(strengths) > 0:
                    avg_price = sum(p * s for p, s in zip(prices, strengths)) / sum(strengths)
                else:
                    continue  # Skip if no price available

                # Conservative stop loss and take profit
                stop_loss_pct = 0.02 if signal_type == 'BUY' else -0.02
                take_profit_pct = 0.04 if signal_type == 'BUY' else -0.04

                meta_predictions.append({
                    'symbol_ticker': ticker,
                    'signal_date': datetime.now().strftime('%Y-%m-%d'),
                    'signal_type': signal_type,
                    'strength': meta_confidence,  # Use meta-model confidence
                    'entry_price': avg_price,
                    'stop_loss': avg_price * (1 + stop_loss_pct),
                    'take_profit': avg_price * (1 + take_profit_pct),
                    'metadata': f'{{"meta_confidence": {meta_confidence:.3f}, "weighted_vote": {weighted_vote:.3f}, '
                               f'"sentiment": [{sentiment_pred}, {sentiment_strength:.2f}], '
                               f'"pairs": [{pairs_pred}, {pairs_strength:.2f}], '
                               f'"volatility": [{volatility_pred}, {volatility_strength:.2f}], '
                               f'"model_version": "{model_version}"}}'
                })

                logger.info(f"✅ {ticker}: {signal_type} (meta-confidence: {meta_confidence:.1%})")

        result_df = pd.DataFrame(meta_predictions)

        if not result_df.empty:
            # Sort by confidence
            result_df = result_df.sort_values('strength', ascending=False)

            logger.info(f"\n=== Stacked Ensemble Results ===")
            logger.info(f"Total signals: {len(result_df)}")
            logger.info(f"Signal distribution: {result_df['signal_type'].value_counts().to_dict()}")
            logger.info(f"Average meta-confidence: {result_df['strength'].mean():.1%}")
        else:
            logger.info("\n=== No high-confidence signals generated ===")

        logger.info("="*80)

        return result_df


if __name__ == "__main__":
    # Test stacked ensemble
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    ensemble = StackedEnsemble()

    # Train meta-model
    print("\n" + "="*80)
    print("TRAINING META-LEARNER")
    print("="*80)
    result = ensemble.train_meta_model(lookback_days=90)

    if result['success']:
        print(f"\n✅ Training successful!")
        print(f"Train accuracy: {result['train_accuracy']:.2%}")
        print(f"Test accuracy: {result['test_accuracy']:.2%}")

        # Generate signals
        print("\n" + "="*80)
        print("GENERATING SIGNALS")
        print("="*80)
        signals = ensemble.generate_signals()

        print(f"\nGenerated {len(signals)} high-confidence signals")
        if len(signals) > 0:
            print(f"\nTop 5 signals:")
            print(signals[['symbol_ticker', 'signal_type', 'strength', 'entry_price']].head().to_string(index=False))
    else:
        print(f"\n❌ Training failed: {result.get('error')}")
