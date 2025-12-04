"""
Incremental ML Model Training System

Efficiently retrains ML models using:
1. Model persistence (save/load weights)
2. Incremental learning on new data only
3. Warm start from previous model
4. Automatic model versioning
5. Performance tracking over time

This avoids retraining on the entire dataset which would be:
- Time consuming (hours vs minutes)
- Memory intensive
- Inefficient (re-learning old patterns)
"""

import pickle
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """Manages incremental training of ML models"""

    def __init__(self,
                 db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 model_dir: str = "strategies/models"):
        """
        Args:
            db_path: Path to database
            model_dir: Directory to store model files
        """
        self.db_path = db_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Ensure model metadata table exists
        self._ensure_metadata_table()

    def _ensure_metadata_table(self):
        """Create table to track model versions and performance"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            model_version INTEGER NOT NULL,
            trained_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            training_start_date DATE,
            training_end_date DATE,
            num_training_samples INTEGER,
            num_new_samples INTEGER,
            is_full_retrain BOOLEAN DEFAULT 0,
            train_accuracy REAL,
            test_accuracy REAL,
            train_sharpe REAL,
            test_sharpe REAL,
            model_path TEXT,
            scaler_path TEXT,
            feature_names TEXT,
            hyperparameters TEXT,
            notes TEXT,
            UNIQUE(strategy_name, model_version)
        )
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_metadata_strategy_version
        ON model_metadata(strategy_name, model_version DESC)
        """)

        conn.commit()
        conn.close()

    def get_latest_model_info(self, strategy_name: str) -> Optional[Dict]:
        """
        Get information about the latest trained model

        Args:
            strategy_name: Name of strategy

        Returns:
            Dictionary with model metadata or None if no model exists
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
        SELECT model_id, model_version, trained_date, training_end_date,
               num_training_samples, train_accuracy, test_accuracy,
               model_path, scaler_path, feature_names, hyperparameters
        FROM model_metadata
        WHERE strategy_name = ?
        ORDER BY model_version DESC
        LIMIT 1
        """, (strategy_name,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'model_id': row[0],
                'model_version': row[1],
                'trained_date': row[2],
                'training_end_date': row[3],
                'num_training_samples': row[4],
                'train_accuracy': row[5],
                'test_accuracy': row[6],
                'model_path': row[7],
                'scaler_path': row[8],
                'feature_names': json.loads(row[9]) if row[9] else None,
                'hyperparameters': json.loads(row[10]) if row[10] else None
            }
        return None

    def load_model(self, strategy_name: str) -> Tuple[Optional[Any], Optional[StandardScaler], Optional[Dict]]:
        """
        Load the latest model and scaler

        Args:
            strategy_name: Name of strategy

        Returns:
            Tuple of (model, scaler, metadata) or (None, None, None) if no model exists
        """
        info = self.get_latest_model_info(strategy_name)

        if not info:
            logger.info(f"No existing model found for {strategy_name}")
            return None, None, None

        model_path = Path(info['model_path'])
        scaler_path = Path(info['scaler_path'])

        if not model_path.exists() or not scaler_path.exists():
            logger.warning(f"Model files not found for {strategy_name}")
            return None, None, None

        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        logger.info(f"✅ Loaded {strategy_name} model v{info['model_version']}")
        logger.info(f"   Trained on {info['num_training_samples']} samples until {info['training_end_date']}")
        logger.info(f"   Test Accuracy: {info['test_accuracy']:.2%}")

        return model, scaler, info

    def save_model(self,
                   strategy_name: str,
                   model: Any,
                   scaler: StandardScaler,
                   training_start_date: str,
                   training_end_date: str,
                   num_training_samples: int,
                   num_new_samples: int,
                   is_full_retrain: bool,
                   train_accuracy: float,
                   test_accuracy: float,
                   feature_names: list,
                   hyperparameters: Dict,
                   train_sharpe: float = 0.0,
                   test_sharpe: float = 0.0,
                   notes: str = "") -> int:
        """
        Save model and metadata

        Args:
            strategy_name: Name of strategy
            model: Trained model object
            scaler: Fitted scaler
            training_start_date: Start date of training data
            training_end_date: End date of training data
            num_training_samples: Total samples in training set
            num_new_samples: Number of new samples (incremental only)
            is_full_retrain: Whether this was a full retrain
            train_accuracy: Training accuracy
            test_accuracy: Test accuracy
            feature_names: List of feature names
            hyperparameters: Model hyperparameters dict
            train_sharpe: Training Sharpe ratio
            test_sharpe: Test Sharpe ratio
            notes: Optional notes

        Returns:
            New model version number
        """
        # Get next version number
        latest_info = self.get_latest_model_info(strategy_name)
        next_version = 1 if not latest_info else latest_info['model_version'] + 1

        # Save model and scaler files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{strategy_name}_v{next_version}_{timestamp}.pkl"
        scaler_filename = f"{strategy_name}_scaler_v{next_version}_{timestamp}.pkl"

        model_path = self.model_dir / model_filename
        scaler_path = self.model_dir / scaler_filename

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Save metadata to database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
        INSERT INTO model_metadata (
            strategy_name, model_version, training_start_date, training_end_date,
            num_training_samples, num_new_samples, is_full_retrain,
            train_accuracy, test_accuracy, train_sharpe, test_sharpe,
            model_path, scaler_path, feature_names, hyperparameters, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            strategy_name, next_version, training_start_date, training_end_date,
            num_training_samples, num_new_samples, is_full_retrain,
            train_accuracy, test_accuracy, train_sharpe, test_sharpe,
            str(model_path), str(scaler_path),
            json.dumps(feature_names), json.dumps(hyperparameters), notes
        ))
        conn.commit()
        conn.close()

        logger.info(f"✅ Saved {strategy_name} model v{next_version}")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Training: {training_start_date} to {training_end_date}")
        logger.info(f"   Samples: {num_training_samples} total, {num_new_samples} new")
        logger.info(f"   Accuracy: Train={train_accuracy:.2%}, Test={test_accuracy:.2%}")

        return next_version

    def get_last_training_date(self, strategy_name: str) -> Optional[str]:
        """
        Get the last training end date for a strategy

        Args:
            strategy_name: Name of strategy

        Returns:
            Last training end date (YYYY-MM-DD) or None if no training history
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT training_end_date
                FROM model_metadata
                WHERE strategy_name = ?
                ORDER BY trained_date DESC
                LIMIT 1
            """, (strategy_name,))

            result = cursor.fetchone()
            conn.close()

            if result:
                return result[0]
            return None

        except Exception as e:
            logger.error(f"Error getting last training date: {e}")
            return None

    def get_new_training_data(self,
                              strategy_name: str,
                              query: str,
                              min_samples: int = 100) -> Tuple[pd.DataFrame, bool]:
        """
        Get new training data since last model training

        Args:
            strategy_name: Name of strategy
            query: SQL query to fetch training data (must include WHERE clause placeholder)
            min_samples: Minimum new samples required for incremental update

        Returns:
            Tuple of (new_data_df, should_retrain_flag)
        """
        latest_info = self.get_latest_model_info(strategy_name)

        if not latest_info:
            # No existing model - full training needed
            logger.info(f"No existing model for {strategy_name} - full training required")
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df, True

        # Get data since last training
        last_training_date = latest_info['training_end_date']

        # Modify query to only get new data
        if "WHERE" in query.upper():
            new_query = query + f" AND feature_date > '{last_training_date}'"
        else:
            new_query = query + f" WHERE feature_date > '{last_training_date}'"

        conn = sqlite3.connect(self.db_path)
        new_df = pd.read_sql_query(new_query, conn)
        conn.close()

        num_new = len(new_df)

        if num_new < min_samples:
            logger.info(f"Only {num_new} new samples for {strategy_name} (min {min_samples} required)")
            logger.info(f"Skipping incremental update - using existing model v{latest_info['model_version']}")
            return pd.DataFrame(), False

        logger.info(f"Found {num_new} new samples for {strategy_name} since {last_training_date}")
        logger.info(f"Will perform incremental update on existing model v{latest_info['model_version']}")

        return new_df, True

    def should_full_retrain(self, strategy_name: str, days_threshold: int = 90) -> bool:
        """
        Determine if a full retrain is needed

        Args:
            strategy_name: Name of strategy
            days_threshold: Days since last full retrain to trigger new full retrain

        Returns:
            True if full retrain is recommended
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get last full retrain
        cursor.execute("""
        SELECT trained_date, model_version
        FROM model_metadata
        WHERE strategy_name = ? AND is_full_retrain = 1
        ORDER BY model_version DESC
        LIMIT 1
        """, (strategy_name,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            # Never had a full retrain
            return True

        last_full_retrain = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        days_since = (datetime.now() - last_full_retrain).days

        if days_since >= days_threshold:
            logger.info(f"Last full retrain was {days_since} days ago (threshold: {days_threshold})")
            logger.info(f"Recommending full retrain for {strategy_name}")
            return True

        logger.info(f"Last full retrain was {days_since} days ago - incremental update sufficient")
        return False

    def incremental_train_xgboost(self,
                                  old_model: xgb.XGBClassifier,
                                  X_new: np.ndarray,
                                  y_new: np.ndarray,
                                  n_estimators_new: int = 50,
                                  max_estimators: int = 200) -> xgb.XGBClassifier:
        """
        Incrementally train XGBoost by adding more trees (WITH CAP to prevent bloat)

        Args:
            old_model: Previously trained XGBoost model
            X_new: New training features
            y_new: New training labels
            n_estimators_new: Number of new trees to add (default 50)
            max_estimators: Maximum total trees allowed (default 200)

        Returns:
            Updated XGBoost model

        IMPORTANT FIX: Models were growing from 657KB -> 29MB because trees kept accumulating.
        Now we cap at max_estimators and reset to base model when limit is reached.
        """
        # Get old model parameters
        old_params = old_model.get_params()
        old_n_estimators = old_params['n_estimators']

        # FIX: Check if model has grown too large - reset to base instead of continuing to bloat
        if old_n_estimators >= max_estimators:
            logger.warning(f"⚠️  Model has {old_n_estimators} trees (max: {max_estimators})")
            logger.warning(f"   Resetting to base model with {max_estimators} trees to prevent bloat")
            logger.warning(f"   Previous models grew from 657KB -> 29MB due to uncapped growth!")

            # Train fresh model with capped size
            old_params['n_estimators'] = max_estimators
            old_params.pop('early_stopping_rounds', None)
            old_params.pop('eval_metric', None)

            new_model = xgb.XGBClassifier(**old_params)
            new_model.fit(X_new, y_new)

            logger.info(f"✅ Reset to base model with {max_estimators} trees")
            return new_model

        # Calculate new total (capped at max)
        new_n_estimators = min(old_n_estimators + n_estimators_new, max_estimators)
        actual_trees_added = new_n_estimators - old_n_estimators

        old_params['n_estimators'] = new_n_estimators

        # Remove early_stopping_rounds for incremental training (we don't have validation set)
        old_params.pop('early_stopping_rounds', None)
        old_params.pop('eval_metric', None)

        # Initialize new model with old parameters
        new_model = xgb.XGBClassifier(**old_params)

        # FIX: Check if the new data contains all the classes the old model was trained on.
        # XGBoost incremental training requires all classes to be present.
        old_classes = set(old_model.classes_)
        new_classes = set(np.unique(y_new))

        if not old_classes.issubset(new_classes):
            logger.warning(f"⚠️  New data is missing classes required by the old model. Cannot perform incremental training.")
            logger.warning(f"   Old model classes: {sorted(list(old_classes))}")
            logger.warning(f"   New data classes:  {sorted(list(new_classes))}")
            logger.warning(f"   Skipping update and returning original model.")
            return old_model

        # Set xgb_model parameter to continue training from old model
        new_model.fit(
            X_new, y_new,
            xgb_model=old_model.get_booster()
        )

        logger.info(f"Added {actual_trees_added} new trees (total: {new_n_estimators}/{max_estimators})")

        if new_n_estimators >= max_estimators:
            logger.warning(f"⚠️  Model at maximum capacity ({max_estimators} trees)")
            logger.warning(f"   Next incremental update will reset to base model")

        return new_model

    def get_training_summary(self, strategy_name: str, limit: int = 10) -> pd.DataFrame:
        """
        Get training history for a strategy

        Args:
            strategy_name: Name of strategy
            limit: Number of recent versions to show

        Returns:
            DataFrame with training history
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT model_version, trained_date, training_end_date,
               num_training_samples, num_new_samples, is_full_retrain,
               train_accuracy, test_accuracy, train_sharpe, test_sharpe
        FROM model_metadata
        WHERE strategy_name = ?
        ORDER BY model_version DESC
        LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=[strategy_name, limit])
        conn.close()

        return df


if __name__ == "__main__":
    """Test incremental trainer"""

    trainer = IncrementalTrainer()

    # Test get latest model info
    for strategy in ['SentimentTradingStrategy', 'VolatilityTradingStrategy']:
        print(f"\n{'='*80}")
        print(f"Testing: {strategy}")
        print('='*80)

        info = trainer.get_latest_model_info(strategy)

        if info:
            print(f"\nLatest Model:")
            print(f"  Version: {info['model_version']}")
            print(f"  Trained: {info['trained_date']}")
            print(f"  Data until: {info['training_end_date']}")
            print(f"  Samples: {info['num_training_samples']}")
            print(f"  Test Accuracy: {info['test_accuracy']:.2%}")
        else:
            print(f"\nNo existing model found")

        # Check if full retrain needed
        needs_retrain = trainer.should_full_retrain(strategy, days_threshold=90)
        print(f"\nFull retrain needed: {needs_retrain}")
