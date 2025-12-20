"""Keras MLP Neural Network for Trading Signal Prediction.

Uses proven hyperparameters from "Python Machine Learning by Example" (R²=0.97):
- Hidden layers: 2 layers of 16 neurons each
- Epochs: 1000
- Learning rate: 0.21
- Dropout: 0.3 for regularization
- Activation: ReLU (hidden layers), Softmax (output)
- Output: 3 classes (BUY=0, HOLD=1, SELL=2)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("WARNING: TensorFlow/Keras not installed. Install with: pip install tensorflow")

logger = logging.getLogger(__name__)


class KerasMLP:
    """Multi-Layer Perceptron for trading signal classification.

    Architecture:
        Input(n_features) -> Dense(16, relu) -> Dropout(0.3) ->
        Dense(16, relu) -> Dropout(0.3) -> Dense(3, softmax)

    Training:
        - Optimizer: Adam(lr=0.21)
        - Loss: sparse_categorical_crossentropy
        - Epochs: 1000 (with early stopping)
    """

    def __init__(self, n_features: int, model_dir: Path):
        """Initialize Keras MLP model.

        Args:
            n_features: Number of input features
            model_dir: Directory to save/load models
        """
        if not KERAS_AVAILABLE:
            raise ImportError("TensorFlow/Keras required. Install: pip install tensorflow")

        self.n_features = n_features
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Proven hyperparameters (R²=0.97)
        self.hidden_size = 16
        self.learning_rate = 0.21
        self.dropout_rate = 0.3
        self.epochs = 1000
        self.batch_size = 32

        self.model = None
        self.history = None

        logger.info("✅ Keras MLP initialized")
        logger.info(f"   Architecture: Input({n_features}) -> Dense(16) -> Dropout(0.3) -> Dense(16) -> Dense(3)")
        logger.info(f"   Hyperparameters: lr={self.learning_rate}, epochs={self.epochs}")

    def _build_model(self) -> keras.Model:
        """Build the MLP model with proven hyperparameters.

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # Input layer (implicit from first Dense layer)
            layers.Dense(
                self.hidden_size,
                activation='relu',
                input_shape=(self.n_features,),
                name='hidden_layer_1'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_1'),

            # Second hidden layer
            layers.Dense(
                self.hidden_size,
                activation='relu',
                name='hidden_layer_2'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_2'),

            # Output layer: 3 classes (BUY, HOLD, SELL)
            layers.Dense(3, activation='softmax', name='output')
        ])

        # Compile with proven hyperparameters
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_categorical_accuracy']
        )

        logger.info("✅ MLP model compiled")
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_split: float = 0.2, verbose: int = 1) -> Dict:
        """Train the MLP model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - values: 0=BUY, 1=HOLD, 2=SELL
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Training history dict with metrics
        """
        logger.info(f"Training MLP on {len(X_train)} samples...")
        logger.info(f"   Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}")

        # Build model
        self.model = self._build_model()

        # Callbacks
        callbacks = [
            # Early stopping: stop if validation loss doesn't improve for 50 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=50,
                restore_best_weights=True,
                verbose=1
            ),

            # Save best model
            ModelCheckpoint(
                filepath=str(self.model_dir / 'best_model.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )

        self.history = history.history

        # Log results
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        logger.info(f"✅ Training complete:")
        logger.info(f"   Final accuracy: {final_acc:.4f} | Val accuracy: {final_val_acc:.4f}")
        logger.info(f"   Final loss: {final_loss:.4f} | Val loss: {final_val_loss:.4f}")
        logger.info(f"   Epochs trained: {len(history.history['loss'])}")

        return self.history

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Tuple of (predicted_classes, prediction_probabilities)
            - predicted_classes: (n_samples,) - 0=BUY, 1=HOLD, 2=SELL
            - prediction_probabilities: (n_samples, 3) - softmax probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")

        # Get probabilities for each class
        probabilities = self.model.predict(X, verbose=0)

        # Get class with highest probability
        predicted_classes = np.argmax(probabilities, axis=1)

        return predicted_classes, probabilities

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilities array (n_samples, 3) for [BUY, HOLD, SELL]
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dict with 'loss' and 'accuracy'
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        loss, accuracy, sparse_acc = self.model.evaluate(X_test, y_test, verbose=0)

        logger.info(f"📊 Test evaluation:")
        logger.info(f"   Loss: {loss:.4f}")
        logger.info(f"   Accuracy: {accuracy:.4f}")

        return {
            'loss': loss,
            'accuracy': accuracy,
            'sparse_categorical_accuracy': sparse_acc
        }

    def save(self, filepath: Optional[Path] = None) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save model (default: model_dir/keras_mlp_YYYYMMDD_HHMMSS.keras)
        """
        if self.model is None:
            raise ValueError("No model to save.")

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.model_dir / f"keras_mlp_{timestamp}.keras"

        self.model.save(filepath)
        logger.info(f"✅ Model saved to {filepath}")

    def load(self, filepath: Path) -> None:
        """Load model from disk.

        Args:
            filepath: Path to saved model file
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"✅ Model loaded from {filepath}")

    def get_feature_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """Estimate feature importance using gradient-based approach.

        Args:
            X_sample: Sample features to analyze

        Returns:
            Feature importance scores (n_features,)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # Use gradient of predictions w.r.t. inputs as importance measure
        with tf.GradientTape() as tape:
            X_tensor = tf.Variable(X_sample, dtype=tf.float32)
            predictions = self.model(X_tensor)

        gradients = tape.gradient(predictions, X_tensor)

        # Average absolute gradient across samples
        importance = np.mean(np.abs(gradients.numpy()), axis=0)

        return importance


def load_latest_model(model_dir: Path, n_features: int) -> Optional[KerasMLP]:
    """Load the most recent Keras MLP model.

    Args:
        model_dir: Directory containing saved models
        n_features: Number of features the model expects

    Returns:
        Loaded KerasMLP instance or None if no models found
    """
    model_dir = Path(model_dir)

    # Find all .keras files
    keras_files = sorted(model_dir.glob("keras_mlp_*.keras"), reverse=True)

    if not keras_files:
        logger.warning(f"No Keras MLP models found in {model_dir}")
        return None

    latest_model_path = keras_files[0]
    logger.info(f"Loading latest Keras MLP: {latest_model_path}")

    mlp = KerasMLP(n_features=n_features, model_dir=model_dir)
    mlp.load(latest_model_path)

    return mlp
