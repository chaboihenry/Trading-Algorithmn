"""
Feature Importance Methods

Three approaches to understand what features matter:

1. MDI (Mean Decrease Impurity): From tree structure
2. MDA (Mean Decrease Accuracy): Permutation-based
3. SFI (Single Feature Importance): Each feature alone

Plus Clustered Feature Importance to handle correlated features.
"""

from RiskLabAI.features import (
    feature_importance_mdi,
    feature_importance_mda,
    feature_importance_sfi,
    get_orthogonal_features,
    cluster_kmeans_base
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using multiple methods.

    Why multiple methods?
    - MDI is fast but biased toward high-cardinality features
    - MDA is unbiased but slow (requires retraining)
    - SFI isolates individual feature contribution

    Clustered importance handles correlated features:
    - Correlated features split importance
    - Clustering groups them to show true importance

    Attributes:
        method: Which importance method to use
    """

    def __init__(self, method: str = 'mda'):
        """
        Initialize analyzer.

        Args:
            method: 'mdi', 'mda', 'sfi', or 'clustered'
        """
        if method not in ['mdi', 'mda', 'sfi', 'clustered']:
            raise ValueError(f"Invalid method: {method}")

        self.method = method
        logger.info(f"FeatureImportanceAnalyzer initialized: method={method}")

    def calculate_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv=None,
        sample_weight: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Calculate feature importance.

        Args:
            model: Fitted model (for MDI) or unfitted (for MDA/SFI)
            X: Feature matrix
            y: Labels
            cv: Cross-validator (for MDA/SFI)
            sample_weight: Optional sample weights

        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Calculating {self.method.upper()} feature importance")

        try:
            if self.method == 'mdi':
                return self._calculate_mdi(model, X)

            elif self.method == 'mda':
                return self._calculate_mda(model, X, y, cv, sample_weight)

            elif self.method == 'sfi':
                return self._calculate_sfi(model, X, y, cv, sample_weight)

            elif self.method == 'clustered':
                return self._calculate_clustered(model, X, y, cv, sample_weight)

        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            raise

    def _calculate_mdi(
        self,
        model,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate Mean Decrease Impurity."""
        importance = feature_importance_mdi(
            fit=model,
            feature_names=X.columns.tolist()
        )

        logger.info(f"MDI complete: top feature = {importance.iloc[0]['feature']}")
        return importance

    def _calculate_mda(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv,
        sample_weight: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Calculate Mean Decrease Accuracy."""
        if cv is None:
            logger.warning("No CV provided for MDA, results may be unreliable")

        importance = feature_importance_mda(
            clf=model,
            X=X,
            y=y,
            cv=cv,
            sample_weight=sample_weight,
            scoring='accuracy'
        )

        logger.info(f"MDA complete: top feature = {importance.iloc[0]['feature']}")
        return importance

    def _calculate_sfi(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv,
        sample_weight: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Calculate Single Feature Importance."""
        if cv is None:
            logger.warning("No CV provided for SFI, results may be unreliable")

        importance = feature_importance_sfi(
            clf=model,
            X=X,
            y=y,
            cv=cv,
            sample_weight=sample_weight,
            scoring='accuracy'
        )

        logger.info(f"SFI complete: top feature = {importance.iloc[0]['feature']}")
        return importance

    def _calculate_clustered(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv,
        sample_weight: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Calculate Clustered Feature Importance."""
        # First cluster features
        corr_matrix = X.corr()
        clusters, _ = cluster_kmeans_base(corr_matrix)

        logger.info(f"Clustered features into {len(clusters.unique())} groups")

        # Calculate MDA on clusters
        importance = feature_importance_mda(
            clf=model,
            X=X,
            y=y,
            cv=cv,
            sample_weight=sample_weight,
            scoring='accuracy',
            clusters=clusters
        )

        logger.info(f"Clustered MDA complete")
        return importance

    def get_orthogonal_features(
        self,
        X: pd.DataFrame,
        variance_threshold: float = 0.95
    ) -> pd.DataFrame:
        """
        Get orthogonal (uncorrelated) features using PCA.

        Args:
            X: Feature matrix
            variance_threshold: Cumulative variance to preserve

        Returns:
            DataFrame with orthogonal features
        """
        logger.info(f"Extracting orthogonal features (variance={variance_threshold})")

        orth_features = get_orthogonal_features(
            X,
            variance_threshold=variance_threshold
        )

        logger.info(f"Reduced {X.shape[1]} features to {orth_features.shape[1]} orthogonal features")

        return orth_features

    def plot_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.

        Args:
            importance_df: Output from calculate_importance()
            top_n: Number of top features to plot
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt

        # Get top N features
        top_features = importance_df.nlargest(top_n, 'importance')

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features ({self.method.upper()})')
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.tight_layout()
            plt.show()

    def compare_methods(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        cv=None,
        sample_weight: Optional[pd.Series] = None
    ) -> dict:
        """
        Compare all importance methods.

        Args:
            model: Model to analyze
            X: Features
            y: Labels
            cv: Cross-validator
            sample_weight: Optional weights

        Returns:
            Dictionary with importance from each method
        """
        logger.info("Comparing all feature importance methods...")

        results = {}

        # MDI (fast, requires fitted model)
        try:
            if hasattr(model, 'feature_importances_'):
                self.method = 'mdi'
                results['mdi'] = self.calculate_importance(model, X, y, cv, sample_weight)
        except Exception as e:
            logger.warning(f"MDI failed: {e}")

        # MDA (slow but reliable)
        try:
            self.method = 'mda'
            results['mda'] = self.calculate_importance(model, X, y, cv, sample_weight)
        except Exception as e:
            logger.warning(f"MDA failed: {e}")

        # SFI (very slow, but shows individual contribution)
        try:
            self.method = 'sfi'
            results['sfi'] = self.calculate_importance(model, X, y, cv, sample_weight)
        except Exception as e:
            logger.warning(f"SFI failed: {e}")

        logger.info(f"Comparison complete: {len(results)} methods succeeded")

        return results
