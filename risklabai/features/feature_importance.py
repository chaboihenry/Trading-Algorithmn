"""
Feature Importance Methods using RiskLabAI

Methods:
1. MDI (Mean Decrease Impurity): From tree structure
2. MDA (Mean Decrease Accuracy): Permutation-based
3. SFI (Single Feature Importance): Each feature alone
"""

from RiskLabAI.features.feature_importance import (
    feature_importance_mdi,
    feature_importance_mda,
    feature_importance_sfi,
    clustered_feature_importance_mda
)
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using RiskLabAI methods.

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
            model: Model to analyze
            X: Feature matrix
            y: Labels
            cv: Cross-validator
            sample_weight: Optional sample weights

        Returns:
            DataFrame with feature importance scores
        """
        logger.info(f"Calculating {self.method.upper()} feature importance")

        try:
            if self.method == 'mdi':
                importance = feature_importance_mdi(
                    model=model,
                    feature_names=X.columns.tolist()
                )

            elif self.method == 'mda':
                importance = feature_importance_mda(
                    clf=model,
                    X=X.values,
                    y=y.values,
                    cv=cv,
                    sample_weight=sample_weight.values if sample_weight is not None else None,
                    scoring='accuracy'
                )
                # Convert to DataFrame
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance
                })

            elif self.method == 'sfi':
                importance = feature_importance_sfi(
                    clf=model,
                    X=X.values,
                    y=y.values,
                    cv=cv,
                    sample_weight=sample_weight.values if sample_weight is not None else None,
                    scoring='accuracy'
                )
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance
                })

            elif self.method == 'clustered':
                importance = clustered_feature_importance_mda(
                    clf=model,
                    X=X.values,
                    y=y.values,
                    cv=cv,
                    sample_weight=sample_weight.values if sample_weight is not None else None,
                    scoring='accuracy'
                )
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importance
                })

            logger.info(f"Feature importance calculated: {len(importance)} features")
            return importance.sort_values('importance', ascending=False)

        except Exception as e:
            logger.error(f"Error calculating importance: {e}")
            raise
