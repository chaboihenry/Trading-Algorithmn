#!/usr/bin/env python3
"""
Check what classes the primary model was trained on.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

# Load model
strategy = RiskLabAIStrategy()
strategy.load_models('models/risklabai_tick_models_365days.pkl')

# Check classes
print("PRIMARY MODEL:")
print(f"  Classes: {strategy.primary_model.classes_}")
print(f"  N Classes: {strategy.primary_model.n_classes_}")

print("\nMETA MODEL:")
print(f"  Classes: {strategy.meta_model.classes_}")
print(f"  N Classes: {strategy.meta_model.n_classes_}")
