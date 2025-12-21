# RiskLabAI API Corrections

## Issue
The RiskLabAI library API structure is different than initially coded. Need to update all import statements.

## Correct API Mappings

### Data Structures (Bars)
```python
# OLD (incorrect)
from RiskLabAI.data import dollar_bars, volume_bars, tick_imbalance_bars

# NEW (correct)
from RiskLabAI.data.structures import StandardBars, FixedImbalanceBars, FixedRunBars
```

### Labeling
```python
# OLD (incorrect)
from RiskLabAI.labeling import get_events, get_bins, triple_barrier

# NEW (correct)
from RiskLabAI.data.labeling import (
    triple_barrier,
    meta_labeling,
    cusum_filter_events_dynamic_threshold,
    vertical_barrier,
    daily_volatility_with_log_returns
)
```

### Fractional Differentiation
```python
# OLD (incorrect)
from RiskLabAI.features import frac_diff_ffd, get_opt_d

# NEW (correct)
from RiskLabAI.data.differentiation import (
    fractional_difference_fixed,
    find_optimal_ffd_simple
)
```

### Feature Importance
```python
# OLD (incorrect)
from RiskLabAI.features import feature_importance_mdi

# NEW (correct)
from RiskLabAI.features.feature_importance import (
    feature_importance_mdi,
    feature_importance_mda,
    feature_importance_sfi,
    clustered_feature_importance_mda
)
```

### Portfolio Optimization
```python
# OLD (incorrect)
from RiskLabAI.portfolio import hrp, nco

# NEW (correct)
from RiskLabAI.optimization import (
    get_optimal_portfolio_weights,  # HRP
    get_optimal_portfolio_weights_nco  # NCO
)
```

### Cross-Validation
```python
# Need to check if purged K-fold exists in RiskLabAI
# May need to implement manually using the concepts
```

## Files to Update

1. [risklabai/data_structures/bars.py](risklabai/data_structures/bars.py)
2. [risklabai/labeling/triple_barrier.py](risklabai/labeling/triple_barrier.py)
3. [risklabai/labeling/meta_labeling.py](risklabai/labeling/meta_labeling.py)
4. [risklabai/features/fractional_diff.py](risklabai/features/fractional_diff.py)
5. [risklabai/sampling/cusum_filter.py](risklabai/sampling/cusum_filter.py)
6. [risklabai/features/feature_importance.py](risklabai/features/feature_importance.py)
7. [risklabai/portfolio/hrp.py](risklabai/portfolio/hrp.py)
8. [risklabai/cross_validation/purged_kfold.py](risklabai/cross_validation/purged_kfold.py) - May need manual implementation

## Next Steps

Due to the extensive API differences, we have two options:

### Option A: Full Rewrite (Recommended)
Rewrite all wrapper files to use the correct RiskLabAI API. This ensures we're using the library as intended.

### Option B: Direct Usage
Skip the wrapper files entirely and use RiskLabAI functions directly in the main strategy. This is simpler but less modular.

## Recommendation

I recommend **Option B** for now to get tests passing quickly, then gradually refactor to Option A for better code organization.
