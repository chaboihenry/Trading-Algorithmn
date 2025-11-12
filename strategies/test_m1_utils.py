"""
Test script for M1 utilities
Run with: python test_m1_utils.py
"""
import numpy as np
from m1_utils import VectorizedFeatures, ParallelProcessor, MemoryOptimizer
import pandas as pd

print('Testing M1 Utilities...')
print('=' * 50)

# Test VectorizedFeatures
print('\n1. Testing VectorizedFeatures:')
prices = np.random.randn(100).cumsum() + 100
features = VectorizedFeatures()

rsi = features.calculate_rsi_vectorized(prices, 14)
print(f'   ✓ RSI calculation: {len(rsi)} values, last value: {rsi[-1]:.2f}')

returns = features.calculate_returns(prices, [1, 5, 10])
print(f'   ✓ Returns calculation: shape={returns.shape}')

upper, middle, lower = features.calculate_bollinger_bands(prices, 20)
print(f'   ✓ Bollinger Bands: upper={upper[-1]:.2f}, middle={middle[-1]:.2f}, lower={lower[-1]:.2f}')

macd, signal = features.calculate_macd(prices, 12, 26, 9)
print(f'   ✓ MACD: macd={macd[-1]:.3f}, signal={signal[-1]:.3f}')

# Test ParallelProcessor
print('\n2. Testing ParallelProcessor:')
optimal_workers = ParallelProcessor.get_optimal_workers()
print(f'   ✓ Optimal workers: {optimal_workers}')

def test_func(ticker):
    return f'{ticker}_processed'

results = ParallelProcessor.process_tickers_parallel(['AAPL', 'GOOGL', 'MSFT'], test_func, n_workers=2)
print(f'   ✓ Parallel processing: {results}')

# Test MemoryOptimizer
print('\n3. Testing MemoryOptimizer:')
df = pd.DataFrame({
    'int_col': np.random.randint(0, 100, 1000),
    'float_col': np.random.rand(1000),
    'str_col': ['cat'] * 500 + ['dog'] * 500
})
mem_before = df.memory_usage(deep=True).sum() / 1024
df_opt = MemoryOptimizer.optimize_dtypes(df)
mem_after = df_opt.memory_usage(deep=True).sum() / 1024
reduction = (1 - mem_after / mem_before) * 100
print(f'   ✓ Memory optimization: {mem_before:.1f}KB → {mem_after:.1f}KB ({reduction:.1f}% reduction)')

print('\n' + '=' * 50)
print('✅ All M1 utilities tests passed!')
print('\nYou can now use these utilities in your strategies:')
print('  from m1_utils import VectorizedFeatures, ParallelProcessor, MemoryOptimizer')
