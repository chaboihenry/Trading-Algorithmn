import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImbalanceBarGenerator:
    """
    Generates Dollar Imbalance Bars with DYNAMIC Thresholds.
    """
    @staticmethod
    def get_dynamic_threshold(ticks: pd.DataFrame, target_bars_per_day=50) -> float:
        """Calculates adaptive threshold based on daily dollar volume."""
        if ticks.empty: return 1_000_000
        
        total_dollar_vol = (ticks['price'] * ticks['size']).sum()
        time_span = ticks.index[-1] - ticks.index[0]
        days = max(1, time_span.days)
        
        avg_daily_dollar_vol = total_dollar_vol / days
        threshold = avg_daily_dollar_vol / target_bars_per_day
        
        # Clamp to reasonable minimum to avoid noise explosion
        return max(threshold, 10_000)

    @staticmethod
    def process_ticks(ticks: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame()

        ticks = ticks.sort_index()
        
        # Auto-calculate threshold if not provided
        if threshold is None:
            threshold = ImbalanceBarGenerator.get_dynamic_threshold(ticks)
        
        # 1. Tick Rule
        price_diff = ticks['price'].diff()
        tick_signs = price_diff.apply(np.sign)
        tick_signs = tick_signs.replace(0, np.nan).ffill().fillna(1)
        
        # 2. Signed Dollar Flow
        signed_dollar_flow = tick_signs * ticks['price'] * ticks['size']
        
        # 3. Fast Iteration (Numpy)
        cum_theta = 0
        bars = []
        
        current_bar = {
            'open': ticks['price'].iloc[0],
            'high': ticks['price'].iloc[0],
            'low': ticks['price'].iloc[0],
            'close': ticks['price'].iloc[0],
            'volume': 0,
            'dollar_val': 0,
            'ticks': 0,
            'start_time': ticks.index[0]
        }
        
        prices = ticks['price'].values
        sizes = ticks['size'].values
        imbalances = signed_dollar_flow.values
        timestamps = ticks.index
        
        for i in range(len(prices)):
            p = prices[i]
            s = sizes[i]
            imbalance = imbalances[i]
            ts = timestamps[i]
            
            cum_theta += imbalance
            
            # Update Bar Stats
            if p > current_bar['high']: current_bar['high'] = p
            if p < current_bar['low']: current_bar['low'] = p
            current_bar['close'] = p
            current_bar['volume'] += s
            current_bar['dollar_val'] += (p * s)
            current_bar['ticks'] += 1
            
            # Check Threshold
            if abs(cum_theta) >= threshold:
                bars.append({
                    'timestamp': ts,
                    'open': current_bar['open'],
                    'high': current_bar['high'],
                    'low': current_bar['low'],
                    'close': current_bar['close'],
                    'volume': current_bar['volume'],
                    'vwap': current_bar['dollar_val'] / current_bar['volume'] if current_bar['volume'] > 0 else p,
                    'ticks': current_bar['ticks'],
                    'threshold_used': threshold
                })
                
                cum_theta = 0
                if i + 1 < len(prices):
                    next_p = prices[i+1]
                    current_bar = {
                        'open': next_p, 'high': next_p, 'low': next_p, 'close': next_p,
                        'volume': 0, 'dollar_val': 0, 'ticks': 0,
                        'start_time': timestamps[i+1]
                    }

        if not bars:
            return pd.DataFrame()
            
        df = pd.DataFrame(bars).set_index('timestamp')
        return df