import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImbalanceBarGenerator:
    """
    Generates Dollar Imbalance Bars with DYNAMIC Thresholds.
    Adapts to the liquidity of the symbol (QQQ vs Small Cap).
    """
    @staticmethod
    def get_dynamic_threshold(ticks: pd.DataFrame, target_bars_per_day=50) -> float:
        """
        Calculates a threshold that results in approx 'target_bars_per_day'.
        """
        total_dollar_vol = (ticks['price'] * ticks['size']).sum()
        
        # Calculate number of days in the dataset
        time_span = ticks.index[-1] - ticks.index[0]
        days = max(1, time_span.days)
        
        avg_daily_dollar_vol = total_dollar_vol / days
        
        # Threshold = Average Dollar Volume per Bar
        threshold = avg_daily_dollar_vol / target_bars_per_day
        
        # Safety floor to prevent tiny bars on bad data
        return max(threshold, 10_000) 

    @staticmethod
    def process_ticks(ticks: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame()

        # Ensure sorted
        ticks = ticks.sort_index()
        
        # --- DYNAMIC THRESHOLD CALCULATION ---
        if threshold is None:
            threshold = ImbalanceBarGenerator.get_dynamic_threshold(ticks)
            # logger.info(f"Dynamic Threshold calculated: ${threshold:,.2f}")
        
        # 1. Tick Rule
        price_diff = ticks['price'].diff()
        tick_signs = price_diff.apply(np.sign)
        tick_signs = tick_signs.replace(0, np.nan).ffill().fillna(1)
        
        # 2. Signed Flow
        signed_dollar_flow = tick_signs * ticks['price'] * ticks['size']
        
        # 3. Bar Generation Loop
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
        
        # Numpy arrays for speed
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
            
            # Update stats
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
                    'threshold_used': threshold # Useful for debugging
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