import pandas as pd
import numpy as np
import logging
from numba import jit

logger = logging.getLogger(__name__)

@jit(nopython=True)
def _generate_dollar_bars_numba(timestamps, prices, volumes, threshold, state_in):
    """
    Implements Dollar Imbalance Bars logic based on Advances in Financial ML (Ch 2.3.2).
    
    1. b_t (Tick Rule): 1 if price up, -1 if down, b_{t-1} if flat.
    2. theta_t (Imbalance): b_t * (price * size)
    3. Sample when |Sum(theta_t)| >= Threshold
    """
    n_ticks = len(prices)
    
    # Handle empty chunk
    if n_ticks == 0:
        return 0, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), \
               np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), \
               np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), \
               np.zeros(0, dtype=np.float64), state_in

    # Unpack State
    cum_theta = state_in[0]      # Current Cumulative Imbalance
    last_sign = int(state_in[1]) # Previous b_t
    cur_high = state_in[2]
    cur_low = state_in[3]
    cur_vol = state_in[4]
    cur_dollar_val = state_in[5]
    cur_open = state_in[6]

    if cur_high == -1.0:
        cur_open = prices[0]
        cur_high = prices[0]
        cur_low = prices[0]
    
    out_ts = np.zeros(n_ticks, dtype=np.int64)
    out_open = np.zeros(n_ticks, dtype=np.float64)
    out_high = np.zeros(n_ticks, dtype=np.float64)
    out_low = np.zeros(n_ticks, dtype=np.float64)
    out_close = np.zeros(n_ticks, dtype=np.float64)
    out_vol = np.zeros(n_ticks, dtype=np.float64)
    out_vwap = np.zeros(n_ticks, dtype=np.float64)
    
    bar_idx = 0
    
    for i in range(n_ticks):
        p = prices[i]
        v = volumes[i]
        ts = timestamps[i]
        
        # --- 1. Tick Rule (b_t) ---
        # "b_t = |Δp|/Δp" -> 1 if Up, -1 if Down
        if i > 0:
            change = p - prices[i-1]
            if change > 0: last_sign = 1
            elif change < 0: last_sign = -1
            # if change == 0, last_sign remains unchanged (b_{t-1})
        
        # --- 2. Imbalance (theta_t) ---
        dollar_flow = p * v
        imbalance = last_sign * dollar_flow
        cum_theta += imbalance
        
        # Track Bar Stats
        if p > cur_high: cur_high = p
        if p < cur_low: cur_low = p
        cur_vol += v
        cur_dollar_val += dollar_flow
        
        # --- 3. Sampling Rule ---
        # Sample if |Σ theta_t| >= Threshold
        if abs(cum_theta) >= threshold:
            out_ts[bar_idx] = ts
            out_open[bar_idx] = cur_open
            out_high[bar_idx] = cur_high
            out_low[bar_idx] = cur_low
            out_close[bar_idx] = p
            out_vol[bar_idx] = cur_vol
            out_vwap[bar_idx] = cur_dollar_val / cur_vol if cur_vol > 0 else p
            
            bar_idx += 1
            
            # Reset
            cum_theta = 0.0
            cur_vol = 0.0
            cur_dollar_val = 0.0
            
            if i < n_ticks - 1:
                cur_open = prices[i+1]
                cur_high = prices[i+1]
                cur_low = prices[i+1]
            else:
                cur_high = -1.0 

    state_out = (float(cum_theta), int(last_sign), float(cur_high), float(cur_low), 
                 float(cur_vol), float(cur_dollar_val), float(cur_open))
    
    return bar_idx, out_ts, out_open, out_high, out_low, out_close, out_vol, out_vwap, state_out

class ImbalanceBarGenerator:
    @staticmethod
    def process_chunk(ticks: pd.DataFrame, threshold: float, state: tuple) -> tuple:
        if ticks.empty: return pd.DataFrame(), state

        if 'volume' not in ticks.columns and 'size' in ticks.columns:
            ticks = ticks.rename(columns={'size': 'volume'})

        ts_arr = ticks['timestamp'].astype(np.int64).values 
        price_arr = ticks['price'].values.astype(np.float64)
        vol_arr = ticks['volume'].values.astype(np.float64)
        
        count, res_ts, res_open, res_high, res_low, res_close, res_vol, res_vwap, new_state = \
            _generate_dollar_bars_numba(ts_arr, price_arr, vol_arr, float(threshold), state)
            
        if count == 0: return pd.DataFrame(), new_state
            
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(res_ts[:count]),
            'open': res_open[:count],
            'high': res_high[:count],
            'low': res_low[:count],
            'close': res_close[:count],
            'volume': res_vol[:count],
            'vwap': res_vwap[:count]
        })
        df.set_index('timestamp', inplace=True)
        return df, new_state