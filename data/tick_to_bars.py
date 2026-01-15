import numpy as np

def generate_bars_from_ticks(ticks, threshold=3000):
    """
    Converts raw ticks (timestamp, price, volume) into Imbalance Bars.
    
    Logic:
    1. Determine trade direction (Tick Rule).
    2. Accumulate signed volume (Imbalance).
    3. When absolute imbalance > threshold, close the bar.
    """
    bars = []
    if not ticks:
        return bars

    # State variables
    current_bar = None
    cum_imbalance = 0
    
    # Tick Rule State: 1=Buy, -1=Sell. Start assuming Buy.
    prev_price = ticks[0][1]
    tick_dir = 1 

    for t in ticks:
        # Unpack tuple from DB (timestamp, price, size)
        ts, price, size = t[0], t[1], t[2]

        # 1. Apply Tick Rule
        if price > prev_price:
            tick_dir = 1
        elif price < prev_price:
            tick_dir = -1
        # If price unchanged, keep previous tick_dir
        
        prev_price = price

        # 2. Accumulate Imbalance
        # Buy volume adds positive imbalance, Sell volume adds negative
        cum_imbalance += tick_dir * size

        # 3. Update Current Bar
        if current_bar is None:
            current_bar = {
                'bar_start': ts,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': size,
                'tick_count': 1
            }
        else:
            current_bar['high'] = max(current_bar['high'], price)
            current_bar['low'] = min(current_bar['low'], price)
            current_bar['close'] = price
            current_bar['volume'] += size
            current_bar['tick_count'] += 1

        # 4. Check Threshold (Sampling)
        if abs(cum_imbalance) >= threshold:
            current_bar['bar_end'] = ts
            # Optional: save the final imbalance for debugging
            current_bar['imbalance'] = cum_imbalance 
            bars.append(current_bar)
            
            # Reset for next bar
            current_bar = None
            cum_imbalance = 0

    return bars