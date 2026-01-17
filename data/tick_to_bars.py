import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ImbalanceBarGenerator:
    """
    Converts raw tick data into Imbalance Bars.
    Implements the standard Tick Rule and Volume/Dollar Imbalance sampling.
    """

    @staticmethod
    def process_ticks(ticks_input, batch_size=1000000, threshold=10000) -> pd.DataFrame:
        """
        Main entry point used by risklabai_bot.py.
        
        Args:
            ticks_input: DataFrame containing ['timestamp', 'price', 'volume']
            threshold: The cumulative imbalance threshold to trigger a new bar.
            
        Returns:
            pd.DataFrame: OHLCV bars indexed by bar_end time.
        """
        if ticks_input is None or ticks_input.empty:
            logger.warning("No ticks provided to ImbalanceBarGenerator.")
            return pd.DataFrame()

        # 1. Convert DataFrame to Numpy for high-speed iteration
        # We expect columns: timestamp, price, volume
        # If 'volume' is missing, look for 'size'
        vol_col = 'volume' if 'volume' in ticks_input.columns else 'size'
        
        try:
            # Create a structured array or list of tuples for speed
            # (Iterating pandas rows is too slow for millions of ticks)
            data_values = ticks_input[['timestamp', 'price', vol_col]].values
        except KeyError as e:
            logger.error(f"Tick data missing required columns: {e}")
            return pd.DataFrame()

        # 2. Run the generation logic
        bars_list = ImbalanceBarGenerator._generate_logic(data_values, threshold)

        # 3. Convert results back to DataFrame
        if not bars_list:
            return pd.DataFrame()

        bars_df = pd.DataFrame(bars_list)
        bars_df.set_index('bar_end', inplace=True)
        bars_df.sort_index(inplace=True)
        
        return bars_df

    @staticmethod
    def _generate_logic(tick_data, threshold):
        """
        Internal loop logic (The Engine).
        tick_data: numpy array of [timestamp, price, volume]
        """
        bars = []
        
        # State Variables
        current_bar = None
        cum_imbalance = 0
        tick_dir = 1 # 1 = Buy, -1 = Sell
        prev_price = float(tick_data[0][1])

        for row in tick_data:
            ts, price, size = row[0], float(row[1]), float(row[2])

            # --- A. Apply Tick Rule ---
            # If price went up, it's a Buy (1). If down, Sell (-1).
            # If price stayed same, keep previous direction.
            if price > prev_price:
                tick_dir = 1
            elif price < prev_price:
                tick_dir = -1
            
            prev_price = price

            # --- B. Accumulate Imbalance ---
            # Signed Volume = Direction * Size
            imbalance = tick_dir * size
            cum_imbalance += imbalance

            # --- C. Update Candle State ---
            if current_bar is None:
                current_bar = {
                    'bar_start': ts,
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': size,
                    'tick_count': 1,
                    'vwap_sum': price * size  # Helper for VWAP
                }
            else:
                current_bar['high'] = max(current_bar['high'], price)
                current_bar['low'] = min(current_bar['low'], price)
                current_bar['close'] = price
                current_bar['volume'] += size
                current_bar['tick_count'] += 1
                current_bar['vwap_sum'] += (price * size)

            # --- D. Check Threshold (Sample the Bar) ---
            if abs(cum_imbalance) >= threshold:
                # Finalize the bar
                current_bar['bar_end'] = ts
                current_bar['vwap'] = current_bar['vwap_sum'] / current_bar['volume']
                
                # Cleanup internal keys
                del current_bar['vwap_sum']
                
                bars.append(current_bar)
                
                # Reset
                current_bar = None
                cum_imbalance = 0
        
        return bars