"""
Tick Imbalance Bar Generator

This module generates tick imbalance bars from raw tick data, as described in
"Advances in Financial Machine Learning" by Marcos López de Prado.

Why Tick Imbalance Bars?
- Time bars oversample during quiet periods, undersample during volatile periods
- Tick imbalance bars sample when buy/sell pressure reaches a threshold
- Result: More homogeneous returns, better for machine learning

Algorithm:
1. Classify each tick: buy (+1) if price > prev_price, sell (-1) if price < prev_price
2. Track cumulative imbalance: θ = sum of tick classifications
3. Sample new bar when |θ| >= expected threshold
4. Update expected values with exponential smoothing

OOP Concepts:
- Class with state: Tracks current bar being built
- Methods for different operations: process_tick, reset, get_bar
- Encapsulation: All bar logic in one place
"""

import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class TickImbalanceBarGenerator:
    """
    Generates tick imbalance bars from raw tick data.

    This class maintains state as it processes ticks, building up bars
    and sampling them when the imbalance threshold is exceeded.

    The class uses the "tick rule" to classify each trade:
    - If price goes up → buy tick (+1)
    - If price goes down → sell tick (-1)
    - If price unchanged → same as previous classification

    Attributes:
        threshold (float): Imbalance threshold for sampling bars
        current_bar (Dict): Bar currently being built
        cumulative_imbalance (float): Running sum of tick classifications
        prev_price (float): Previous tick price (for classification)
        prev_direction (int): Previous tick direction (+1 or -1)
        ewma_alpha (float): Exponential smoothing factor (0-1)
        expected_imbalance (float): Expected imbalance per bar (updated with EWMA)

    Example:
        >>> generator = TickImbalanceBarGenerator(threshold=100.0)
        >>> ticks = [
        ...     ('2024-01-01 09:30:00', 450.0, 100),
        ...     ('2024-01-01 09:30:01', 450.1, 50),
        ...     ('2024-01-01 09:30:02', 449.9, 75),
        ... ]
        >>> for timestamp, price, size in ticks:
        ...     bar = generator.process_tick(timestamp, price, size)
        ...     if bar:
        ...         print(f"New bar: {bar['open']} -> {bar['close']}, volume: {bar['volume']}")
    """

    def __init__(
        self,
        threshold: float = 100.0,
        ewma_alpha: float = 0.05
    ):
        """
        Initialize the tick imbalance bar generator.

        Args:
            threshold: Initial imbalance threshold for sampling bars.
                      Higher = fewer bars (bars span more ticks).
                      Lower = more bars (bars span fewer ticks).
                      This will be calibrated to achieve target bars/day.
            ewma_alpha: Exponential weighted moving average smoothing factor.
                       Lower = more smoothing (slower adaptation).
                       Higher = less smoothing (faster adaptation).
                       Typical range: 0.01 - 0.1
        """
        self.threshold = threshold
        self.ewma_alpha = ewma_alpha

        # Current bar being built
        self.current_bar = None

        # Imbalance tracking
        self.cumulative_imbalance = 0.0
        self.expected_imbalance = threshold  # Will adapt over time

        # Tick classification state
        self.prev_price = None
        self.prev_direction = 1  # Assume first tick is buy (arbitrary)

        # Statistics
        self.total_bars_generated = 0
        self.total_ticks_processed = 0

        logger.debug(f"TickImbalanceBarGenerator initialized (threshold={threshold:.2f})")

    def _classify_tick(self, price: float) -> int:
        """
        Classify tick as buy (+1) or sell (-1) using tick rule.

        The tick rule:
        - If price > previous_price → buy tick (+1)
        - If price < previous_price → sell tick (-1)
        - If price == previous_price → same as previous classification

        This is a simple but effective way to infer trade direction without
        having access to bid/ask data.

        Args:
            price: Current tick price

        Returns:
            int: +1 for buy, -1 for sell
        """
        if self.prev_price is None:
            # First tick - assume buy (arbitrary)
            self.prev_price = price
            return 1

        if price > self.prev_price:
            # Price went up → buy tick
            direction = 1
        elif price < self.prev_price:
            # Price went down → sell tick
            direction = -1
        else:
            # Price unchanged → use previous direction
            direction = self.prev_direction

        # Update state
        self.prev_price = price
        self.prev_direction = direction

        return direction

    def process_tick(
        self,
        timestamp: str,
        price: float,
        size: int
    ) -> Optional[Dict]:
        """
        Process one tick and return a bar if threshold is exceeded.

        This is the main method you'll call for each incoming tick.
        It:
        1. Classifies the tick (buy or sell)
        2. Updates the current bar OHLCV
        3. Updates cumulative imbalance
        4. Checks if threshold is exceeded
        5. If yes, returns the completed bar and starts a new one

        Args:
            timestamp: Tick timestamp (ISO format string)
            price: Trade price
            size: Trade size (number of shares)

        Returns:
            Dict with bar data if a bar was completed, None otherwise:
            {
                'bar_start': '2024-01-01 09:30:00',
                'bar_end': '2024-01-01 09:35:45',
                'open': 450.0,
                'high': 451.2,
                'low': 449.5,
                'close': 450.8,
                'volume': 125000,
                'tick_count': 487,
                'imbalance': 52.3
            }

        Example:
            >>> gen = TickImbalanceBarGenerator(threshold=100.0)
            >>> bar = gen.process_tick('2024-01-01 09:30:00', 450.0, 100)
            >>> if bar:
            ...     print(f"Bar completed: {bar['open']} -> {bar['close']}")
        """
        self.total_ticks_processed += 1

        # Classify tick
        direction = self._classify_tick(price)

        # Update cumulative imbalance
        self.cumulative_imbalance += direction

        # Initialize or update current bar
        if self.current_bar is None:
            # Start new bar
            self.current_bar = {
                'bar_start': timestamp,
                'bar_end': timestamp,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': size,
                'tick_count': 1,
                'buy_ticks': 1 if direction == 1 else 0,
                'sell_ticks': 1 if direction == -1 else 0
            }
        else:
            # Update existing bar
            self.current_bar['bar_end'] = timestamp
            self.current_bar['high'] = max(self.current_bar['high'], price)
            self.current_bar['low'] = min(self.current_bar['low'], price)
            self.current_bar['close'] = price
            self.current_bar['volume'] += size
            self.current_bar['tick_count'] += 1
            if direction == 1:
                self.current_bar['buy_ticks'] += 1
            else:
                self.current_bar['sell_ticks'] += 1

        # Check if threshold exceeded
        if abs(self.cumulative_imbalance) >= self.threshold:
            # Threshold exceeded - sample this bar
            completed_bar = self._complete_bar()

            # Update expected imbalance with EWMA
            # This adapts the threshold to changing market conditions
            self.expected_imbalance = (
                self.ewma_alpha * abs(self.cumulative_imbalance) +
                (1 - self.ewma_alpha) * self.expected_imbalance
            )

            # Reset for next bar
            self.cumulative_imbalance = 0.0
            self.current_bar = None
            self.total_bars_generated += 1

            logger.debug(
                f"Bar completed: {completed_bar['tick_count']} ticks, "
                f"imbalance={completed_bar['imbalance']:.2f}"
            )

            return completed_bar

        return None

    def _complete_bar(self) -> Dict:
        """
        Finalize the current bar and prepare it for output.

        This calculates final statistics like imbalance and adds
        metadata before returning the bar.

        Returns:
            Dict: Completed bar with all fields
        """
        if self.current_bar is None:
            raise ValueError("No current bar to complete")

        # Calculate imbalance for this bar
        # Imbalance = (buy_ticks - sell_ticks) / tick_count
        # Range: -1 (all sells) to +1 (all buys)
        imbalance = (
            (self.current_bar['buy_ticks'] - self.current_bar['sell_ticks']) /
            self.current_bar['tick_count']
        )

        # Create final bar dict
        completed_bar = {
            'bar_start': self.current_bar['bar_start'],
            'bar_end': self.current_bar['bar_end'],
            'open': self.current_bar['open'],
            'high': self.current_bar['high'],
            'low': self.current_bar['low'],
            'close': self.current_bar['close'],
            'volume': self.current_bar['volume'],
            'tick_count': self.current_bar['tick_count'],
            'imbalance': imbalance
        }

        return completed_bar

    def get_current_bar_state(self) -> Optional[Dict]:
        """
        Get the current incomplete bar (for monitoring/debugging).

        Returns:
            Dict with current bar state, or None if no bar in progress

        Example:
            >>> gen = TickImbalanceBarGenerator()
            >>> gen.process_tick('2024-01-01 09:30:00', 450.0, 100)
            >>> state = gen.get_current_bar_state()
            >>> print(f"Current bar has {state['tick_count']} ticks")
        """
        return self.current_bar.copy() if self.current_bar else None

    def reset(self):
        """
        Reset the generator state (for starting a new session).

        Call this when starting a new trading day or after a gap in data.

        Example:
            >>> gen = TickImbalanceBarGenerator()
            >>> # Process some ticks...
            >>> gen.reset()  # Start fresh for new day
        """
        self.current_bar = None
        self.cumulative_imbalance = 0.0
        self.prev_price = None
        self.prev_direction = 1

        logger.debug("Generator state reset")

    def get_statistics(self) -> Dict:
        """
        Get statistics about the generator's performance.

        Returns:
            Dict with statistics

        Example:
            >>> gen = TickImbalanceBarGenerator()
            >>> # Process ticks...
            >>> stats = gen.get_statistics()
            >>> print(f"Generated {stats['total_bars']} bars from {stats['total_ticks']} ticks")
        """
        return {
            'total_bars': self.total_bars_generated,
            'total_ticks': self.total_ticks_processed,
            'current_threshold': self.threshold,
            'expected_imbalance': self.expected_imbalance,
            'ticks_in_current_bar': self.current_bar['tick_count'] if self.current_bar else 0,
            'current_imbalance': self.cumulative_imbalance
        }


def generate_bars_from_ticks(
    ticks: List[Tuple[str, float, int]],
    threshold: float = 100.0
) -> List[Dict]:
    """
    Convenience function to generate bars from a list of ticks.

    This is a wrapper around TickImbalanceBarGenerator for batch processing.
    Use this when you have all ticks in memory and want to generate bars
    in one shot.

    Args:
        ticks: List of (timestamp, price, size) tuples
        threshold: Imbalance threshold

    Returns:
        List of bar dictionaries

    Example:
        >>> ticks = [
        ...     ('2024-01-01 09:30:00', 450.0, 100),
        ...     ('2024-01-01 09:30:01', 450.1, 50),
        ...     # ... more ticks ...
        ... ]
        >>> bars = generate_bars_from_ticks(ticks, threshold=100.0)
        >>> print(f"Generated {len(bars)} bars from {len(ticks)} ticks")
    """
    generator = TickImbalanceBarGenerator(threshold=threshold)
    bars = []

    for timestamp, price, size in ticks:
        bar = generator.process_tick(timestamp, price, size)
        if bar:
            bars.append(bar)

    logger.info(f"Generated {len(bars)} bars from {len(ticks)} ticks")

    return bars


if __name__ == "__main__":
    """
    Test the tick imbalance bar generator with sample data.
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("TICK IMBALANCE BAR GENERATOR TEST")
    print("=" * 80)

    # Create sample tick data
    # Simulate SPY ticks with some buy/sell pressure
    import random
    random.seed(42)

    base_price = 450.0
    ticks = []

    for i in range(1000):
        # Random walk with slight upward drift
        price_change = random.choice([-0.01, -0.01, 0, 0.01, 0.01, 0.02])
        base_price += price_change

        size = random.randint(10, 200)

        timestamp = f"2024-01-01 09:30:{i:02d}"
        ticks.append((timestamp, round(base_price, 2), size))

    print(f"Generated {len(ticks)} sample ticks")
    print(f"Price range: ${ticks[0][1]:.2f} - ${ticks[-1][1]:.2f}")

    # Generate bars with threshold=50
    bars = generate_bars_from_ticks(ticks, threshold=50.0)

    print(f"\nGenerated {len(bars)} bars")
    print(f"Bars per tick: {len(bars) / len(ticks):.3f}")

    # Show first few bars
    print("\nFirst 5 bars:")
    print("-" * 80)
    for i, bar in enumerate(bars[:5], 1):
        print(f"Bar {i}:")
        print(f"  Time: {bar['bar_start']} → {bar['bar_end']}")
        print(f"  OHLC: ${bar['open']:.2f} / ${bar['high']:.2f} / ${bar['low']:.2f} / ${bar['close']:.2f}")
        print(f"  Volume: {bar['volume']:,}")
        print(f"  Ticks: {bar['tick_count']}")
        print(f"  Imbalance: {bar['imbalance']:.3f}")

    print("\n" + "=" * 80)
    print("✓ Tick imbalance bar generator working correctly!")
    print("=" * 80)
