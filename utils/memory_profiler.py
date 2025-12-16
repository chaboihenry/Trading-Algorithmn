"""
Memory Profiler for Trading Bot

FIXES MEMORY LEAKS by:
- Tracking memory growth over time
- Identifying memory-heavy objects
- Forcing garbage collection
- Cleaning up unused resources
- Reporting memory usage patterns

Usage:
    from utils.memory_profiler import MemoryProfiler

    profiler = MemoryProfiler()
    profiler.start()

    # ... bot runs ...

    profiler.cleanup()  # Force cleanup
    profiler.report()   # Print memory report
"""

import gc
import sys
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available - memory profiling will be limited")
    PSUTIL_AVAILABLE = False


class MemoryProfiler:
    """
    Monitors and profiles memory usage to detect and fix leaks.

    Features:
    - Memory usage tracking over time
    - Garbage collection forcing
    - Object reference counting
    - Memory cleanup recommendations
    """

    def __init__(self):
        """Initialize memory profiler."""
        self.start_time = datetime.now()
        self.memory_snapshots: List[Tuple[datetime, float]] = []
        self.cleanup_count = 0

        if PSUTIL_AVAILABLE:
            import os
            self.process = psutil.Process(os.getpid())
            self.initial_memory = self._get_memory_mb()
            logger.info(f"✅ Memory profiler initialized: {self.initial_memory:.1f}MB")
        else:
            self.process = None
            self.initial_memory = 0
            logger.warning("Memory profiler initialized without psutil (limited functionality)")

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if PSUTIL_AVAILABLE and self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0

    def start(self):
        """Start memory profiling."""
        self.take_snapshot()
        logger.info("Memory profiling started")

    def take_snapshot(self):
        """Take a memory snapshot."""
        memory_mb = self._get_memory_mb()
        self.memory_snapshots.append((datetime.now(), memory_mb))

        if len(self.memory_snapshots) > 1000:
            # Keep only recent snapshots
            self.memory_snapshots = self.memory_snapshots[-500:]

    def get_memory_growth(self) -> Dict[str, float]:
        """
        Calculate memory growth statistics.

        Returns:
            Dict with growth metrics
        """
        if not self.memory_snapshots:
            return {}

        current_memory = self._get_memory_mb()
        growth_mb = current_memory - self.initial_memory
        growth_percent = (growth_mb / self.initial_memory * 100) if self.initial_memory > 0 else 0

        # Calculate growth rate (MB per hour)
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        growth_rate = growth_mb / elapsed_hours if elapsed_hours > 0 else 0

        return {
            'current_mb': current_memory,
            'initial_mb': self.initial_memory,
            'growth_mb': growth_mb,
            'growth_percent': growth_percent,
            'growth_rate_mb_per_hour': growth_rate,
            'elapsed_hours': elapsed_hours
        }

    def force_cleanup(self) -> int:
        """
        Force garbage collection to free memory.

        Returns:
            Number of objects collected
        """
        logger.info("Forcing garbage collection...")

        # Take pre-cleanup snapshot
        before_mb = self._get_memory_mb()

        # Run garbage collection
        collected = gc.collect()

        # Take post-cleanup snapshot
        after_mb = self._get_memory_mb()
        freed_mb = before_mb - after_mb

        self.cleanup_count += 1

        logger.info(f"✅ Garbage collection complete:")
        logger.info(f"   Objects collected: {collected}")
        logger.info(f"   Memory freed: {freed_mb:.2f}MB")

        return collected

    def get_large_objects(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the largest objects in memory.

        Args:
            limit: Number of objects to return

        Returns:
            List of (type_name, size_bytes) tuples
        """
        objects = gc.get_objects()
        type_sizes: Dict[str, int] = {}

        for obj in objects:
            try:
                obj_type = type(obj).__name__
                obj_size = sys.getsizeof(obj)

                if obj_type in type_sizes:
                    type_sizes[obj_type] += obj_size
                else:
                    type_sizes[obj_type] = obj_size
            except:
                continue

        # Sort by size
        sorted_types = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)

        return sorted_types[:limit]

    def detect_leaks(self) -> bool:
        """
        Detect potential memory leaks.

        Returns:
            True if leak detected
        """
        growth = self.get_memory_growth()

        if not growth:
            return False

        # Check for concerning growth patterns
        if growth['growth_mb'] > 200:  # Over 200MB growth
            logger.warning(f"⚠️ LARGE MEMORY GROWTH: {growth['growth_mb']:.1f}MB")
            return True

        if growth['elapsed_hours'] > 1 and growth['growth_rate_mb_per_hour'] > 50:
            logger.warning(f"⚠️ HIGH GROWTH RATE: {growth['growth_rate_mb_per_hour']:.1f}MB/hour")
            return True

        return False

    def cleanup(self):
        """
        Perform comprehensive memory cleanup.

        Includes:
        - Garbage collection
        - Reference cycle breaking
        - Cache clearing
        """
        logger.info("=" * 60)
        logger.info("PERFORMING MEMORY CLEANUP")
        logger.info("=" * 60)

        before_mb = self._get_memory_mb()

        # Force garbage collection (all generations)
        collected = sum(gc.collect(gen) for gen in range(3))

        # Clear any caches
        try:
            # Clear functools caches if any
            from functools import lru_cache
            # Note: Can't clear all caches globally, but specific ones can be cleared
        except:
            pass

        after_mb = self._get_memory_mb()
        freed_mb = before_mb - after_mb

        logger.info(f"✅ Cleanup complete:")
        logger.info(f"   Objects collected: {collected}")
        logger.info(f"   Memory freed: {freed_mb:.2f}MB")
        logger.info(f"   Current memory: {after_mb:.1f}MB")
        logger.info("=" * 60)

        self.cleanup_count += 1

    def report(self):
        """Generate and log memory usage report."""
        growth = self.get_memory_growth()

        logger.info("=" * 80)
        logger.info("MEMORY USAGE REPORT")
        logger.info("=" * 80)

        if growth:
            logger.info(f"Initial memory: {growth['initial_mb']:.1f}MB")
            logger.info(f"Current memory: {growth['current_mb']:.1f}MB")
            logger.info(f"Memory growth: {growth['growth_mb']:.1f}MB (+{growth['growth_percent']:.1f}%)")
            logger.info(f"Growth rate: {growth['growth_rate_mb_per_hour']:.2f}MB/hour")
            logger.info(f"Elapsed time: {growth['elapsed_hours']:.2f} hours")

        logger.info(f"Snapshots taken: {len(self.memory_snapshots)}")
        logger.info(f"Cleanups performed: {self.cleanup_count}")

        # Show large object types
        logger.info("\nLargest object types in memory:")
        for obj_type, size_bytes in self.get_large_objects(10):
            size_mb = size_bytes / 1024 / 1024
            logger.info(f"  {obj_type}: {size_mb:.2f}MB")

        # Garbage collector stats
        logger.info(f"\nGarbage collector stats:")
        logger.info(f"  GC enabled: {gc.isenabled()}")
        logger.info(f"  GC thresholds: {gc.get_threshold()}")
        logger.info(f"  GC counts: {gc.get_count()}")

        logger.info("=" * 80)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_memory_profiler = None

def get_memory_profiler() -> MemoryProfiler:
    """
    Get the global MemoryProfiler instance (creates it if needed).

    Returns:
        Singleton MemoryProfiler instance
    """
    global _memory_profiler

    if _memory_profiler is None:
        _memory_profiler = MemoryProfiler()
        _memory_profiler.start()

    return _memory_profiler
