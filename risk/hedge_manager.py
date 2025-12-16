"""
Hedge Manager - Profit when the market drops using inverse ETFs.

FIXES BUG: Hedge logic was in combined_strategy but never ran due to no cash.
This module makes hedging a standalone, testable component.

WHY HEDGING MATTERS:
When the market crashes -10%, a traditional portfolio loses $10K on $100K.
With inverse ETFs, you PROFIT from the crash, offsetting losses.

INVERSE ETFs EXPLAINED:
- Regular ETF: SPY tracks S&P 500. SPY up 1% when market up 1%.
- Inverse ETF: SH is -1x S&P 500. SH up 1% when market DOWN 1%.
- Leveraged inverse: SPXS is -3x S&P 500. SPXS up 3% when market down 1%.

OOP CONCEPTS:
- HedgeManager is a CLASS with methods for detecting bearish markets
- It maintains STATE (self.current_hedge_value) between calls
- Uses COMPOSITION - contains a MarketDataClient instance
"""

import logging
from typing import Dict, List, Optional, Tuple
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from data.market_data import get_market_data_client, get_latest_price
from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_PAPER,
    INVERSE_ETFS,
    DEFAULT_INVERSE_ETF,
    MAX_INVERSE_ALLOCATION,
    MARKET_SENTIMENT_SYMBOLS,
    BEARISH_MARKET_THRESHOLD,
    RSI_OVERBOUGHT,
    ENABLE_EXTENDED_HOURS
)

logger = logging.getLogger(__name__)


class HedgeManager:
    """
    Manages inverse ETF positions to profit from market downturns.

    STRATEGY:
    1. Sample major stocks (AAPL, MSFT, GOOGL, NVDA, TSLA)
    2. Calculate % that are overbought (RSI > 70)
    3. If ≥60% overbought → market is topping out → hedge
    4. Buy inverse ETF (SH) to profit when market crashes
    5. Exit hedge when market recovers
    """

    def __init__(self):
        """Initialize hedge manager with Alpaca connection."""
        self.trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            paper=ALPACA_PAPER
        )
        self.market_data = get_market_data_client()
        self.last_volatility_check = None
        self.last_sentiment_check = None
        logger.info("✅ Hedge manager initialized")

    def _get_market_volatility(self) -> Tuple[float, str]:
        """
        Get current market volatility level.

        FIXED (Problem 9): Track volatility to determine check frequency.

        Returns:
            Tuple of (volatility_percent, level_description)
            - volatility_percent: Recent price swing percentage
            - level_description: "low", "normal", "high", or "extreme"
        """
        try:
            from datetime import datetime, timedelta
            import pandas as pd

            # Get VIX (volatility index) or SPY recent moves
            symbol = 'SPY'  # S&P 500 ETF as market proxy

            # Get last 5 days of bars to calculate recent volatility
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=7),
                end=datetime.now()
            )

            bars = data_client.get_stock_bars(request)

            if not bars or not hasattr(bars, 'data') or symbol not in bars.data:
                logger.warning("Could not fetch volatility data, assuming normal")
                return (1.0, "normal")

            # Extract close prices
            symbol_bars = bars.data[symbol]
            closes = [bar.close for bar in symbol_bars]

            if len(closes) < 2:
                return (1.0, "normal")

            # Calculate daily % changes
            changes = []
            for i in range(1, len(closes)):
                pct_change = abs((closes[i] - closes[i-1]) / closes[i-1] * 100)
                changes.append(pct_change)

            # Average daily volatility
            avg_volatility = sum(changes) / len(changes) if changes else 1.0

            # Categorize volatility
            # Typical SPY moves: 0.5-1.5% daily
            if avg_volatility < 0.5:
                level = "low"
            elif avg_volatility < 1.5:
                level = "normal"
            elif avg_volatility < 3.0:
                level = "high"
            else:
                level = "extreme"

            logger.info(f"📊 Market volatility: {avg_volatility:.2f}% daily ({level})")
            self.last_volatility_check = datetime.now()

            return (avg_volatility, level)

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return (1.0, "normal")

    def _is_pre_market(self) -> bool:
        """
        Check if we're currently in pre-market hours.

        FIXED (Problem 9): Detect pre-market to catch early moves.

        Pre-market: 4:00 AM - 9:30 AM ET, Monday-Friday

        Returns:
            True if in pre-market hours
        """
        try:
            import pytz
            from datetime import datetime

            eastern = pytz.timezone('US/Eastern')
            now_eastern = datetime.now(eastern)

            # Weekend check
            if now_eastern.weekday() >= 5:  # Saturday=5, Sunday=6
                return False

            # Pre-market hours: 4:00 AM - 9:30 AM ET
            pre_market_start = now_eastern.replace(hour=4, minute=0, second=0)
            market_open = now_eastern.replace(hour=9, minute=30, second=0)

            return pre_market_start <= now_eastern < market_open

        except Exception as e:
            logger.error(f"Error checking pre-market: {e}")
            return False

    def _is_monday_morning(self) -> bool:
        """
        Check if it's Monday morning before market open.

        FIXED (Problem 9): Detect Monday morning to handle weekend gaps.

        Returns:
            True if Monday before 9:30 AM ET
        """
        try:
            import pytz
            from datetime import datetime

            eastern = pytz.timezone('US/Eastern')
            now_eastern = datetime.now(eastern)

            # Check if Monday (weekday == 0)
            if now_eastern.weekday() != 0:
                return False

            # Check if before market open
            market_open = now_eastern.replace(hour=9, minute=30, second=0)
            return now_eastern < market_open

        except Exception as e:
            logger.error(f"Error checking Monday morning: {e}")
            return False

    def should_check_sentiment_now(self) -> Tuple[bool, str]:
        """
        Determine if we should check market sentiment now.

        FIXED (Problem 9): Dynamic timing based on market conditions.

        Checks more frequently when:
        - High/extreme volatility (every 15-30 min)
        - Pre-market hours (catch early moves)
        - Monday morning (weekend gap protection)
        - First check of the day

        Returns:
            Tuple of (should_check, reason)
        """
        from datetime import datetime, timedelta

        now = datetime.now()

        # Always check on first call
        if self.last_sentiment_check is None:
            return (True, "first_check")

        # Calculate time since last check
        time_since_last = (now - self.last_sentiment_check).total_seconds() / 60  # minutes

        # Monday morning check (weekend gap protection)
        if self._is_monday_morning():
            if time_since_last > 30:  # Check every 30 min on Monday morning
                return (True, "monday_morning_gap_protection")

        # Pre-market check (catch early moves)
        if self._is_pre_market():
            if time_since_last > 30:  # Check every 30 min in pre-market
                return (True, "pre_market_monitoring")

        # Check volatility
        volatility_pct, volatility_level = self._get_market_volatility()

        # High volatility: check more frequently
        if volatility_level == "extreme":
            if time_since_last > 15:  # Every 15 minutes
                return (True, f"extreme_volatility_{volatility_pct:.1f}%")

        if volatility_level == "high":
            if time_since_last > 30:  # Every 30 minutes
                return (True, f"high_volatility_{volatility_pct:.1f}%")

        # Normal conditions: hourly check
        if time_since_last > 60:
            return (True, "normal_hourly_check")

        # Not time to check yet
        minutes_until_next = 60 - time_since_last
        return (False, f"next_check_in_{minutes_until_next:.0f}_minutes")

    def get_recommended_check_interval(self) -> int:
        """
        Get recommended minutes until next sentiment check.

        FIXED (Problem 9): Dynamic interval based on market conditions.

        Returns:
            Minutes until next check (15, 30, or 60)
        """
        # Check current conditions
        is_monday_am = self._is_monday_morning()
        is_pre_market = self._is_pre_market()
        _, volatility_level = self._get_market_volatility()

        # Extreme volatility: check every 15 minutes
        if volatility_level == "extreme":
            return 15

        # High volatility, pre-market, or Monday morning: every 30 minutes
        if volatility_level == "high" or is_pre_market or is_monday_am:
            return 30

        # Normal conditions: hourly
        return 60

    def _get_rsi_from_db(self, symbol: str) -> Optional[float]:
        """
        Get RSI for a symbol from the database with staleness check.

        FIXED (Problem 8): Now checks if data is stale (>1 hour old during market hours)
        and falls back to real-time calculation if needed.

        Args:
            symbol: Stock ticker

        Returns:
            RSI value (0-100) or None if not available
        """
        try:
            import sqlite3
            from datetime import datetime, timedelta
            from config.settings import DB_PATH

            # FIXED: Use context manager to prevent connection leaks
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()

                # FIXED: Also fetch timestamp to check staleness
                query = """
                    SELECT rsi_14, indicator_date
                    FROM technical_indicators
                    WHERE symbol_ticker = ?
                    ORDER BY indicator_date DESC
                    LIMIT 1
                """

                cursor.execute(query, (symbol,))
                result = cursor.fetchone()
                # Connection automatically closed when exiting 'with' block

            if result and result[0] is not None:
                rsi = float(result[0])
                indicator_date_str = result[1]

                # FIXED: Check data freshness
                try:
                    indicator_date = datetime.strptime(indicator_date_str, '%Y-%m-%d')
                    now = datetime.now()
                    data_age_hours = (now - indicator_date).total_seconds() / 3600

                    # FIXED: During market hours, data should be fresh (within 1 hour)
                    # Check if market is currently open
                    is_market_hours = self._is_market_hours()

                    if is_market_hours and data_age_hours > 1.0:
                        logger.warning(f"⚠️ {symbol} RSI data is {data_age_hours:.1f} hours old - STALE!")
                        logger.warning(f"Fetching real-time data instead...")
                        return self._calculate_rsi_realtime(symbol)

                    elif data_age_hours > 24:
                        # Data is more than 1 day old (even outside market hours)
                        logger.warning(f"⚠️ {symbol} RSI data is {data_age_hours:.1f} hours old - very stale")
                        logger.warning(f"Attempting real-time calculation...")
                        return self._calculate_rsi_realtime(symbol)

                    else:
                        logger.debug(f"{symbol} RSI: {rsi:.1f} (age: {data_age_hours:.1f}h)")
                        return rsi

                except ValueError as e:
                    logger.warning(f"Could not parse date '{indicator_date_str}': {e}")
                    # Return RSI anyway but log warning
                    logger.warning(f"Using RSI {rsi:.1f} without staleness check")
                    return rsi

            # No data in database - try real-time calculation
            logger.warning(f"No RSI data in database for {symbol} - calculating real-time")
            return self._calculate_rsi_realtime(symbol)

        except Exception as e:
            logger.error(f"Error getting RSI for {symbol}: {e}")
            return None

    def _is_market_hours(self) -> bool:
        """
        Check if market is currently in trading hours.

        Returns:
            True if market is open, False otherwise
        """
        try:
            from datetime import datetime
            import pytz

            # Get current time in US/Eastern (market timezone)
            eastern = pytz.timezone('US/Eastern')
            now_eastern = datetime.now(eastern)

            # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
            if now_eastern.weekday() >= 5:  # Saturday=5, Sunday=6
                return False

            market_open = now_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_eastern.replace(hour=16, minute=0, second=0, microsecond=0)

            return market_open <= now_eastern <= market_close

        except Exception as e:
            logger.warning(f"Could not check market hours: {e}")
            # Assume market is open to be safe (will use fresher data)
            return True

    def _calculate_rsi_realtime(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate RSI from real-time price data when database is stale.

        FIXED (Problem 8): Fallback method to get fresh RSI when DB data is old.

        Args:
            symbol: Stock ticker
            period: RSI period (default 14)

        Returns:
            RSI value (0-100) or None if calculation fails
        """
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from datetime import datetime, timedelta
            import pandas as pd
            from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

            logger.info(f"Calculating real-time RSI for {symbol}...")

            # Get historical bars (need period+1 days for RSI calculation)
            data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

            # Request daily bars - need extra days to account for weekends/holidays
            # Request 30 calendar days to ensure we get at least 14 trading days
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=datetime.now() - timedelta(days=30),
                end=datetime.now()
            )

            bars = data_client.get_stock_bars(request)

            # Check if we got data back (Alpaca returns BarSet with .data attribute)
            if not bars or not hasattr(bars, 'data') or symbol not in bars.data:
                logger.error(f"No price data returned for {symbol}")
                return None

            # Get bars for this symbol
            symbol_bars = bars.data[symbol]
            if not symbol_bars:
                logger.error(f"Empty bar data for {symbol}")
                return None

            # Convert to DataFrame
            # Alpaca returns list of Bar objects, need to extract data
            bar_data = []
            for bar in symbol_bars:
                bar_data.append({
                    'close': bar.close,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'volume': bar.volume,
                    'timestamp': bar.timestamp
                })

            df = pd.DataFrame(bar_data)
            if len(df) < period:
                logger.error(f"Insufficient data for RSI calculation: {len(df)} bars (need {period})")
                return None

            # Calculate RSI
            closes = df['close'].values
            deltas = pd.Series(closes).diff()

            gain = deltas.where(deltas > 0, 0).rolling(window=period).mean()
            loss = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Get most recent RSI
            current_rsi = float(rsi.iloc[-1])

            logger.info(f"✅ Real-time RSI for {symbol}: {current_rsi:.1f}")
            return current_rsi

        except Exception as e:
            logger.error(f"Failed to calculate real-time RSI for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def check_market_sentiment(self) -> Dict[str, any]:
        """
        Analyze overall market sentiment by sampling major stocks.

        FIXED (Problem 9): Now tracks check time for dynamic scheduling.

        Returns:
            Dict with sentiment analysis:
            {
                'bearish_count': int,
                'total_checked': int,
                'bearish_ratio': float,
                'is_bearish': bool,
                'symbols_overbought': List[str],
                'check_reason': str,
                'next_check_minutes': int
            }
        """
        from datetime import datetime

        bearish_count = 0
        total_checked = 0
        symbols_overbought = []

        # FIXED (Problem 9): Check if we should run now
        should_check, reason = self.should_check_sentiment_now()
        if not should_check:
            logger.info(f"⏭️  Skipping sentiment check: {reason}")
            return {
                'bearish_count': 0,
                'total_checked': 0,
                'bearish_ratio': 0.0,
                'is_bearish': False,
                'symbols_overbought': [],
                'check_reason': reason,
                'next_check_minutes': self.get_recommended_check_interval(),
                'skipped': True
            }

        logger.info(f"Checking market sentiment... (reason: {reason})")

        for symbol in MARKET_SENTIMENT_SYMBOLS:
            rsi = self._get_rsi_from_db(symbol)

            if rsi is not None:
                total_checked += 1

                if rsi > RSI_OVERBOUGHT:
                    bearish_count += 1
                    symbols_overbought.append(symbol)
                    logger.info(f"  📊 {symbol}: RSI {rsi:.1f} (overbought)")
                else:
                    logger.info(f"  📊 {symbol}: RSI {rsi:.1f}")

        # FIXED (Problem 9): Update check timestamp
        self.last_sentiment_check = datetime.now()
        next_check_min = self.get_recommended_check_interval()

        if total_checked == 0:
            logger.warning("⚠️  No RSI data available for sentiment analysis")
            return {
                'bearish_count': 0,
                'total_checked': 0,
                'bearish_ratio': 0.0,
                'is_bearish': False,
                'symbols_overbought': [],
                'check_reason': reason,
                'next_check_minutes': next_check_min,
                'skipped': False
            }

        bearish_ratio = bearish_count / total_checked
        is_bearish = bearish_ratio >= BEARISH_MARKET_THRESHOLD

        logger.info(f"Market sentiment: {bearish_count}/{total_checked} overbought ({bearish_ratio:.1%})")

        if is_bearish:
            logger.warning(f"🔴 BEARISH MARKET DETECTED ({bearish_ratio:.1%} overbought)")
        else:
            logger.info(f"✅ Market is healthy ({bearish_ratio:.1%} overbought)")

        # FIXED (Problem 9): Include timing info in response
        logger.info(f"⏰ Next sentiment check in {next_check_min} minutes")

        return {
            'bearish_count': bearish_count,
            'total_checked': total_checked,
            'bearish_ratio': bearish_ratio,
            'is_bearish': is_bearish,
            'symbols_overbought': symbols_overbought,
            'check_reason': reason,
            'next_check_minutes': next_check_min,
            'skipped': False
        }

    def get_current_hedge_allocation(self) -> Tuple[float, float]:
        """
        Calculate current allocation to inverse ETFs.

        Returns:
            Tuple of (hedge_value, hedge_pct)
        """
        try:
            positions = self.market_data.get_positions()
            portfolio_value = self.market_data.get_portfolio_value()

            hedge_value = 0.0

            for position in positions:
                if position.symbol in INVERSE_ETFS.values():
                    pos_value = float(position.qty) * float(position.current_price)
                    hedge_value += pos_value
                    logger.debug(f"  Hedge: {position.symbol} = ${pos_value:,.2f}")

            hedge_pct = hedge_value / portfolio_value if portfolio_value > 0 else 0.0

            logger.info(f"Current hedge allocation: ${hedge_value:,.2f} ({hedge_pct:.1%})")
            return hedge_value, hedge_pct

        except Exception as e:
            logger.error(f"Error calculating hedge allocation: {e}")
            return 0.0, 0.0

    def create_hedge(self, cash_available: float, portfolio_value: float) -> bool:
        """
        Create a hedge position using inverse ETF.

        Args:
            cash_available: Cash available for trading
            portfolio_value: Total portfolio value

        Returns:
            True if hedge created successfully
        """
        try:
            # Get current hedge allocation
            hedge_value, hedge_pct = self.get_current_hedge_allocation()

            # Check if we're already at max allocation
            if hedge_pct >= MAX_INVERSE_ALLOCATION:
                logger.info(f"Already at max hedge allocation ({hedge_pct:.1%})")
                return False

            # Calculate how much more we can allocate
            max_hedge_value = portfolio_value * MAX_INVERSE_ALLOCATION
            room_for_hedge = max_hedge_value - hedge_value

            # Allocate up to 10% of portfolio per hedge trade
            target_value = min(
                portfolio_value * 0.10,  # 10% of portfolio
                room_for_hedge,           # Remaining room
                cash_available * 0.5      # Max 50% of available cash
            )

            if target_value < 100:
                logger.warning("Not enough cash/room for hedge position")
                return False

            # Use default inverse ETF (SH = 1x inverse S&P 500)
            hedge_symbol = INVERSE_ETFS[DEFAULT_INVERSE_ETF]

            # Get current price
            price = get_latest_price(self.market_data, hedge_symbol)
            if not price:
                logger.error(f"Could not get price for {hedge_symbol}")
                return False

            # Calculate quantity
            quantity = int(target_value / price)

            if quantity < 1:
                logger.warning(f"Quantity too small: {quantity}")
                return False

            logger.warning(f"🛡️  Creating hedge position:")
            logger.warning(f"  Symbol: {hedge_symbol}")
            logger.warning(f"  Quantity: {quantity} shares")
            logger.warning(f"  Price: ${price:.2f}")
            logger.warning(f"  Value: ${quantity * price:,.2f}")
            logger.warning(f"  This will PROFIT if market drops!")

            # Create market order to buy inverse ETF
            order_data = MarketOrderRequest(
                symbol=hedge_symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data)

            logger.info(f"✅ Hedge order submitted (Order ID: {order.id})")
            return True

        except Exception as e:
            logger.error(f"Failed to create hedge: {e}")
            return False

    def exit_hedges(self) -> bool:
        """
        Exit all inverse ETF positions (market is bullish again).

        Returns:
            True if all hedges exited successfully
        """
        try:
            positions = self.market_data.get_positions()
            hedge_positions = [p for p in positions if p.symbol in INVERSE_ETFS.values()]

            if not hedge_positions:
                logger.info("No hedge positions to exit")
                return True

            logger.info(f"📈 Market bullish - exiting {len(hedge_positions)} hedge position(s)")

            success = True
            for position in hedge_positions:
                try:
                    symbol = position.symbol
                    quantity = float(position.qty)

                    logger.info(f"Exiting hedge: {symbol} ({quantity:.2f} shares)")

                    # Create market sell order
                    order_data = MarketOrderRequest(
                        symbol=symbol,
                        qty=quantity,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )

                    order = self.trading_client.submit_order(order_data)
                    logger.info(f"✅ Hedge exit order submitted (Order ID: {order.id})")

                except Exception as e:
                    logger.error(f"Failed to exit hedge {symbol}: {e}")
                    success = False

            return success

        except Exception as e:
            logger.error(f"Error exiting hedges: {e}")
            return False

    def manage_hedges(self, cash_available: float, portfolio_value: float) -> Dict[str, any]:
        """
        Main entry point - check market and manage hedges accordingly.

        FIXED (Problem 9): Now returns sentiment with timing info.

        Call this every trading iteration to maintain appropriate hedging.

        Args:
            cash_available: Cash available for trading
            portfolio_value: Total portfolio value

        Returns:
            Sentiment dict with timing info (includes 'next_check_minutes')
        """
        logger.info("=" * 60)
        logger.info("HEDGE MANAGEMENT")
        logger.info("=" * 60)

        # Check market sentiment (with dynamic timing)
        sentiment = self.check_market_sentiment()

        # If skipped, return early
        if sentiment.get('skipped'):
            logger.info("=" * 60)
            return sentiment

        # Get current hedge status
        hedge_value, hedge_pct = self.get_current_hedge_allocation()

        # Decision logic
        if sentiment['is_bearish']:
            # Market is overbought - hedge if possible
            if hedge_pct < MAX_INVERSE_ALLOCATION:
                logger.warning(f"🔴 Bearish market - creating/increasing hedge")
                self.create_hedge(cash_available, portfolio_value)
            else:
                logger.info(f"Already fully hedged ({hedge_pct:.1%})")

        elif sentiment['bearish_ratio'] < 0.3:
            # Market is healthy - exit hedges if we have any
            if hedge_pct > 0:
                logger.info(f"📈 Market recovered - exiting hedges")
                self.exit_hedges()
            else:
                logger.info("No hedges, market healthy - no action needed")

        else:
            # Market is neutral - maintain current hedges
            logger.info(f"Market neutral ({sentiment['bearish_ratio']:.1%} overbought) - maintaining hedges")

        logger.info("=" * 60)

        # FIXED (Problem 9): Return sentiment with timing info
        return sentiment


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def check_and_hedge():
    """
    One-line function to manage hedges.

    Call this at each trading iteration.
    """
    try:
        manager = HedgeManager()

        # Get portfolio info
        market_data = get_market_data_client()
        cash = market_data.get_cash()
        portfolio_value = market_data.get_portfolio_value()

        # Manage hedges
        manager.manage_hedges(cash, portfolio_value)

    except Exception as e:
        logger.error(f"Error in hedge management: {e}")
