import sqlite3
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional
import logging
import yfinance as yf
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PairsPaperTrader:
    """
    Paper trading system for pairs trading strategy

    Database: /Volumes/Vault/85_assets_prediction.db
    Tables Used:
        - pairs_statistics: Source of trading signals
        - paper_trading_positions: Track open/closed positions
        - paper_trading_performance: Track daily P&L and metrics

    Strategy:
        - Identifies high-confidence diverged pairs (z-score > 2.0)
        - Executes long/short positions with virtual capital
        - Tracks entry/exit prices and P&L
        - Risk management: position sizing, stop losses, max positions

    Initial Capital: $1,000
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db",
                 initial_capital: float = 1000.0) -> None:
        """Initialize paper trading system"""
        self.db_path = db_path
        self.initial_capital = initial_capital

        # Trading parameters
        self.z_score_entry_threshold = 1.5  # Enter when |z-score| > 1.5 (relaxed)
        self.z_score_exit_threshold = 0.5   # Exit when |z-score| < 0.5
        self.max_open_positions = 5         # Maximum concurrent positions
        self.position_size_pct = 0.15       # Use 15% of capital per trade
        self.stop_loss_pct = 0.10           # Stop loss at 10% loss
        self.take_profit_pct = 0.20         # Take profit at 20% gain

        # Cointegration requirements (relaxed for paper trading)
        self.min_cointegration_pvalue = 0.20  # p-value < 0.20 (relaxed from 0.05)
        self.min_half_life = 3               # At least 3 days mean reversion
        self.max_half_life = 100             # Max 100 days mean reversion

        logger.info(f"Initialized PairsPaperTrader")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        logger.info(f"Position size: {self.position_size_pct:.1%} per trade")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _initialize_trading_tables(self) -> None:
        """Create paper trading tables if they don't exist"""
        conn = self._get_db_connection()
        cursor = conn.cursor()

        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trading_positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_ticker_1 TEXT NOT NULL,
                symbol_ticker_2 TEXT NOT NULL,
                position_type TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                entry_z_score REAL,
                entry_price_1 REAL NOT NULL,
                entry_price_2 REAL NOT NULL,
                shares_1 INTEGER NOT NULL,
                shares_2 INTEGER NOT NULL,
                capital_allocated REAL NOT NULL,
                hedge_ratio REAL,
                status TEXT NOT NULL,
                exit_date TEXT,
                exit_z_score REAL,
                exit_price_1 REAL,
                exit_price_2 REAL,
                realized_pnl REAL,
                exit_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Performance tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trading_performance (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                performance_date TEXT NOT NULL,
                total_capital REAL NOT NULL,
                open_positions_count INTEGER NOT NULL,
                closed_positions_count INTEGER NOT NULL,
                daily_pnl REAL,
                cumulative_pnl REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Paper trading tables initialized")

    def _get_current_capital(self) -> float:
        """Calculate current available capital"""
        try:
            conn = self._get_db_connection()

            # Get latest performance record
            perf_df = pd.read_sql(
                "SELECT total_capital FROM paper_trading_performance ORDER BY performance_date DESC LIMIT 1",
                conn
            )

            if perf_df.empty:
                # First run, return initial capital
                conn.close()
                return self.initial_capital

            total_capital = perf_df.iloc[0]['total_capital']

            # Subtract capital allocated to open positions
            open_positions_df = pd.read_sql(
                "SELECT SUM(capital_allocated) as allocated FROM paper_trading_positions WHERE status = 'OPEN'",
                conn
            )

            conn.close()

            allocated = open_positions_df.iloc[0]['allocated'] or 0
            available_capital = total_capital - allocated

            return max(0, available_capital)

        except Exception as e:
            logger.error(f"Error calculating current capital: {str(e)}")
            return self.initial_capital

    def _get_latest_price(self, ticker: str) -> Optional[float]:
        """Get the latest price for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')

            if hist.empty:
                logger.warning(f"No price data for {ticker}")
                return None

            return float(hist['Close'].iloc[-1])

        except Exception as e:
            logger.warning(f"Error getting price for {ticker}: {str(e)}")
            return None

    def _get_trading_signals(self) -> pd.DataFrame:
        """
        Get high-confidence pairs trading signals from pairs_statistics table

        Returns:
            DataFrame with actionable trading signals
        """
        try:
            conn = self._get_db_connection()

            # Get latest signals with strong cointegration and divergence
            # Look for diverged pairs regardless of entry_signal
            query = """
                SELECT
                    symbol_ticker_1,
                    symbol_ticker_2,
                    stat_date,
                    spread_zscore,
                    CASE
                        WHEN spread_zscore > ? THEN 'Short_Spread'
                        WHEN spread_zscore < -? THEN 'Long_Spread'
                        ELSE 'None'
                    END as entry_signal,
                    cointegration_pvalue,
                    hedge_ratio,
                    half_life_mean_reversion,
                    spread_volatility_20d,
                    spread_direction
                FROM pairs_statistics
                WHERE stat_date = (SELECT MAX(stat_date) FROM pairs_statistics)
                  AND cointegration_pvalue < ?
                  AND half_life_mean_reversion > ?
                  AND half_life_mean_reversion < ?
                  AND ABS(spread_zscore) > ?
                ORDER BY ABS(spread_zscore) DESC
            """

            signals_df = pd.read_sql(
                query,
                conn,
                params=(self.z_score_entry_threshold,  # For CASE WHEN spread_zscore > ?
                       self.z_score_entry_threshold,   # For CASE WHEN spread_zscore < -?
                       self.min_cointegration_pvalue,
                       self.min_half_life,
                       self.max_half_life,
                       self.z_score_entry_threshold)
            )

            conn.close()

            logger.info(f"Found {len(signals_df)} high-confidence trading signals")
            return signals_df

        except Exception as e:
            logger.error(f"Error getting trading signals: {str(e)}")
            return pd.DataFrame()

    def _check_if_pair_already_open(self, ticker1: str, ticker2: str) -> bool:
        """Check if we already have an open position for this pair"""
        try:
            conn = self._get_db_connection()

            query = """
                SELECT COUNT(*) as count
                FROM paper_trading_positions
                WHERE status = 'OPEN'
                  AND ((symbol_ticker_1 = ? AND symbol_ticker_2 = ?)
                   OR (symbol_ticker_1 = ? AND symbol_ticker_2 = ?))
            """

            result = pd.read_sql(query, conn, params=(ticker1, ticker2, ticker2, ticker1))
            conn.close()

            return result.iloc[0]['count'] > 0

        except Exception as e:
            logger.error(f"Error checking existing position: {str(e)}")
            return True  # Err on the side of caution

    def _calculate_position_size(self, price1: float, price2: float,
                                 hedge_ratio: float, available_capital: float) -> Tuple[int, int, float]:
        """
        Calculate position sizes for the pair

        Args:
            price1: Current price of asset 1
            price2: Current price of asset 2
            hedge_ratio: Hedge ratio from cointegration
            available_capital: Available capital

        Returns:
            Tuple of (shares1, shares2, capital_allocated)
        """
        # Capital to use for this trade
        trade_capital = available_capital * self.position_size_pct

        # For pairs trading: Long asset1, Short asset2 (or vice versa)
        # We need to balance the position according to hedge ratio
        # Spread = Asset1 - hedge_ratio * Asset2

        # Allocate half capital to each leg
        capital_per_leg = trade_capital / 2

        # Calculate shares for asset 1
        shares1 = int(capital_per_leg / price1)

        # Calculate shares for asset 2 based on hedge ratio
        shares2 = int((shares1 * price1 * hedge_ratio) / price2)

        # Calculate actual capital allocated
        capital_allocated = (shares1 * price1) + (shares2 * price2)

        return shares1, shares2, capital_allocated

    def execute_entry_signals(self) -> int:
        """
        Execute entry signals for high-confidence pairs

        Returns:
            Number of positions opened
        """
        logger.info("Checking for entry signals...")

        # Get available capital
        available_capital = self._get_current_capital()
        logger.info(f"Available capital: ${available_capital:,.2f}")

        # Get current open positions count
        conn = self._get_db_connection()
        open_count = pd.read_sql(
            "SELECT COUNT(*) as count FROM paper_trading_positions WHERE status = 'OPEN'",
            conn
        ).iloc[0]['count']
        conn.close()

        if open_count >= self.max_open_positions:
            logger.info(f"Maximum positions ({self.max_open_positions}) already open")
            return 0

        # Get trading signals
        signals_df = self._get_trading_signals()

        if signals_df.empty:
            logger.info("No trading signals found")
            return 0

        positions_opened = 0

        for idx, signal in signals_df.iterrows():
            # Check if we've hit max positions
            if open_count + positions_opened >= self.max_open_positions:
                logger.info(f"Reached max positions limit")
                break

            # Check if we have enough capital
            if available_capital < 100:  # Minimum $100 per trade
                logger.info("Insufficient capital for more positions")
                break

            ticker1 = signal['symbol_ticker_1']
            ticker2 = signal['symbol_ticker_2']

            # Check if already open
            if self._check_if_pair_already_open(ticker1, ticker2):
                logger.info(f"Position already open for {ticker1}-{ticker2}, skipping")
                continue

            # Get current prices
            price1 = self._get_latest_price(ticker1)
            price2 = self._get_latest_price(ticker2)

            if not price1 or not price2:
                logger.warning(f"Could not get prices for {ticker1}-{ticker2}, skipping")
                continue

            # Calculate position size
            shares1, shares2, capital_allocated = self._calculate_position_size(
                price1, price2, signal['hedge_ratio'], available_capital
            )

            if shares1 < 1 or shares2 < 1:
                logger.warning(f"Position too small for {ticker1}-{ticker2}, skipping")
                continue

            # Determine position type based on signal
            # Long_Spread = Buy asset1, Sell asset2
            # Short_Spread = Sell asset1, Buy asset2
            position_type = signal['entry_signal']

            # Record position
            try:
                conn = self._get_db_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT INTO paper_trading_positions (
                        symbol_ticker_1, symbol_ticker_2, position_type,
                        entry_date, entry_z_score, entry_price_1, entry_price_2,
                        shares_1, shares_2, capital_allocated, hedge_ratio, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (
                    ticker1, ticker2, position_type,
                    datetime.now().strftime('%Y-%m-%d'),
                    signal['spread_zscore'],
                    price1, price2,
                    shares1, shares2,
                    capital_allocated,
                    signal['hedge_ratio']
                ))

                conn.commit()
                conn.close()

                logger.info(f"OPENED {position_type}: {ticker1}/{ticker2}")
                logger.info(f"  Z-score: {signal['spread_zscore']:.2f}")
                logger.info(f"  Shares: {shares1} x {ticker1} @ ${price1:.2f}, {shares2} x {ticker2} @ ${price2:.2f}")
                logger.info(f"  Capital allocated: ${capital_allocated:,.2f}")

                positions_opened += 1
                available_capital -= capital_allocated

            except Exception as e:
                logger.error(f"Error recording position for {ticker1}-{ticker2}: {str(e)}")
                continue

            # Rate limiting
            time.sleep(0.5)

        return positions_opened

    def _get_current_spread_zscore(self, ticker1: str, ticker2: str) -> Optional[float]:
        """Get the current spread z-score for a pair"""
        try:
            conn = self._get_db_connection()

            query = """
                SELECT spread_zscore
                FROM pairs_statistics
                WHERE symbol_ticker_1 = ? AND symbol_ticker_2 = ?
                  AND stat_date = (SELECT MAX(stat_date) FROM pairs_statistics)
            """

            result = pd.read_sql(query, conn, params=(ticker1, ticker2))
            conn.close()

            if result.empty:
                return None

            return float(result.iloc[0]['spread_zscore'])

        except Exception as e:
            logger.warning(f"Error getting z-score for {ticker1}-{ticker2}: {str(e)}")
            return None

    def check_exit_conditions(self) -> int:
        """
        Check open positions for exit conditions

        Exit conditions:
        1. Z-score reverts to near zero (< 0.5)
        2. Stop loss hit (10% loss)
        3. Take profit hit (20% gain)

        Returns:
            Number of positions closed
        """
        logger.info("Checking exit conditions for open positions...")

        # Get open positions
        conn = self._get_db_connection()
        open_positions = pd.read_sql(
            "SELECT * FROM paper_trading_positions WHERE status = 'OPEN'",
            conn
        )
        conn.close()

        if open_positions.empty:
            logger.info("No open positions to check")
            return 0

        positions_closed = 0

        for idx, position in open_positions.iterrows():
            ticker1 = position['symbol_ticker_1']
            ticker2 = position['symbol_ticker_2']

            # Get current prices
            current_price1 = self._get_latest_price(ticker1)
            current_price2 = self._get_latest_price(ticker2)

            if not current_price1 or not current_price2:
                logger.warning(f"Could not get current prices for {ticker1}-{ticker2}")
                continue

            # Get current z-score
            current_zscore = self._get_current_spread_zscore(ticker1, ticker2)

            # Calculate P&L
            # For Long_Spread: profit when spread narrows (asset1 up, asset2 down relative)
            # For Short_Spread: profit when spread widens (asset1 down, asset2 up relative)

            entry_value1 = position['shares_1'] * position['entry_price_1']
            entry_value2 = position['shares_2'] * position['entry_price_2']

            current_value1 = position['shares_1'] * current_price1
            current_value2 = position['shares_2'] * current_price2

            if position['position_type'] == 'Long_Spread':
                # Long asset1, Short asset2
                pnl = (current_value1 - entry_value1) - (current_value2 - entry_value2)
            else:  # Short_Spread
                # Short asset1, Long asset2
                pnl = (entry_value1 - current_value1) + (current_value2 - entry_value2)

            pnl_pct = pnl / position['capital_allocated']

            # Check exit conditions
            exit_reason = None

            if current_zscore is not None and abs(current_zscore) < self.z_score_exit_threshold:
                exit_reason = "Mean Reversion"
            elif pnl_pct <= -self.stop_loss_pct:
                exit_reason = "Stop Loss"
            elif pnl_pct >= self.take_profit_pct:
                exit_reason = "Take Profit"

            if exit_reason:
                # Close position
                try:
                    conn = self._get_db_connection()
                    cursor = conn.cursor()

                    cursor.execute("""
                        UPDATE paper_trading_positions
                        SET status = 'CLOSED',
                            exit_date = ?,
                            exit_z_score = ?,
                            exit_price_1 = ?,
                            exit_price_2 = ?,
                            realized_pnl = ?,
                            exit_reason = ?
                        WHERE position_id = ?
                    """, (
                        datetime.now().strftime('%Y-%m-%d'),
                        current_zscore,
                        current_price1,
                        current_price2,
                        pnl,
                        exit_reason,
                        position['position_id']
                    ))

                    conn.commit()
                    conn.close()

                    logger.info(f"CLOSED {position['position_type']}: {ticker1}/{ticker2}")
                    logger.info(f"  Exit reason: {exit_reason}")
                    logger.info(f"  P&L: ${pnl:,.2f} ({pnl_pct:.2%})")
                    logger.info(f"  Entry z-score: {position['entry_z_score']:.2f} -> Exit z-score: {current_zscore:.2f}")

                    positions_closed += 1

                except Exception as e:
                    logger.error(f"Error closing position {position['position_id']}: {str(e)}")
                    continue

            time.sleep(0.5)

        return positions_closed

    def update_performance_metrics(self) -> None:
        """Calculate and store daily performance metrics"""
        logger.info("Updating performance metrics...")

        try:
            conn = self._get_db_connection()

            # Get all closed positions
            closed_positions = pd.read_sql(
                "SELECT * FROM paper_trading_positions WHERE status = 'CLOSED'",
                conn
            )

            # Get open positions value
            open_positions = pd.read_sql(
                "SELECT * FROM paper_trading_positions WHERE status = 'OPEN'",
                conn
            )

            # Calculate total capital
            total_realized_pnl = closed_positions['realized_pnl'].sum() if not closed_positions.empty else 0

            # Get unrealized P&L for open positions
            unrealized_pnl = 0
            for idx, pos in open_positions.iterrows():
                price1 = self._get_latest_price(pos['symbol_ticker_1'])
                price2 = self._get_latest_price(pos['symbol_ticker_2'])

                if price1 and price2:
                    entry_val1 = pos['shares_1'] * pos['entry_price_1']
                    entry_val2 = pos['shares_2'] * pos['entry_price_2']
                    current_val1 = pos['shares_1'] * price1
                    current_val2 = pos['shares_2'] * price2

                    if pos['position_type'] == 'Long_Spread':
                        pnl = (current_val1 - entry_val1) - (current_val2 - entry_val2)
                    else:
                        pnl = (entry_val1 - current_val1) + (current_val2 - entry_val2)

                    unrealized_pnl += pnl

            total_capital = self.initial_capital + total_realized_pnl + unrealized_pnl

            # Calculate cumulative P&L
            cumulative_pnl = total_capital - self.initial_capital

            # Calculate win rate
            win_rate = None
            if not closed_positions.empty:
                winners = len(closed_positions[closed_positions['realized_pnl'] > 0])
                win_rate = winners / len(closed_positions)

            # Get previous day's capital for daily P&L
            prev_perf = pd.read_sql(
                "SELECT total_capital FROM paper_trading_performance ORDER BY performance_date DESC LIMIT 1",
                conn
            )

            daily_pnl = None
            if not prev_perf.empty:
                daily_pnl = total_capital - prev_perf.iloc[0]['total_capital']

            # Insert performance record
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO paper_trading_performance (
                    performance_date, total_capital, open_positions_count,
                    closed_positions_count, daily_pnl, cumulative_pnl, win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime('%Y-%m-%d'),
                total_capital,
                len(open_positions),
                len(closed_positions),
                daily_pnl,
                cumulative_pnl,
                win_rate
            ))

            conn.commit()
            conn.close()

            logger.info(f"Performance updated:")
            logger.info(f"  Total capital: ${total_capital:,.2f}")
            logger.info(f"  Cumulative P&L: ${cumulative_pnl:,.2f} ({(cumulative_pnl/self.initial_capital):.2%})")
            logger.info(f"  Open positions: {len(open_positions)}")
            logger.info(f"  Closed positions: {len(closed_positions)}")
            if win_rate is not None:
                logger.info(f"  Win rate: {win_rate:.2%}")

        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")

    def run_trading_cycle(self) -> None:
        """
        Execute one complete trading cycle:
        1. Check exit conditions for open positions
        2. Look for new entry signals
        3. Update performance metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting trading cycle - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}\n")

        # Initialize tables if needed
        self._initialize_trading_tables()

        # Check exits first
        closed_count = self.check_exit_conditions()
        logger.info(f"Closed {closed_count} positions")

        # Look for entries
        opened_count = self.execute_entry_signals()
        logger.info(f"Opened {opened_count} positions")

        # Update metrics
        self.update_performance_metrics()

        logger.info(f"\n{'='*60}")
        logger.info(f"Trading cycle complete")
        logger.info(f"{'='*60}\n")

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio status as dictionary"""
        conn = self._get_db_connection()

        # Get latest performance
        perf = pd.read_sql(
            "SELECT * FROM paper_trading_performance ORDER BY performance_date DESC LIMIT 1",
            conn
        )

        # Get open positions
        open_pos = pd.read_sql(
            """
            SELECT symbol_ticker_1, symbol_ticker_2, position_type,
                   entry_date, entry_z_score, capital_allocated
            FROM paper_trading_positions
            WHERE status = 'OPEN'
            ORDER BY entry_date DESC
            """,
            conn
        )

        # Get recent closed positions
        closed_pos = pd.read_sql(
            """
            SELECT symbol_ticker_1, symbol_ticker_2, position_type,
                   entry_date, exit_date, realized_pnl, exit_reason
            FROM paper_trading_positions
            WHERE status = 'CLOSED'
            ORDER BY exit_date DESC
            LIMIT 10
            """,
            conn
        )

        conn.close()

        # Prepare summary dictionary
        summary = {
            'total_capital': 0.0,
            'cumulative_pnl': 0.0,
            'return_pct': 0.0,
            'open_positions': 0,
            'closed_positions': 0,
            'win_rate': None
        }

        if not perf.empty:
            p = perf.iloc[0]
            summary['total_capital'] = float(p['total_capital'])
            summary['cumulative_pnl'] = float(p['cumulative_pnl'])
            summary['return_pct'] = (float(p['cumulative_pnl']) / self.initial_capital) * 100
            summary['open_positions'] = int(p['open_positions_count'])
            summary['closed_positions'] = int(p['closed_positions_count'])
            if p['win_rate'] is not None:
                summary['win_rate'] = float(p['win_rate']) * 100

        # Print summary
        print(f"\n{'='*60}")
        print(f"PAPER TRADING PORTFOLIO SUMMARY")
        print(f"{'='*60}\n")

        print(f"Total Capital: ${summary['total_capital']:,.2f}")
        print(f"Cumulative P&L: ${summary['cumulative_pnl']:,.2f} ({summary['return_pct']:.2f}%)")
        print(f"Open Positions: {summary['open_positions']}")
        print(f"Closed Positions: {summary['closed_positions']}")
        if summary['win_rate'] is not None:
            print(f"Win Rate: {summary['win_rate']:.2f}%")

        if not open_pos.empty:
            print(f"\n{'='*60}")
            print(f"OPEN POSITIONS ({len(open_pos)}):")
            print(f"{'='*60}")
            print(open_pos.to_string(index=False))

        if not closed_pos.empty:
            print(f"\n{'='*60}")
            print(f"RECENT CLOSED POSITIONS (Last 10):")
            print(f"{'='*60}")
            print(closed_pos.to_string(index=False))

        print(f"\n{'='*60}\n")

        return summary
