"""
Pairs Trading Strategy for Lumibot

Statistical arbitrage strategy that trades cointegrated pairs of stocks.
Adapted from the existing pairs_trading.py to work with Lumibot architecture.

Strategy logic:
1. Identifies cointegrated pairs using Engle-Granger test
2. Calculates spread and z-score
3. Enters positions when spread deviates significantly from mean
4. Exits when spread reverts to mean
"""

import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from lumibot.strategies import Strategy
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class PairsStrategy(Strategy):
    """
    Lumibot implementation of statistical arbitrage pairs trading.

    This strategy:
    - Finds cointegrated pairs using historical price data
    - Monitors spread deviation (z-score)
    - Enters mean-reversion trades when spread is extreme
    - Exits when spread normalizes
    """

    # Strategy parameters
    SLEEPTIME = "24H"  # Check once per day
    LOOKBACK_DAYS = 120  # Historical data for cointegration test
    ZSCORE_ENTRY = 1.5   # Enter trades at this z-score threshold
    ZSCORE_EXIT = 0.5    # Exit trades at this z-score threshold
    MIN_CORRELATION = 0.7  # Minimum correlation to consider pair
    MIN_QUALITY_SCORE = 0.6  # Minimum quality score for trading
    MAX_PAIRS = 5  # Maximum number of pairs to trade simultaneously
    POSITION_SIZE = 0.1  # Use 10% of portfolio per pair (5% per leg)

    def initialize(self, parameters: Dict = None):
        """
        Initialize pairs strategy.

        Args:
            parameters: Optional dict with:
                - db_path: Path to database (default: /Volumes/Vault/85_assets_prediction.db)
                - lookback_days: Days of history for cointegration (default: 120)
                - symbols: List of symbols to consider (default: all in assets table)
        """
        self.sleeptime = self.SLEEPTIME

        # Get parameters
        params = parameters or {}
        self.db_path = params.get('db_path', '/Volumes/Vault/85_assets_prediction.db')
        self.lookback_days = params.get('lookback_days', self.LOOKBACK_DAYS)
        self.symbols = params.get('symbols', None)

        # Track active pairs and spreads
        self.active_pairs = {}  # {(s1, s2): {'hedge_ratio': float, 'position': 'long'/'short'}}
        self.cointegrated_pairs = []  # List of (s1, s2, correlation, pvalue, quality_score)

        # Find cointegrated pairs on initialization
        self._find_cointegrated_pairs_from_db()

        logger.info("Pairs Strategy initialized")
        logger.info(f"Found {len(self.cointegrated_pairs)} cointegrated pairs")
        logger.info(f"Will trade top {self.MAX_PAIRS} pairs")

    def _get_price_data_from_db(self, days: int = None) -> pd.DataFrame:
        """
        Fetch price data from database.

        Args:
            days: Number of days to fetch (default: self.lookback_days)

        Returns:
            DataFrame with columns [symbol_ticker, date, close]
        """
        if days is None:
            days = self.lookback_days

        conn = sqlite3.connect(self.db_path)

        if self.symbols:
            # Fetch specific symbols
            placeholders = ','.join(['?' for _ in self.symbols])
            query = f"""
                SELECT symbol_ticker, price_date, close
                FROM raw_price_data
                WHERE symbol_ticker IN ({placeholders})
                  AND price_date >= date('now', '-{days + 20} days')
                ORDER BY symbol_ticker, price_date
            """
            df = pd.read_sql(query, conn, params=self.symbols)
        else:
            # Fetch all symbols
            query = f"""
                SELECT symbol_ticker, price_date, close
                FROM raw_price_data
                WHERE price_date >= date('now', '-{days + 20} days')
                ORDER BY symbol_ticker, price_date
            """
            df = pd.read_sql(query, conn)

        conn.close()
        return df

    def _augmented_dickey_fuller(self, series: np.ndarray) -> Tuple[float, float]:
        """
        Simplified ADF test for stationarity.

        Returns:
            Tuple of (test_statistic, p_value)
        """
        n = len(series)
        if n < 10:
            return 0.0, 1.0

        delta_y = np.diff(series)
        y_lag = series[:-1]

        try:
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            beta = np.linalg.lstsq(X, delta_y, rcond=None)[0]
            residuals = delta_y - X @ beta

            se = np.sqrt(np.sum(residuals**2) / (len(residuals) - len(beta)))
            denominator = np.sqrt(np.sum((y_lag - y_lag.mean())**2))

            if denominator < 1e-10:
                return 0.0, 1.0

            t_stat = beta[1] / (se / denominator)

            # Approximate p-values
            if t_stat < -3.5:
                p_value = 0.01
            elif t_stat < -2.9:
                p_value = 0.05
            else:
                p_value = 0.15

            return t_stat, p_value
        except:
            return 0.0, 1.0

    def _engle_granger_test(self, y: np.ndarray, x: np.ndarray) -> Tuple[float, float, np.ndarray, float]:
        """
        Engle-Granger cointegration test.

        Returns:
            Tuple of (adf_stat, p_value, residuals, hedge_ratio)
        """
        try:
            X_with_const = np.column_stack([np.ones(len(x)), x])
            coefficients = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            alpha, beta = coefficients[0], coefficients[1]

            residuals = y - (alpha + beta * x)
            adf_stat, p_value = self._augmented_dickey_fuller(residuals)

            return adf_stat, p_value, residuals, beta
        except:
            return 0.0, 1.0, np.array([]), 0.0

    def _calculate_half_life(self, residuals: np.ndarray) -> float:
        """Calculate mean reversion half-life."""
        if len(residuals) < 2:
            return 999

        lag_residuals = residuals[:-1]
        delta_residuals = np.diff(residuals)

        try:
            theta = np.linalg.lstsq(
                lag_residuals.reshape(-1, 1),
                delta_residuals,
                rcond=None
            )[0][0]

            if theta < 0:
                half_life = -np.log(2) / np.log(1 + theta)
                return half_life
            else:
                return 999
        except:
            return 999

    def _calculate_quality_score(self, correlation: float, p_value: float, half_life: float) -> float:
        """
        Calculate pair quality score.

        Args:
            correlation: Price correlation
            p_value: Cointegration p-value
            half_life: Mean reversion half-life

        Returns:
            Quality score (0-1, higher is better)
        """
        corr_score = abs(correlation)
        coint_score = 1 - min(p_value, 1.0)

        # Prefer half-life between 5-30 days
        if 5 <= half_life <= 30:
            reversion_score = 1.0
        elif half_life < 5:
            reversion_score = 0.7
        else:
            reversion_score = max(0.3, 30 / half_life)

        total_score = (
            0.3 * corr_score +
            0.5 * coint_score +
            0.2 * reversion_score
        )

        return total_score

    def _find_cointegrated_pairs_from_db(self):
        """
        Find cointegrated pairs from database and store them.
        """
        logger.info("Finding cointegrated pairs...")

        # Get price data
        prices = self._get_price_data_from_db()

        if prices.empty:
            logger.warning("No price data available")
            return

        # Pivot to price matrix (handle duplicates by taking the mean)
        price_matrix = prices.groupby(['price_date', 'symbol_ticker'])['close'].mean().unstack()
        price_matrix = price_matrix.ffill().dropna(axis=1)

        if len(price_matrix.columns) < 2:
            logger.warning("Insufficient symbols for pair analysis")
            return

        symbols = price_matrix.columns.tolist()
        pairs = []

        logger.info(f"Testing {len(symbols) * (len(symbols) - 1) // 2} potential pairs")

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]

                p1 = price_matrix[s1].values
                p2 = price_matrix[s2].values

                # Check correlation
                correlation = np.corrcoef(p1, p2)[0, 1]

                if correlation >= self.MIN_CORRELATION:
                    # Test cointegration
                    adf_stat, p_value, residuals, hedge_ratio = self._engle_granger_test(p1, p2)

                    # Significant cointegration (p < 0.05)
                    if p_value < 0.05:
                        # Calculate quality metrics
                        half_life = self._calculate_half_life(residuals)
                        quality_score = self._calculate_quality_score(correlation, p_value, half_life)

                        pairs.append((s1, s2, correlation, p_value, quality_score, hedge_ratio))

        # Sort by quality
        pairs = sorted(pairs, key=lambda x: x[4], reverse=True)

        self.cointegrated_pairs = pairs[:self.MAX_PAIRS * 3]  # Keep top pairs

        logger.info(f"Found {len(self.cointegrated_pairs)} high-quality pairs")
        if len(self.cointegrated_pairs) > 0:
            top = self.cointegrated_pairs[0]
            logger.info(f"Top pair: {top[0]}-{top[1]} (quality: {top[4]:.3f}, correlation: {top[2]:.3f})")

    def _calculate_current_spread(self, s1: str, s2: str, hedge_ratio: float) -> Tuple[float, float]:
        """
        Calculate current spread and z-score for a pair.

        Args:
            s1: Symbol 1
            s2: Symbol 2
            hedge_ratio: Hedge ratio from cointegration test

        Returns:
            Tuple of (spread, zscore)
        """
        # Get recent prices
        prices = self._get_price_data_from_db(days=self.lookback_days)

        # Filter to our pair
        pair_prices = prices[prices['symbol_ticker'].isin([s1, s2])]
        price_matrix = pair_prices.groupby(['price_date', 'symbol_ticker'])['close'].mean().unstack()

        if s1 not in price_matrix.columns or s2 not in price_matrix.columns:
            return 0.0, 0.0

        p1 = price_matrix[s1].dropna()
        p2 = price_matrix[s2].dropna()

        # Align dates
        common_dates = p1.index.intersection(p2.index)
        p1 = p1.loc[common_dates]
        p2 = p2.loc[common_dates]

        if len(p1) < 20:
            return 0.0, 0.0

        # Calculate spread
        spread = p1 - hedge_ratio * p2

        # Calculate z-score
        spread_mean = spread.mean()
        spread_std = spread.std()

        if spread_std < 1e-10:
            return spread.iloc[-1], 0.0

        current_spread = spread.iloc[-1]
        zscore = (current_spread - spread_mean) / spread_std

        return current_spread, zscore

    def on_trading_iteration(self):
        """
        Main trading logic called every 24 hours.

        Checks all cointegrated pairs and:
        1. Enters new positions when spread is extreme
        2. Exits existing positions when spread normalizes
        """
        logger.info("=" * 80)
        logger.info(f"PAIRS STRATEGY - Trading Iteration at {datetime.now()}")
        logger.info("=" * 80)

        cash = self.get_cash()
        portfolio_value = self.get_portfolio_value()

        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
        logger.info(f"Available cash: ${cash:,.2f}")
        logger.info(f"Active pairs: {len(self.active_pairs)}")

        # Check existing positions for exits
        pairs_to_remove = []
        for pair_key, pair_info in self.active_pairs.items():
            s1, s2 = pair_key
            hedge_ratio = pair_info['hedge_ratio']
            position_type = pair_info['position']  # 'long' or 'short'

            spread, zscore = self._calculate_current_spread(s1, s2, hedge_ratio)

            logger.info(f"Checking pair {s1}-{s2}: zscore={zscore:.2f}, position={position_type}")

            # Exit condition: zscore returned to normal range
            if abs(zscore) < self.ZSCORE_EXIT:
                logger.info(f"Exiting pair {s1}-{s2} (zscore normalized: {zscore:.2f})")

                # Close both legs
                pos1 = self.get_position(s1)
                pos2 = self.get_position(s2)

                if pos1:
                    self.submit_order(self.create_order(s1, pos1.quantity, "sell"))
                if pos2:
                    self.submit_order(self.create_order(s2, pos2.quantity, "sell"))

                pairs_to_remove.append(pair_key)

        # Remove closed pairs
        for pair_key in pairs_to_remove:
            del self.active_pairs[pair_key]

        # Check for new entry opportunities
        if len(self.active_pairs) < self.MAX_PAIRS:
            for pair_data in self.cointegrated_pairs:
                s1, s2, correlation, p_value, quality_score, hedge_ratio = pair_data

                # Skip if already trading this pair
                if (s1, s2) in self.active_pairs or (s2, s1) in self.active_pairs:
                    continue

                # Skip low quality pairs
                if quality_score < self.MIN_QUALITY_SCORE:
                    continue

                # Check current spread
                spread, zscore = self._calculate_current_spread(s1, s2, hedge_ratio)

                logger.info(f"Evaluating pair {s1}-{s2}: zscore={zscore:.2f}, quality={quality_score:.3f}")

                # Entry conditions
                position_size_dollars = portfolio_value * self.POSITION_SIZE

                if zscore > self.ZSCORE_ENTRY:
                    # Spread too high: short s1, long s2
                    logger.info(f"ENTERING SHORT SPREAD: {s1}-{s2}")

                    try:
                        # Get prices
                        price1 = self.get_last_price(s1)
                        price2 = self.get_last_price(s2)

                        # Calculate quantities
                        qty1 = (position_size_dollars / 2) / price1
                        qty2 = (position_size_dollars / 2) / price2

                        # Execute trades
                        self.submit_order(self.create_order(s1, qty1, "sell"))
                        self.submit_order(self.create_order(s2, qty2, "buy"))

                        # Track position
                        self.active_pairs[(s1, s2)] = {
                            'hedge_ratio': hedge_ratio,
                            'position': 'short',
                            'entry_zscore': zscore
                        }

                        logger.info(f"Opened short spread: SELL {qty1:.2f} {s1}, BUY {qty2:.2f} {s2}")

                    except Exception as e:
                        logger.error(f"Error entering position for {s1}-{s2}: {e}")

                elif zscore < -self.ZSCORE_ENTRY:
                    # Spread too low: long s1, short s2
                    logger.info(f"ENTERING LONG SPREAD: {s1}-{s2}")

                    try:
                        price1 = self.get_last_price(s1)
                        price2 = self.get_last_price(s2)

                        qty1 = (position_size_dollars / 2) / price1
                        qty2 = (position_size_dollars / 2) / price2

                        self.submit_order(self.create_order(s1, qty1, "buy"))
                        self.submit_order(self.create_order(s2, qty2, "sell"))

                        self.active_pairs[(s1, s2)] = {
                            'hedge_ratio': hedge_ratio,
                            'position': 'long',
                            'entry_zscore': zscore
                        }

                        logger.info(f"Opened long spread: BUY {qty1:.2f} {s1}, SELL {qty2:.2f} {s2}")

                    except Exception as e:
                        logger.error(f"Error entering position for {s1}-{s2}: {e}")

                # Stop after filling max pairs
                if len(self.active_pairs) >= self.MAX_PAIRS:
                    break

        logger.info("=" * 80)
        logger.info(f"Iteration complete. Active pairs: {len(self.active_pairs)}")
        logger.info("=" * 80)
