import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicatorCalculator:
    """
    Calculates technical indicators for all assets and populates technical_indicators table

    Database: /Volumes/Vault/85_assets_prediction.db
    Source Table: raw_price_data
    Target Table: technical_indicators

    Indicators Calculated:
        - Moving Averages: SMA (10, 20, 50, 200), EMA (12, 26)
        - Momentum: RSI (7, 14), MACD, Stochastic
        - Volatility: Bollinger Bands, ATR
        - Trend: ADX, +DI, -DI
        - Volume: OBV, Volume SMA, Volume Ratio
        - Price Momentum: 5d, 20d, 60d
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the calculator with database path"""
        self.db_path = db_path
        logger.info(f"Initialized TechnicalIndicatorCalculator")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_all_tickers(self) -> List[str]:
        """Get all tickers from assets table"""
        try:
            conn = self._get_db_connection()
            query = "SELECT symbol_ticker FROM assets ORDER BY symbol_ticker"
            df = pd.read_sql(query, conn)
            conn.close()
            tickers = df['symbol_ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} tickers from assets table")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving tickers: {str(e)}")
            raise

    def _load_price_data(self, ticker: str) -> pd.DataFrame:
        """
        Load OHLCV price data for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with price data
        """
        try:
            conn = self._get_db_connection()
            query = """
                SELECT price_date, open, high, low, close, volume
                FROM raw_price_data
                WHERE symbol_ticker = ?
                ORDER BY price_date
            """
            df = pd.read_sql(query, conn, params=(ticker,))
            conn.close()

            if df.empty:
                logger.warning(f"No price data found for {ticker}")
                return pd.DataFrame()

            # Convert date
            df['price_date'] = pd.to_datetime(df['price_date'])
            return df

        except Exception as e:
            logger.error(f"Error loading price data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def _calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window, min_periods=window).mean()

    def _calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices: Price series
            period: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Returns:
            Dictionary with 'macd', 'signal', 'histogram'
        """
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series,
                             close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator

        Returns:
            Dictionary with '%K' and '%D'
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=3).mean()

        return {
            'k': stoch_k,
            'd': stoch_d
        }

    def _calculate_bollinger_bands(self, prices: pd.Series,
                                   period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands

        Returns:
            Dictionary with 'upper', 'middle', 'lower', 'width', 'percent'
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = upper - lower

        # BB Percent: where price is relative to bands (0-1)
        bb_percent = (prices - lower) / (upper - lower)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'percent': bb_percent
        }

    def _calculate_atr(self, high: pd.Series, low: pd.Series,
                      close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def _calculate_adx(self, high: pd.Series, low: pd.Series,
                      close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index (ADX) and Directional Indicators

        Returns:
            Dictionary with 'adx', 'plus_di', 'minus_di'
        """
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate True Range
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Smooth with Wilder's moving average
        atr = true_range.rolling(window=period).mean()
        plus_dm_smooth = plus_dm.rolling(window=period).mean()
        minus_dm_smooth = minus_dm.rolling(window=period).mean()

        # Calculate DI
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)

        Args:
            close: Close prices
            volume: Volume

        Returns:
            OBV values
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    def calculate_indicators(self, ticker: str) -> pd.DataFrame:
        """
        Calculate all technical indicators for a single ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with all indicators
        """
        logger.info(f"Calculating technical indicators for {ticker}")

        # Load price data
        df = self._load_price_data(ticker)

        if df.empty:
            logger.warning(f"No data available for {ticker}")
            return pd.DataFrame()

        # Initialize result dataframe
        result = pd.DataFrame({
            'symbol_ticker': ticker,
            'indicator_date': df['price_date']
        })

        # Moving Averages
        result['sma_10'] = self._calculate_sma(df['close'], 10)
        result['sma_20'] = self._calculate_sma(df['close'], 20)
        result['sma_50'] = self._calculate_sma(df['close'], 50)
        result['sma_200'] = self._calculate_sma(df['close'], 200)
        result['ema_12'] = self._calculate_ema(df['close'], 12)
        result['ema_26'] = self._calculate_ema(df['close'], 26)

        # RSI
        result['rsi_14'] = self._calculate_rsi(df['close'], 14)
        result['rsi_7'] = self._calculate_rsi(df['close'], 7)

        # MACD
        macd_data = self._calculate_macd(df['close'])
        result['macd'] = macd_data['macd']
        result['macd_signal'] = macd_data['signal']
        result['macd_histogram'] = macd_data['histogram']

        # Stochastic
        stoch_data = self._calculate_stochastic(df['high'], df['low'], df['close'])
        result['stochastic_k'] = stoch_data['k']
        result['stochastic_d'] = stoch_data['d']

        # Bollinger Bands
        bb_data = self._calculate_bollinger_bands(df['close'])
        result['bb_upper'] = bb_data['upper']
        result['bb_middle'] = bb_data['middle']
        result['bb_lower'] = bb_data['lower']
        result['bb_width'] = bb_data['width']
        result['bb_percent'] = bb_data['percent']

        # ATR
        result['atr_14'] = self._calculate_atr(df['high'], df['low'], df['close'], 14)
        result['atr_20'] = self._calculate_atr(df['high'], df['low'], df['close'], 20)

        # ADX and Directional Indicators
        adx_data = self._calculate_adx(df['high'], df['low'], df['close'])
        result['adx_14'] = adx_data['adx']
        result['plus_di'] = adx_data['plus_di']
        result['minus_di'] = adx_data['minus_di']

        # Volume Indicators
        result['obv'] = self._calculate_obv(df['close'], df['volume'])
        result['volume_sma_20'] = self._calculate_sma(df['volume'], 20)
        result['volume_ratio'] = df['volume'] / result['volume_sma_20']

        # Price Distance to Moving Averages
        result['price_distance_to_sma_50'] = (df['close'] - result['sma_50']) / result['sma_50']
        result['price_distance_to_sma_200'] = (df['close'] - result['sma_200']) / result['sma_200']

        # Price Momentum (rate of change)
        result['price_momentum_5d'] = df['close'].pct_change(periods=5)
        result['price_momentum_20d'] = df['close'].pct_change(periods=20)
        result['price_momentum_60d'] = df['close'].pct_change(periods=60)

        logger.info(f"Calculated {len(result)} days of indicators for {ticker}")
        return result

    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate technical indicators for all assets

        Returns:
            DataFrame with indicators for all assets
        """
        logger.info("Starting technical indicator calculation for all assets")

        tickers = self._get_all_tickers()
        all_indicators = []

        for idx, ticker in enumerate(tickers, 1):
            try:
                indicators = self.calculate_indicators(ticker)

                if not indicators.empty:
                    all_indicators.append(indicators)

                # Log progress every 10 tickers
                if idx % 10 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} tickers processed")

            except Exception as e:
                logger.error(f"Error calculating indicators for {ticker}: {str(e)}")

        if all_indicators:
            df = pd.concat(all_indicators, ignore_index=True)
            logger.info(f"Calculated indicators for {len(df)} total records")
            return df
        else:
            logger.warning("No indicators calculated")
            return pd.DataFrame()

    def populate_indicators_table(self, replace: bool = False) -> None:
        """
        Populate the technical_indicators table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting technical_indicators table population")

        try:
            # Calculate all indicators
            indicators_df = self.calculate_all_indicators()

            if indicators_df.empty:
                logger.error("No indicator data calculated. Aborting database insertion.")
                return

            # Convert date to string format for database
            indicators_df['indicator_date'] = indicators_df['indicator_date'].dt.strftime('%Y-%m-%d')

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM technical_indicators")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, indicator_date FROM technical_indicators",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['indicator_date'])
                )

                # Filter to only new records
                new_records_mask = ~indicators_df.apply(
                    lambda row: (row['symbol_ticker'], row['indicator_date']) in existing_keys,
                    axis=1
                )
                new_indicators_df = indicators_df[new_records_mask]

                if len(new_indicators_df) > 0:
                    new_indicators_df.to_sql('technical_indicators', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_indicators_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                indicators_df.to_sql('technical_indicators', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(indicators_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM technical_indicators")
            final_count = cursor.fetchone()[0]
            logger.info(f"technical_indicators table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    COUNT(*) as record_count,
                    MIN(indicator_date) as earliest_date,
                    MAX(indicator_date) as latest_date
                FROM technical_indicators
                GROUP BY symbol_ticker
                ORDER BY symbol_ticker
                LIMIT 10
            """, conn)

            logger.info(f"\nTechnical Indicators Summary (first 10 tickers):")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # Show recent indicator values for a sample ticker
            sample_df = pd.read_sql("""
                SELECT
                    indicator_date,
                    ROUND(sma_50, 2) as sma_50,
                    ROUND(rsi_14, 2) as rsi_14,
                    ROUND(macd, 2) as macd,
                    ROUND(bb_percent, 2) as bb_pct,
                    ROUND(adx_14, 2) as adx
                FROM technical_indicators
                WHERE symbol_ticker = 'AAPL'
                ORDER BY indicator_date DESC
                LIMIT 5
            """, conn)

            logger.info(f"\nRecent Indicators for AAPL (sample):")
            logger.info(f"\n{sample_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated technical_indicators table")

        except Exception as e:
            logger.error(f"Error populating technical_indicators table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize calculator
    calculator = TechnicalIndicatorCalculator()

    print(f"\n{'='*60}")
    print(f"Technical Indicators Calculation Script")
    print(f"{'='*60}")
    print(f"Database: {calculator.db_path}")

    # Populate database
    print(f"\n{'='*60}")
    print("Calculating technical indicators for all assets...")
    print(f"{'='*60}\n")

    calculator.populate_indicators_table(replace=True)

    print(f"\n{'='*60}")
    print("Technical indicators calculation complete!")
    print(f"{'='*60}\n")