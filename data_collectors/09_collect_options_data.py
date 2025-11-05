import sqlite3
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptionsDataCollector:
    """
    Collects options data for all 85 assets (stocks and ETFs)

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: options_data
    Data Sources:
        - yfinance: Options chain data, implied volatility
        - Calculated: Put/call ratios, IV metrics
    Coverage: All 85 assets in the trading universe
    Note: Polygon.io options endpoint requires premium tier
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path"""
        self.db_path = db_path

        # Polygon.io API key
        self.polygon_api_key = "GqaA97fQfGJTiMc0KX4_kpUhuuhpd5NW"
        self.polygon_base_url = "https://api.polygon.io"

        logger.info(f"Initialized OptionsDataCollector")
        logger.info(f"Data source: yfinance (Polygon.io options require premium tier)")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def _get_stock_tickers(self) -> List[str]:
        """Get all tickers from assets table (stocks and ETFs)"""
        try:
            conn = self._get_db_connection()
            query = """
                SELECT symbol_ticker
                FROM assets
                WHERE asset_type IN ('Stock', 'ETF')
                ORDER BY symbol_ticker
            """
            df = pd.read_sql(query, conn)
            conn.close()
            tickers = df['symbol_ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} tickers from assets table")
            return tickers
        except Exception as e:
            logger.error(f"Error retrieving stock tickers: {str(e)}")
            raise

    def calculate_iv_percentile(self, current_iv: float, iv_history: List[float], lookback_days: int) -> Optional[float]:
        """
        Calculate IV percentile (what percentage of historical IVs are below current IV)
        """
        if not iv_history or current_iv is None:
            return None

        try:
            below_current = sum(1 for iv in iv_history if iv < current_iv)
            percentile = (below_current / len(iv_history)) * 100
            return percentile
        except:
            return None

    def calculate_historical_volatility(self, ticker: str, days: int = 30) -> Optional[float]:
        """
        Calculate historical volatility (realized volatility) from price data

        Args:
            ticker: Stock ticker symbol
            days: Lookback period in days

        Returns:
            Annualized historical volatility (as decimal, e.g., 0.25 = 25%)
        """
        try:
            # Get historical price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days+10)  # Extra buffer for calculation

            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty or len(hist) < days:
                return None

            # Calculate daily returns
            hist['returns'] = hist['Close'].pct_change()

            # Calculate standard deviation of returns
            daily_vol = hist['returns'].std()

            # Annualize (multiply by sqrt of trading days per year)
            annual_vol = daily_vol * (252 ** 0.5)

            return round(annual_vol, 4)

        except Exception as e:
            logger.warning(f"Could not calculate HV for {ticker}: {str(e)}")
            return None

    def get_iv_history(self, ticker: str, lookback_days: int = 252) -> List[float]:
        """
        Get historical IV data by collecting options data over time
        Note: This is a simplified version - in production, you'd store historical IV in database

        For now, we'll use a proxy: calculate multiple HV periods to simulate IV history
        """
        try:
            iv_history = []

            # Use different lookback periods as proxy for IV history
            periods = [20, 30, 40, 50, 60, 90, 120, 180, 252]

            for period in periods:
                hv = self.calculate_historical_volatility(ticker, days=period)
                if hv:
                    iv_history.append(hv)

            return iv_history

        except Exception as e:
            logger.warning(f"Could not get IV history for {ticker}: {str(e)}")
            return []

    def collect_options_data_polygon(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Collect options data for a single ticker using Polygon.io

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with options metrics or None if failed
        """
        try:
            logger.info(f"Collecting options data for {ticker} from Polygon.io")

            # Get current date
            options_date = datetime.now().strftime('%Y-%m-%d')

            # Get stock snapshot for current price
            snapshot_url = f"{self.polygon_base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
            snapshot_params = {'apiKey': self.polygon_api_key}

            snapshot_response = requests.get(snapshot_url, params=snapshot_params, timeout=30)

            if snapshot_response.status_code != 200:
                logger.warning(f"Polygon snapshot API returned status {snapshot_response.status_code} for {ticker}")
                return None

            snapshot_data = snapshot_response.json()

            if snapshot_data.get('status') != 'OK' or not snapshot_data.get('ticker'):
                logger.warning(f"No snapshot data for {ticker}")
                return None

            current_price = snapshot_data['ticker'].get('day', {}).get('c')

            if not current_price:
                logger.warning(f"Could not get current price for {ticker} from Polygon")
                return None

            # Get options contracts (snapshot)
            # Note: Polygon.io's options data requires premium tier for full chain
            # We'll collect what we can with the standard API
            options_snapshot_url = f"{self.polygon_base_url}/v3/snapshot/options/{ticker}"
            options_params = {'apiKey': self.polygon_api_key}

            options_response = requests.get(options_snapshot_url, params=options_params, timeout=30)

            if options_response.status_code != 200:
                logger.warning(f"Polygon options API returned status {options_response.status_code} for {ticker}")
                return None

            options_data = options_response.json()

            if options_data.get('status') != 'OK' or not options_data.get('results'):
                logger.warning(f"No options data available for {ticker} from Polygon")
                return None

            results = options_data['results']

            # Separate calls and puts
            calls = [opt for opt in results if opt.get('details', {}).get('contract_type') == 'call']
            puts = [opt for opt in results if opt.get('details', {}).get('contract_type') == 'put']

            if not calls or not puts:
                logger.warning(f"Insufficient options data for {ticker}")
                return None

            # Calculate metrics from options snapshot
            # Get 30-day expirations (closest to 30 days)
            target_exp = datetime.now() + timedelta(days=30)

            calls_30d = [opt for opt in calls
                        if opt.get('details', {}).get('expiration_date')
                        and abs((datetime.strptime(opt['details']['expiration_date'], '%Y-%m-%d') - target_exp).days) < 15]

            puts_30d = [opt for opt in puts
                       if opt.get('details', {}).get('expiration_date')
                       and abs((datetime.strptime(opt['details']['expiration_date'], '%Y-%m-%d') - target_exp).days) < 15]

            # Calculate average IVs
            iv_30d = None
            if calls_30d:
                ivs = [opt.get('implied_volatility') for opt in calls_30d if opt.get('implied_volatility')]
                iv_30d = sum(ivs) / len(ivs) if ivs else None

            # Get 60-day and 90-day IVs
            target_60d = datetime.now() + timedelta(days=60)
            calls_60d = [opt for opt in calls
                        if opt.get('details', {}).get('expiration_date')
                        and abs((datetime.strptime(opt['details']['expiration_date'], '%Y-%m-%d') - target_60d).days) < 15]

            iv_60d = None
            if calls_60d:
                ivs = [opt.get('implied_volatility') for opt in calls_60d if opt.get('implied_volatility')]
                iv_60d = sum(ivs) / len(ivs) if ivs else None

            target_90d = datetime.now() + timedelta(days=90)
            calls_90d = [opt for opt in calls
                        if opt.get('details', {}).get('expiration_date')
                        and abs((datetime.strptime(opt['details']['expiration_date'], '%Y-%m-%d') - target_90d).days) < 15]

            iv_90d = None
            if calls_90d:
                ivs = [opt.get('implied_volatility') for opt in calls_90d if opt.get('implied_volatility')]
                iv_90d = sum(ivs) / len(ivs) if ivs else None

            # Calculate put/call ratios from 30-day options
            call_volume = sum([opt.get('day', {}).get('volume', 0) for opt in calls_30d])
            put_volume = sum([opt.get('day', {}).get('volume', 0) for opt in puts_30d])

            put_call_ratio_volume = put_volume / call_volume if call_volume > 0 else None

            call_oi = sum([opt.get('open_interest', 0) for opt in calls_30d])
            put_oi = sum([opt.get('open_interest', 0) for opt in puts_30d])

            put_call_ratio_oi = put_oi / call_oi if call_oi > 0 else None

            put_call_ratio = None
            if put_call_ratio_volume and put_call_ratio_oi:
                put_call_ratio = (put_call_ratio_volume + put_call_ratio_oi) / 2

            # Find ATM options
            atm_strike = min([opt['details']['strike_price'] for opt in calls_30d],
                           key=lambda x: abs(x - current_price)) if calls_30d else None

            atm_iv = None
            if atm_strike:
                atm_calls = [opt for opt in calls_30d if opt['details']['strike_price'] == atm_strike]
                if atm_calls and atm_calls[0].get('implied_volatility'):
                    atm_iv = atm_calls[0]['implied_volatility']

            # Calculate skew (25-delta approximation)
            otm_put_strike = current_price * 0.9
            otm_call_strike = current_price * 1.1

            otm_puts = [opt for opt in puts_30d if opt['details']['strike_price'] <= otm_put_strike]
            otm_calls = [opt for opt in calls_30d if opt['details']['strike_price'] >= otm_call_strike]

            otm_put_iv = otm_puts[0].get('implied_volatility') if otm_puts else None
            otm_call_iv = otm_calls[0].get('implied_volatility') if otm_calls else None

            skew_25delta = None
            if otm_put_iv and otm_call_iv:
                skew_25delta = otm_put_iv - otm_call_iv

            # IV term structure
            iv_term_structure_slope = None
            if iv_90d and iv_30d:
                iv_term_structure_slope = iv_90d - iv_30d

            # Create record
            record = {
                'symbol_ticker': ticker,
                'options_date': options_date,
                'implied_volatility_30d': iv_30d,
                'implied_volatility_60d': iv_60d,
                'implied_volatility_90d': iv_90d,
                'iv_rank_1y': None,
                'iv_percentile_1y': None,
                'iv_rank_3y': None,
                'iv_percentile_3y': None,
                'hv_iv_spread': None,
                'put_call_ratio': put_call_ratio,
                'put_call_ratio_volume': put_call_ratio_volume,
                'put_call_ratio_oi': put_call_ratio_oi,
                'put_volume': int(put_volume) if put_volume else None,
                'call_volume': int(call_volume) if call_volume else None,
                'total_options_volume': int(call_volume + put_volume) if (call_volume and put_volume) else None,
                'put_oi': int(put_oi) if put_oi else None,
                'call_oi': int(call_oi) if call_oi else None,
                'put_call_oi_ratio': put_call_ratio_oi,
                'atm_iv': atm_iv,
                'skew_25delta': skew_25delta,
                'iv_term_structure_slope': iv_term_structure_slope
            }

            logger.info(f"Collected options data for {ticker} from Polygon: IV={iv_30d:.2%}, P/C={put_call_ratio:.2f}" if iv_30d and put_call_ratio else f"Collected options data for {ticker} from Polygon")
            return record

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error collecting options data for {ticker}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error collecting options data for {ticker} from Polygon: {str(e)}")
            return None

    def collect_options_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Collect options data for a single ticker using yfinance

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with options metrics or None if failed
        """
        try:
            logger.info(f"Collecting options data for {ticker}")

            stock = yf.Ticker(ticker)

            # Get current date
            options_date = datetime.now().strftime('%Y-%m-%d')

            # Get available expiration dates
            expirations = stock.options

            if not expirations or len(expirations) == 0:
                logger.warning(f"No options data available for {ticker}")
                return None

            # Get nearest expiration (typically 30-45 days out for standard metrics)
            # Find expiration closest to 30 days
            target_date = datetime.now() + timedelta(days=30)
            nearest_exp = min(expirations,
                            key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))

            # Get options chain for nearest expiration
            opt_chain = stock.option_chain(nearest_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts

            if calls.empty or puts.empty:
                logger.warning(f"Empty options chain for {ticker}")
                return None

            # Get current stock price
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            if not current_price:
                logger.warning(f"Could not get current price for {ticker}")
                return None

            # Find ATM (at-the-money) options
            atm_strike = min(calls['strike'].values, key=lambda x: abs(x - current_price))

            # Get ATM call and put
            atm_call = calls[calls['strike'] == atm_strike].iloc[0] if len(calls[calls['strike'] == atm_strike]) > 0 else None
            atm_put = puts[puts['strike'] == atm_strike].iloc[0] if len(puts[puts['strike'] == atm_strike]) > 0 else None

            # Get ATM implied volatility
            atm_iv = None
            if atm_call is not None and 'impliedVolatility' in atm_call:
                atm_iv = atm_call['impliedVolatility']

            # Calculate average IV for different maturities
            # 30-day IV (current chain)
            iv_30d = calls['impliedVolatility'].mean() if 'impliedVolatility' in calls.columns else None

            # Try to get 60 and 90 day IVs
            iv_60d = None
            iv_90d = None

            # Find expiration closest to 60 days
            target_60d = datetime.now() + timedelta(days=60)
            if len(expirations) > 1:
                exp_60d = min(expirations,
                             key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_60d).days))
                try:
                    chain_60d = stock.option_chain(exp_60d)
                    iv_60d = chain_60d.calls['impliedVolatility'].mean() if 'impliedVolatility' in chain_60d.calls.columns else None
                except:
                    pass

            # Find expiration closest to 90 days
            target_90d = datetime.now() + timedelta(days=90)
            if len(expirations) > 2:
                exp_90d = min(expirations,
                             key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_90d).days))
                try:
                    chain_90d = stock.option_chain(exp_90d)
                    iv_90d = chain_90d.calls['impliedVolatility'].mean() if 'impliedVolatility' in chain_90d.calls.columns else None
                except:
                    pass

            # Calculate put/call ratios
            total_call_volume = calls['volume'].sum() if 'volume' in calls.columns else 0
            total_put_volume = puts['volume'].sum() if 'volume' in puts.columns else 0

            put_call_ratio_volume = total_put_volume / total_call_volume if total_call_volume > 0 else None

            # Open interest ratios
            total_call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
            total_put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0

            put_call_ratio_oi = total_put_oi / total_call_oi if total_call_oi > 0 else None

            # Overall put/call ratio (average of volume and OI)
            put_call_ratio = None
            if put_call_ratio_volume and put_call_ratio_oi:
                put_call_ratio = (put_call_ratio_volume + put_call_ratio_oi) / 2

            # Calculate skew (25-delta put IV vs call IV)
            # Approximate 25-delta by using strikes ~10% OTM
            otm_put_strike = current_price * 0.9
            otm_call_strike = current_price * 1.1

            otm_put_iv = None
            otm_call_iv = None

            otm_puts = puts[puts['strike'] <= otm_put_strike]
            if not otm_puts.empty and 'impliedVolatility' in otm_puts.columns:
                otm_put_iv = otm_puts.iloc[0]['impliedVolatility']

            otm_calls = calls[calls['strike'] >= otm_call_strike]
            if not otm_calls.empty and 'impliedVolatility' in otm_calls.columns:
                otm_call_iv = otm_calls.iloc[0]['impliedVolatility']

            skew_25delta = None
            if otm_put_iv and otm_call_iv:
                skew_25delta = otm_put_iv - otm_call_iv

            # IV term structure slope (90d IV - 30d IV)
            iv_term_structure_slope = None
            if iv_90d and iv_30d:
                iv_term_structure_slope = iv_90d - iv_30d

            # Calculate historical volatility (30-day)
            hv_30d = self.calculate_historical_volatility(ticker, days=30)

            # Calculate HV-IV spread
            hv_iv_spread = None
            if hv_30d and iv_30d:
                hv_iv_spread = hv_30d - iv_30d

            # Get IV history for rank/percentile calculations
            iv_history_1y = self.get_iv_history(ticker, lookback_days=252)
            iv_history_3y = self.get_iv_history(ticker, lookback_days=756)

            # Calculate IV rank and percentile
            iv_rank_1y = None
            iv_percentile_1y = None
            if iv_30d and iv_history_1y:
                iv_percentile_1y = self.calculate_iv_percentile(iv_30d, iv_history_1y, 252)
                if len(iv_history_1y) > 0:
                    min_iv = min(iv_history_1y)
                    max_iv = max(iv_history_1y)
                    if max_iv > min_iv:
                        iv_rank_1y = ((iv_30d - min_iv) / (max_iv - min_iv)) * 100

            iv_rank_3y = None
            iv_percentile_3y = None
            if iv_30d and iv_history_3y:
                iv_percentile_3y = self.calculate_iv_percentile(iv_30d, iv_history_3y, 756)
                if len(iv_history_3y) > 0:
                    min_iv = min(iv_history_3y)
                    max_iv = max(iv_history_3y)
                    if max_iv > min_iv:
                        iv_rank_3y = ((iv_30d - min_iv) / (max_iv - min_iv)) * 100

            # Create options record
            record = {
                'symbol_ticker': ticker,
                'options_date': options_date,
                'implied_volatility_30d': iv_30d,
                'implied_volatility_60d': iv_60d,
                'implied_volatility_90d': iv_90d,
                'iv_rank_1y': round(iv_rank_1y, 2) if iv_rank_1y else None,
                'iv_percentile_1y': round(iv_percentile_1y, 2) if iv_percentile_1y else None,
                'iv_rank_3y': round(iv_rank_3y, 2) if iv_rank_3y else None,
                'iv_percentile_3y': round(iv_percentile_3y, 2) if iv_percentile_3y else None,
                'hv_iv_spread': round(hv_iv_spread, 4) if hv_iv_spread else None,
                'put_call_ratio': put_call_ratio,
                'put_call_ratio_volume': put_call_ratio_volume,
                'put_call_ratio_oi': put_call_ratio_oi,
                'put_volume': int(total_put_volume) if total_put_volume else None,
                'call_volume': int(total_call_volume) if total_call_volume else None,
                'total_options_volume': int(total_call_volume + total_put_volume) if (total_call_volume and total_put_volume) else None,
                'put_oi': int(total_put_oi) if total_put_oi else None,
                'call_oi': int(total_call_oi) if total_call_oi else None,
                'put_call_oi_ratio': put_call_ratio_oi,
                'atm_iv': atm_iv,
                'skew_25delta': skew_25delta,
                'iv_term_structure_slope': iv_term_structure_slope
            }

            logger.info(f"Collected options data for {ticker}: IV={iv_30d:.2%}, P/C={put_call_ratio:.2f}" if iv_30d and put_call_ratio else f"Collected options data for {ticker}")
            return record

        except Exception as e:
            logger.error(f"Error collecting options data for {ticker}: {str(e)}")
            return None

    def collect_all_options_data(self) -> pd.DataFrame:
        """
        Collect options data for all tickers with active options
        Uses Polygon.io first, falls back to yfinance if needed
        """
        logger.info("Starting options data collection")

        # Get tickers
        tickers = self._get_stock_tickers()

        all_options_data = []
        success_count = 0
        failed_tickers = []
        polygon_count = 0
        yfinance_count = 0

        for idx, ticker in enumerate(tickers, 1):
            try:
                # Note: Polygon.io options snapshot endpoint requires premium/paid tier
                # Using yfinance for options data (free and reliable)
                options_data = self.collect_options_data(ticker)

                if options_data:
                    all_options_data.append(options_data)
                    success_count += 1
                    yfinance_count += 1
                else:
                    failed_tickers.append(ticker)

                # Log progress every 5 tickers
                if idx % 5 == 0:
                    logger.info(f"Progress: {idx}/{len(tickers)} tickers processed ({success_count} successful)")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                failed_tickers.append(ticker)

        # Convert to DataFrame
        if all_options_data:
            options_df = pd.DataFrame(all_options_data)
            options_df = options_df.sort_values(['symbol_ticker', 'options_date'])

            logger.info(f"\n{'='*60}")
            logger.info(f"Collection Summary:")
            logger.info(f"  Total tickers: {len(tickers)}")
            logger.info(f"  Successful: {success_count}")
            logger.info(f"    - Polygon.io: {polygon_count}")
            logger.info(f"    - yfinance: {yfinance_count}")
            logger.info(f"  Failed: {len(failed_tickers)}")
            if failed_tickers:
                logger.info(f"  Failed tickers: {', '.join(failed_tickers)}")
            logger.info(f"  Total options records: {len(options_df)}")
            logger.info(f"{'='*60}\n")

            return options_df
        else:
            logger.warning("No options data collected")
            return pd.DataFrame()

    def populate_options_data_table(self, replace: bool = False) -> None:
        """
        Populate the options_data table in the database

        Args:
            replace: If True, replace existing data. If False, append only new data.
        """
        logger.info("Starting options_data table population")

        try:
            # Collect all options data
            options_df = self.collect_all_options_data()

            if options_df.empty:
                logger.error("No options data collected. Aborting database insertion.")
                return

            # Connect to database
            conn = self._get_db_connection()
            cursor = conn.cursor()

            # Check existing data
            cursor.execute("SELECT COUNT(*) FROM options_data")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing records. Appending new data only.")

                # Get existing ticker-date combinations
                existing_df = pd.read_sql(
                    "SELECT symbol_ticker, options_date FROM options_data",
                    conn
                )
                existing_keys = set(
                    zip(existing_df['symbol_ticker'], existing_df['options_date'])
                )

                # Filter to only new records
                new_records_mask = ~options_df.apply(
                    lambda row: (row['symbol_ticker'], row['options_date']) in existing_keys,
                    axis=1
                )
                new_options_df = options_df[new_records_mask]

                if len(new_options_df) > 0:
                    new_options_df.to_sql('options_data', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_options_df)} new records")
                else:
                    logger.info("No new records to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                options_df.to_sql('options_data', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced with' if replace else 'Inserted'} {len(options_df)} records")

            # Verify insertion and show summary
            cursor.execute("SELECT COUNT(*) FROM options_data")
            final_count = cursor.fetchone()[0]
            logger.info(f"options_data table now contains {final_count} total records")

            # Show summary statistics
            summary_df = pd.read_sql("""
                SELECT
                    symbol_ticker,
                    ROUND(implied_volatility_30d * 100, 2) as iv_30d_pct,
                    ROUND(put_call_ratio, 2) as pc_ratio,
                    total_options_volume,
                    ROUND(skew_25delta * 100, 2) as skew_pct
                FROM options_data
                ORDER BY total_options_volume DESC
                LIMIT 15
            """, conn)

            logger.info(f"\nOptions Data Summary (top 15 by volume):")
            logger.info(f"\n{summary_df.to_string(index=False)}")

            # IV statistics
            iv_stats_df = pd.read_sql("""
                SELECT
                    AVG(implied_volatility_30d * 100) as avg_iv_30d,
                    MIN(implied_volatility_30d * 100) as min_iv_30d,
                    MAX(implied_volatility_30d * 100) as max_iv_30d,
                    AVG(put_call_ratio) as avg_pc_ratio
                FROM options_data
                WHERE implied_volatility_30d IS NOT NULL
            """, conn)

            logger.info(f"\nOverall IV Statistics:")
            logger.info(f"\n{iv_stats_df.to_string(index=False)}")

            conn.close()
            logger.info("Successfully populated options_data table")

        except Exception as e:
            logger.error(f"Error populating options_data table: {str(e)}")
            raise


if __name__ == "__main__":
    # Initialize collector
    collector = OptionsDataCollector()

    print(f"\n{'='*60}")
    print(f"Options Data Collection Script")
    print(f"{'='*60}")
    print(f"Database: {collector.db_path}")
    print(f"Data Source: yfinance")
    print(f"Target: All 85 assets (stocks and ETFs)")

    # Populate database
    print(f"\n{'='*60}")
    print("Collecting options data...")
    print(f"{'='*60}\n")

    collector.populate_options_data_table(replace=True)

    print(f"\n{'='*60}")
    print("Options data collection complete!")
    print(f"{'='*60}\n")
