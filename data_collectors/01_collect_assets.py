import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetCollector:
    """
    Populates the assets table with 85 tickers organized by category

    Database: /Volumes/Vault/85_assets_prediction.db
    Table: assets
    """

    def __init__(self, db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> None:
        """Initialize the collector with database path and all 85 assets"""
        self.db_path = db_path

        # ===================================================================
        # core large-cap stocks (30 assets)
        # ===================================================================
        self.tech_leaders = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA','AMZN',
                             'TSLA', 'ORCL', 'CRM', 'ADBE']
        self.financials = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW']
        self.healthcare = ['UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'MRK']
        self.consumer = ['WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT']

        # ===================================================================
        # high-volume momentum stocks (15 assets)
        # ===================================================================
        self.volatile_tech = ['AMD', 'INTC', 'NFLX', 'ROKU', 'SNAP', 'PINS',
                              'SHOP', 'PYPL']
        self.meme_stocks = ['GME', 'AMC', 'BB', 'BBBY', 'NOK', 'PLTR', 'SOFI']

        # ===================================================================
        # sector ETFs (11 assets)
        # ===================================================================
        self.sector_etfs = [
            'SPY',   # S&P 500
            'QQQ',   # Nasdaq-100
            'DIA',   # Dow Jones
            'IWM',   # Russell 2000
            'XLF',   # Financials
            'XLK',   # Technology
            'XLE',   # Energy
            'XLV',   # Healthcare
            'XLY',   # Consumer Discretionary
            'XLI',   # Industrials
            'XLRE'   # Real Estate
        ]

        # ===================================================================
        # volatility & hedging instruments (6 assets)
        # ===================================================================
        self.volatility_hedging = [
            'VXX',   # Volatility (VIX proxy)
            'TLT',   # 20+ Year Treasury
            'GLD',   # Gold
            'SLV',   # Silver
            'UUP',   # US Dollar Index
            'HYG'    # High Yield Bonds
        ]

        # ===================================================================
        # energy & commodities (8 assets)
        # ===================================================================
        self.energy_commodities = [
            'XOM', 'CVX',  # Oil majors
            'OXY', 'SLB',  # Oil volatility plays
            'USO',         # Oil ETF
            'UNG',         # Natural Gas
            'FCX',         # Copper/materials
            'VALE'         # Materials
        ]

        # ===================================================================
        # pairs trading candidates (15 assets)
        # ===================================================================
        self.pairs_trading = [
            'KO', 'PEP',   # Beverages
            'V', 'MA',     # Payments
            'UPS', 'FDX',  # Shipping
            'T', 'VZ',     # Telecom
            'F', 'GM',     # Auto
            'BA', 'LMT',   # Aerospace
            'CVS', 'CI',   # Healthcare
            'COST'         # Retail (pairs with WMT)
        ]

        # Category mapping for database
        self.category_map = {
            **{t: 'Tech Leaders' for t in self.tech_leaders},
            **{t: 'Financials' for t in self.financials},
            **{t: 'Healthcare' for t in self.healthcare},
            **{t: 'Consumer' for t in self.consumer},
            **{t: 'Volatile Tech' for t in self.volatile_tech},
            **{t: 'Meme Stocks' for t in self.meme_stocks},
            **{t: 'Sector ETFs' for t in self.sector_etfs},
            **{t: 'Volatility/Hedging' for t in self.volatility_hedging},
            **{t: 'Energy/Commodities' for t in self.energy_commodities},
            **{t: 'Pairs Trading' for t in self.pairs_trading}
        }

        # Asset type mapping
        self.type_map = {
            **{t: 'Stock' for t in (self.tech_leaders + self.financials +
                                    self.healthcare + self.consumer +
                                    self.volatile_tech + self.meme_stocks +
                                    self.energy_commodities + self.pairs_trading)},
            **{t: 'ETF' for t in (self.sector_etfs + self.volatility_hedging)}
        }

        # aggregate all tickers
        self.all_tickers = (
            self.tech_leaders + self.financials + self.healthcare +
            self.consumer + self.volatile_tech + self.meme_stocks +
            self.sector_etfs + self.volatility_hedging +
            self.energy_commodities + self.pairs_trading
        )

        logger.info(f"Initialized AssetCollector with {len(self.all_tickers)} total assets")

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create and return database connection"""
        return sqlite3.connect(self.db_path)

    def create_assets_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with all asset information"""
        logger.info("Creating assets DataFrame")

        assets_data = []
        for ticker in self.all_tickers:
            asset_info = {
                'symbol_ticker': ticker,
                'name': None,  # Will be populated later
                'asset_type': self.type_map[ticker],
                'sector': None,  # Will be populated later
                'category': self.category_map[ticker],
                'is_pairs_candidate': 1 if ticker in self.pairs_trading else 0,
                'primary_pair': None,  # Will be populated later
                'secondary_pair': None,  # Will be populated later
                'avg_daily_volume_3m': None,  # Will be populated later
                'avg_dollar_volume_3m': None  # Will be populated later
            }
            assets_data.append(asset_info)

        df = pd.DataFrame(assets_data)
        logger.info(f"Created DataFrame with {len(df)} assets")
        return df

    def populate_assets_table(self, replace: bool = False) -> None:
        """
        Populate the assets table in the database

        Args:
            replace: If True, replace existing data. If False, append only new assets.
        """
        logger.info("Starting assets table population")

        try:
            # Create DataFrame
            assets_df = self.create_assets_dataframe()

            # Connect to database
            conn = self._get_db_connection()

            # Check if table exists and has data
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM assets")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not replace:
                logger.info(f"Found {existing_count} existing assets. Inserting only new assets.")
                # Get existing tickers
                existing_df = pd.read_sql("SELECT symbol_ticker FROM assets", conn)
                existing_tickers = set(existing_df['symbol_ticker'].values)

                # Filter to only new tickers
                new_assets_df = assets_df[~assets_df['symbol_ticker'].isin(existing_tickers)]

                if len(new_assets_df) > 0:
                    new_assets_df.to_sql('assets', conn, if_exists='append', index=False)
                    logger.info(f"Inserted {len(new_assets_df)} new assets")
                else:
                    logger.info("No new assets to insert")
            else:
                # Replace or insert all
                if_exists = 'replace' if replace else 'append'
                assets_df.to_sql('assets', conn, if_exists=if_exists, index=False)
                logger.info(f"{'Replaced' if replace else 'Inserted'} {len(assets_df)} assets")

            # Verify insertion
            cursor.execute("SELECT COUNT(*) FROM assets")
            final_count = cursor.fetchone()[0]
            logger.info(f"Assets table now contains {final_count} total assets")

            # Show summary by category
            summary_df = pd.read_sql("""
                SELECT category, asset_type, COUNT(*) as count
                FROM assets
                GROUP BY category, asset_type
                ORDER BY category
            """, conn)
            logger.info(f"\nAssets by category:\n{summary_df.to_string()}")

            conn.close()
            logger.info("Successfully populated assets table")

        except Exception as e:
            logger.error(f"Error populating assets table: {str(e)}")
            raise

    def get_all_assets(self) -> List[str]:
        """Return all 85 asset tickers"""
        return self.all_tickers

    def get_assets_by_category(self, category: str) -> List[str]:
        """Get assets by category"""
        category_map = {
            'tech': self.tech_leaders,
            'financials': self.financials,
            'healthcare': self.healthcare,
            'consumer': self.consumer,
            'volatile_tech': self.volatile_tech,
            'meme': self.meme_stocks,
            'sector_etfs': self.sector_etfs,
            'volatility': self.volatility_hedging,
            'energy': self.energy_commodities,
            'pairs': self.pairs_trading
        }
        return category_map.get(category, [])


if __name__ == "__main__":
    # Initialize collector
    collector = AssetCollector()

    print(f"\n{'='*60}")
    print(f"Asset Collection Script - 85 Assets")
    print(f"{'='*60}")
    print(f"\nTotal assets: {len(collector.all_tickers)}")
    print(f"Database: {collector.db_path}")

    # Show breakdown
    print(f"\nBreakdown:")
    print(f"  Tech Leaders: {len(collector.tech_leaders)}")
    print(f"  Financials: {len(collector.financials)}")
    print(f"  Healthcare: {len(collector.healthcare)}")
    print(f"  Consumer: {len(collector.consumer)}")
    print(f"  Volatile Tech: {len(collector.volatile_tech)}")
    print(f"  Meme Stocks: {len(collector.meme_stocks)}")
    print(f"  Sector ETFs: {len(collector.sector_etfs)}")
    print(f"  Volatility/Hedging: {len(collector.volatility_hedging)}")
    print(f"  Energy/Commodities: {len(collector.energy_commodities)}")
    print(f"  Pairs Trading: {len(collector.pairs_trading)}")

    # Populate database
    print(f"\n{'='*60}")
    print("Populating assets table...")
    print(f"{'='*60}\n")

    collector.populate_assets_table(replace=True)

    print(f"\n{'='*60}")
    print("Asset collection complete!")
    print(f"{'='*60}\n")
