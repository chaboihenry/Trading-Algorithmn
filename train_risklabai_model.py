#!/usr/bin/env python3
"""
Train RiskLabAI Model

This script trains the RiskLabAI models using historical data from Alpaca.
Run this before starting the live trading bot to ensure models are ready.

Usage:
    python train_risklabai_model.py
    python train_risklabai_model.py --symbol AAPL --bars 1000
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from risklabai.strategy.risklabai_strategy import RiskLabAIStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_risklabai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Colors for terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def get_historical_data(symbol: str, num_bars: int = 1000) -> pd.DataFrame:
    """
    Fetch historical data from Alpaca.

    Args:
        symbol: Stock symbol
        num_bars: Number of bars to fetch

    Returns:
        DataFrame with OHLCV data
    """
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        raise ValueError("Alpaca credentials not found in .env file!")

    logger.info(f"Fetching {num_bars} days of historical data for {symbol}...")

    # Create data client
    data_client = StockHistoricalDataClient(api_key, api_secret)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_bars + 100)  # Extra buffer for weekends

    # Request bars
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )

    bars = data_client.get_stock_bars(request)

    # Convert to DataFrame
    df = bars.df

    if symbol in df.index.get_level_values(0):
        df = df.loc[symbol]

    # Reset index to have datetime as column
    df = df.reset_index()

    logger.info(f"  Fetched {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def train_model(
    symbol: str = 'SPY',
    num_bars: int = 1000,
    model_path: str = 'models/risklabai_models.pkl'
):
    """
    Train RiskLabAI model with historical data.

    Args:
        symbol: Symbol to train on
        num_bars: Number of historical bars
        model_path: Path to save model
    """
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'RISKLABAI MODEL TRAINING'.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")

    # Create models directory if it doesn't exist
    models_dir = Path(model_path).parent
    models_dir.mkdir(exist_ok=True)

    # Fetch historical data
    try:
        bars = get_historical_data(symbol, num_bars)
    except Exception as e:
        logger.error(f"Failed to fetch historical data: {e}")
        return False

    if len(bars) < 500:
        logger.error(f"Insufficient data: only {len(bars)} bars (need at least 500)")
        return False

    # Rename columns to match expected format
    bars = bars.rename(columns={'timestamp': 'date'})
    bars = bars.set_index('date')

    # Market data sometimes has duplicates (e.g., same day appearing twice)
    if bars.index.duplicated().any():
        num_dupes = bars.index.duplicated().sum()
        logger.warning(f"Found {num_dupes} duplicate timestamps, keeping last occurrence")
        bars = bars[~bars.index.duplicated(keep='last')]
        logger.info(f"After removing duplicates: {len(bars)} bars")

    # Initialize strategy
    print(f"\n{Colors.BLUE}Initializing RiskLabAI strategy...{Colors.END}")
    strategy = RiskLabAIStrategy(
        profit_taking=2.0,
        stop_loss=2.0,
        max_holding=10,
        n_cv_splits=5
    )

    # Train models
    print(f"\n{Colors.BLUE}Training models on {len(bars)} bars of {symbol}...{Colors.END}")
    print(f"{Colors.YELLOW}This may take a few minutes...{Colors.END}\n")

    results = strategy.train(bars)

    if results['success']:
        print(f"\n{Colors.GREEN}✓ TRAINING SUCCESSFUL!{Colors.END}")
        print(f"{Colors.BOLD}Results:{Colors.END}")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Primary Model Accuracy: {results['primary_accuracy']:.1%}")
        print(f"  Meta Model Accuracy: {results['meta_accuracy']:.1%}")
        print(f"  Top Features: {', '.join(results['top_features'][:3])}")

        # Save model
        strategy.save_models(model_path)
        print(f"\n{Colors.GREEN}✓ Model saved to {model_path}{Colors.END}")

        print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.GREEN}{'READY FOR LIVE TRADING'.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")

        return True
    else:
        print(f"\n{Colors.YELLOW}⚠ TRAINING FAILED{Colors.END}")
        print(f"Reason: {results.get('reason', 'Unknown')}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train RiskLabAI model with historical data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--symbol',
        default='SPY',
        help='Symbol to train on (default: SPY)'
    )
    parser.add_argument(
        '--bars',
        type=int,
        default=1000,
        help='Number of historical bars (default: 1000)'
    )
    parser.add_argument(
        '--model-path',
        default='models/risklabai_models.pkl',
        help='Path to save model (default: models/risklabai_models.pkl)'
    )

    args = parser.parse_args()

    success = train_model(
        symbol=args.symbol,
        num_bars=args.bars,
        model_path=args.model_path
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
