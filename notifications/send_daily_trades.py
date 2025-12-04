#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Daily Trade Notification Script

Runs after the complete automation process and sends:
- Top 5 trades from EnsembleStrategy (best performing strategy)
- Portfolio performance metrics (using walk-forward validation)
- Comprehensive daily report

Sends to: henry.vianna123@gmail.com
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backtesting.trade_ranker import TradeRanker
from backtesting.strategy_validator import StrategyValidator
from notifications.email_sender import EmailSender
import sqlite3


def get_stacked_ensemble_test_accuracy(db_path: str = "/Volumes/Vault/85_assets_prediction.db") -> tuple:
    """
    Get latest test accuracy from model_metadata table + model version

    Returns:
        Tuple of (test_accuracy, model_version, trained_date)
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT test_accuracy, model_version, trained_date
            FROM model_metadata
            WHERE strategy_name = 'StackedEnsembleStrategy'
            ORDER BY trained_date DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        conn.close()

        if result:
            return (result[0], result[1], result[2])
        else:
            # If no metadata entry yet, try to load from joblib file
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / 'strategies'))

            try:
                import joblib
                models_dir = Path(__file__).parent.parent / 'strategies' / 'models'
                meta_model_path = models_dir / 'stacked_ensemble_meta.joblib'

                if meta_model_path.exists():
                    saved_data = joblib.load(meta_model_path)
                    test_acc = saved_data.get('test_accuracy', 0.0)
                    trained_date = saved_data.get('trained_date', 'Unknown')
                    return (test_acc, 'N/A', trained_date)
            except Exception as e:
                print(f"Warning: Could not load from joblib: {e}")

            return (0.0, 'N/A', 'Unknown')

    except Exception as e:
        print(f"Warning: Could not retrieve test accuracy: {e}")
        return (0.0, 'N/A', 'Unknown')


def send_daily_trade_notification(
    recipient_email: str = "henry.vianna123@gmail.com",
    num_trades: int = 5,
    total_capital: float = 1_000
) -> bool:
    """
    Send daily trade notification email

    Args:
        recipient_email: Email address to send to
        num_trades: Number of top trades to include
        total_capital: Total trading capital

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("DAILY TRADE NOTIFICATION SYSTEM")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Recipient: {recipient_email}")
    print(f"Strategy: StackedEnsembleStrategy (Top {num_trades} trades)")
    print(f"Capital: ${total_capital:,.0f}")
    print("="*80)

    try:
        # Step 1: Get top trades from StackedEnsembleStrategy
        print("\n" + "▶"*40)
        print("STEP 1: SELECTING TOP TRADES (StackedEnsembleStrategy)")
        print("▶"*40)

        ranker = TradeRanker()
        top_trades = ranker.get_ensemble_top_trades(
            num_trades=num_trades,
            total_capital=total_capital
        )

        if len(top_trades) == 0:
            print("\n⚠️  No trades available from StackedEnsembleStrategy")
            print("Sending notification with no trade recommendations...")

        # Get ensemble test accuracy (meta-learner performance)
        ensemble_test_acc, ensemble_version, ensemble_trained_date = get_stacked_ensemble_test_accuracy()
        print(f"\nStackedEnsemble Model Info:")
        print(f"  Test Accuracy: {ensemble_test_acc:.1%}")
        print(f"  Version: {ensemble_version}")
        print(f"  Trained: {ensemble_trained_date}")

        # Step 2: Get portfolio performance metrics using walk-forward validation
        # This ensures consistency with the daily report metrics s
        print("\n" + "▶"*40)
        print("STEP 2: WALK-FORWARD VALIDATION (StackedEnsembleStrategy)")
        print("▶"*40)

        validator = StrategyValidator()
        # Try StackedEnsembleStrategy first (new name)
        validation_result = validator.quick_validation('StackedEnsembleStrategy', lookback_days=90)

        # If StackedEnsemble has too few validatable trades (signals too fresh), skip validation
        if not validation_result.get('success') or validation_result.get('num_trades', 0) < 5:
            if validation_result.get('success') == False and 'No valid returns' in validation_result.get('error', ''):
                print("\n⚠️  StackedEnsemble signals too new to validate (need forward price data)")
                print("    Skipping validation - signals will be validated after market close tomorrow")
                validation_result = {'success': False, 'skip_validation': True, 'reason': 'Signals too fresh'}
            else:
                print("\n⚠️  Insufficient validatable trades for StackedEnsembleStrategy")
                print("    This is expected for newly implemented strategies")
                validation_result = {'success': False, 'skip_validation': True, 'reason': 'Insufficient history'}

        # Convert validation result to format expected by email_sender
        performance = None
        if validation_result.get('success'):
            metrics = validation_result.get('metrics', {})
            strategy_name = validation_result.get('strategy_name', 'StackedEnsembleStrategy')
            performance = {
                'success': True,
                'strategy_metrics': {
                    strategy_name: {
                        'sharpe': metrics.get('sharpe_ratio', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'avg_return': metrics.get('total_return', 0),
                        'num_trades': validation_result.get('num_trades', 0)
                    }
                },
                'validation_passed': validation_result.get('passes_validation', False),
                'validation_reason': validation_result.get('validation_reason', '')
            }

        # Step 3: Send email notification (no report attachment)
        print("\n" + "▶"*40)
        print("STEP 3: SENDING EMAIL NOTIFICATION")
        print("▶"*40)

        email_sender = EmailSender()

        success = email_sender.send_trade_notification(
            recipient_email=recipient_email,
            trades=top_trades,
            performance_metrics=performance if performance and performance.get('success') else None,
            report_path=None,  # Removed: report attachments are too verbose
            ensemble_test_accuracy=ensemble_test_acc,
            ensemble_version=ensemble_version if isinstance(ensemble_version, int) else 0
        )

        # Summary
        print("\n" + "="*80)
        if success:
            print("✅ NOTIFICATION SENT SUCCESSFULLY")
            print("="*80)
            print(f"\nEmail Details:")
            print(f"  To:         {recipient_email}")
            print(f"  Trades:     {len(top_trades)} recommendations")
            if len(top_trades) > 0:
                total_allocation = top_trades['position_size'].sum()
                print(f"  Allocation: {total_allocation:.2%} (${total_allocation * total_capital:,.0f})")
            print(f"  Report:     Not attached (available in backtesting/results/)")
        else:
            print("❌ NOTIFICATION FAILED")
            print("="*80)
            print("\nPlease check:")
            print("  1. TRADING_EMAIL_SENDER environment variable is set")
            print("  2. TRADING_EMAIL_PASSWORD environment variable is set")
            print("  3. Gmail app-specific password is correct")
            print("  4. Internet connection is available")

        print("\n" + "="*80 + "\n")

        return success

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    import os

    # Check if email credentials are configured
    sender = os.getenv('TRADING_EMAIL_SENDER')
    password = os.getenv('TRADING_EMAIL_PASSWORD')

    if not sender or not password:
        print("\n" + "="*80)
        print("❌ EMAIL CREDENTIALS NOT CONFIGURED")
        print("="*80)
        print("\nTo enable email notifications, set these environment variables:")
        print("\n  export TRADING_EMAIL_SENDER='your-email@gmail.com'")
        print("  export TRADING_EMAIL_PASSWORD='your-app-specific-password'")
        print("\nFor Gmail, create an app-specific password:")
        print("  1. Go to https://myaccount.google.com/apppasswords")
        print("  2. Select 'Mail' and your device")
        print("  3. Copy the 16-character password")
        print("  4. Set it as TRADING_EMAIL_PASSWORD")
        print("\nAdd these to your shell profile (~/.zshrc or ~/.bash_profile):")
        print("  export TRADING_EMAIL_SENDER='your-email@gmail.com'")
        print("  export TRADING_EMAIL_PASSWORD='xxxx xxxx xxxx xxxx'")
        print("\n" + "="*80 + "\n")
        sys.exit(1)

    # Send daily notification
    success = send_daily_trade_notification(
        recipient_email="henry.vianna123@gmail.com",
        num_trades=5,
        total_capital=1_000
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
