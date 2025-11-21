#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Daily Trade Notification Script

Runs after the complete automation process and sends:
- Top 5 trades from EnsembleStrategy (best performing strategy)
- Portfolio performance metrics
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
from backtesting.backtest_engine import BacktestEngine
from notifications.email_sender import EmailSender


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
    print(f"Strategy: EnsembleStrategy (Top {num_trades} trades)")
    print(f"Capital: ${total_capital:,.0f}")
    print("="*80)

    try:
        # Step 1: Get top trades from EnsembleStrategy
        print("\n" + "▶"*40)
        print("STEP 1: SELECTING TOP TRADES (EnsembleStrategy)")
        print("▶"*40)

        ranker = TradeRanker()
        top_trades = ranker.get_ensemble_top_trades(
            num_trades=num_trades,
            total_capital=total_capital
        )

        if len(top_trades) == 0:
            print("\n⚠️  No trades available from EnsembleStrategy")
            print("Sending notification with no trade recommendations...")

        # Step 2: Get portfolio performance metrics
        print("\n" + "▶"*40)
        print("STEP 2: ANALYZING PORTFOLIO PERFORMANCE")
        print("▶"*40)

        engine = BacktestEngine()
        performance = engine.analyze_portfolio_performance(lookback_days=90)

        # Step 3: Get latest report path (if exists)
        print("\n" + "▶"*40)
        print("STEP 3: LOCATING LATEST REPORT")
        print("▶"*40)

        reports_dir = Path("backtesting/results")
        latest_report = None

        if reports_dir.exists():
            md_reports = sorted(reports_dir.glob("daily_report_*.md"), reverse=True)
            if md_reports:
                latest_report = str(md_reports[0])
                print(f"✅ Found latest report: {latest_report}")
            else:
                print("ℹ️  No recent reports found")

        # Step 4: Send email notification
        print("\n" + "▶"*40)
        print("STEP 4: SENDING EMAIL NOTIFICATION")
        print("▶"*40)

        email_sender = EmailSender()

        success = email_sender.send_trade_notification(
            recipient_email=recipient_email,
            trades=top_trades,
            performance_metrics=performance if performance.get('success') else None,
            report_path=latest_report
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
            print(f"  Report:     {'Attached' if latest_report else 'No attachment'}")
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
