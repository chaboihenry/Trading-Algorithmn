#!/Users/henry/miniconda3/envs/trading/bin/python
"""
Email Notification System for Trading Signals

Sends daily top 5 trade recommendations via email with:
- Professional HTML formatting
- Trade details (symbol, strategy, position size, entry/stop/target)
- Performance metrics and risk assessment
- Links to detailed reports
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd
import os


class EmailSender:
    """Send trading signal notifications via email"""

    def __init__(self,
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587,
                 sender_email: Optional[str] = None,
                 sender_password: Optional[str] = None):
        """
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address (reads from env if None)
            sender_password: App-specific password (reads from env if None)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

        # Read credentials from environment variables
        self.sender_email = sender_email or os.getenv('TRADING_EMAIL_SENDER')
        self.sender_password = sender_password or os.getenv('TRADING_EMAIL_PASSWORD')

        if not self.sender_email or not self.sender_password:
            raise ValueError(
                "Email credentials not found. Set TRADING_EMAIL_SENDER and "
                "TRADING_EMAIL_PASSWORD environment variables, or pass them to __init__"
            )

    def create_trade_email_html(self,
                                trades: pd.DataFrame,
                                ensemble_test_accuracy: float,
                                report_date: str,
                                performance_30d: Optional[Dict] = None) -> str:
        """
        Create HTML email content for top trades

        Args:
            trades: DataFrame with top trades
            ensemble_test_accuracy: Stacked ensemble meta-learner test accuracy
            report_date: Report generation date

        Returns:
            HTML string
        """
        # HTML header with styling
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px 10px 0 0;
            margin: -30px -30px 30px -30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .metric-value.positive {{
            color: #28a745;
        }}
        .metric-value.negative {{
            color: #dc3545;
        }}
        .trade-card {{
            background: #f8f9fa;
            border-left: 5px solid #28a745;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
        }}
        .trade-card.short {{
            border-left-color: #dc3545;
        }}
        .trade-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .trade-symbol {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .trade-badge {{
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}
        .trade-badge.short {{
            background: #dc3545;
        }}
        .trade-details {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 15px;
        }}
        .detail-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px;
            background: white;
            border-radius: 5px;
        }}
        .detail-label {{
            color: #666;
            font-weight: 500;
        }}
        .detail-value {{
            color: #333;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #eee;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .warning-title {{
            font-weight: bold;
            color: #856404;
            margin-bottom: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Daily Trading Signals</h1>
            <p>{report_date}</p>
        </div>
"""

        # 30-day performance summary (if available)
        if performance_30d and performance_30d.get('num_trades', 0) > 0:
            num_trades_30d = performance_30d['num_trades']
            num_wins_30d = performance_30d['num_wins']
            win_rate_30d = performance_30d['win_rate']
            total_return_30d = performance_30d['total_return']

            win_rate_color = '#28a745' if win_rate_30d >= 0.60 else '#ffc107' if win_rate_30d >= 0.50 else '#dc3545'
            return_color = '#28a745' if total_return_30d > 0 else '#dc3545'

            html += f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <div style="text-align: center; font-size: 14px; opacity: 0.9; margin-bottom: 10px;">
            Last 30 Days Performance
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
            <div>
                <div style="font-size: 24px; font-weight: bold;">{num_trades_30d}</div>
                <div style="font-size: 12px; opacity: 0.8;">trades, {num_wins_30d} wins</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {win_rate_color};">{win_rate_30d:.1%}</div>
                <div style="font-size: 12px; opacity: 0.8;">win rate</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: bold; color: {return_color};">{total_return_30d:+.2%}</div>
                <div style="font-size: 12px; opacity: 0.8;">total return</div>
            </div>
        </div>
    </div>
"""

        # Single line showing ensemble test accuracy
        html += f"""
    <div style="text-align: center; margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 8px;">
        <div style="font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">
            Stacked Ensemble Performance
        </div>
        <div style="font-size: 32px; font-weight: bold; color: {'#28a745' if ensemble_test_accuracy > 0.65 else '#dc3545'};">
            {ensemble_test_accuracy:.1%}
        </div>
        <div style="font-size: 12px; color: #999; margin-top: 5px;">
            Test Accuracy (Meta-Learner)
        </div>
    </div>
"""

        # Trades section
        if len(trades) > 0:
            html += f"""
        <h2 style="color: #333; margin-top: 30px;">TOP {len(trades)} SIGNALS FOR {report_date.upper()}</h2>
"""

            for i, row in trades.iterrows():
                direction = "LONG" if row['signal'] == 'BUY' else "SHORT"
                badge_class = "" if direction == "LONG" else " short"
                meta_conf = row.get('meta_confidence', 0.0)

                html += f"""
        <div class="trade-card{badge_class}">
            <div class="trade-header">
                <div class="trade-symbol">#{i+1} {row['symbol']}</div>
                <div class="trade-badge{badge_class}">{direction}</div>
            </div>

            <!-- Meta-Confidence Display -->
            <div style="text-align: center; margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                <span style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px;">
                    Meta-Confidence
                </span>
                <span style="font-size: 18px; font-weight: bold; color: #667eea; margin-left: 10px;">
                    {meta_conf:.1%}
                </span>
            </div>

            <div class="trade-details">
                <div class="detail-item">
                    <span class="detail-label">Entry:</span>
                    <span class="detail-value">${row['close']:.2f}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Stop Loss:</span>
                    <span class="detail-value">${row['stop_loss_price']:.2f}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Take Profit:</span>
                    <span class="detail-value">${row['take_profit_price']:.2f}</span>
                </div>
            </div>
        </div>
"""
        else:
            html += """
        <div class="warning">
            <div class="warning-title">No Trade Recommendations</div>
            <p>No qualifying signals found for today. The system will continue monitoring for opportunities.</p>
        </div>
"""

        # Footer
        html += f"""
        <div class="footer">
            <p><strong>⚠️ Risk Disclaimer:</strong> Past performance is not indicative of future results.
            This is an automated trading system. Always verify signals before executing trades.</p>
            <p style="margin-top: 15px; color: #999;">
                Generated by Automated Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def send_trade_notification(self,
                               recipient_email: str,
                               trades: pd.DataFrame,
                               ensemble_test_accuracy: float = 0.0,
                               ensemble_version: int = 0,
                               performance_30d: Optional[Dict] = None,
                               performance_metrics: Optional[Dict] = None,
                               report_path: Optional[str] = None,
                               subject: Optional[str] = None) -> bool:
        """
        Send trade notification email

        Args:
            recipient_email: Email address to send to
            trades: DataFrame with top trades
            ensemble_test_accuracy: Stacked ensemble test accuracy
            performance_metrics: Portfolio performance metrics (deprecated, kept for compatibility)
            report_path: Path to detailed report file (optional attachment)
            subject: Email subject line (auto-generated if None)

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create message
            message = MIMEMultipart("alternative")

            # Subject
            if subject is None:
                num_trades = len(trades)
                date_str = datetime.now().strftime('%Y-%m-%d')
                version_str = f"v{ensemble_version}" if ensemble_version > 0 else ""

                # Add 30-day performance to subject if available
                if performance_30d and performance_30d.get('num_trades', 0) > 0:
                    perf_str = f"30d: {performance_30d['num_trades']}T, {performance_30d['win_rate']:.0%}WR, {performance_30d['total_return']:+.1%}"
                    subject = f"📊 {num_trades} Signals | {perf_str} | {version_str} ({date_str})"
                else:
                    subject = f"📊 Daily Signals: {num_trades} Trades | Ensemble {version_str} ({date_str})"

            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient_email

            # Create HTML content
            report_date = datetime.now().strftime('%A, %B %d, %Y')
            html_content = self.create_trade_email_html(
                trades=trades,
                ensemble_test_accuracy=ensemble_test_accuracy,
                report_date=report_date,
                performance_30d=performance_30d
            )

            # Attach HTML
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)

            # Attach report file if provided
            if report_path and Path(report_path).exists():
                with open(report_path, "rb") as f:
                    attachment = MIMEBase("application", "octet-stream")
                    attachment.set_payload(f.read())
                    encoders.encode_base64(attachment)

                    filename = Path(report_path).name
                    attachment.add_header(
                        "Content-Disposition",
                        f"attachment; filename= {filename}",
                    )
                    message.attach(attachment)

            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    recipient_email,
                    message.as_string()
                )

            print(f"✅ Email sent successfully to {recipient_email}")
            return True

        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}")
            return False

    def send_test_email(self, recipient_email: str) -> bool:
        """
        Send a test email to verify configuration

        Args:
            recipient_email: Email address to send test to

        Returns:
            True if sent successfully
        """
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from notifications.send_daily_trades import (
            get_stacked_ensemble_test_accuracy,
            get_stacked_ensemble_model_version,
            get_30day_performance_summary
        )

        # Create dummy trade data
        test_trades = pd.DataFrame([{
            'symbol': 'TEST',
            'strategy_name': 'StackedEnsembleStrategy',
            'signal': 'BUY',
            'position_size': 0.10,
            'position_value': 10000,
            'num_shares': 100,
            'close': 100.00,
            'stop_loss_price': 95.00,
            'take_profit_price': 110.00,
            'score': 0.5678,
            'signal_strength': 0.85,
            'meta_confidence': 0.72
        }])

        # Get real test accuracy, version, and 30-day performance from database
        test_accuracy = get_stacked_ensemble_test_accuracy()
        model_version = get_stacked_ensemble_model_version()
        perf_30d = get_30day_performance_summary()

        return self.send_trade_notification(
            recipient_email=recipient_email,
            trades=test_trades,
            ensemble_test_accuracy=test_accuracy,
            ensemble_version=model_version,
            performance_30d=perf_30d,
            subject="🧪 Test Email - Trading System"
        )


if __name__ == "__main__":
    """Test email sending when executed directly"""
    import sys

    # Check if email credentials are set
    sender = os.getenv('TRADING_EMAIL_SENDER')
    password = os.getenv('TRADING_EMAIL_PASSWORD')

    if not sender or not password:
        print("\n❌ Email credentials not configured!")
        print("\nTo use email notifications, set these environment variables:")
        print("  export TRADING_EMAIL_SENDER='your-email@gmail.com'")
        print("  export TRADING_EMAIL_PASSWORD='your-app-specific-password'")
        print("\nFor Gmail, create an app-specific password at:")
        print("  https://myaccount.google.com/apppasswords")
        sys.exit(1)

    # Send test email
    email_sender = EmailSender()
    recipient = "henry.vianna123@gmail.com"

    print(f"\nSending test email to {recipient}...")
    success = email_sender.send_test_email(recipient)

    if success:
        print("\n✅ Test email sent successfully!")
        print("Check your inbox at henry.vianna123@gmail.com")
    else:
        print("\n❌ Test email failed!")
