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
                                performance_metrics: Dict[str, any],
                                report_date: str) -> str:
        """
        Create HTML email content for top trades

        Args:
            trades: DataFrame with top trades
            performance_metrics: Portfolio performance metrics
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

        # Portfolio metrics section
        if performance_metrics:
            metrics = performance_metrics.get('overall_metrics', {})
            html += """
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value{}">{:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Return</div>
                <div class="metric-value{}">{:.2%}</div>
            </div>
        </div>
""".format(
                " positive" if metrics.get('sharpe_ratio', 0) > 0 else " negative",
                metrics.get('sharpe_ratio', 0),
                metrics.get('win_rate', 0),
                " positive" if metrics.get('total_return', 0) > 0 else " negative",
                metrics.get('total_return', 0)
            )

        # Trades section
        if len(trades) > 0:
            html += """
        <h2 style="color: #333; margin-top: 30px;">Top 5 Recommended Trades (EnsembleStrategy)</h2>
"""

            for i, row in trades.iterrows():
                direction = "LONG" if row['signal'] > 0 else "SHORT"
                badge_class = "" if direction == "LONG" else " short"

                html += f"""
        <div class="trade-card{badge_class}">
            <div class="trade-header">
                <div class="trade-symbol">#{i+1} {row['symbol']}</div>
                <div class="trade-badge{badge_class}">{direction}</div>
            </div>
            <div class="detail-item" style="background: white; margin-bottom: 10px;">
                <span class="detail-label">Strategy:</span>
                <span class="detail-value">{row['strategy_name']}</span>
            </div>
            <div class="trade-details">
                <div class="detail-item">
                    <span class="detail-label">Position Size:</span>
                    <span class="detail-value">{row['position_size']:.2%}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Capital:</span>
                    <span class="detail-value">${row['position_value']:,.0f}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Shares:</span>
                    <span class="detail-value">{row['num_shares']:,}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Entry Price:</span>
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
                <div class="detail-item">
                    <span class="detail-label">Kelly Score:</span>
                    <span class="detail-value">{row['score']:.4f}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Confidence:</span>
                    <span class="detail-value">{row['signal_strength']:.2f}</span>
                </div>
            </div>
        </div>
"""

            total_allocation = trades['position_size'].sum()
            html += f"""
        <div class="detail-item" style="background: #e7f3ff; padding: 15px; margin-top: 20px; border-radius: 8px;">
            <span class="detail-label" style="font-size: 16px;">Total Capital Allocated:</span>
            <span class="detail-value" style="font-size: 18px;">{total_allocation:.2%} (${total_allocation * 100_000:,.0f})</span>
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
                               performance_metrics: Optional[Dict] = None,
                               report_path: Optional[str] = None,
                               subject: Optional[str] = None) -> bool:
        """
        Send trade notification email

        Args:
            recipient_email: Email address to send to
            trades: DataFrame with top trades
            performance_metrics: Portfolio performance metrics
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
                subject = f"📊 Daily Trading Signals: {num_trades} Recommendations ({date_str})"

            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = recipient_email

            # Create HTML content
            report_date = datetime.now().strftime('%A, %B %d, %Y')
            html_content = self.create_trade_email_html(
                trades=trades,
                performance_metrics=performance_metrics,
                report_date=report_date
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
        # Create dummy trade data
        test_trades = pd.DataFrame([{
            'symbol': 'TEST',
            'strategy_name': 'EnsembleStrategy',
            'signal': 1,
            'position_size': 0.10,
            'position_value': 10000,
            'num_shares': 100,
            'close': 100.00,
            'stop_loss_price': 95.00,
            'take_profit_price': 110.00,
            'score': 0.5678,
            'signal_strength': 0.85
        }])

        test_metrics = {
            'overall_metrics': {
                'sharpe_ratio': 1.45,
                'win_rate': 0.58,
                'total_return': 0.12
            }
        }

        return self.send_trade_notification(
            recipient_email=recipient_email,
            trades=test_trades,
            performance_metrics=test_metrics,
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
