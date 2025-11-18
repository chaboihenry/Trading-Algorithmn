"""
Email Notification System

Sends daily trade recommendations via email with:
- Top 5 trades from EnsembleStrategy
- Professional HTML formatting
- Performance metrics
- Risk assessments
"""

from notifications.email_sender import EmailSender
from notifications.send_daily_trades import send_daily_trade_notification

__all__ = [
    'EmailSender',
    'send_daily_trade_notification'
]
