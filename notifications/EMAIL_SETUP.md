# Email Notification Setup Guide

This guide will help you configure email notifications for daily trading signals.

## Overview

The email notification system sends daily trade recommendations to **henry.vianna123@gmail.com** every morning at **10:05 AM** after the complete automation process finishes.

**What you'll receive:**
- Top 5 trades from EnsembleStrategy (the best performing strategy)
- Professional HTML-formatted email with trade details
- Portfolio performance metrics (Sharpe ratio, win rate, returns)
- Entry prices, stop losses, and take profit targets
- Position sizes calculated using Kelly Criterion
- Attached daily report (Markdown format)

---

## Gmail App-Specific Password Setup

Since you're using Gmail, you need to create an **app-specific password** (regular Gmail password won't work).

### Step 1: Enable 2-Factor Authentication

1. Go to [Google Account Security](https://myaccount.google.com/security)
2. Under "Signing in to Google," select **2-Step Verification**
3. Follow the prompts to enable 2FA if not already enabled

### Step 2: Generate App-Specific Password

1. Go to [App Passwords](https://myaccount.google.com/apppasswords)
2. You may need to sign in again
3. At the bottom, you'll see "App passwords"
4. Select app: **Mail**
5. Select device: **Mac** (or your device)
6. Click **Generate**
7. Google will display a 16-character password like: `xxxx xxxx xxxx xxxx`
8. **Copy this password** (you won't be able to see it again)

---

## Configure Environment Variables

### Option 1: Add to Shell Profile (Recommended)

Add these lines to your shell profile (`~/.zshrc` or `~/.bash_profile`):

```bash
# Trading System Email Credentials
export TRADING_EMAIL_SENDER='your-email@gmail.com'
export TRADING_EMAIL_PASSWORD='xxxx xxxx xxxx xxxx'  # App-specific password from step 2
```

**Replace with your actual credentials:**
- `TRADING_EMAIL_SENDER`: Your Gmail address (e.g., `henry.work@gmail.com`)
- `TRADING_EMAIL_PASSWORD`: The 16-character app password from Step 2

**Example:**
```bash
export TRADING_EMAIL_SENDER='henry.work@gmail.com'
export TRADING_EMAIL_PASSWORD='abcd efgh ijkl mnop'
```

After adding, reload your shell:
```bash
source ~/.zshrc  # or source ~/.bash_profile
```

### Option 2: Set for Current Session Only

```bash
export TRADING_EMAIL_SENDER='your-email@gmail.com'
export TRADING_EMAIL_PASSWORD='xxxx xxxx xxxx xxxx'
```

This will only work for the current terminal session.

---

## Verify Configuration

Test that everything is working:

```bash
cd "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"
python notifications/email_sender.py
```

You should see:
```
Sending test email to henry.vianna123@gmail.com...
✅ Email sent successfully to henry.vianna123@gmail.com

✅ Test email sent successfully!
Check your inbox at henry.vianna123@gmail.com
```

**Check your email** at henry.vianna123@gmail.com to verify you received the test email.

---

## Troubleshooting

### Error: "Email credentials not found"

**Problem:** Environment variables are not set.

**Solution:**
1. Check if variables are set: `echo $TRADING_EMAIL_SENDER`
2. If empty, add them to your shell profile (see Option 1 above)
3. Reload your shell: `source ~/.zshrc`

### Error: "Authentication failed" or "Username and Password not accepted"

**Problem:** Using regular Gmail password instead of app-specific password.

**Solution:**
1. Go back to [App Passwords](https://myaccount.google.com/apppasswords)
2. Generate a NEW app-specific password
3. Update `TRADING_EMAIL_PASSWORD` with the new password
4. Make sure there are NO extra spaces in the password

### Error: "SMTPServerDisconnected"

**Problem:** Network or SMTP connection issue.

**Solution:**
1. Check your internet connection
2. Verify Gmail SMTP is not blocked by firewall
3. Try again in a few minutes

### Email sent successfully but not received

**Problem:** Email may be in spam or promotions folder.

**Solution:**
1. Check your **Spam** folder
2. Check your **Promotions** tab (if using Gmail)
3. Add your sender email to contacts to avoid spam filtering

---

## Email Schedule

The email notification is automatically sent:

- **Time:** 10:05 AM ET (Eastern Time)
- **Frequency:** Daily, Monday-Friday
- **Trigger:** After "Validate Strategies & Generate Report" completes (Tier 7)
- **Recipient:** henry.vianna123@gmail.com

You can view the schedule in:
```
master_orchestrator/dependency_graph.yaml
```

Look for:
```yaml
# Tier 7: Email Notification (needs tier 6, runs after all processing complete)
- name: "Send Daily Trade Notification"
  script: "notifications/send_daily_trades.py"
  time: "10:05:00"
  tier: 7
  dependencies: ["Validate Strategies & Generate Report"]
  max_runtime_seconds: 60
  critical: false
```

---

## Email Content Preview

Your daily email will include:

### Header
- **Subject:** 📊 Daily Trading Signals: 5 Recommendations (2025-11-17)
- **Date:** Monday, November 17, 2025

### Metrics Dashboard
- Sharpe Ratio
- Win Rate
- Total Return

### Top 5 Trades
For each trade:
- Symbol and direction (LONG/SHORT)
- Strategy name (EnsembleStrategy)
- Position size (% of capital)
- Number of shares
- Entry price
- Stop loss price
- Take profit price
- Kelly score (confidence)
- Signal strength

### Footer
- Risk disclaimer
- Total capital allocated
- Cash remaining
- Report generation timestamp

### Attachment
- Daily report in Markdown format with full details

---

## Manual Testing

To manually send a test email:

```bash
cd "/Users/henry/Desktop/Programming & Networking/Personal Projects/Integrated Trading Agent"
python notifications/send_daily_trades.py
```

This will:
1. Get top 5 trades from EnsembleStrategy
2. Analyze portfolio performance
3. Generate HTML email
4. Send to henry.vianna123@gmail.com

---

## Security Notes

1. **App passwords are safer than regular passwords** - They only work for the specific app and can be revoked anytime
2. **Never commit passwords to Git** - Environment variables keep credentials out of code
3. **Revoke unused app passwords** - Go to [App Passwords](https://myaccount.google.com/apppasswords) and delete any you're not using

---

## Changing Recipient Email

To send to a different email address, edit:

```python
# File: notifications/send_daily_trades.py
# Line: ~195

success = send_daily_trade_notification(
    recipient_email="new-email@example.com",  # Change this
    num_trades=5,
    total_capital=100_000
)
```

---

## Support

If you continue to have issues:

1. Check the logs in `logs/` directory
2. Verify environment variables are set correctly
3. Test with the email_sender.py test script
4. Ensure your Gmail account has 2FA enabled
5. Make sure you're using an app-specific password, not your regular password

---

**Last Updated:** November 17, 2025
