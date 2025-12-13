#!/usr/bin/env python
"""
Quick script to check all positions against risk management thresholds
Run this anytime to see if any positions should trigger stop-loss or take-profit
"""

import os
from alpaca.trading.client import TradingClient

# Risk management thresholds (must match combined_strategy.py)
STOP_LOSS_PCT = 0.05  # 5%
TAKE_PROFIT_PCT = 0.15  # 15%

def main():
    # Get API credentials
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        print("❌ Error: ALPACA_API_KEY and ALPACA_API_SECRET must be set")
        return

    # Connect to Alpaca
    client = TradingClient(api_key, api_secret, paper=True)

    # Get all positions
    positions = client.get_all_positions()

    if not positions:
        print("No positions found")
        return

    print("=" * 80)
    print("POSITION RISK CHECK")
    print("=" * 80)
    print(f"Stop-Loss Threshold: -{STOP_LOSS_PCT:.1%}")
    print(f"Take-Profit Threshold: +{TAKE_PROFIT_PCT:.1%}")
    print("=" * 80)
    print()

    triggers = []

    for pos in positions:
        symbol = pos.symbol
        entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        quantity = float(pos.qty)
        unrealized_pl = float(pos.unrealized_pl)

        # Calculate P&L percentage
        pnl_pct = (current_price - entry_price) / entry_price

        # Check triggers
        status = "✅ HOLD"
        action = None

        if pnl_pct <= -STOP_LOSS_PCT:
            status = "🛑 STOP-LOSS"
            action = "SELL NOW"
            triggers.append((symbol, pnl_pct, unrealized_pl, 'stop_loss'))
        elif pnl_pct >= TAKE_PROFIT_PCT:
            status = "💰 TAKE-PROFIT"
            action = "SELL NOW"
            triggers.append((symbol, pnl_pct, unrealized_pl, 'take_profit'))
        elif pnl_pct <= -0.03:  # Warning at -3%
            status = "⚠️  WARNING"
        elif pnl_pct >= 0.10:  # Approaching take-profit at +10%
            status = "📈 GAINING"

        print(f"{status:20s} {symbol:6s}")
        print(f"  Entry: ${entry_price:8.2f}  Current: ${current_price:8.2f}")
        print(f"  P&L: {pnl_pct:+7.2%} (${unrealized_pl:+9.2f})")
        print(f"  Qty: {quantity:.2f} shares")
        if action:
            print(f"  🚨 ACTION: {action}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if triggers:
        print(f"⚠️  {len(triggers)} POSITION(S) NEED IMMEDIATE ACTION:")
        print()
        total_loss = 0
        for symbol, pnl_pct, pl_dollar, reason in triggers:
            emoji = "🛑" if reason == 'stop_loss' else "💰"
            print(f"{emoji} {symbol}: {pnl_pct:+.2%} (${pl_dollar:+.2f})")
            total_loss += pl_dollar
        print()
        print(f"Total P&L from triggered positions: ${total_loss:+.2f}")
        print()
        print("The bot will automatically sell these at the next iteration (12:00 PM)")
    else:
        print("✅ No positions have triggered risk management thresholds")
        print()
        total_pl = sum(float(pos.unrealized_pl) for pos in positions)
        print(f"Total unrealized P&L: ${total_pl:+.2f}")

    print("=" * 80)

if __name__ == "__main__":
    main()
