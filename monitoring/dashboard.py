#!/usr/bin/env python
"""
Portfolio Monitoring Dashboard - Real-time view of trading bot status.

This dashboard provides a comprehensive view of:
- Account health (cash, buying power, portfolio value)
- Current positions with profit/loss
- Risk protection status (stop-loss and take-profit coverage)
- Hedge allocation (inverse ETF positions)
- Recent trading activity

Run this anytime to check on your bot's status.
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import get_market_data_client
from risk.stop_loss_manager import StopLossManager
from risk.hedge_manager import HedgeManager
from config.settings import (
    INVERSE_ETFS,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_POSITION_PCT,
    MAX_INVERSE_ALLOCATION
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress debug/info logs for cleaner output
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def format_currency(value: float) -> str:
    """Format value as currency with color coding."""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"


def format_percentage(value: float) -> str:
    """Format percentage with color coding."""
    if value >= 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_account_overview():
    """Display account overview section."""
    print_header("ACCOUNT OVERVIEW")

    try:
        market_data = get_market_data_client()

        cash = market_data.get_cash()
        buying_power = market_data.get_buying_power()
        portfolio_value = market_data.get_portfolio_value()

        # Calculate positions value
        positions_value = portfolio_value - cash
        cash_pct = (cash / portfolio_value * 100) if portfolio_value > 0 else 0
        positions_pct = 100 - cash_pct

        print(f"\n  Portfolio Value:  {format_currency(portfolio_value)}")
        print(f"  Cash Available:   {format_currency(cash)} ({cash_pct:.1f}%)")
        print(f"  In Positions:     {format_currency(positions_value)} ({positions_pct:.1f}%)")
        print(f"  Buying Power:     {format_currency(buying_power)}")

        # Account status
        account_info = market_data.get_account_info()
        if account_info.get('account_blocked') or account_info.get('trade_suspended'):
            print(f"\n  ⚠️  WARNING: Account has restrictions!")
        else:
            print(f"\n  ✅ Account Status: ACTIVE")

        if account_info.get('pattern_day_trader'):
            print(f"  📊 Pattern Day Trader: YES (Day trades used: {account_info.get('day_trade_count', 0)})")

        return portfolio_value, cash

    except Exception as e:
        print(f"\n  ❌ Error fetching account data: {e}")
        return 0.0, 0.0


def print_positions(portfolio_value: float):
    """Display current positions with P/L."""
    print_header("CURRENT POSITIONS")

    try:
        market_data = get_market_data_client()
        positions = market_data.get_positions()

        if not positions:
            print("\n  No open positions")
            return positions

        print(f"\n  Total Positions: {len(positions)}")
        print()

        # Table header
        print(f"  {'Symbol':<8} {'Qty':<10} {'Entry':<10} {'Current':<10} {'P/L ($)':<12} {'P/L (%)':<10} {'Value':<12} {'Alloc':<8}")
        print(f"  {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 11} {'-' * 9} {'-' * 11} {'-' * 7}")

        total_pl = 0.0
        total_value = 0.0

        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            entry_price = float(pos.avg_entry_price)
            current_price = float(pos.current_price)
            unrealized_pl = float(pos.unrealized_pl)
            unrealized_plpc = float(pos.unrealized_plpc) * 100  # Convert to percentage
            market_value = float(pos.market_value)
            allocation = (market_value / portfolio_value * 100) if portfolio_value > 0 else 0

            total_pl += unrealized_pl
            total_value += market_value

            # Color coding for position size
            alloc_marker = "⚠️ " if allocation > MAX_POSITION_PCT * 100 else ""

            # Format the row
            print(f"  {symbol:<8} {qty:<10.2f} ${entry_price:<9.2f} ${current_price:<9.2f} "
                  f"{format_currency(unrealized_pl):<12} {format_percentage(unrealized_plpc):<10} "
                  f"{format_currency(market_value):<12} {alloc_marker}{allocation:<6.1f}%")

        # Summary
        print(f"  {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 11} {'-' * 9} {'-' * 11} {'-' * 7}")
        total_plpc = (total_pl / (total_value - total_pl) * 100) if (total_value - total_pl) > 0 else 0
        print(f"  {'TOTAL':<8} {'':<10} {'':<10} {'':<10} {format_currency(total_pl):<12} "
              f"{format_percentage(total_plpc):<10} {format_currency(total_value):<12}")

        if abs(allocation - 100) > 0.1:
            total_alloc = (total_value / portfolio_value * 100) if portfolio_value > 0 else 0
            print(f"\n  Total Allocation: {total_alloc:.1f}%")

        return positions

    except Exception as e:
        print(f"\n  ❌ Error fetching positions: {e}")
        import traceback
        traceback.print_exc()
        return []


def print_protection_status(positions: List):
    """Display risk protection status."""
    print_header("RISK PROTECTION STATUS")

    try:
        if not positions:
            print("\n  No positions to protect")
            return

        stop_loss_mgr = StopLossManager()
        protection_status = stop_loss_mgr.verify_protection(positions)

        # Count protection status
        fully_protected = sum(1 for s in protection_status.values() if s['fully_protected'])
        has_stop_only = sum(1 for s in protection_status.values() if s['has_stop_loss'] and not s['has_take_profit'])
        has_tp_only = sum(1 for s in protection_status.values() if s['has_take_profit'] and not s['has_stop_loss'])
        unprotected = sum(1 for s in protection_status.values() if not s['has_stop_loss'] and not s['has_take_profit'])

        print(f"\n  Protection Summary:")
        print(f"    ✅ Fully Protected:      {fully_protected}/{len(positions)}")
        print(f"    ⚠️  Stop-loss only:       {has_stop_only}/{len(positions)}")
        print(f"    ⚠️  Take-profit only:     {has_tp_only}/{len(positions)}")
        print(f"    🚫 No protection:        {unprotected}/{len(positions)}")

        if unprotected > 0:
            print(f"\n  ⚠️  WARNING: {unprotected} position(s) lack protection!")
            print(f"     Run: python tests/test_bug_fixes.py to create missing orders")

        # Show details for each position
        if len(positions) <= 20:  # Only show details if not too many positions
            print(f"\n  Position Details:")
            print(f"  {'Symbol':<8} {'Stop-Loss':<12} {'Take-Profit':<12} {'Status':<20}")
            print(f"  {'-' * 7} {'-' * 11} {'-' * 11} {'-' * 19}")

            for pos in positions:
                symbol = pos.symbol
                status = protection_status[symbol]

                stop_status = "✅ Active" if status['has_stop_loss'] else "🚫 Missing"
                tp_status = "✅ Active" if status['has_take_profit'] else "🚫 Missing"

                if status['fully_protected']:
                    overall_status = "✅ Fully Protected"
                elif not status['has_stop_loss'] and not status['has_take_profit']:
                    overall_status = "🚫 NO PROTECTION"
                else:
                    overall_status = "⚠️  Partial"

                print(f"  {symbol:<8} {stop_status:<12} {tp_status:<12} {overall_status:<20}")

        print(f"\n  Risk Parameters:")
        print(f"    Stop-Loss:      {STOP_LOSS_PCT * 100:.1f}% below entry")
        print(f"    Take-Profit:    {TAKE_PROFIT_PCT * 100:.1f}% above entry")
        print(f"    Max Position:   {MAX_POSITION_PCT * 100:.1f}% of portfolio")

    except Exception as e:
        print(f"\n  ❌ Error checking protection: {e}")
        import traceback
        traceback.print_exc()


def print_hedge_status(portfolio_value: float):
    """Display hedge allocation status."""
    print_header("HEDGE ALLOCATION")

    try:
        hedge_mgr = HedgeManager()

        # Get current hedge allocation
        hedge_value, hedge_pct = hedge_mgr.get_current_hedge_allocation()

        # Get market sentiment
        sentiment = hedge_mgr.check_market_sentiment()

        print(f"\n  Market Sentiment:")
        print(f"    Overbought:     {sentiment['bearish_count']}/{sentiment['total_checked']} major stocks")
        print(f"    Bearish Ratio:  {sentiment['bearish_ratio']:.1%}")
        print(f"    Market Status:  {'🔴 BEARISH' if sentiment['is_bearish'] else '✅ HEALTHY'}")

        if sentiment['symbols_overbought']:
            print(f"    Overbought:     {', '.join(sentiment['symbols_overbought'])}")

        print(f"\n  Hedge Positions:")
        print(f"    Current Value:  {format_currency(hedge_value)}")
        print(f"    Allocation:     {hedge_pct:.1%} / {MAX_INVERSE_ALLOCATION:.1%} max")

        # Show inverse ETF positions
        market_data = get_market_data_client()
        positions = market_data.get_positions()
        hedge_positions = [p for p in positions if p.symbol in INVERSE_ETFS.values()]

        if hedge_positions:
            print(f"\n  Active Hedges:")
            for pos in hedge_positions:
                qty = float(pos.qty)
                current_price = float(pos.current_price)
                value = float(pos.market_value)
                pl = float(pos.unrealized_pl)
                pl_pct = float(pos.unrealized_plpc) * 100

                print(f"    {pos.symbol}: {qty:.2f} shares @ ${current_price:.2f} = {format_currency(value)} "
                      f"(P/L: {format_currency(pl)} / {format_percentage(pl_pct)})")

        # Recommendation
        if sentiment['is_bearish'] and hedge_pct < MAX_INVERSE_ALLOCATION:
            print(f"\n  💡 Recommendation: Consider adding hedge (market is bearish)")
        elif not sentiment['is_bearish'] and hedge_pct > 0:
            print(f"\n  💡 Recommendation: Consider exiting hedges (market recovered)")
        else:
            print(f"\n  ✅ Hedge allocation appropriate for current market conditions")

    except Exception as e:
        print(f"\n  ❌ Error checking hedge status: {e}")
        import traceback
        traceback.print_exc()


def print_recent_activity():
    """Display recent trading activity."""
    print_header("RECENT ACTIVITY (Last 10 Orders)")

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_PAPER

        client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)

        # Get recent orders (all statuses)
        request = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            limit=10
        )
        orders = client.get_orders(request)

        if not orders:
            print("\n  No recent orders")
            return

        print(f"\n  {'Date':<20} {'Symbol':<8} {'Side':<6} {'Qty':<8} {'Price':<10} {'Status':<12} {'Type':<12}")
        print(f"  {'-' * 19} {'-' * 7} {'-' * 5} {'-' * 7} {'-' * 9} {'-' * 11} {'-' * 11}")

        for order in orders:
            # Parse timestamp
            created_at = order.created_at.strftime("%Y-%m-%d %H:%M:%S") if order.created_at else "N/A"
            symbol = order.symbol
            side = order.side.value if hasattr(order.side, 'value') else str(order.side)
            qty = float(order.qty) if order.qty else 0

            # Get price (filled price if available, otherwise limit/stop price)
            if order.filled_avg_price:
                price = f"${float(order.filled_avg_price):.2f}"
            elif order.limit_price:
                price = f"${float(order.limit_price):.2f}"
            elif order.stop_price:
                price = f"${float(order.stop_price):.2f}"
            else:
                price = "Market"

            status = order.status.value if hasattr(order.status, 'value') else str(order.status)
            order_type = order.type.value if hasattr(order.type, 'value') else str(order.type)

            # Status emoji
            status_emoji = ""
            if status == "filled":
                status_emoji = "✅"
            elif status == "canceled" or status == "cancelled":
                status_emoji = "❌"
            elif status in ["pending_new", "accepted", "new"]:
                status_emoji = "⏳"

            print(f"  {created_at:<20} {symbol:<8} {side:<6} {qty:<8.2f} {price:<10} "
                  f"{status_emoji}{status:<11} {order_type:<12}")

    except Exception as e:
        print(f"\n  ❌ Error fetching recent activity: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the complete dashboard."""
    # Print title
    print("\n" + "=" * 80)
    print(" " * 25 + "TRADING BOT DASHBOARD")
    print(" " * 25 + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Display all sections
    portfolio_value, cash = print_account_overview()
    positions = print_positions(portfolio_value)
    print_protection_status(positions)
    print_hedge_status(portfolio_value)
    print_recent_activity()

    # Footer
    print("\n" + "=" * 80)
    print("  💡 Tip: Run 'python tests/test_bug_fixes.py' to verify and fix protection")
    print("  💡 Tip: Run 'python monitoring/performance_tracker.py' for historical metrics")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDashboard interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
