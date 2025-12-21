#!/usr/bin/env python3
"""
Alpaca Account Reset Script

Aggressively cleans up your Alpaca paper trading account:
- Cancels ALL open orders
- Sells ALL positions at market price
- Provides detailed account status

Safe to run - this is paper trading only!

Usage:
    python reset_account.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    """Print colored header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.END}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def get_alpaca_client():
    """Get Alpaca trading client using .env credentials."""
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        print_error("Alpaca credentials not found in .env file!")
        print_info("Make sure .env file exists with:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_API_SECRET=your_secret")
        sys.exit(1)

    # Always use paper trading for safety
    return TradingClient(api_key, api_secret, paper=True)


def print_account_status(client):
    """Print current account status."""
    try:
        account = client.get_account()

        print_header("CURRENT ACCOUNT STATUS")

        print(f"{Colors.BOLD}Account Number:{Colors.END} {account.account_number}")
        print(f"{Colors.BOLD}Status:{Colors.END} {account.status}")
        print(f"{Colors.BOLD}Cash:{Colors.END} ${float(account.cash):,.2f}")
        print(f"{Colors.BOLD}Portfolio Value:{Colors.END} ${float(account.equity):,.2f}")
        print(f"{Colors.BOLD}Buying Power:{Colors.END} ${float(account.buying_power):,.2f}")
        print(f"{Colors.BOLD}Day Trading Buying Power:{Colors.END} ${float(account.daytrading_buying_power):,.2f}")
        print(f"{Colors.BOLD}Pattern Day Trader:{Colors.END} {account.pattern_day_trader}")

        return True

    except Exception as e:
        print_error(f"Error getting account status: {e}")
        return False


def cancel_all_orders(client):
    """Cancel all open orders."""
    print_header("CANCELING ALL OPEN ORDERS")

    try:
        # Get all open orders (no filter parameter, just get all)
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            limit=500
        )
        orders = client.get_orders(filter=request)

        if not orders:
            print_info("No open orders to cancel")
            return True

        print_info(f"Found {len(orders)} open order(s)")

        # Display orders before canceling
        for order in orders:
            print(f"\n{Colors.YELLOW}Order ID:{Colors.END} {order.id}")
            print(f"  Symbol: {order.symbol}")
            print(f"  Side: {order.side}")
            print(f"  Quantity: {order.qty}")
            print(f"  Type: {order.order_type}")
            if order.limit_price:
                print(f"  Limit Price: ${float(order.limit_price):.2f}")
            print(f"  Status: {order.status}")

        # Cancel all orders
        print(f"\n{Colors.BOLD}Canceling all orders...{Colors.END}")
        client.cancel_orders()

        print_success(f"Successfully canceled {len(orders)} order(s)")
        return True

    except Exception as e:
        print_error(f"Error canceling orders: {e}")
        return False


def close_all_positions(client):
    """Close all open positions at market price."""
    print_header("CLOSING ALL POSITIONS")

    try:
        # Get all open orders to check for existing sell orders
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
        open_orders = client.get_orders(filter=request)

        # Build a set of symbols that already have sell orders
        symbols_with_sell_orders = set()
        for order in open_orders:
            if order.side.value == 'sell':
                symbols_with_sell_orders.add(order.symbol)

        # Get all positions
        positions = client.get_all_positions()

        if not positions:
            print_info("No open positions to close")
            return True

        print_info(f"Found {len(positions)} position(s)")

        # Display positions
        total_value = 0
        for position in positions:
            qty = float(position.qty)
            current_price = float(position.current_price)
            market_value = float(position.market_value)
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc) * 100

            total_value += market_value

            print(f"\n{Colors.CYAN}Symbol:{Colors.END} {position.symbol}")
            print(f"  Quantity: {qty:,.0f}")
            print(f"  Current Price: ${current_price:.2f}")
            print(f"  Market Value: ${market_value:,.2f}")
            print(f"  Unrealized P/L: ${unrealized_pl:,.2f} ({unrealized_plpc:+.2f}%)")

        print(f"\n{Colors.BOLD}Total Position Value: ${total_value:,.2f}{Colors.END}")

        # Close all positions
        print(f"\n{Colors.BOLD}Closing all positions at market price...{Colors.END}")

        closed_count = 0
        failed_count = 0

        for position in positions:
            try:
                symbol = position.symbol
                qty = abs(float(position.qty))
                side = OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY

                # Skip if already have a sell order for this symbol
                if symbol in symbols_with_sell_orders:
                    print_info(f"Skipping {symbol} - already has pending sell order")
                    closed_count += 1
                    continue

                # Create market order to close
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    time_in_force=TimeInForce.DAY  # DAY orders required for fractional shares
                )

                order = client.submit_order(order_request)
                print_success(f"Submitted {side} order for {qty} shares of {symbol} (Order ID: {order.id})")
                closed_count += 1

            except Exception as e:
                print_error(f"Failed to close position in {symbol}: {e}")
                failed_count += 1

        print(f"\n{Colors.BOLD}Summary:{Colors.END}")
        print_success(f"Successfully submitted close orders for {closed_count} position(s)")
        if failed_count > 0:
            print_error(f"Failed to close {failed_count} position(s)")

        if closed_count > 0:
            print_warning("\nNote: Orders submitted as market orders.")
            print_warning("They will execute when the market opens (if currently closed).")

        return True

    except Exception as e:
        print_error(f"Error closing positions: {e}")
        return False


def main():
    """Main execution function."""
    print_header("ALPACA PAPER TRADING ACCOUNT RESET")

    print_info("This script will:")
    print("  1. Cancel ALL open orders")
    print("  2. Close ALL positions at market price")
    print("  3. Show final account status")
    print(f"\n{Colors.YELLOW}This is PAPER TRADING - no real money at risk!{Colors.END}\n")

    # Get Alpaca client
    try:
        client = get_alpaca_client()
        print_success("Connected to Alpaca paper trading")
    except Exception as e:
        print_error(f"Failed to connect to Alpaca: {e}")
        return

    # Show initial account status
    if not print_account_status(client):
        return

    # Check market status
    try:
        clock = client.get_clock()
        print(f"\n{Colors.BOLD}Market Status:{Colors.END}")
        print(f"  Is Open: {clock.is_open}")
        print(f"  Next Open: {clock.next_open}")
        print(f"  Next Close: {clock.next_close}")

        if not clock.is_open:
            print_warning("\n⚠️  Market is currently CLOSED")
            print_warning("Orders will be submitted but won't execute until market opens")
    except Exception as e:
        print_warning(f"Could not get market status: {e}")

    # Confirmation
    print(f"\n{Colors.BOLD}{Colors.RED}Are you sure you want to proceed?{Colors.END}")
    response = input(f"{Colors.YELLOW}Type 'YES' to continue: {Colors.END}")

    if response.strip() != 'YES':
        print_info("Operation canceled by user")
        return

    # Execute cleanup
    print()

    # Step 1: Cancel all orders
    cancel_all_orders(client)

    # Wait for order cancellations to propagate
    print(f"\n{Colors.BLUE}⏳ Waiting 5 seconds for order cancellations to propagate...{Colors.END}")
    time.sleep(5)

    # Verify all orders are canceled
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)
    remaining_orders = client.get_orders(filter=request)

    if remaining_orders:
        print_warning(f"Found {len(remaining_orders)} orders still pending cancellation")
        print_info("Attempting to cancel them individually...")

        for order in remaining_orders:
            try:
                client.cancel_order_by_id(order.id)
                print_success(f"Individually canceled order {order.id} ({order.symbol})")
            except Exception as e:
                print_warning(f"Could not cancel {order.symbol} order {order.id}: {e}")

        print_info("Waiting another 5 seconds...")
        time.sleep(5)

    # Step 2: Close all positions
    close_all_positions(client)

    # Step 3: Show final status
    print()
    print_account_status(client)

    print_header("RESET COMPLETE")
    print_success("Your Alpaca paper trading account has been reset!")
    print_info("\nYour account is now clean and ready for RiskLabAI trading")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Operation canceled by user{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
