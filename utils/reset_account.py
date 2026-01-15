"""
Reset Account (The "Nuke" Button)
Cancels all orders and liquidates all positions immediately.
"""

import sys
from pathlib import Path
from alpaca.trading.client import TradingClient

# --- PATH SETUP ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

def main():
    if not ALPACA_API_KEY:
        print("❌ Error: API Keys not found in config.")
        return

    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
    account = client.get_account()

    print(f"--- RESETTING ACCOUNT ({account.account_number}) ---")
    print(f"Starting Equity: ${float(account.equity):,.2f}")

    try:
        # The "One-Shot" command: Closes every position AND cancels every order
        client.close_all_positions(cancel_orders=True)
        print("✅ Liquidation orders submitted and open orders cancelled.")
    except Exception as e:
        # Often throws error if no positions exist, which is fine
        print(f"ℹ️  Note: {e}")
        print("Checking for remaining open orders...")
        client.cancel_orders()

    print("--- DONE ---")

if __name__ == "__main__":
    main()