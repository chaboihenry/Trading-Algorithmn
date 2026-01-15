"""
Master Symbol List
This file defines the universe of assets the bot will monitor for data collection 
and eventually for buy/sell positions.
"""

SYMBOLS = [
    # --- Indices / ETFs ---
    "QQQ", "SPY", "IWM", "DIA", "XLF", "XLK", "XLE", "XLV", "XLY", "XLP",
    "XLI", "XLU", "XLB", "TLT", "HYG", "EEM", "VXX", "SQQQ", "TQQQ", "UVXY",
    
    # --- Magnificent 7 & Big Tech ---
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META",
    
    # --- Semiconductors ---
    "AMD", "INTC", "MU", "QCOM", "AVGO", "TXN", "AMAT", "LRCX", "TSM", "SMH",
    
    # --- Financials ---
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK",
    
    # --- Consumer Discretionary & Staples ---
    "HD", "LOW", "MCD", "SBUX", "NKE", "WMT", "TGT", "COST", "PG", "KO", "PEP",
    
    # --- Energy & Utilities ---
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "NEE", "DUK", "SO",
    
    # --- Healthcare & Pharma ---
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH", "CVS", "AMGN", "GILD",
    
    # --- Industrials & Aerospace ---
    "BA", "CAT", "DE", "GE", "HON", "LMT", "RTX", "UPS", "FDX",
    
    # --- Telecom & Media ---
    "T", "VZ", "DIS", "CMCSA", "NFLX",
    
    # --- EV / Auto ---
    "F", "GM", "RIVN", "LCID", "NIO",
    
    # --- Crypto Related ---
    "COIN", "MARA", "RIOT", "MSTR",
    
    # --- High Beta / Growth / Meme ---
    "PLTR", "SOFI", "DKNG", "UBER", "LYFT", "HOOD", "ROKU", "SHOP", "SQ",
    "AFRM", "UPST", "AI", "GME", "AMC"
]

# Helper function if you ever need to count them programmatically
def get_symbol_count():
    return len(SYMBOLS)