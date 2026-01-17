import os
from dotenv import load_dotenv

config_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(config_dir, ".env")
load_dotenv(env_path)

# 1. Load Individual Variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DB_PATH = os.getenv("DB_PATH")

# 2. Create the Config Dictionary
ALPACA_CONFIG = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_SECRET_KEY,
    "PAPER": True
}

# 3. Validation
if not DB_PATH:
    print("WARNING: DB_PATH not found in .env loaded from:", env_path)
if not ALPACA_API_KEY:
    print("WARNING: ALPACA_API_KEY not found. Check your .env file.")