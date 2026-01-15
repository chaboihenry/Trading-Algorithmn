import os
from dotenv import load_dotenv

# Get the directory where THIS file (settings.py) lives
config_dir = os.path.dirname(os.path.abspath(__file__))

# Point directly to the .env file in this same directory
env_path = os.path.join(config_dir, ".env")

# Load it
load_dotenv(env_path)

# Export variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DB_PATH = os.getenv("DB_PATH")

# Validate (optional, but good practice to ensure it worked)
if not DB_PATH:
    print("WARNING: DB_PATH not found in .env loaded from:", env_path)
