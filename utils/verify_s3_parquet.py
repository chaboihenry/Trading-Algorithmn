import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# --- 1. TARGET SPECIFIC .ENV PATH ---
# Get the project root directory (one level up from 'utils')
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / "config" / ".env"

print(f"Loading credentials from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)

import s3fs
import pyarrow.parquet as pq

# --- CONFIG ---
S3_BUCKET = "risklabai-models"
AWS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET = os.getenv('AWS_SECRET_ACCESS_KEY')

# Debug Print (Masked)
if AWS_KEY:
    print(f"✅ API Key Loaded: {AWS_KEY[:4]}...{AWS_KEY[-4:]}")
else:
    print("❌ API Key NOT Found. Check your .env file path.")
    sys.exit(1)

# Initialize S3 with explicit credentials
fs = s3fs.S3FileSystem(
    key=AWS_KEY,
    secret=AWS_SECRET
)

def check_bucket():
    print(f"\n🔎 Scanning S3 Bucket: {S3_BUCKET}...")
    
    try:
        # List all parquet files using the glob pattern
        files = fs.glob(f"{S3_BUCKET}/*/parquet/ticks.parquet")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    if not files:
        print("⚠️ No Parquet files found in bucket.")
        return

    print(f"Found {len(files)} Parquet files.")
    print("-" * 65)
    print(f"{'SYMBOL':<10} | {'SIZE (MB)':<10} | {'ROWS':<15} | {'STATUS':<10}")
    print("-" * 65)

    for f in files:
        # Extract symbol. Path format: bucket/SYMBOL/parquet/ticks.parquet
        parts = f.split('/')
        # Handle path variations
        if len(parts) >= 3:
            # If path is risklabai-models/AAPL/parquet/ticks.parquet -> index 1 is AAPL
            symbol = parts[1]
        else:
            symbol = "UNKNOWN"

        try:
            # Get File Size
            size_bytes = fs.du(f)
            size_mb = size_bytes / (1024 * 1024)
            
            # Read Parquet Metadata (Fast, no download)
            # We open the file stream directly from S3
            with fs.open(f, 'rb') as s3_file:
                pfile = pq.ParquetFile(s3_file)
                rows = pfile.metadata.num_rows
            
            status = "✅ OK"
            # Flag if file is too small (under 50k rows for a whole year is suspicious)
            if rows < 50000: 
                status = "⚠️ LOW DATA"
            if size_mb < 1.0:
                status = "⚠️ TINY FILE"
            
            print(f"{symbol:<10} | {size_mb:>9.2f} | {rows:>15,} | {status}")
            
        except Exception as e:
            print(f"{symbol:<10} | {'ERROR':>9} | {'N/A':>15} | ❌ CORRUPT")

if __name__ == "__main__":
    check_bucket()