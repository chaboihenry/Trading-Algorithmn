import argparse
import logging
import sys
import os
import boto3
import io
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
env_path = project_root / "config" / ".env"
if env_path.exists(): load_dotenv(dotenv_path=env_path)
else: load_dotenv()

if str(project_root) not in sys.path: sys.path.append(str(project_root))
from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging
from config.aws_config import S3_MODEL_BUCKET, AWS_REGION
from data.tick_to_bars import ImbalanceBarGenerator

logger = setup_logging(script_name="build_bars")

def get_s3_client():
    key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not key_id or not secret_key:
        logger.error("❌ AWS Credentials not found."); return None
    return boto3.client('s3', region_name=os.getenv("AWS_REGION", "us-east-1"), aws_access_key_id=key_id, aws_secret_access_key=secret_key)

def get_s3_keys(s3_client, bucket, symbol):
    prefixes = [f"data/{symbol}/parquet/", f"{symbol}/parquet/"]
    all_keys = []
    for prefix in prefixes:
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            for page in pages:
                if 'Contents' not in page: continue
                for obj in page['Contents']:
                    if obj['Key'].endswith('.parquet'): all_keys.append(obj['Key'])
            if all_keys:
                logger.info(f"[{symbol}] Found {len(all_keys)} files in 's3://{bucket}/{prefix}'")
                break 
        except Exception: pass
    return sorted(all_keys)

def read_parquet_from_s3(s3_client, bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        with io.BytesIO(response['Body'].read()) as buffer: return pd.read_parquet(buffer)
    except Exception as e:
        logger.error(f"Failed to read s3://{bucket}/{key}: {e}"); return pd.DataFrame()

def calculate_intrinsic_threshold(df, expected_ticks_per_bar=10000):
    """
    Calculates threshold based on Advances in Financial ML:
    Threshold = E[T] * |2P[b=1] - 1| * E[v]
    
    This adapts to the asset's specific buy/sell bias and volume profile.
    """
    if df.empty: return 1000000.0

    # 1. Clean Data
    if 'size' in df.columns and 'volume' not in df.columns: df = df.rename(columns={'size': 'volume'})
    if 'price' not in df.columns or 'volume' not in df.columns: return 1000000.0
    
    # 2. Determine Tick Signs (b_t)
    # We need to replicate the tick rule to count Buys vs Sells
    price_diff = df['price'].diff().fillna(0)
    
    # Vectorized Tick Rule
    # This is an approximation for stats: 1 if diff > 0, -1 if diff < 0. 
    # (Ignoring 0s for probability calculation is standard estimation)
    buys = (price_diff > 0).sum()
    sells = (price_diff < 0).sum()
    total_active_ticks = buys + sells
    
    if total_active_ticks == 0: return 1000000.0
    
    # 3. Calculate Statistics
    # P[b=1]: Probability of Buy
    prob_buy = buys / total_active_ticks
    
    # E[v]: Expected Dollar Volume per tick
    dollar_vol = df['price'] * df['volume']
    avg_dollar_vol_per_tick = dollar_vol.mean()
    
    # 4. Calculate Threshold Formula
    # |2P[b=1] - 1| captures the "informed" bias.
    # If P[b=1] is 0.5 (random), threshold -> 0 (sample every tick? No, implies no imbalance).
    # Lopez de Prado suggests using the Max of this term or a minimum expectation.
    imbalance_factor = abs(2 * prob_buy - 1)
    
    # Fail-safe: If market is perfectly balanced (factor=0), we default to a small baseline
    imbalance_factor = max(imbalance_factor, 0.01) 
    
    threshold = expected_ticks_per_bar * imbalance_factor * avg_dollar_vol_per_tick
    
    logger.info(f"   [Intrinsic Calc] P[Buy]={prob_buy:.3f} | E[v]=${avg_dollar_vol_per_tick:.2f}")
    logger.info(f"   [Intrinsic Calc] Imbalance Factor={imbalance_factor:.4f} | Window={expected_ticks_per_bar}")
    logger.info(f"   [Intrinsic Calc] FINAL THRESHOLD: ${threshold:,.0f}")
    
    return max(threshold, 1000.0)

def process_symbol(symbol):
    # E[T]: Expected ticks per bar. 
    # We set this to 5000 to define "Microstructure Granularity".
    # This is NOT "Bars per day". It means "After 5000 ticks worth of theoretical imbalance, create a bar".
    TICKS_PER_BAR_TARGET = 5000 
    
    logger.info(f"--- Processing {symbol} (Intrinsic Imbalance) ---")
    s3 = get_s3_client()
    if not s3: return
    bucket = os.getenv("S3_MODEL_BUCKET", "trading-agent-models")
    keys = get_s3_keys(s3, bucket, symbol)
    if not keys: logger.error(f"[{symbol}] No parquet files found."); return

    # 1. Sample Data for Statistics
    last_key = keys[-1]
    logger.info(f"   Sampling {Path(last_key).name} for stats...")
    sample_df = read_parquet_from_s3(s3, bucket, last_key)
    
    # 2. Calculate Theoretical Threshold
    threshold = calculate_intrinsic_threshold(sample_df, TICKS_PER_BAR_TARGET)

    # 3. Generate Bars
    state = (0.0, 1, -1.0, -1.0, 0.0, 0.0, 0.0)
    all_bars = []
    
    for i, key in enumerate(keys):
        df = read_parquet_from_s3(s3, bucket, key)
        if df.empty: continue
        
        if 'size' in df.columns and 'volume' not in df.columns: df = df.rename(columns={'size': 'volume'})
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']): df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')

        bars, state = ImbalanceBarGenerator.process_chunk(df, threshold, state)
        if not bars.empty: all_bars.append(bars)
        print(f"   Processed S3 File {i+1}/{len(keys)}: {Path(key).name}", end='\r')

    print("") 
    if not all_bars: logger.warning(f"[{symbol}] No bars generated."); return

    final_df = pd.concat(all_bars)
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    out_key = f"{symbol}/bars/{symbol}_dollar_bars.parquet"
    
    logger.info(f"[{symbol}] Uploading to s3://{bucket}/{out_key} ...")
    try:
        with io.BytesIO() as out_buffer:
            final_df.to_parquet(out_buffer)
            out_buffer.seek(0)
            s3.upload_fileobj(out_buffer, bucket, out_key)
        logger.info(f"[{symbol}] COMPLETE -> Uploaded {len(final_df):,} bars to S3.")
    except Exception as e: logger.error(f"[{symbol}] Failed to upload: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Symbol")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    targets = [args.symbol.upper()] if args.symbol else SYMBOLS
    for sym in targets: process_symbol(sym)

if __name__ == "__main__":
    main()