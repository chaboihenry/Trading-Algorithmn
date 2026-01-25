import argparse
import logging
import os
import sys
import boto3
import io
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone

# Path Setup
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
env_path = project_root / "config" / ".env"
if env_path.exists(): load_dotenv(dotenv_path=env_path)
else: load_dotenv()

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging
from config.aws_config import AWS_REGION, S3_MODEL_BUCKET
from strategies.risklabai_bot import RiskLabAIModel
from data.model_storage import ModelStorage

logger = setup_logging(script_name="backtest")

TEST_SPLIT = 0.20 # Last 20% of data for Out-of-Sample testing

def get_s3_client():
    key_id = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not key_id or not secret_key: return None
    return boto3.client(
        's3',
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key
    )

def load_bars_from_s3(symbol):
    s3 = get_s3_client()
    if not s3:
        logger.error("AWS Credentials missing.")
        return pd.DataFrame()

    bucket = os.getenv("S3_MODEL_BUCKET", "trading-agent-models")
    key = f"{symbol}/bars/{symbol}_dollar_bars.parquet"
    
    try:
        logger.info(f"[{symbol}] Downloading bars from s3://{bucket}/{key} ...")
        obj = s3.get_object(Bucket=bucket, Key=key)
        with io.BytesIO(obj['Body'].read()) as buffer:
            bars = pd.read_parquet(buffer)
        
        # Ensure Index is clean
        if not isinstance(bars.index, pd.DatetimeIndex):
            if 'timestamp' in bars.columns: bars.set_index('timestamp', inplace=True)
            else: bars.index = pd.to_datetime(bars.index)
        
        if bars.index.tz is not None: bars.index = bars.index.tz_localize(None)
        
        return bars
    except Exception as e:
        logger.error(f"[{symbol}] Failed to load bars from S3: {e}")
        return pd.DataFrame()

def calculate_alpaca_fees(price, qty, side):
    """
    Estimates costs: Slippage + Regulatory Fees
    """
    notional = price * qty
    # Assume 1 basis point slippage (0.01%)
    slippage = notional * 0.0001 
    
    reg_fees = 0.0
    if side == -1: # Sell side fees
        sec_fee = notional * (8.00 / 1_000_000)
        taf_fee = min(qty * 0.000166, 8.30)
        reg_fees = sec_fee + taf_fee
        
    return slippage + reg_fees

def simulate_trades(prices, signals, confidences, volatility, threshold, pt_mult, sl_mult, time_limit_bars):
    trades = []
    active_trade = None
    
    # Iterate through the test set
    for i in range(len(prices)):
        price = prices.iloc[i]
        current_time = prices.index[i]
        
        # 1. Manage Active Trade
        if active_trade:
            entry_price = active_trade['entry_price']
            side = active_trade['side']
            qty = active_trade['qty']
            
            # Check Exit Conditions
            hit_tp = (price >= active_trade['tp']) if side == 1 else (price <= active_trade['tp'])
            hit_sl = (price <= active_trade['sl']) if side == 1 else (price >= active_trade['sl'])
            
            # Calculate time elapsed in bars
            bars_held = i - active_trade['entry_idx']
            time_expired = bars_held >= time_limit_bars
            
            exit_price = None
            exit_reason = None
            
            if hit_tp:
                exit_price = active_trade['tp']
                exit_reason = 'TP'
            elif hit_sl:
                exit_price = active_trade['sl']
                exit_reason = 'SL'
            elif time_expired:
                exit_price = price
                exit_reason = 'TIME'
            
            if exit_price:
                # Calculate PnL
                gross_pnl = (exit_price - entry_price) * qty * side
                exit_cost = calculate_alpaca_fees(exit_price, qty, -side)
                total_cost = active_trade['entry_cost'] + exit_cost
                net_pnl = gross_pnl - total_cost
                
                notional = entry_price * qty
                net_ret = net_pnl / notional
                
                trades.append({
                    'entry_time': active_trade['entry_time'],
                    'exit_time': current_time,
                    'side': side,
                    'ret': net_ret,
                    'reason': exit_reason,
                    'conf': active_trade['conf']
                })
                active_trade = None
                continue # Trade closed, move to next bar

        # 2. Check for New Entry (if flat)
        if active_trade is None:
            sig = signals.iloc[i]
            conf = confidences.iloc[i]
            
            # Check Threshold
            if sig != 0 and conf > threshold:
                vol = volatility.iloc[i]
                qty = 100 # Fixed size for sim
                
                # Dynamic Barriers based on Volatility
                if sig == 1:
                    tp = price * (1 + vol * pt_mult)
                    sl = price * (1 - vol * sl_mult)
                else:
                    tp = price * (1 - vol * pt_mult)
                    sl = price * (1 + vol * sl_mult)
                
                entry_cost = calculate_alpaca_fees(price, qty, sig)
                
                active_trade = {
                    'entry_price': price,
                    'entry_time': current_time,
                    'entry_idx': i,
                    'side': sig,
                    'tp': tp,
                    'sl': sl,
                    'qty': qty,
                    'entry_cost': entry_cost,
                    'conf': conf
                }

    return pd.DataFrame(trades)

def run_simulation(symbol):
    logger.info(f"--- Backtesting {symbol} ---")
    
    # 1. Load Data
    bars = load_bars_from_s3(symbol)
    if bars.empty: return

    # 2. Load Trained Model
    storage = ModelStorage()
    model_data = storage.load_model(symbol)
    if not model_data:
        logger.error(f"[{symbol}] No trained model found.")
        return
        
    # Reconstruct the RiskLabAIModel wrapper
    rl_model = RiskLabAIModel(symbol)
    rl_model.primary_model = model_data['primary_model']
    rl_model.meta_model = model_data['meta_model']
    rl_model.feature_names = model_data['feature_names']
    rl_model.label_encoder = model_data['label_encoder']

    # 3. Prepare Test Data
    # We must generate features exactly like training
    logger.info(f"[{symbol}] Generating Features for Backtest...")
    features = rl_model.generate_features(bars)
    if features.empty: return
    
    # Split Data (OOS Test)
    split_idx = int(len(features) * (1 - TEST_SPLIT))
    test_features = features.iloc[split_idx:]
    test_bars = bars.loc[test_features.index]
    
    if len(test_features) < 100:
        logger.warning("Not enough test data.")
        return

    logger.info(f"[{symbol}] Testing on {len(test_features)} bars ({test_bars.index[0]} -> {test_bars.index[-1]})")

    # 4. Generate Signals
    logger.info(f"[{symbol}] Predicting Signals...")
    try:
        X_test = test_features[rl_model.feature_names]
        
        # Primary Signal (Argmax)
        # Use predict_proba + argmax to be robust
        probs = rl_model.primary_model.predict_proba(X_test)
        pred_indices = np.argmax(probs, axis=1)
        signals = rl_model.label_encoder.inverse_transform(pred_indices)
        
        # Meta Confidence
        confidences = np.zeros(len(signals))
        if rl_model.meta_model:
            if hasattr(rl_model.meta_model, "predict_proba"):
                confidences = rl_model.meta_model.predict_proba(X_test)[:, 1]
            else:
                confidences = rl_model.meta_model.predict(X_test)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return

    # 5. Run Sim Loop
    # Align signals to DataFrame index
    signals_series = pd.Series(signals, index=test_features.index)
    conf_series = pd.Series(confidences, index=test_features.index)
    close_prices = test_bars['close']
    
    # Calculate volatility for dynamic barriers (replicate Labeler logic)
    # Using simple EWM std dev matching training
    volatility = close_prices.pct_change().ewm(span=100).std()
    
    # Benchmarks
    bh_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
    
    best_res = None
    best_net_ret = -999
    
    print(f"\n{'Threshold':<10} | {'Trades':<6} | {'Win%':<6} | {'Net Return':<10} | {'Sharpe':<8} | {'Benchmark':<10}")
    print("-" * 75)
    
    # Test different confidence thresholds
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
        trade_log = simulate_trades(
            prices=close_prices,
            signals=signals_series,
            confidences=conf_series,
            volatility=volatility,
            threshold=thresh,
            pt_mult=rl_model.labeler.pt_mult,
            sl_mult=rl_model.labeler.sl_mult,
            # FIXED: Use .max_holding instead of .max_holding_period
            time_limit_bars=rl_model.labeler.max_holding
        )
        
        if trade_log.empty:
            continue
            
        # Stats
        cum_ret = (1 + trade_log['ret']).prod() - 1
        win_rate = len(trade_log[trade_log['ret'] > 0]) / len(trade_log)
        
        # Sharpe (approx)
        if trade_log['ret'].std() > 0:
            # Approx annualization: 252 days * 50 bars/day (rough estimate for imbalance bars)
            sharpe = (trade_log['ret'].mean() / trade_log['ret'].std()) * np.sqrt(252 * 50) 
        else:
            sharpe = 0
            
        print(f"{thresh:<10} | {len(trade_log):<6} | {win_rate:<6.1%} | {cum_ret:<10.2%} | {sharpe:<8.2f} | {bh_return:<10.2%}")
        
        if cum_ret > best_net_ret:
            best_net_ret = cum_ret
            best_res = {
                "thresh": thresh,
                "ret": cum_ret,
                "trades": len(trade_log),
                "win": win_rate
            }

    if best_res:
        print("\n" + "="*60)
        print(f"BEST RESULT for {symbol}")
        print(f"Threshold:    {best_res['thresh']}")
        print(f"Net Return:   {best_res['ret']:.2%}")
        print(f"Trades:       {best_res['trades']}")
        print(f"Win Rate:     {best_res['win']:.1%}")
        print("="*60)
    else:
        print(f"\n[{symbol}] No profitable trades found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Symbol to test")
    args = parser.parse_args()
    
    targets = [args.symbol.upper()] if args.symbol else SYMBOLS
    for sym in targets:
        run_simulation(sym)

if __name__ == "__main__":
    main()