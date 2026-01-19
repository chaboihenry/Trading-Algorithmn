import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone

# ML Imports
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Path Setup
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import DB_PATH
from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging
from data.tick_storage import TickStorage
from data.tick_to_bars import ImbalanceBarGenerator
from strategies.risklabai_bot import RiskLabAIModel
from utils.financial_ml import PurgedCrossValidator

logger = setup_logging(script_name="backtest")

TEST_SPLIT = 0.20
TRANSACTION_COST = 0.0002 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Symbol to test")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()

def calculate_alpaca_fees(price, qty, side):
    notional = price * qty
    slippage = notional * 0.0001
    reg_fees = 0.0
    if side == -1:
        sec_fee = notional * (8.00 / 1_000_000)
        taf_fee = min(qty * 0.000166, 8.30)
        reg_fees = sec_fee + taf_fee
    return slippage + reg_fees

def simulate_triple_barrier_trades(prices, signals, confidences, volatility, threshold, pt_mult, sl_mult, time_limit_bars):
    trades = []
    active_trade = None
    
    for i in range(len(prices)):
        price = prices.iloc[i]
        current_time = prices.index[i]
        
        if active_trade:
            entry_price = active_trade['entry_price']
            side = active_trade['side']
            qty = active_trade['qty']
            
            hit_tp = (price >= active_trade['tp']) if side == 1 else (price <= active_trade['tp'])
            hit_sl = (price <= active_trade['sl']) if side == 1 else (price >= active_trade['sl'])
            time_expired = (i - active_trade['entry_idx']) >= time_limit_bars
            
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
                    'reason': exit_reason
                })
                active_trade = None
                continue

        if active_trade is None:
            sig = signals.iloc[i]
            conf = confidences.iloc[i]
            
            if sig != 0 and conf > threshold:
                vol = volatility.iloc[i]
                qty = 100 
                
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
                    'entry_cost': entry_cost
                }

    return pd.DataFrame(trades)

def run_simulation(symbol):
    logger.info(f"--- Simulating {symbol} ---")
    
    storage = TickStorage(DB_PATH)
    ticks = storage.load_ticks(symbol)
    storage.close()
    
    if ticks.empty or len(ticks) < 1000:
        logger.warning(f"[{symbol}] Not enough data (Need backfill).")
        return None

    # Dynamic Dollar Bars
    bars = ImbalanceBarGenerator.process_ticks(ticks, threshold=None)
    if bars.empty: return None
    if bars.index.tz is not None: bars.index = bars.index.tz_localize(None)
    
    model = RiskLabAIModel()
    features = model.generate_features(bars)
    if features.empty: return None
        
    close_prices = bars.loc[features.index, 'close']
    volatility = model.labeler.get_volatility(close_prices)
    events = model.cusum.get_events(close_prices)
    labels = model.labeler.label(close_prices, pd.DataFrame(index=events))
    
    common_idx = labels.index.intersection(features.index)
    X = features.loc[common_idx, model.feature_names]
    
    le = LabelEncoder()
    y_raw = labels.loc[common_idx, 'bin']
    y_encoded = le.fit_transform(y_raw)
    y = pd.Series(y_encoded, index=common_idx)
    t1 = labels.loc[common_idx, 't1']
    
    if X.index.tz is not None: X.index = X.index.tz_localize(None)
    
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    t1_train = t1.iloc[:split_idx]
    
    if len(X_test) < 10: return None
    logger.info(f"[{symbol}] Train: {len(X_train)} | Test: {len(X_test)}")

    primary_base = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, n_jobs=1))
    ])
    
    meta_features = np.zeros(len(y_train))
    valid_indices_mask = np.zeros(len(y_train), dtype=bool)
    cv = PurgedCrossValidator(n_splits=5, embargo_pct=0.01)
    
    try:
        cv_gen = cv.get_cv(t1_train)
        for train_idx, val_idx in cv_gen.split(X_train, y_train):
            temp_model = clone(primary_base)
            temp_model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            val_preds = temp_model.predict(X_train.iloc[val_idx])
            meta_features[val_idx] = val_preds
            valid_indices_mask[val_idx] = True
    except Exception: return None

    primary_final = clone(primary_base)
    primary_final.fit(X_train, y_train)
    
    meta_final = LogisticRegression(class_weight='balanced', solver='liblinear')
    X_meta_train = X_train.iloc[valid_indices_mask]
    y_meta_train = (meta_features[valid_indices_mask] == y_train.iloc[valid_indices_mask]).astype(int)
    meta_final.fit(X_meta_train, y_meta_train)
    
    p_preds_encoded = primary_final.predict(X_test)
    signals = pd.Series(le.inverse_transform(p_preds_encoded), index=X_test.index)
    
    if hasattr(meta_final, "predict_proba"):
        conf_arr = meta_final.predict_proba(X_test)[:, 1]
    else:
        conf_arr = meta_final.predict(X_test)
    confidences = pd.Series(conf_arr, index=X_test.index)

    logger.info(f"[{symbol}] Running Stateful Simulation...")
    
    test_prices = close_prices.loc[X_test.index]
    bh_return = (test_prices.iloc[-1] / test_prices.iloc[0]) - 1
    test_vol = volatility.loc[X_test.index]
    
    # 5 Days Approx
    avg_bars_per_day = len(bars) / 90 
    time_limit_bars = int(avg_bars_per_day * 5)

    best_res = None
    best_ret = -999
    
    print(f"\n{'Threshold':<10} | {'Trades':<6} | {'Win%':<6} | {'Return':<8} | {'L/S Ratio':<10}")
    print("-" * 60)
    
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        trade_log = simulate_triple_barrier_trades(
            prices=test_prices,
            signals=signals,
            confidences=confidences,
            volatility=test_vol,
            threshold=threshold,
            pt_mult=model.labeler.pt_mult,
            sl_mult=model.labeler.sl_mult,
            time_limit_bars=time_limit_bars
        )
        
        if trade_log.empty: continue
            
        cum_ret = (1 + trade_log['ret']).prod() - 1
        win_rate = len(trade_log[trade_log['ret'] > 0]) / len(trade_log)
        
        n_longs = len(trade_log[trade_log['side'] == 1])
        n_shorts = len(trade_log[trade_log['side'] == -1])
        ls_ratio = f"{n_longs}/{n_shorts}"
        
        print(f"{threshold:<10} | {len(trade_log):<6} | {win_rate:<6.1%} | {cum_ret:<8.2%} | {ls_ratio:<10}")
        
        if cum_ret > best_ret and len(trade_log) > 5:
            best_ret = cum_ret
            best_res = {
                "Symbol": symbol,
                "Best_Thresh": threshold,
                "Net Return": f"{cum_ret:.2%}",
                "Trades": len(trade_log),
                "Win Rate": f"{win_rate:.1%}",
                "L_S": ls_ratio,
                "Benchmark": f"{bh_return:.2%}"
            }

    if best_res:
        print("\n" + "="*60)
        print(f"RESULTS FOR {symbol}")
        print(f"Benchmark (Buy & Hold): {best_res['Benchmark']}")
        print(f"Bot Best Return:        {best_res['Net Return']} (Thresh: {best_res['Best_Thresh']})")
        print(f"Trades (Long/Short):    {best_res['L_S']}")
        print("="*60)
        return best_res
    
    return None

def main():
    args = parse_args()
    targets = [args.symbol.upper()] if args.symbol else SYMBOLS
    for sym in targets:
        try:
            run_simulation(sym)
        except Exception as e:
            logger.error(f"Error {sym}: {e}")

if __name__ == "__main__":
    main()