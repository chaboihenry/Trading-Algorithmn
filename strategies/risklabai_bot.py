import logging
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# --- PATH SETUP ---
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- ML & Data Science Imports ---
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier

# --- Lumibot Imports ---
from lumibot.strategies import Strategy

# --- Project Imports ---
from data.tick_storage import TickStorage
from data.tick_to_bars import generate_bars_from_ticks
from data.model_storage import ModelStorage
from config.settings import DB_PATH
from config.all_symbols import SYMBOLS
from utils.market_calendar import is_market_open, now_et

# --- RiskLabAI Component Imports ---
from utils.financial_ml import (
    TripleBarrierLabeler,
    MetaLabeler,
    CUSUMEventFilter,
    PurgedCrossValidator,
    HRPPortfolio
)

# --- CONSTANTS ---
INITIAL_IMBALANCE_THRESHOLD = 3000
MAX_POSITION_SIZE_PCT = 0.10
STOP_LOSS_COOLDOWN_DAYS = 7

logger = logging.getLogger(__name__)

# =============================================================================
# MARKET REGIME FILTER
# =============================================================================
class MarketRegime:
    def __init__(self, benchmark_symbol="SPY"):
        self.symbol = benchmark_symbol
        self.is_bullish = True
        self.last_price = 0.0
        self.ma_200 = 0.0

    def update(self, trader):
        try:
            # Fetch last 200 days for trend analysis
            end = datetime.now()
            # Buffer for holidays/weekends
            start = end - timedelta(days=300)
            
            bars = trader.get_historical_prices(self.symbol, 200, "day")
            if bars is None or len(bars) < 200:
                self.is_bullish = True
                return

            closes = bars['close']
            current_price = closes.iloc[-1]
            ma_200 = closes.mean()
            
            # Bullish if price > 200MA (with 3% buffer to avoid chop)
            self.is_bullish = current_price > (ma_200 * 0.97)
            
            status = "BULLISH ✅" if self.is_bullish else "BEARISH ❌ (SHORTING ENABLED)"
            logger.info(f"MARKET REGIME ({self.symbol}): {status} [Price: {current_price:.2f} | 200MA: {ma_200:.2f}]")

        except Exception as e:
            logger.error(f"MarketRegime check failed: {e}")

# =============================================================================
# HELPER CLASSES
# =============================================================================
class KellyCriterion:
    @staticmethod
    def calculate_kelly(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5) -> float:
        if avg_win <= 0 or win_rate <= 0 or win_rate >= 1: return 0.05
        loss_rate = 1 - win_rate
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        kelly = (win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
        return max(0.01, min(kelly * fraction, 0.15))

class CircuitBreaker:
    def __init__(self, max_daily_loss=0.03, max_drawdown=0.10, max_consecutive_losses=5, max_trades_per_hour=10):
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.max_trades_per_hour = max_trades_per_hour
        self.is_tripped = False
        self.trip_reason = None

    def check(self, portfolio_val, start_val, peak_val, consec_losses, recent_trades):
        if start_val > 0:
            daily_pnl = (portfolio_val - start_val) / start_val
            if daily_pnl <= -self.max_daily_loss: return True, f"Daily loss exceeded: {daily_pnl:.2%}"
        if peak_val > 0:
            drawdown = (portfolio_val - peak_val) / peak_val
            if drawdown <= -self.max_drawdown: return True, f"Max drawdown exceeded: {drawdown:.2%}"
        if consec_losses >= self.max_consecutive_losses: return True, f"Consecutive losses limit hit: {consecutive_losses}"
        return False, ""

    def trip(self, reason):
        self.is_tripped = True
        self.trip_reason = reason
        logger.error(f"⛔ CIRCUIT BREAKER TRIPPED: {reason}")

    def reset(self):
        self.is_tripped = False
        self.trip_reason = None
        logger.info("✅ Circuit breaker reset")

class RiskLabAIModel:
    def __init__(self, profit_taking=2.5, stop_loss=2.5, max_holding=10, n_cv_splits=5, margin_threshold=0.03):
        self.labeler = TripleBarrierLabeler(profit_taking, stop_loss, max_holding)
        self.meta_labeler = MetaLabeler()
        self.cv = PurgedCrossValidator(n_splits=n_cv_splits)
        self.hrp = HRPPortfolio()
        self.primary_model = None
        self.meta_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.margin_threshold = margin_threshold

    def prepare_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=bars.index)
        close = bars['close']
        volume = bars.get('volume', pd.Series(0, index=bars.index))
        features['log_ret_1'] = np.log(close / close.shift(1))
        features['log_ret_5'] = np.log(close / close.shift(5))
        returns_std = features['log_ret_1'].rolling(window=20).std()
        features['norm_volatility'] = features['log_ret_1'] / (returns_std + 1e-8)
        features['vol_regime'] = returns_std / returns_std.rolling(window=100).mean()
        features['vol_imbalance'] = np.sign(close.diff()) * volume
        self.feature_names = features.columns.tolist()
        return features.dropna()

    def train(self, bars: pd.DataFrame, min_samples: int = 100) -> Dict:
        if bars.index.tz is not None:
            bars.index = bars.index.tz_convert('UTC').tz_localize(None)
        features = self.prepare_features(bars)
        if len(features) < min_samples:
            return {'success': False, 'reason': 'insufficient_samples'}
        
        labels = self.labeler.label(close=bars['close'], events=pd.DataFrame(index=features.index))
        aligned_idx = features.index.intersection(labels.index)
        features = features.loc[aligned_idx]
        labels = labels.loc[aligned_idx]
        
        X = features
        y = labels['bin']
        self.label_encoder = LabelEncoder().fit(y)
        X_scaled = StandardScaler().fit_transform(X)
        self.scaler = StandardScaler().fit(X)
        
        self.primary_model = XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1)
        self.primary_model.fit(X_scaled, self.label_encoder.transform(y))
        
        preds = self.primary_model.predict(X_scaled)
        meta_y = (preds == self.label_encoder.transform(y)).astype(int)
        self.meta_model = LogisticRegression().fit(X_scaled, meta_y)
        
        return {'success': True, 'n_samples': len(features), 'primary_accuracy': 0.0}

    def predict(self, bars: pd.DataFrame, prob_threshold=0.015, meta_threshold=0.5) -> Tuple[int, float]:
        if not self.primary_model: return 0, 0.0
        features = self.prepare_features(bars)
        if features.empty: return 0, 0.0
        X_scaled = self.scaler.transform(features.iloc[[-1]])
        
        probs = self.primary_model.predict_proba(X_scaled)[0]
        # Identify class indices for 1 (Long) and -1 (Short)
        try:
            # Assumes classes are [-1, 0, 1] or similar
            classes = list(self.label_encoder.classes_)
            idx_up = classes.index(1) if 1 in classes else None
            idx_down = classes.index(-1) if -1 in classes else None
            
            prob_up = probs[idx_up] if idx_up is not None else 0
            prob_down = probs[idx_down] if idx_down is not None else 0
        except:
            return 0, 0.0
        
        direction = 0
        if prob_up > prob_down + self.margin_threshold: direction = 1
        elif prob_down > prob_up + self.margin_threshold: direction = -1
        
        if direction == 0: return 0, 0.0
        
        meta_conf = self.meta_model.predict_proba(X_scaled)[0][1]
        if meta_conf < meta_threshold: return 0, 0.0
        
        return direction, meta_conf

    def train_from_ticks(self, symbol, threshold=None, min_samples=100):
        storage = TickStorage(DB_PATH)
        ticks = storage.load_ticks(symbol)
        storage.close()
        if not ticks: return {'success': False, 'reason': 'no_ticks'}
        bars = generate_bars_from_ticks(ticks, threshold=threshold or INITIAL_IMBALANCE_THRESHOLD)
        df = pd.DataFrame(bars)
        if df.empty: return {'success': False, 'reason': 'no_bars'}
        df['bar_end'] = pd.to_datetime(df['bar_end'])
        df = df.set_index('bar_end').sort_index()
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return self.train(df, min_samples=min_samples)

# =============================================================================
# MAIN BOT STRATEGY
# =============================================================================
class RiskLabAICombined(Strategy):
    SLEEPTIME = "1M"

    def initialize(self, parameters: Optional[Dict] = None):
        if parameters is None: parameters = getattr(self, 'parameters', {})
        self.symbols = parameters.get('symbols', SYMBOLS)
        self.use_kelly = parameters.get('use_kelly_sizing', True)
        
        self.circuit_breaker = CircuitBreaker()
        self.market_regime = MarketRegime("SPY")
        
        self.model_storage = ModelStorage(local_dir=os.environ.get("MODELS_PATH", "models"))
        self.symbol_models = {}
        self.stop_loss_cooldowns = {}
        self.daily_start_value = None
        self.peak_portfolio_value = None
        self.consecutive_losses = 0
        self.last_buy_check = None
        
        self._load_models()

    def on_trading_iteration(self):
        if not is_market_open(): return

        val = self.get_portfolio_value()
        if self.daily_start_value is None: self.daily_start_value = val
        if self.peak_portfolio_value is None: self.peak_portfolio_value = val
        self.peak_portfolio_value = max(self.peak_portfolio_value, val)

        # 1. Circuit Breaker
        trip, reason = self.circuit_breaker.check(val, self.daily_start_value, self.peak_portfolio_value, self.consecutive_losses, [])
        if trip:
            if not self.circuit_breaker.is_tripped: self.circuit_breaker.trip(reason)
            return

        # 2. Market Regime (Hourly)
        now = datetime.now()
        if self.last_buy_check is None or (now - self.last_buy_check) > timedelta(minutes=60):
            self.market_regime.update(self)
            self.last_buy_check = now

        # 3. Check Exits (Take Profit / Stop Loss)
        self._check_positions()

        # 4. Check Entries
        # We allow signal generation even in Bear Markets, to catch shorts.
        self._generate_signals()

    def _generate_signals(self):
        for symbol in self.symbols:
            if symbol in self.stop_loss_cooldowns: continue
            if symbol not in self.symbol_models: continue
            
            try:
                bars = self._get_bars(symbol)
                if bars is None: continue
                
                model = self.symbol_models[symbol]
                signal, conf = model.predict(bars)
                
                # --- MARKET REGIME FILTER APPLIED HERE ---
                if signal == 1: 
                    # LONG SIGNAL: Only if Market is Bullish
                    if self.market_regime.is_bullish:
                        self._execute_entry(symbol, conf, "long")
                    else:
                        logger.info(f"{symbol}: Long signal IGNORED (Bear Market)")
                        
                elif signal == -1:
                    # SHORT SIGNAL: Valid (Great for Bear Markets)
                    self._execute_entry(symbol, conf, "short")
                    
            except Exception as e:
                logger.error(f"Signal error {symbol}: {e}")

    def _execute_entry(self, symbol, confidence, side):
        val = self.get_portfolio_value()
        
        # Position Sizing
        size_pct = MAX_POSITION_SIZE_PCT
        if self.use_kelly:
            size_pct = KellyCriterion.calculate_kelly(0.55, 0.04, 0.02, fraction=0.5)
        
        usd_size = val * size_pct * confidence
        price = self.get_last_price(symbol)
        qty = int(usd_size / price)
        
        if qty > 0:
            current = self.get_position(symbol)
            # Only enter if flat
            if current is None or current.quantity == 0:
                if side == "long":
                    logger.info(f"🔵 BUY {symbol}: {qty} shares (Conf: {confidence:.2f})")
                    self.submit_order(self.create_order(symbol, qty, "buy"))
                elif side == "short":
                    logger.info(f"🔻 SHORT {symbol}: {qty} shares (Conf: {confidence:.2f})")
                    self.submit_order(self.create_order(symbol, qty, "sell"))

    def _check_positions(self):
        """
        Manages Exits:
        - Longs: Sell to Close
        - Shorts: Buy to Cover
        """
        if not hasattr(self.broker, 'api'): return
        
        for pos in self.broker.api.get_all_positions():
            symbol = pos.symbol
            pnl = float(pos.unrealized_plpc)
            qty = float(pos.qty) # Negative if short
            
            is_long = qty > 0
            abs_qty = abs(qty)
            
            close_action = "sell" if is_long else "buy"
            
            # Logic: If Short (-2%), PnL shows +2% profit. Logic holds.
            
            if pnl >= 0.04:
                logger.info(f"🟢 TAKE PROFIT {symbol} ({pnl:.2%})")
                self.submit_order(self.create_order(symbol, abs_qty, close_action))
                
            elif pnl <= -0.02:
                logger.info(f"🔴 STOP LOSS {symbol} ({pnl:.2%})")
                self.submit_order(self.create_order(symbol, abs_qty, close_action))
                self.stop_loss_cooldowns[symbol] = datetime.now()
                self.consecutive_losses += 1

    def _get_bars(self, symbol):
        try:
            storage = TickStorage(DB_PATH)
            ticks = storage.load_ticks(symbol)
            storage.close()
            if not ticks: return None
            # Limit to recent history for speed
            bars = generate_bars_from_ticks(ticks[-5000:], threshold=INITIAL_IMBALANCE_THRESHOLD)
            df = pd.DataFrame(bars)
            df['bar_end'] = pd.to_datetime(df['bar_end'])
            df = df.set_index('bar_end').sort_index()
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df
        except: return None

    def _load_models(self):
        for symbol in self.symbols:
            try:
                data = self.model_storage.load_model(symbol, prefer_s3=True)
                if data:
                    m = RiskLabAIModel()
                    m.primary_model = data['primary_model']
                    m.meta_model = data['meta_model']
                    m.scaler = data['scaler']
                    m.label_encoder = data['label_encoder']
                    self.symbol_models[symbol] = m
            except: pass
        logger.info(f"Loaded {len(self.symbol_models)} models")

    def _load_state(self): pass
    def _save_state(self): pass