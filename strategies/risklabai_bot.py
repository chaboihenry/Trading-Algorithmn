import logging
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# --- PATH SETUP (CRITICAL FIX) ---
# This forces Python to look 2 levels up for 'data' and 'utils'
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
INITIAL_IMBALANCE_THRESHOLD = 3000   # Volume threshold for bars
CUSUM_EVENT_WINDOW_SECONDS = 86400   # 24 hours
STOP_LOSS_COOLDOWN_DAYS = 7          # Days to wait after stopping out
MAX_POSITION_SIZE_PCT = 0.10         # Max 10% allocation per trade (fallback)

logger = logging.getLogger(__name__)

class KellyCriterion:
    @staticmethod
    def calculate_kelly(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.5) -> float:
        if avg_win <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.05
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
        self.trip_timestamp = None

    def check(self, portfolio_val, start_val, peak_val, consec_losses, recent_trades):
        if start_val > 0:
            daily_pnl = (portfolio_val - start_val) / start_val
            if daily_pnl <= -self.max_daily_loss:
                return True, f"Daily loss exceeded: {daily_pnl:.2%}"
        if peak_val > 0:
            drawdown = (portfolio_val - peak_val) / peak_val
            if drawdown <= -self.max_drawdown:
                return True, f"Max drawdown exceeded: {drawdown:.2%}"
        if consec_losses >= self.max_consecutive_losses:
            return True, f"Consecutive losses limit hit: {consecutive_losses}"
        hour_ago = datetime.now() - timedelta(hours=1)
        trades_last_hour = len([t for t in recent_trades if t['timestamp'] > hour_ago])
        if trades_last_hour >= self.max_trades_per_hour:
            return True, f"Too many trades/hour: {trades_last_hour}"
        return False, ""

    def trip(self, reason):
        self.is_tripped = True
        self.trip_reason = reason
        self.trip_timestamp = datetime.now()
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
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features['rsi'] = (rsi / 100.0) - 0.5
        rolling_mean = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        features['bb_pos'] = (close - rolling_mean) / (rolling_std + 1e-8)
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
        y_encoded = self.label_encoder.transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=False)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.primary_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        self.primary_model.fit(X_train_scaled, y_train)
        primary_preds = self.primary_model.predict(X_train_scaled)
        meta_X = X_train_scaled 
        meta_y = (primary_preds == y_train).astype(int)
        self.meta_model = LogisticRegression(class_weight='balanced')
        self.meta_model.fit(meta_X, meta_y)
        acc = self.primary_model.score(X_test_scaled, y_test)
        return {'success': True, 'primary_accuracy': acc, 'n_samples': len(features)}

    def predict(self, bars: pd.DataFrame, prob_threshold=0.015, meta_threshold=0.5) -> Tuple[int, float]:
        if not self.primary_model: return 0, 0.0
        features = self.prepare_features(bars)
        if features.empty: return 0, 0.0
        X_scaled = self.scaler.transform(features.iloc[[-1]])
        probs = self.primary_model.predict_proba(X_scaled)[0]
        class_probs = {cls: p for cls, p in zip(self.label_encoder.classes_, probs)}
        direction = 0
        if class_probs.get(1, 0) > self.margin_threshold: direction = 1
        elif class_probs.get(-1, 0) > self.margin_threshold: direction = -1
        if direction == 0: return 0, 0.0
        meta_conf = self.meta_model.predict_proba(X_scaled)[0][1]
        if meta_conf < meta_threshold: return 0, 0.0
        return direction, meta_conf

    def train_from_ticks(self, symbol: str, threshold: float = None, min_samples: int = 100):
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

class RiskLabAICombined(Strategy):
    SLEEPTIME = "1M"
    def initialize(self, parameters: Optional[Dict] = None):
        if parameters is None: parameters = getattr(self, 'parameters', {})
        self.symbols = parameters.get('symbols', SYMBOLS)
        self.profit_taking = parameters.get('profit_taking', 2.5)
        self.stop_loss = parameters.get('stop_loss', 2.5)
        self.max_holding = parameters.get('max_holding', 10)
        self.prob_threshold = parameters.get('prob_threshold', 0.015)
        self.meta_threshold = parameters.get('meta_threshold', 0.001)
        self.use_kelly = parameters.get('use_kelly_sizing', True)
        self.kelly_fraction = parameters.get('kelly_fraction', 0.5)
        self.cusum_filter = CUSUMEventFilter()
        self.model_storage = ModelStorage(local_dir=os.environ.get("MODELS_PATH", "models"))
        self.symbol_models = {}
        self.stop_loss_cooldowns = {}
        self.daily_start_value = None
        self.peak_portfolio_value = None
        self.consecutive_losses = 0
        self.trades_today = 0
        self.last_buy_check = None
        self.dynamic_win_rate = 0.50
        self.circuit_breaker = CircuitBreaker(max_daily_loss=parameters.get('daily_loss_limit_pct', 0.03), max_drawdown=parameters.get('max_drawdown_pct', 0.10))
        self._load_models()
        self._load_state()

    def on_trading_iteration(self):
        if not is_market_open():
            logger.info("Market closed.")
            return
        val = self.get_portfolio_value()
        if self.daily_start_value is None or (self.last_buy_check and self.last_buy_check.date() != datetime.now().date()):
            self.daily_start_value = val
            self.trades_today = 0
            self.circuit_breaker.reset()
        if self.peak_portfolio_value is None: self.peak_portfolio_value = val
        self.peak_portfolio_value = max(self.peak_portfolio_value, val)
        trip, reason = self.circuit_breaker.check(val, self.daily_start_value, self.peak_portfolio_value, self.consecutive_losses, [])
        if trip:
            if not self.circuit_breaker.is_tripped:
                self.circuit_breaker.trip(reason)
                self._save_state()
            return
        if self.circuit_breaker.is_tripped: return
        self._check_positions()
        now = datetime.now()
        if self.last_buy_check is None or (now - self.last_buy_check) > timedelta(hours=1):
            logger.info("Checking buy signals...")
            self._generate_signals()
            self.last_buy_check = now

    def _generate_signals(self):
        for symbol in self.symbols:
            if symbol in self.stop_loss_cooldowns:
                if (datetime.now() - self.stop_loss_cooldowns[symbol]).days < STOP_LOSS_COOLDOWN_DAYS: continue
                del self.stop_loss_cooldowns[symbol]
            if symbol not in self.symbol_models: continue
            try:
                bars = self._get_bars(symbol)
                if bars is None or len(bars) < 50: continue
                model = self.symbol_models[symbol]
                signal, conf = model.predict(bars, self.prob_threshold, self.meta_threshold)
                if signal == 1: self._execute_buy(symbol, conf)
            except Exception as e: logger.error(f"Error {symbol}: {e}")

    def _execute_buy(self, symbol, confidence):
        val = self.get_portfolio_value()
        size_pct = MAX_POSITION_SIZE_PCT
        if self.use_kelly:
            size_pct = KellyCriterion.calculate_kelly(self.dynamic_win_rate, 0.04, 0.02, self.kelly_fraction)
        usd_size = val * size_pct * confidence
        price = self.get_last_price(symbol)
        qty = int(usd_size / price)
        if qty > 0:
            current = self.get_position(symbol)
            if current is None or current.quantity == 0:
                logger.info(f"🔵 BUY {symbol}: {qty} shares (Conf: {confidence:.2f})")
                self.submit_order(self.create_order(symbol, qty, "buy"))
                self.trades_today += 1

    def _check_positions(self):
        if not hasattr(self.broker, 'api'): return
        for pos in self.broker.api.get_all_positions():
            symbol = pos.symbol
            pnl = float(pos.unrealized_plpc)
            qty = float(pos.qty)
            reason = None
            if pnl >= 0.04: reason = "TAKE PROFIT"
            elif pnl <= -0.02: reason = "STOP LOSS"
            if reason:
                logger.info(f"🔴 CLOSE {symbol}: {reason} ({pnl:.2%})")
                self.submit_order(self.create_order(symbol, abs(qty), "sell"))
                if "STOP" in reason:
                    self.stop_loss_cooldowns[symbol] = datetime.now()
                    self.consecutive_losses += 1
                else: self.consecutive_losses = 0

    def _get_bars(self, symbol):
        try:
            storage = TickStorage(DB_PATH)
            ticks = storage.load_ticks(symbol)
            storage.close()
            if not ticks: return None
            bars = generate_bars_from_ticks(ticks, threshold=INITIAL_IMBALANCE_THRESHOLD)
            df = pd.DataFrame(bars)
            df['bar_end'] = pd.to_datetime(df['bar_end'])
            df = df.set_index('bar_end').sort_index()
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            return df.tail(100)
        except Exception: return None

    def _load_models(self):
        for symbol in self.symbols:
            try:
                data = self.model_storage.load_model(symbol, prefer_s3=True)
                if data:
                    model = RiskLabAIModel(self.profit_taking, self.stop_loss, self.max_holding)
                    model.primary_model = data.get('primary_model')
                    model.meta_model = data.get('meta_model')
                    model.scaler = data.get('scaler')
                    model.label_encoder = data.get('label_encoder')
                    self.symbol_models[symbol] = model
            except Exception: pass
        logger.info(f"Loaded {len(self.symbol_models)} models.")

    def _load_state(self):
        if os.path.exists("bot_state.json"):
            try:
                with open("bot_state.json") as f:
                    state = json.load(f)
                    cd = state.get("cooldowns", {})
                    self.stop_loss_cooldowns = {k: datetime.fromisoformat(v) for k, v in cd.items()}
            except Exception: pass

    def _save_state(self):
        state = {"cooldowns": {k: v.isoformat() for k, v in self.stop_loss_cooldowns.items()}}
        with open("bot_state.json", "w") as f: json.dump(state, f)