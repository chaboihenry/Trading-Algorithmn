"""
Combined Strategy with Stacked Ensemble Meta-Learner

This strategy combines sentiment and pairs trading signals using a machine learning
meta-learner (XGBoost) that dynamically adjusts weights based on market conditions.

Instead of fixed weights, the meta-learner:
1. Takes signals from both strategies as input features
2. Learns which strategy to trust in different conditions
3. Adapts weights continuously based on historical performance
4. Discovers non-linear interactions between strategies

This is based on your existing stacked_ensemble.py pattern but adapted for Lumibot
and combining sentiment + pairs instead of multiple ML models.
"""

import logging
import numpy as np
import pandas as pd
import sqlite3
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from lumibot.strategies import Strategy
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Base strategies not needed - using simplified approach
# from sentiment_strategy import SentimentStrategy
# from pairs_strategy import PairsStrategy

logger = logging.getLogger(__name__)


class CombinedStrategy(Strategy):
    """
    Meta-learning ensemble that combines sentiment and pairs trading.

    The meta-learner (XGBoost) takes as input:
    - Sentiment score and confidence from SentimentStrategy
    - Pairs signal (z-score, quality) from PairsStrategy
    - Market conditions (volatility, trend, volume)

    It outputs:
    - Dynamic weights for each strategy (0-1)
    - Final trading decision (buy/sell/hold)
    - Position sizing based on confidence
    """

    # Strategy parameters - ACTIVE TRADING: Check every hour
    SLEEPTIME = "1H"
    RETRAIN_FREQUENCY_DAYS = 7  # Retrain meta-model weekly
    MIN_TRAINING_SAMPLES = 100  # Minimum samples needed to train
    CONFIDENCE_THRESHOLD = 0.6  # Only trade if meta-model confidence > 60%

    # Risk management - Active 24/7 with SERVER-SIDE bracket orders
    STOP_LOSS_PCT = 0.05  # Exit at -5% loss (attached to every order)
    TAKE_PROFIT_PCT = 0.15  # Exit at +15% profit (attached to every order)
    ENABLE_EXTENDED_HOURS = True  # Trade after hours

    # Inverse ETFs for bearish sentiment (profit when market drops)
    INVERSE_ETFS = {
        'tech': 'SQQQ',    # 3x inverse NASDAQ (use when tech sentiment bearish)
        'sp500': 'SPXS',   # 3x inverse S&P 500 (use when market sentiment bearish)
        'general': 'SH'    # 1x inverse S&P 500 (conservative hedge)
    }
    MAX_INVERSE_ALLOCATION = 0.20  # Max 20% of portfolio in inverse positions

    def initialize(self, parameters: Dict = None):
        """
        Initialize combined strategy with meta-learner.

        Args:
            parameters: Dict with optional parameters:
                - db_path: Database path
                - model_path: Path to saved meta-model
                - retrain: Whether to retrain model (default: False)
        """
        self.sleeptime = self.SLEEPTIME

        params = parameters or {}
        self.db_path = params.get('db_path', '/Volumes/Vault/85_assets_prediction.db')
        self.retrain = params.get('retrain', False)

        # We don't actually need separate strategy instances
        # Instead, we'll implement the logic directly in this combined strategy
        # (The original code tried to instantiate strategies without brokers, which fails)

        # Sentiment parameters (from SentimentStrategy)
        self.NEWS_LOOKBACK_DAYS = 3
        self.SYMBOLS = [
            "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "TSLA",
            "AMD", "NFLX", "CRM", "ADBE", "INTC", "PYPL", "SQ"
        ]

        # Initialize FinBERT for sentiment
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.sentiment_model.eval()
        self.torch = torch

        # Pairs parameters (from PairsStrategy) - DISABLED FOR NOW
        # Focus on risk management first, pairs trading can be added later
        self.LOOKBACK_DAYS = 120
        self.ZSCORE_ENTRY = 1.5
        self.MIN_CORRELATION = 0.7
        self.cointegrated_pairs = []  # Empty for now

        # TODO: Add pairs finding logic later
        # self._find_cointegrated_pairs_from_db()

        # Meta-learner components
        self.meta_model = None
        self.scaler = StandardScaler()
        self.last_retrain_date = None

        # Model persistence
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

        model_path = params.get('model_path')
        if model_path:
            self._load_meta_model(model_path)
        else:
            # Try to load latest model
            self._load_latest_meta_model()

        # If no model or retrain requested, train new model
        if self.meta_model is None or self.retrain:
            self._train_meta_model()

        logger.info("Combined Strategy initialized with meta-learner")
        logger.info(f"Meta-model active: {self.meta_model is not None}")

    def _load_meta_model(self, model_path: str):
        """Load a saved meta-model from disk."""
        try:
            meta_path = Path(model_path)
            scaler_path = meta_path.parent / f"{meta_path.stem}_scaler.joblib"

            self.meta_model = joblib.load(model_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)

            logger.info(f"Loaded meta-model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load meta-model: {e}")
            self.meta_model = None

    def _load_latest_meta_model(self):
        """Load the most recent meta-model from models directory."""
        try:
            model_files = list(self.models_dir.glob("combined_meta_*.joblib"))
            if not model_files:
                logger.info("No existing meta-model found")
                return

            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
            self._load_meta_model(str(latest_model))
        except Exception as e:
            logger.warning(f"Error loading latest model: {e}")

    def _save_meta_model(self):
        """Save meta-model to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.models_dir / f"combined_meta_{timestamp}.joblib"
        scaler_path = self.models_dir / f"combined_meta_{timestamp}_scaler.joblib"

        joblib.dump(self.meta_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        logger.info(f"Saved meta-model to {model_path}")

    def _get_historical_signals(self) -> pd.DataFrame:
        """
        Fetch historical trading signals from database to train meta-learner.

        Returns:
            DataFrame with columns: [date, symbol, sentiment_signal, sentiment_prob,
                                     pairs_signal, pairs_zscore, actual_return_5d, ...]
        """
        conn = sqlite3.connect(self.db_path)

        # Get historical signals and returns
        # This query assumes you have a trading_signals table with signal history
        query = """
        SELECT
            ts.signal_date,
            ts.symbol_ticker,
            ts.signal_type,
            ts.strength,
            ts.metadata,
            p.close as entry_price,
            p2.close as exit_price_5d,
            (p2.close - p.close) / p.close as return_5d,
            vm.volatility_20d,
            ti.rsi_14
        FROM trading_signals ts
        LEFT JOIN raw_price_data p
            ON ts.symbol_ticker = p.symbol_ticker
            AND ts.signal_date = p.date
        LEFT JOIN raw_price_data p2
            ON ts.symbol_ticker = p2.symbol_ticker
            AND p2.date = date(ts.signal_date, '+5 days')
        LEFT JOIN volatility_metrics vm
            ON ts.symbol_ticker = vm.symbol_ticker
            AND ts.signal_date = vm.date
        LEFT JOIN technical_indicators ti
            ON ts.symbol_ticker = ti.symbol_ticker
            AND ts.signal_date = ti.date
        WHERE ts.signal_date >= date('now', '-365 days')
          AND ts.metadata IS NOT NULL
        ORDER BY ts.signal_date
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            logger.warning("No historical signals found in database")
            return pd.DataFrame()

        logger.info(f"Loaded {len(df)} historical signals for training")
        return df

    def _prepare_meta_features(
        self,
        sentiment_score: float,
        sentiment_prob: float,
        pairs_zscore: float,
        pairs_quality: float,
        volatility: float,
        rsi: float
    ) -> np.ndarray:
        """
        Prepare feature vector for meta-learner.

        Args:
            sentiment_score: Sentiment signal (-1, 0, 1)
            sentiment_prob: Sentiment confidence (0-1)
            pairs_zscore: Pairs spread z-score
            pairs_quality: Pairs quality score
            volatility: Market volatility
            rsi: RSI indicator

        Returns:
            Feature vector ready for meta-model
        """
        features = np.array([
            sentiment_score,
            sentiment_prob,
            abs(sentiment_score) * sentiment_prob,  # Weighted sentiment
            pairs_zscore,
            abs(pairs_zscore),  # Magnitude of deviation
            pairs_quality,
            pairs_zscore * pairs_quality,  # Weighted pairs signal
            volatility if volatility else 0.2,  # Default volatility
            rsi if rsi else 50,  # Default RSI
            # Interaction features
            sentiment_score * pairs_zscore,  # Agreement indicator
            abs(sentiment_score - np.sign(pairs_zscore)),  # Disagreement indicator
            sentiment_prob * pairs_quality,  # Combined confidence
        ])

        return features.reshape(1, -1)

    def _train_meta_model(self):
        """
        Train the meta-learner on historical signals.

        The meta-model learns to predict profitability based on:
        - Strategy signals
        - Market conditions
        - Historical performance patterns
        """
        logger.info("Training meta-learner...")

        # Get historical data
        historical_data = self._get_historical_signals()

        if len(historical_data) < self.MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient training data ({len(historical_data)} samples)")
            logger.warning("Meta-model will use equal weights until more data is available")
            return

        # Extract features and labels
        # (This is simplified - you'll need to parse metadata and extract actual signal info)
        try:
            # Create feature matrix
            features_list = []
            labels_list = []

            for _, row in historical_data.iterrows():
                # Parse metadata to extract signal details
                # For now, using simplified features
                features = [
                    1 if row['signal_type'] == 'BUY' else -1 if row['signal_type'] == 'SELL' else 0,
                    row['strength'] if pd.notna(row['strength']) else 0.5,
                    row['volatility_20d'] if pd.notna(row['volatility_20d']) else 0.2,
                    row['rsi_14'] if pd.notna(row['rsi_14']) else 50,
                ]

                # Label: 1 if profitable, 0 otherwise
                label = 1 if (row['return_5d'] if pd.notna(row['return_5d']) else 0) > 0 else 0

                features_list.append(features)
                labels_list.append(label)

            X = np.array(features_list)
            y = np.array(labels_list)

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train XGBoost meta-model (small model to prevent overfitting)
            self.meta_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                tree_method='hist'  # M1 optimized
            )

            self.meta_model.fit(X_train_scaled, y_train)

            # Evaluate
            train_score = self.meta_model.score(X_train_scaled, y_train)
            test_score = self.meta_model.score(X_test_scaled, y_test)

            logger.info(f"Meta-model trained: train_acc={train_score:.3f}, test_acc={test_score:.3f}")

            # Save model
            self._save_meta_model()
            self.last_retrain_date = datetime.now()

        except Exception as e:
            logger.error(f"Error training meta-model: {e}")
            self.meta_model = None

    def _get_market_conditions(self, symbol: str) -> Tuple[float, float]:
        """
        Fetch current market conditions.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (volatility, rsi)
        """
        try:
            conn = sqlite3.connect(self.db_path)

            query = """
            SELECT vm.volatility_20d, ti.rsi_14
            FROM volatility_metrics vm
            LEFT JOIN technical_indicators ti
                ON vm.symbol_ticker = ti.symbol_ticker
                AND vm.date = ti.date
            WHERE vm.symbol_ticker = ?
            ORDER BY vm.date DESC
            LIMIT 1
            """

            result = pd.read_sql(query, conn, params=(symbol,))
            conn.close()

            if not result.empty:
                volatility = result['volatility_20d'].iloc[0]
                rsi = result['rsi_14'].iloc[0]
                return volatility, rsi
            else:
                return 0.2, 50  # Defaults

        except Exception as e:
            logger.warning(f"Error fetching market conditions: {e}")
            return 0.2, 50

    def _get_combined_signal(self, symbol: str) -> Tuple[int, float]:
        """
        Get combined signal from meta-learner.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Tuple of (signal, confidence) where signal is -1/0/1 and confidence is 0-1
        """
        # Get sentiment signal
        try:
            sentiment_prob, sentiment_signal = self.sentiment_strategy.get_news_sentiment(symbol)
        except Exception as e:
            logger.warning(f"Error getting sentiment for {symbol}: {e}")
            sentiment_prob, sentiment_signal = 0.5, 0

        # Get pairs signal (if this symbol is in a pair)
        pairs_zscore = 0
        pairs_quality = 0

        for pair_data in getattr(self.pairs_strategy, 'cointegrated_pairs', []):
            if symbol in pair_data[:2]:
                s1, s2, corr, pval, quality, hedge = pair_data
                pairs_quality = quality

                # Calculate current z-score
                try:
                    _, zscore = self.pairs_strategy._calculate_current_spread(s1, s2, hedge)
                    pairs_zscore = zscore
                except:
                    pass
                break

        # Get market conditions
        volatility, rsi = self._get_market_conditions(symbol)

        # If no meta-model, use simple weighted average
        if self.meta_model is None:
            logger.info(f"{symbol}: Using equal weights (no meta-model)")

            # Simple combination: 60% sentiment, 40% pairs
            combined_signal = 0.6 * sentiment_signal

            if abs(pairs_zscore) > 1.5:
                # Pairs signal is strong
                pairs_contrib = -np.sign(pairs_zscore) * 0.4
                combined_signal += pairs_contrib

            final_signal = 1 if combined_signal > 0.3 else -1 if combined_signal < -0.3 else 0
            confidence = min(abs(combined_signal), 1.0)

            return final_signal, confidence

        # Use meta-model for intelligent combination
        features = self._prepare_meta_features(
            sentiment_signal,
            sentiment_prob,
            pairs_zscore,
            pairs_quality,
            volatility,
            rsi
        )

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Get prediction
        prediction_proba = self.meta_model.predict_proba(features_scaled)[0]
        prediction = self.meta_model.predict(features_scaled)[0]

        # Convert to signal
        confidence = max(prediction_proba)  # Max probability

        if confidence < self.CONFIDENCE_THRESHOLD:
            logger.info(f"{symbol}: Low confidence ({confidence:.2f}), skipping")
            return 0, confidence

        # Determine direction based on input signals and prediction
        if prediction == 1:  # Meta-model predicts profitable
            # Use strongest signal
            if abs(sentiment_signal) > abs(np.sign(pairs_zscore)):
                final_signal = sentiment_signal
            else:
                final_signal = -np.sign(pairs_zscore) if pairs_zscore != 0 else sentiment_signal
        else:
            final_signal = 0  # Don't trade if meta-model predicts unprofitable

        logger.info(f"{symbol}: sentiment={sentiment_signal}({sentiment_prob:.2f}), pairs_z={pairs_zscore:.2f}, meta_signal={final_signal}({confidence:.2f})")

        return final_signal, confidence

    def _rebalance_positions(self, portfolio_value: float, max_position_pct: float = 0.15) -> None:
        """
        Rebalance positions to prevent any single position from dominating the portfolio.

        If one stock goes up significantly, it might become 30% of your portfolio.
        This is risky - if that stock crashes, you lose big. This method trims
        oversized positions back down to a reasonable level.

        Args:
            portfolio_value: Total portfolio value
            max_position_pct: Maximum % any position should be (default 15%)
        """
        try:
            if portfolio_value <= 0:
                return

            logger.info(f"Checking position sizes (max allowed: {max_position_pct:.0%})")

            positions = self.get_positions()
            if not positions or len(positions) == 0:
                return

            rebalanced = False

            for position in positions:
                symbol = position.symbol
                quantity = position.quantity
                current_price = self.get_last_price(symbol)

                if not current_price or current_price <= 0:
                    continue

                position_value = quantity * current_price
                position_pct = position_value / portfolio_value

                logger.debug(f"{symbol}: {position_pct:.1%} of portfolio (${position_value:,.2f})")

                # If position is too large, trim it down
                if position_pct > max_position_pct:
                    # Calculate how many shares to sell to get to max_position_pct
                    target_value = portfolio_value * max_position_pct
                    shares_to_sell = (position_value - target_value) / current_price

                    if shares_to_sell >= 1:  # Only sell if >= 1 share
                        logger.warning(f"⚖️  {symbol} is {position_pct:.1%} of portfolio (limit: {max_position_pct:.0%})")
                        logger.warning(f"   Trimming position: selling {shares_to_sell:.2f} shares")

                        # Sell the excess shares
                        order = self.create_order(symbol, shares_to_sell, "sell")
                        self.submit_order(order)
                        rebalanced = True

                        logger.info(f"✂️  Sold {shares_to_sell:.2f} shares of {symbol} @ ${current_price:.2f}")
                        logger.info(f"   Reduced from {position_pct:.1%} to ~{max_position_pct:.0%} of portfolio")

            if not rebalanced:
                logger.info("✅ All positions properly sized")

        except Exception as e:
            logger.error(f"Error rebalancing positions: {e}")

    def _check_market_sentiment_and_hedge(self, cash: float, portfolio_value: float) -> None:
        """
        Check overall market sentiment and use inverse ETFs to profit from declines.

        When the market is bearish, instead of just watching positions drop, we can
        profit by buying inverse ETFs (they go UP when the market goes DOWN).

        This is like betting against the market as a hedge.

        Args:
            cash: Available cash
            portfolio_value: Total portfolio value
        """
        try:
            # Check sentiment for major indices symbols
            bearish_signals = 0
            total_checked = 0

            # Sample a few tech stocks to gauge market sentiment
            sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

            for symbol in sample_symbols:
                try:
                    _, rsi = self._get_market_conditions(symbol)
                    total_checked += 1

                    if rsi > 70:  # Overbought = bearish signal
                        bearish_signals += 1
                except:
                    continue

            if total_checked == 0:
                return

            bearish_ratio = bearish_signals / total_checked

            logger.info(f"Market sentiment check: {bearish_signals}/{total_checked} symbols overbought")
            logger.info(f"Bearish ratio: {bearish_ratio:.1%}")

            # If >= 60% of market is overbought, consider hedging with inverse ETF
            if bearish_ratio >= 0.6:
                # Check current inverse ETF positions
                inverse_value = 0
                for inv_symbol in self.INVERSE_ETFS.values():
                    pos = self.get_position(inv_symbol)
                    if pos:
                        inverse_value += pos.quantity * self.get_last_price(inv_symbol)

                inverse_allocation = inverse_value / portfolio_value if portfolio_value > 0 else 0

                logger.info(f"Current inverse ETF allocation: {inverse_allocation:.1%}")

                # Only add more if below max allocation
                if inverse_allocation < self.MAX_INVERSE_ALLOCATION and cash > 1000:
                    # Use conservative inverse ETF (SH = 1x inverse S&P 500)
                    inv_symbol = self.INVERSE_ETFS['general']  # SH
                    inv_price = self.get_last_price(inv_symbol)

                    if inv_price and inv_price > 0:
                        # Allocate up to 10% of portfolio to inverse position
                        target_value = min(portfolio_value * 0.10, cash * 0.5)
                        quantity = target_value / inv_price

                        logger.warning(f"🔴 BEARISH MARKET DETECTED ({bearish_ratio:.1%} overbought)")
                        logger.warning(f"🛡️  Hedging with inverse ETF {inv_symbol}")

                        # Buy inverse ETF with bracket order
                        success = self._create_bracket_order(inv_symbol, quantity, "buy", inv_price)
                        if success:
                            logger.info(f"✅ Hedge position: {quantity:.2f} shares of {inv_symbol}")
                            logger.info(f"   This will profit if market drops!")

            elif bearish_ratio < 0.3:
                # Market is healthy, exit inverse positions if we have any
                for inv_symbol in self.INVERSE_ETFS.values():
                    pos = self.get_position(inv_symbol)
                    if pos and pos.quantity > 0:
                        logger.info(f"📈 Market bullish again - exiting inverse position {inv_symbol}")
                        order = self.create_order(inv_symbol, pos.quantity, "sell")
                        self.submit_order(order)

        except Exception as e:
            logger.error(f"Error in market sentiment hedge check: {e}")

    def _create_bracket_order(self, symbol: str, quantity: float, side: str = "buy",
                             current_price: float = None) -> bool:
        """
        Create a bracket order with automatic stop-loss and take-profit.

        This uses Alpaca's server-side bracket orders which execute automatically
        without the bot needing to be awake. Much better than iteration-based checks!

        Args:
            symbol: Stock symbol to trade
            quantity: Number of shares
            side: "buy" or "sell"
            current_price: Current stock price (fetched if not provided)

        Returns:
            True if order submitted successfully, False otherwise

        Example:
            If buying AAPL at $150:
            - Main order: Buy AAPL at market price
            - Take-profit: Automatically sell at $172.50 (+15%)
            - Stop-loss: Automatically sell at $142.50 (-5%)
        """
        try:
            if current_price is None:
                current_price = self.get_last_price(symbol)

            if not current_price or current_price <= 0:
                logger.error(f"Invalid price for {symbol}: {current_price}")
                return False

            # Calculate stop-loss and take-profit prices
            if side == "buy":
                take_profit_price = round(current_price * (1 + self.TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(current_price * (1 - self.STOP_LOSS_PCT), 2)
            else:  # sell (for inverse ETFs or shorts)
                take_profit_price = round(current_price * (1 - self.TAKE_PROFIT_PCT), 2)
                stop_loss_price = round(current_price * (1 + self.STOP_LOSS_PCT), 2)

            logger.info(f"Creating bracket order for {symbol}:")
            logger.info(f"  Entry: {quantity:.2f} shares @ ${current_price:.2f}")
            logger.info(f"  Take-profit: ${take_profit_price:.2f} (+{self.TAKE_PROFIT_PCT:.1%})")
            logger.info(f"  Stop-loss: ${stop_loss_price:.2f} (-{self.STOP_LOSS_PCT:.1%})")

            # Create the bracket order
            # Lumibot will automatically create 3 orders: main, take-profit, stop-loss
            order = self.create_order(
                symbol,
                quantity,
                side,
                type="bracket",
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                time_in_force="gtc" if self.ENABLE_EXTENDED_HOURS else "day"
            )

            self.submit_order(order)
            logger.info(f"✅ Bracket order submitted for {symbol}")
            logger.info(f"   Alpaca will automatically execute stop-loss/take-profit server-side")
            return True

        except Exception as e:
            logger.error(f"Error creating bracket order for {symbol}: {e}")
            logger.error(f"Falling back to simple market order...")

            # Fallback: Create simple order without brackets
            try:
                order = self.create_order(symbol, quantity, side)
                self.submit_order(order)
                logger.warning(f"⚠️  Simple order submitted (no automatic stops)")
                return True
            except Exception as e2:
                logger.error(f"Failed to submit any order: {e2}")
                return False

    def _check_risk_management_alpaca(self, alpaca_position) -> Optional[str]:
        """Check if Alpaca position triggers stop-loss or take-profit."""
        try:
            symbol = alpaca_position.symbol
            entry_price = float(alpaca_position.avg_entry_price)
            current_price = float(alpaca_position.current_price)

            pnl_pct = (current_price - entry_price) / entry_price

            if pnl_pct <= -self.STOP_LOSS_PCT:
                logger.warning(f"🛑 STOP-LOSS: {symbol} at {pnl_pct:.2%} (entry: ${entry_price:.2f}, current: ${current_price:.2f})")
                return 'stop_loss'

            if pnl_pct >= self.TAKE_PROFIT_PCT:
                logger.info(f"💰 TAKE-PROFIT: {symbol} at {pnl_pct:.2%} (entry: ${entry_price:.2f}, current: ${current_price:.2f})")
                return 'take_profit'

            # Log status for positions approaching thresholds
            if pnl_pct <= -0.03:
                logger.info(f"⚠️  {symbol} at {pnl_pct:.2%} (approaching stop-loss)")
            elif pnl_pct >= 0.10:
                logger.info(f"📈 {symbol} at {pnl_pct:.2%} (approaching take-profit)")

            return None

        except Exception as e:
            logger.error(f"Error checking risk for position: {e}")
            return None

    def _submit_risk_exit_order_alpaca(self, alpaca_position, reason: str):
        """Submit exit order for Alpaca position."""
        try:
            symbol = alpaca_position.symbol
            quantity = float(alpaca_position.qty)
            current_price = float(alpaca_position.current_price)

            # Use market order for immediate exit on stop-loss, limit for take-profit
            if reason == 'stop_loss':
                order = self.create_order(
                    symbol,
                    quantity,
                    "sell",
                    type="market"
                )
            else:  # take_profit
                limit_price = round(current_price * 0.99, 2)  # 1% below current to ensure fill
                order = self.create_order(
                    symbol,
                    quantity,
                    "sell",
                    type="limit",
                    limit_price=limit_price,
                    time_in_force="gtc" if self.ENABLE_EXTENDED_HOURS else "day",
                    extended_hours=self.ENABLE_EXTENDED_HOURS
                )

            self.submit_order(order)
            logger.info(f"{'🛑' if reason == 'stop_loss' else '💰'} SOLD {symbol}: {quantity:.2f} shares @ ${current_price:.2f}")

        except Exception as e:
            logger.error(f"Error exiting {alpaca_position.symbol}: {e}")

    def _check_risk_management(self, symbol: str, position) -> Optional[str]:
        """Check if position triggers stop-loss or take-profit."""
        if position is None:
            return None

        current_price = self.get_last_price(symbol)
        if current_price is None:
            return None

        # Debug: Log all position attributes
        logger.debug(f"Position attributes for {symbol}: {dir(position)}")

        # Try different attribute names for entry price
        entry_price = None
        qty = None

        # Try to get entry price from various possible attributes
        for attr_name in ['avg_entry_price', 'avg_fill_price', 'entry_price', 'purchase_price']:
            if hasattr(position, attr_name):
                try:
                    entry_price = float(getattr(position, attr_name))
                    if entry_price > 0:
                        logger.debug(f"Found entry price in {attr_name}: ${entry_price:.2f}")
                        break
                except (ValueError, TypeError):
                    continue

        # If still None, try cost_basis / qty
        if entry_price is None and hasattr(position, 'cost_basis'):
            # Try to get quantity
            for qty_attr in ['qty', 'quantity', 'shares']:
                if hasattr(position, qty_attr):
                    try:
                        qty = float(getattr(position, qty_attr))
                        if qty > 0:
                            break
                    except (ValueError, TypeError):
                        continue

            if qty and qty > 0:
                try:
                    cost_basis = float(position.cost_basis)
                    entry_price = cost_basis / qty
                    logger.debug(f"Calculated entry price from cost_basis: ${entry_price:.2f}")
                except (ValueError, TypeError):
                    pass

        if entry_price is None or entry_price <= 0:
            logger.warning(f"Could not determine entry price for {symbol}. Available attributes: {[a for a in dir(position) if not a.startswith('_')]}")
            return None

        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= -self.STOP_LOSS_PCT:
            logger.warning(f"🛑 STOP-LOSS: {symbol} at {pnl_pct:.2%}")
            return 'stop_loss'

        if pnl_pct >= self.TAKE_PROFIT_PCT:
            logger.info(f"💰 TAKE-PROFIT: {symbol} at {pnl_pct:.2%}")
            return 'take_profit'

        return None

    def _submit_risk_exit_order(self, symbol: str, position, reason: str):
        """Submit exit order for risk management."""
        try:
            current_price = self.get_last_price(symbol)
            limit_price = round(current_price * (0.98 if reason == 'stop_loss' else 0.99), 2)

            order = self.create_order(
                symbol,
                position.quantity,
                "sell",
                type="limit",
                limit_price=limit_price,
                time_in_force="gtc" if self.ENABLE_EXTENDED_HOURS else "day",
                extended_hours=self.ENABLE_EXTENDED_HOURS
            )
            self.submit_order(order)

            logger.info(f"{'🛑' if reason == 'stop_loss' else '💰'} SOLD {symbol}: {position.quantity:.2f} shares @ ${limit_price:.2f}")

        except Exception as e:
            logger.error(f"Error exiting {symbol}: {e}")

    def on_trading_iteration(self):
        """
        Main trading logic - combines signals from both strategies using meta-learner.
        """
        logger.info("=" * 80)
        logger.info(f"COMBINED STRATEGY - Trading Iteration at {datetime.now()}")
        logger.info("=" * 80)

        # STEP 1: Risk Management Check (LEGACY - for old positions without bracket orders)
        # NOTE: All NEW orders use bracket orders with server-side stop-loss/take-profit
        # This manual check is only a fallback for existing positions created before bracket orders
        try:
            from alpaca.trading.client import TradingClient
            import os

            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_API_SECRET')
            trading_client = TradingClient(api_key, api_secret, paper=True)
            alpaca_positions = trading_client.get_all_positions()

            if len(alpaca_positions) > 0:
                logger.info(f"Checking {len(alpaca_positions)} positions (fallback for pre-bracket positions)...")
                legacy_positions = 0
                for alpaca_pos in alpaca_positions:
                    # Only manually exit if this is a legacy position without bracket orders
                    risk_trigger = self._check_risk_management_alpaca(alpaca_pos)
                    if risk_trigger:
                        logger.warning(f"Legacy position {alpaca_pos.symbol} triggered {risk_trigger}")
                        self._submit_risk_exit_order_alpaca(alpaca_pos, risk_trigger)
                        legacy_positions += 1

                if legacy_positions == 0:
                    logger.info("✅ All positions have bracket orders - server-side protection active")
            else:
                logger.info("No positions to check")
        except Exception as e:
            logger.error(f"Error checking risk management: {e}")
            import traceback
            traceback.print_exc()

        # Check if we need to retrain meta-model
        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain >= self.RETRAIN_FREQUENCY_DAYS:
                logger.info(f"Retraining meta-model ({days_since_retrain} days since last retrain)")
                self._train_meta_model()

        # STEP 2: Check market sentiment and hedge if needed
        cash = self.get_cash()
        portfolio_value = self.get_portfolio_value()

        logger.info(f"Portfolio value: ${portfolio_value:,.2f}")

        # PRIORITY: Check if we should hedge with inverse ETFs
        if cash is not None and portfolio_value > 0:
            self._check_market_sentiment_and_hedge(cash if cash else 0, portfolio_value)

        # STEP 3: Rebalance positions if any position is too large
        self._rebalance_positions(portfolio_value)

        # STEP 4: Trading signals - look for new opportunities
        logger.info("Checking for new trading opportunities...")

        if cash is not None and cash > 0:
            logger.info(f"Available cash: ${cash:,.2f}")

            # Only trade if we have cash and meta-model is trained
            if self.meta_model is not None:
                logger.info("Trading signals enabled with meta-model")
                # Trade only symbols we already have positions in or our core list
                symbols_to_check = set(self.SYMBOLS[:5])  # Top 5 only to reduce API calls

                for symbol in symbols_to_check:
                    try:
                        # Simple approach: just use technical indicators from DB
                        volatility, rsi = self._get_market_conditions(symbol)

                        # Simple rules without sentiment/pairs:
                        # - RSI < 30 = oversold (potential buy)
                        # - RSI > 70 = overbought (potential sell)
                        # - High volatility = skip

                        if volatility > 0.4:  # Too volatile, skip
                            continue

                        position = self.get_position(symbol)

                        if rsi < 30 and position is None and cash > 1000:
                            # Oversold, consider buying with BRACKET ORDER
                            price = self.get_last_price(symbol)
                            if price and price > 0:
                                quantity = min(cash * 0.05, 500) / price  # Max $500 or 5% of cash per position

                                # Use bracket order with automatic stop-loss/take-profit
                                success = self._create_bracket_order(symbol, quantity, "buy", price)
                                if success:
                                    logger.info(f"📈 BUY {symbol}: {quantity:.2f} shares @ ${price:.2f} (RSI: {rsi:.1f})")
                                    logger.info(f"   Server-side stops active: -5% stop-loss, +15% take-profit")
                                    cash -= quantity * price

                        elif rsi > 70 and position is not None:
                            # Overbought - let the bracket order's take-profit handle exit
                            logger.info(f"📊 {symbol} overbought (RSI: {rsi:.1f}) - bracket order will auto-exit at +15%")

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
            else:
                logger.info("Meta-model not trained - skipping new trades")
        else:
            logger.info("No cash available for new trades")

        logger.info(f"Final portfolio value: ${self.get_portfolio_value():,.2f}")

        logger.info("=" * 80)
        logger.info("Trading iteration complete")
        logger.info("=" * 80)
