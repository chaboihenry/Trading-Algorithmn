"""
Enhanced Sentiment Trading Bot - Main Integration
================================================

This module combines all the improvements into a complete trading system
that you can integrate with your existing stacked ensemble model.

Components:
1. FinBERT Sentiment Analysis (sentiment_engine.py)
2. Multi-Source News Fetching (news_fetcher.py)
3. Risk Management (risk_manager.py)
4. Trading Execution (this file)
"""

import os
from datetime import datetime
from typing import Optional, Tuple
from dataclasses import dataclass

# Import our custom modules
from sentiment_engine import FinBERTSentimentAnalyzer, SentimentResult, SentimentLabel
from news_fetcher import (
    MultiSourceNewsFetcher,
    AlpacaNewsProvider,
    FinnhubNewsProvider,
    AlphaVantageNewsProvider
)
from risk_manager import (
    RiskManager,
    BracketOrder,
    TradeDirection,
    PositionTracker
)


@dataclass
class TradingConfig:
    """
    Configuration for the trading bot.
    
    Centralize all your settings here for easy tuning.
    
    Attributes:
    -----------
    symbol : str
        Stock to trade (default "SPY")
    confidence_threshold : float
        Minimum confidence for trades (0.95 = 95%)
    take_profit_pct : float
        Take profit percentage (0.20 = 20%)
    stop_loss_pct : float
        Stop loss percentage (0.05 = 5%)
    cash_at_risk : float
        Fraction of cash to use per trade (0.50 = 50%)
    news_days_back : int
        How many days of news to analyze (3 is recommended)
    """
    symbol: str = "SPY"
    confidence_threshold: float = 0.95
    take_profit_pct: float = 0.20
    stop_loss_pct: float = 0.05
    cash_at_risk: float = 0.50
    news_days_back: int = 3
    max_positions: int = 5


class EnhancedSentimentTrader:
    """
    Complete trading bot with enhanced sentiment analysis.
    
    This class orchestrates all components:
    1. Fetches news from multiple sources
    2. Analyzes sentiment using FinBERT
    3. Makes trading decisions based on high-confidence signals
    4. Manages risk with asymmetric bracket orders
    
    How to Integrate with Your Ensemble Model:
    ------------------------------------------
    You can use this as an additional signal in your ensemble:
    
    >>> trader = EnhancedSentimentTrader()
    >>> signal = trader.get_trading_signal("AAPL", current_price=150.0)
    >>> 
    >>> # Add to your ensemble
    >>> sentiment_score = signal.sentiment_score  # -1 to +1
    >>> your_ensemble.add_feature("finbert_sentiment", sentiment_score)
    
    Example Usage:
    --------------
    >>> config = TradingConfig(symbol="SPY", confidence_threshold=0.95)
    >>> trader = EnhancedSentimentTrader(config)
    >>> 
    >>> # Get trading signal
    >>> signal = trader.get_trading_signal("SPY", current_price=450.0)
    >>> print(signal.action, signal.confidence)
    """
    
    def __init__(self, config: Optional[TradingConfig] = None):
        """
        Initialize the trading bot.
        
        Parameters:
        -----------
        config : TradingConfig, optional
            Configuration settings. Uses defaults if not provided.
        """
        self.config = config or TradingConfig()
        
        print("="*60)
        print("INITIALIZING ENHANCED SENTIMENT TRADER")
        print("="*60)
        
        # Initialize sentiment analyzer
        print("\n1. Loading FinBERT sentiment analyzer...")
        self.sentiment_analyzer = FinBERTSentimentAnalyzer(
            confidence_threshold=self.config.confidence_threshold
        )
        
        # Initialize news fetcher
        print("\n2. Setting up news fetchers...")
        self._setup_news_fetcher()
        
        # Initialize risk manager
        print("\n3. Configuring risk management...")
        self.risk_manager = RiskManager(
            take_profit_pct=self.config.take_profit_pct,
            stop_loss_pct=self.config.stop_loss_pct,
            cash_at_risk=self.config.cash_at_risk
        )
        
        # Initialize position tracker
        self.position_tracker = PositionTracker(
            max_positions=self.config.max_positions
        )
        
        # Track state
        self.last_trade = None  # "buy" or "sell"
        
        print("\n" + "="*60)
        print("✓ TRADER INITIALIZED SUCCESSFULLY")
        print("="*60 + "\n")
    
    def _setup_news_fetcher(self):
        """Set up news fetcher with available providers."""
        self.news_fetcher = MultiSourceNewsFetcher()
        
        # Try to add each provider
        self.news_fetcher.add_provider(AlpacaNewsProvider(paper=True))
        self.news_fetcher.add_provider(FinnhubNewsProvider())
        self.news_fetcher.add_provider(AlphaVantageNewsProvider())
        
        if len(self.news_fetcher.providers) == 0:
            print("\n⚠️  WARNING: No news providers configured!")
            print("   The bot will not be able to fetch real news.")
            print("   See API_SETUP_GUIDE.md for configuration help.")
    
    def get_sentiment(self, symbol: Optional[str] = None) -> SentimentResult:
        """
        Get sentiment analysis for a symbol.
        
        Parameters:
        -----------
        symbol : str, optional
            Stock ticker. Uses config default if not provided.
            
        Returns:
        --------
        SentimentResult
            Complete sentiment analysis result.
        """
        symbol = symbol or self.config.symbol
        
        # Fetch headlines
        print(f"\n📰 Fetching news for {symbol}...")
        headlines = self.news_fetcher.get_headlines(
            symbol,
            days_back=self.config.news_days_back
        )
        
        if not headlines:
            print("   ⚠️ No headlines found - returning neutral sentiment")
            return SentimentResult(
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.0,
                probabilities={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                num_headlines=0,
                should_trade=False
            )
        
        print(f"   Found {len(headlines)} headlines")
        
        # Analyze sentiment
        print(f"\n🧠 Analyzing sentiment...")
        result = self.sentiment_analyzer.analyze(headlines)
        
        print(f"   Sentiment: {result.sentiment.value}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Should Trade: {result.should_trade}")
        
        return result
    
    def get_trading_signal(
        self,
        symbol: Optional[str] = None,
        current_price: Optional[float] = None
    ) -> "TradingSignal":
        """
        Generate a trading signal based on sentiment.
        
        This is the main method for getting actionable trading signals.
        
        Parameters:
        -----------
        symbol : str, optional
            Stock ticker. Uses config default if not provided.
        current_price : float, optional
            Current stock price. Required for order creation.
            
        Returns:
        --------
        TradingSignal
            Complete signal with action, confidence, and optional order.
        """
        symbol = symbol or self.config.symbol
        
        # Get sentiment analysis
        sentiment_result = self.get_sentiment(symbol)
        
        # Determine action
        action = "hold"
        direction = None
        
        if sentiment_result.should_trade:
            if sentiment_result.sentiment == SentimentLabel.POSITIVE:
                action = "buy"
                direction = TradeDirection.LONG
            elif sentiment_result.sentiment == SentimentLabel.NEGATIVE:
                action = "sell"
                direction = TradeDirection.SHORT
        
        # Convert sentiment to numeric score for ensemble integration
        # Positive sentiment → +1, Negative → -1, weighted by confidence
        if sentiment_result.sentiment == SentimentLabel.POSITIVE:
            sentiment_score = sentiment_result.confidence
        elif sentiment_result.sentiment == SentimentLabel.NEGATIVE:
            sentiment_score = -sentiment_result.confidence
        else:
            sentiment_score = 0.0
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            direction=direction,
            sentiment=sentiment_result.sentiment,
            confidence=sentiment_result.confidence,
            sentiment_score=sentiment_score,
            probabilities=sentiment_result.probabilities,
            num_headlines=sentiment_result.num_headlines,
            should_trade=sentiment_result.should_trade,
            timestamp=datetime.now()
        )
    
    def create_order(
        self,
        signal: "TradingSignal",
        current_price: float,
        available_cash: float,
        portfolio_value: Optional[float] = None
    ) -> Optional[BracketOrder]:
        """
        Create a bracket order based on a trading signal.
        
        Parameters:
        -----------
        signal : TradingSignal
            The trading signal from get_trading_signal()
        current_price : float
            Current stock price
        available_cash : float
            Cash available for trading
        portfolio_value : float, optional
            Total portfolio value
            
        Returns:
        --------
        Optional[BracketOrder]
            A bracket order if signal suggests trading, None otherwise.
        """
        if not signal.should_trade or signal.direction is None:
            print("⚠️ Signal does not recommend trading")
            return None
        
        # Check if we can open a position
        if not self.position_tracker.can_open_position(signal.symbol):
            print(f"⚠️ Cannot open position for {signal.symbol}")
            print("   (Already have position or at max positions)")
            return None
        
        # Handle existing opposite position
        if signal.symbol in self.position_tracker.positions:
            existing = self.position_tracker.positions[signal.symbol]
            if existing["direction"] != signal.direction:
                print(f"⚠️ Closing existing {existing['direction'].value} position")
                self.position_tracker.close_position(signal.symbol, current_price)
        
        # Create the bracket order
        order = self.risk_manager.create_bracket_order(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=current_price,
            available_cash=available_cash,
            portfolio_value=portfolio_value
        )
        
        if order:
            # Track the new position
            self.position_tracker.open_position(
                signal.symbol,
                signal.direction,
                order.quantity,
                current_price
            )
            self.last_trade = signal.action
        
        return order
    
    def get_ensemble_features(self, symbol: Optional[str] = None) -> dict:
        """
        Get sentiment features for your stacked ensemble model.
        
        This method returns features that you can directly add to your
        existing feature matrix for the ensemble model.
        
        Parameters:
        -----------
        symbol : str, optional
            Stock ticker. Uses config default if not provided.
            
        Returns:
        --------
        dict
            Dictionary of features suitable for ML models:
            - finbert_sentiment_score: -1 (bearish) to +1 (bullish)
            - finbert_confidence: 0.0 to 1.0
            - finbert_positive_prob: Probability of positive sentiment
            - finbert_negative_prob: Probability of negative sentiment
            - finbert_neutral_prob: Probability of neutral sentiment
            - finbert_news_count: Number of headlines analyzed
            - finbert_signal: 1 (buy), -1 (sell), 0 (hold)
        """
        signal = self.get_trading_signal(symbol)
        
        # Convert action to numeric signal
        if signal.action == "buy":
            numeric_signal = 1
        elif signal.action == "sell":
            numeric_signal = -1
        else:
            numeric_signal = 0
        
        return {
            "finbert_sentiment_score": signal.sentiment_score,
            "finbert_confidence": signal.confidence,
            "finbert_positive_prob": signal.probabilities["positive"],
            "finbert_negative_prob": signal.probabilities["negative"],
            "finbert_neutral_prob": signal.probabilities["neutral"],
            "finbert_news_count": signal.num_headlines,
            "finbert_signal": numeric_signal
        }


@dataclass
class TradingSignal:
    """
    Complete trading signal with all relevant information.
    
    Attributes:
    -----------
    symbol : str
        Stock ticker
    action : str
        "buy", "sell", or "hold"
    direction : Optional[TradeDirection]
        LONG, SHORT, or None (for hold)
    sentiment : SentimentLabel
        POSITIVE, NEGATIVE, or NEUTRAL
    confidence : float
        Model confidence (0.0 to 1.0)
    sentiment_score : float
        Numeric score for ensemble (-1.0 to +1.0)
    probabilities : dict
        Probability for each sentiment class
    num_headlines : int
        How many headlines were analyzed
    should_trade : bool
        Whether confidence exceeds threshold
    timestamp : datetime
        When signal was generated
    """
    symbol: str
    action: str
    direction: Optional[TradeDirection]
    sentiment: SentimentLabel
    confidence: float
    sentiment_score: float
    probabilities: dict
    num_headlines: int
    should_trade: bool
    timestamp: datetime
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"TradingSignal({self.symbol})\n"
            f"  Action: {self.action.upper()}\n"
            f"  Sentiment: {self.sentiment.value} ({self.confidence:.1%})\n"
            f"  Score for Ensemble: {self.sentiment_score:+.3f}\n"
            f"  Headlines Analyzed: {self.num_headlines}\n"
            f"  Should Trade: {'YES' if self.should_trade else 'NO'}"
        )


def demo_integration():
    """
    Demonstrate how to integrate with your existing ensemble model.
    
    This shows the typical workflow:
    1. Get sentiment features
    2. Add to your existing features
    3. Run through your ensemble
    4. Combine predictions
    """
    print("\n" + "="*70)
    print("DEMO: Integrating FinBERT Sentiment with Your Ensemble Model")
    print("="*70)
    
    # Create trader
    config = TradingConfig(
        symbol="SPY",
        confidence_threshold=0.90  # Lower threshold for demo
    )
    trader = EnhancedSentimentTrader(config)
    
    # Get sentiment features
    print("\n📊 Getting sentiment features for SPY...")
    features = trader.get_ensemble_features("SPY")
    
    print("\nFeatures for your ensemble model:")
    print("-" * 40)
    for name, value in features.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")
    
    # Show how to integrate
    print("\n📝 Integration example (pseudocode):")
    print("-" * 40)
    print("""
    # In your existing ensemble code:
    
    # 1. Get your existing features
    technical_features = get_technical_indicators(symbol)
    fundamental_features = get_fundamentals(symbol)
    
    # 2. Add FinBERT sentiment features
    sentiment_features = trader.get_ensemble_features(symbol)
    
    # 3. Combine all features
    all_features = {
        **technical_features,
        **fundamental_features,
        **sentiment_features
    }
    
    # 4. Run through your stacked ensemble
    prediction = your_ensemble_model.predict(all_features)
    
    # The sentiment features give your model additional signal
    # about market sentiment that technical indicators miss!
    """)
    
    # Get full trading signal
    print("\n📈 Getting full trading signal...")
    signal = trader.get_trading_signal("SPY")
    print("\n" + str(signal))


if __name__ == "__main__":
    demo_integration()
