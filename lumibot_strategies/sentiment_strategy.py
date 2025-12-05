"""
Sentiment Trading Strategy using FinBERT and Lumibot

This strategy:
1. Wakes up once per day
2. Fetches news from the last 3 days
3. Runs FinBERT sentiment analysis
4. Makes trading decisions based on sentiment scores
5. Sleeps for 24 hours and repeats

Based on the proven YouTube tutorial strategy that achieved ~234% returns.
"""

import logging
from datetime import datetime, timedelta
from lumibot.strategies import Strategy
from lumibot.entities import TradingFee
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple
import requests
from collections import Counter

logger = logging.getLogger(__name__)


class SentimentStrategy(Strategy):
    """
    FinBERT-based sentiment trading strategy for Lumibot.

    This strategy analyzes news sentiment using the FinBERT model to make
    trading decisions. It focuses on a small universe of liquid stocks and
    makes one decision per day based on aggregate sentiment from recent news.
    """

    # Class-level constants
    CASH_AT_RISK = 0.5  # Use 50% of available cash per position
    SLEEPTIME = "24H"   # Check once per day
    NEWS_LOOKBACK_DAYS = 3  # Analyze news from last 3 days

    # Stock universe (liquid tech stocks with good news coverage)
    SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "TSLA",
        "AMD", "NFLX", "CRM", "ADBE", "INTC", "PYPL", "SQ"
    ]

    def initialize(self, parameters: Dict = None):
        """
        Initialize the strategy. Called once when strategy starts.

        Args:
            parameters: Optional dict of parameters (not used currently)
        """
        self.sleeptime = self.SLEEPTIME

        # Initialize FinBERT model for sentiment analysis
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.model.eval()  # Set to evaluation mode

        # Track positions
        self.last_trade_date = {}

        logger.info("Sentiment Strategy initialized")
        logger.info(f"Trading universe: {len(self.SYMBOLS)} symbols")
        logger.info(f"Sleep time: {self.SLEEPTIME}")

    def position_sizing(self) -> float:
        """
        Calculate position size based on available cash.

        Returns:
            Dollar amount to invest per position
        """
        cash = self.get_cash()
        last_price_multiplier = 1  # Could fetch last price if needed
        quantity = round(cash * self.CASH_AT_RISK / last_price_multiplier, 0)
        return quantity

    def get_news_sentiment(self, symbol: str) -> Tuple[float, int]:
        """
        Fetch news and calculate aggregate sentiment score using FinBERT.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (probability, sentiment) where:
            - probability: Float between 0-1 indicating confidence
            - sentiment: Int (1=positive, 0=neutral, -1=negative)
        """
        # Get dates for news lookup
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.NEWS_LOOKBACK_DAYS)

        # Fetch news from Alpaca (built into Lumibot)
        try:
            news = self.api.get_news(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
        except Exception as e:
            logger.warning(f"Could not fetch news for {symbol}: {e}")
            return 0.0, 0  # Neutral if no news

        if not news or len(news) == 0:
            logger.info(f"No news found for {symbol}")
            return 0.0, 0  # Neutral if no news

        logger.info(f"Analyzing {len(news)} news articles for {symbol}")

        # Analyze each headline with FinBERT
        sentiments = []
        probabilities = []

        for article in news[:20]:  # Limit to 20 most recent articles
            headline = article.headline

            # Tokenize and analyze
            inputs = self.tokenizer(headline, return_tensors="pt", padding=True, truncation=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get prediction
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            positive = predictions[:, 0].tolist()[0]  # positive
            negative = predictions[:, 1].tolist()[0]  # negative
            neutral = predictions[:, 2].tolist()[0]   # neutral

            # Determine sentiment (-1, 0, 1)
            max_prob = max(positive, negative, neutral)

            if max_prob == positive:
                sentiment = 1
                prob = positive
            elif max_prob == negative:
                sentiment = -1
                prob = negative
            else:
                sentiment = 0
                prob = neutral

            sentiments.append(sentiment)
            probabilities.append(prob)

        # Aggregate: majority vote weighted by confidence
        if not sentiments:
            return 0.0, 0

        # Calculate weighted sentiment
        total_weight = sum(probabilities)
        if total_weight == 0:
            return 0.0, 0

        weighted_sentiment = sum(s * p for s, p in zip(sentiments, probabilities)) / total_weight
        avg_probability = sum(probabilities) / len(probabilities)

        # Convert to discrete signal
        if weighted_sentiment > 0.15:  # Threshold for bullish
            final_sentiment = 1
        elif weighted_sentiment < -0.15:  # Threshold for bearish
            final_sentiment = -1
        else:
            final_sentiment = 0

        logger.info(f"{symbol} sentiment: {final_sentiment} (confidence: {avg_probability:.3f})")

        return avg_probability, final_sentiment

    def on_trading_iteration(self):
        """
        Main trading logic. Called every sleeptime interval (24 hours).

        This is where the strategy makes trading decisions:
        1. Check each symbol for news sentiment
        2. Buy if sentiment is positive
        3. Sell if sentiment is negative or position has turned negative
        4. Hold if sentiment is neutral
        """
        logger.info("=" * 80)
        logger.info(f"SENTIMENT STRATEGY - Trading Iteration at {datetime.now()}")
        logger.info("=" * 80)

        cash = self.get_cash()
        logger.info(f"Available cash: ${cash:,.2f}")

        # Get current positions
        positions = self.get_positions()
        position_symbols = [p.symbol for p in positions]

        logger.info(f"Current positions: {len(position_symbols)}")

        # Check each symbol in our universe
        for symbol in self.SYMBOLS:
            try:
                # Get sentiment
                probability, sentiment = self.get_news_sentiment(symbol)

                # Get current position if any
                position = self.get_position(symbol)

                # Decision logic
                if sentiment == 1 and probability > 0.7:
                    # Strong positive sentiment - BUY
                    if position is None:
                        # No position - enter new position
                        last_price = self.get_last_price(symbol)
                        quantity = self.position_sizing() / last_price

                        order = self.create_order(
                            asset=symbol,
                            quantity=quantity,
                            side="buy"
                        )
                        self.submit_order(order)
                        logger.info(f"BUY {symbol}: {quantity:.2f} shares @ ${last_price:.2f} (sentiment: {sentiment}, prob: {probability:.3f})")
                    else:
                        logger.info(f"HOLD {symbol}: Already have position (sentiment: {sentiment})")

                elif sentiment == -1 and position is not None:
                    # Negative sentiment and we have a position - SELL
                    order = self.create_order(
                        asset=symbol,
                        quantity=position.quantity,
                        side="sell"
                    )
                    self.submit_order(order)
                    logger.info(f"SELL {symbol}: {position.quantity:.2f} shares (sentiment: {sentiment}, prob: {probability:.3f})")

                elif position is not None and sentiment != 1:
                    # Have position but sentiment turned neutral/negative - SELL
                    order = self.create_order(
                        asset=symbol,
                        quantity=position.quantity,
                        side="sell"
                    )
                    self.submit_order(order)
                    logger.info(f"SELL {symbol}: {position.quantity:.2f} shares (sentiment neutral/negative)")

                else:
                    # No action needed
                    if position is None:
                        logger.info(f"SKIP {symbol}: Sentiment not strong enough (sentiment: {sentiment}, prob: {probability:.3f})")
                    else:
                        logger.info(f"HOLD {symbol}: Maintaining position (sentiment: {sentiment})")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        logger.info("=" * 80)
        logger.info(f"Trading iteration complete. Next check in {self.SLEEPTIME}")
        logger.info("=" * 80)
