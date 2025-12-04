"""
Enhanced Sentiment Analysis Engine for Trading
==============================================

This module implements the key improvements from the high-performing trading bot:
1. FinBERT for financial-specific sentiment analysis
2. Aggregated sentiment across multiple headlines (sum logits → softmax)
3. High-confidence filtering (99%+ threshold)
4. Multiple news source support
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


class SentimentLabel(Enum):
    """
    Enum class for sentiment labels.
    
    Why use Enum instead of strings?
    --------------------------------
    - Type safety: Python will catch typos like "postive" at development time
    - IDE support: Autocomplete shows you all valid options
    - Cleaner code: SentimentLabel.POSITIVE vs "positive"
    """
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SentimentResult:
    """
    Data class to hold sentiment analysis results.
    
    What is @dataclass?
    -------------------
    It's a decorator that automatically generates:
    - __init__() method (so you don't write self.x = x for each attribute)
    - __repr__() method (nice string representation for debugging)
    - __eq__() method (compare two SentimentResult objects)
    
    Attributes:
    -----------
    sentiment : SentimentLabel
        The predicted sentiment (POSITIVE, NEGATIVE, or NEUTRAL)
    confidence : float
        How confident the model is (0.0 to 1.0)
    probabilities : dict
        The probability for each label {"positive": 0.8, "negative": 0.1, "neutral": 0.1}
    num_headlines : int
        How many headlines were analyzed
    should_trade : bool
        Whether confidence meets the threshold for trading
    """
    sentiment: SentimentLabel
    confidence: float
    probabilities: dict
    num_headlines: int
    should_trade: bool


class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT.
    
    This class wraps the ProsusAI/finbert model and implements the key
    improvements that make the trading bot successful:
    
    1. Uses FinBERT (trained on financial text) instead of generic sentiment
    2. Aggregates multiple headlines by summing logits before softmax
    3. Filters trades based on a high confidence threshold
    
    Example Usage:
    --------------
    >>> analyzer = FinBERTSentimentAnalyzer(confidence_threshold=0.95)
    >>> headlines = ["Tesla stock surges on strong earnings", "Investors bullish on EV market"]
    >>> result = analyzer.analyze(headlines)
    >>> print(result.sentiment, result.confidence)
    SentimentLabel.POSITIVE 0.987
    
    Attributes:
    -----------
    device : str
        'cuda:0' if GPU available, else 'cpu'
    tokenizer : AutoTokenizer
        Converts text to numbers (tokens) that the model understands
    model : AutoModelForSequenceClassification
        The actual FinBERT neural network
    labels : list
        Maps model output indices to sentiment labels [positive, negative, neutral]
    confidence_threshold : float
        Minimum confidence required to recommend a trade (default 0.95)
    """
    
    # Class variable: maps index to label (same order as FinBERT outputs)
    LABELS = ["positive", "negative", "neutral"]
    
    def __init__(self, confidence_threshold: float = 0.95, use_gpu: bool = True):
        """
        Initialize the FinBERT sentiment analyzer.
        
        Parameters:
        -----------
        confidence_threshold : float, default=0.95
            Minimum confidence to trigger a trade signal.
            The original bot uses 0.999, but 0.95 is a good starting point.
            
        use_gpu : bool, default=True
            Whether to use GPU if available. Set False for testing on CPU.
            
        What happens during initialization:
        -----------------------------------
        1. Check if GPU is available (CUDA)
        2. Download FinBERT model from HuggingFace (first time only)
        3. Load model into memory (GPU or CPU)
        4. Store the confidence threshold
        """
        # Step 1: Determine device (GPU or CPU)
        # torch.cuda.is_available() returns True if you have an NVIDIA GPU with CUDA
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda:0"  # Use first GPU
            print("✓ Using GPU for inference (faster)")
        else:
            self.device = "cpu"
            print("✓ Using CPU for inference (slower but works everywhere)")
        
        # Step 2: Load the tokenizer
        # The tokenizer converts text → numbers (tokens)
        # Example: "Tesla surges" → [7592, 15216, 102]
        print("Loading FinBERT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        # Step 3: Load the model
        # .to(device) moves the model to GPU/CPU
        print("Loading FinBERT model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        ).to(self.device)
        
        # Step 4: Set model to evaluation mode
        # This disables dropout and batch normalization training behavior
        # IMPORTANT: Always do this when making predictions!
        self.model.eval()
        
        # Step 5: Store the confidence threshold
        self.confidence_threshold = confidence_threshold
        
        print(f"✓ FinBERT loaded successfully (threshold={confidence_threshold})")
    
    def analyze(self, headlines: List[str]) -> SentimentResult:
        """
        Analyze sentiment from a list of news headlines.
        
        This is the CORE method that implements the bot's secret sauce:
        1. Tokenize all headlines at once (batch processing)
        2. Get model predictions (logits) for each headline
        3. SUM the logits across headlines (aggregation)
        4. Apply softmax to get probabilities
        5. Return result with trade recommendation
        
        Parameters:
        -----------
        headlines : List[str]
            A list of news headline strings.
            Example: ["Tesla stock rises 5%", "Musk announces new factory"]
            
        Returns:
        --------
        SentimentResult
            Contains sentiment, confidence, probabilities, and trade signal.
            
        Why sum logits instead of averaging probabilities?
        --------------------------------------------------
        Logits are the raw outputs BEFORE softmax. They can be negative or positive.
        
        Example with 3 headlines:
        - Headline 1 logits: [2.0, -1.0, 0.5]  (leaning positive)
        - Headline 2 logits: [1.5, -0.5, 0.3]  (leaning positive)
        - Headline 3 logits: [0.8, -0.2, 0.1]  (slightly positive)
        
        Summed: [4.3, -1.7, 0.9]
        After softmax: [0.96, 0.003, 0.03] → Very confident positive!
        
        The sum AMPLIFIES consistent signals. If all headlines agree, the
        confidence goes UP. If they disagree, it stays NEUTRAL.
        """
        # Handle edge case: no headlines
        if not headlines or len(headlines) == 0:
            return SentimentResult(
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.0,
                probabilities={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                num_headlines=0,
                should_trade=False
            )
        
        # Filter out empty strings and None values
        headlines = [h for h in headlines if h and isinstance(h, str) and len(h.strip()) > 0]
        
        if len(headlines) == 0:
            return SentimentResult(
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.0,
                probabilities={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                num_headlines=0,
                should_trade=False
            )
        
        # Step 1: Tokenize all headlines at once
        # return_tensors="pt" → Return PyTorch tensors
        # padding=True → Make all sequences the same length (required for batching)
        # truncation=True → Cut off headlines that are too long (FinBERT max=512 tokens)
        # .to(self.device) → Move data to same device as model (GPU/CPU)
        tokens = self.tokenizer(
            headlines,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Step 2: Get model predictions
        # torch.no_grad() tells PyTorch we're just predicting, not training
        # This saves memory and speeds up computation
        with torch.no_grad():
            # model() returns a SequenceClassifierOutput object
            # .logits gives us the raw prediction scores (before softmax)
            # Shape: [num_headlines, 3] where 3 = positive/negative/neutral
            outputs = self.model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )
            logits = outputs.logits
        
        # Step 3: Aggregate by summing logits
        # torch.sum(logits, dim=0) sums across headlines (dimension 0)
        # Result shape: [3] - one summed score per sentiment class
        aggregated_logits = torch.sum(logits, dim=0)
        
        # Step 4: Convert to probabilities with softmax
        # Softmax: e^x / sum(e^x) - converts scores to probabilities that sum to 1
        # dim=-1 means apply softmax to the last dimension
        probabilities = torch.nn.functional.softmax(aggregated_logits, dim=-1)
        
        # Step 5: Extract results
        # torch.argmax() finds the index of the highest probability
        # .item() converts a single-element tensor to a Python number
        confidence = probabilities.max().item()
        sentiment_idx = probabilities.argmax().item()
        sentiment_label = SentimentLabel(self.LABELS[sentiment_idx])
        
        # Create probability dictionary for transparency
        prob_dict = {
            "positive": probabilities[0].item(),
            "negative": probabilities[1].item(),
            "neutral": probabilities[2].item()
        }
        
        # Step 6: Determine if we should trade
        # Only trade if confidence exceeds threshold AND sentiment is not neutral
        should_trade = (
            confidence >= self.confidence_threshold and 
            sentiment_label != SentimentLabel.NEUTRAL
        )
        
        return SentimentResult(
            sentiment=sentiment_label,
            confidence=confidence,
            probabilities=prob_dict,
            num_headlines=len(headlines),
            should_trade=should_trade
        )
    
    def analyze_single(self, headline: str) -> SentimentResult:
        """
        Analyze a single headline (convenience method).
        
        Parameters:
        -----------
        headline : str
            A single news headline.
            
        Returns:
        --------
        SentimentResult
            The sentiment analysis result.
        """
        return self.analyze([headline])
    
    def get_trade_signal(self, headlines: List[str]) -> Tuple[str, float]:
        """
        Get a simple trade signal (for compatibility with existing code).
        
        This method mimics the original bot's interface:
        - Returns (probability, sentiment) tuple
        - Use this if you want to integrate with existing code
        
        Parameters:
        -----------
        headlines : List[str]
            List of news headlines.
            
        Returns:
        --------
        Tuple[str, float]
            (confidence_score, sentiment_string)
            Example: (0.95, "positive")
        """
        result = self.analyze(headlines)
        return result.confidence, result.sentiment.value


# Convenience function for quick testing (matches original bot's interface)
def estimate_sentiment(news: List[str]) -> Tuple[float, str]:
    """
    Quick sentiment estimation (compatible with original bot).
    
    This function creates a temporary analyzer and returns results
    in the same format as the original finbert_utils.py.
    
    Note: For production, create ONE FinBERTSentimentAnalyzer instance
    and reuse it (faster, uses less memory).
    
    Parameters:
    -----------
    news : List[str]
        List of news headlines.
        
    Returns:
    --------
    Tuple[float, str]
        (probability, sentiment) matching original interface.
    """
    # Use a module-level singleton to avoid reloading model
    global _default_analyzer
    if '_default_analyzer' not in globals():
        _default_analyzer = FinBERTSentimentAnalyzer(confidence_threshold=0.999)
    
    result = _default_analyzer.analyze(news)
    return result.confidence, result.sentiment.value


if __name__ == "__main__":
    # Test the analyzer
    print("\n" + "="*60)
    print("TESTING FINBERT SENTIMENT ANALYZER")
    print("="*60 + "\n")
    
    # Create analyzer with 95% threshold (easier to see results in testing)
    analyzer = FinBERTSentimentAnalyzer(confidence_threshold=0.95)
    
    # Test cases
    test_cases = [
        {
            "name": "Strong Positive",
            "headlines": [
                "Tesla stock surges 10% on record earnings",
                "Investors bullish as EV demand exceeds expectations",
                "Analysts upgrade Tesla to strong buy"
            ]
        },
        {
            "name": "Strong Negative",
            "headlines": [
                "Markets crash amid recession fears",
                "Investors flee stocks as inflation soars",
                "Economic outlook turns grim"
            ]
        },
        {
            "name": "Mixed Sentiment",
            "headlines": [
                "Tesla beats earnings expectations",
                "But concerns remain about competition",
                "Analysts divided on future outlook"
            ]
        },
        {
            "name": "Neutral News",
            "headlines": [
                "Company announces quarterly meeting date",
                "Board to review annual report"
            ]
        }
    ]
    
    for test in test_cases:
        print(f"\n📰 Test: {test['name']}")
        print(f"   Headlines: {test['headlines'][:2]}...")
        
        result = analyzer.analyze(test['headlines'])
        
        print(f"   Sentiment: {result.sentiment.value}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Should Trade: {'✅ YES' if result.should_trade else '❌ NO'}")
        print(f"   Probabilities: pos={result.probabilities['positive']:.2%}, "
              f"neg={result.probabilities['negative']:.2%}, "
              f"neu={result.probabilities['neutral']:.2%}")
