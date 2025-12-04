"""
Risk Management Module for Trading Bot
======================================

This module implements the risk management strategies that contribute
to the trading bot's success:

1. Asymmetric Risk/Reward (4:1 ratio by default)
2. Position sizing based on available capital
3. Bracket orders (take profit + stop loss)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class TradeDirection(Enum):
    """
    Trade direction enum.
    
    LONG = Betting the price will go UP
    SHORT = Betting the price will go DOWN
    """
    LONG = "long"
    SHORT = "short"


@dataclass
class BracketOrder:
    """
    Represents a bracket order with entry, take-profit, and stop-loss.
    
    What is a Bracket Order?
    ------------------------
    A bracket order is actually THREE orders in one:
    1. Entry order: Buy/sell to open position
    2. Take-profit order: Close position when profit target is reached
    3. Stop-loss order: Close position when loss limit is reached
    
    When one of the exit orders fills, the other is automatically canceled.
    
    Attributes:
    -----------
    symbol : str
        Stock ticker (e.g., "SPY")
    quantity : int
        Number of shares to trade
    direction : TradeDirection
        LONG (buy) or SHORT (sell)
    entry_price : float
        Price at which to enter the trade
    take_profit_price : float
        Price at which to take profits
    stop_loss_price : float
        Price at which to stop losses
    risk_reward_ratio : float
        Ratio of potential reward to potential risk
    """
    symbol: str
    quantity: int
    direction: TradeDirection
    entry_price: float
    take_profit_price: float
    stop_loss_price: float
    risk_reward_ratio: float
    
    def potential_profit(self) -> float:
        """Calculate potential profit in dollars."""
        if self.direction == TradeDirection.LONG:
            return (self.take_profit_price - self.entry_price) * self.quantity
        else:  # SHORT
            return (self.entry_price - self.take_profit_price) * self.quantity
    
    def potential_loss(self) -> float:
        """Calculate potential loss in dollars (always positive)."""
        if self.direction == TradeDirection.LONG:
            return (self.entry_price - self.stop_loss_price) * self.quantity
        else:  # SHORT
            return (self.stop_loss_price - self.entry_price) * self.quantity
    
    def __str__(self) -> str:
        """Human-readable representation of the order."""
        return (
            f"BracketOrder({self.direction.value.upper()} {self.quantity} {self.symbol}\n"
            f"  Entry: ${self.entry_price:.2f}\n"
            f"  Take Profit: ${self.take_profit_price:.2f} "
            f"(+${self.potential_profit():.2f})\n"
            f"  Stop Loss: ${self.stop_loss_price:.2f} "
            f"(-${self.potential_loss():.2f})\n"
            f"  Risk/Reward: 1:{self.risk_reward_ratio:.1f})"
        )


class RiskManager:
    """
    Manages risk for the trading bot.
    
    This class calculates:
    - How many shares to buy (position sizing)
    - Where to set take-profit (exit with gains)
    - Where to set stop-loss (exit to limit losses)
    
    The Key Insight: Asymmetric Risk/Reward
    ---------------------------------------
    The original bot uses:
    - Take profit: +20% 
    - Stop loss: -5%
    
    This gives a 4:1 reward-to-risk ratio. Why does this matter?
    
    Even if you're wrong 75% of the time:
    - 75 trades lose 5% each = 375% total loss
    - 25 trades win 20% each = 500% total gain
    - Net: +125% gain!
    
    This is the mathematical edge that makes the strategy profitable
    even with imperfect sentiment predictions.
    
    Attributes:
    -----------
    take_profit_pct : float
        Percentage gain to trigger take-profit (e.g., 0.20 for 20%)
    stop_loss_pct : float
        Percentage loss to trigger stop-loss (e.g., 0.05 for 5%)
    cash_at_risk : float
        Fraction of available cash to use per trade (e.g., 0.5 for 50%)
    max_position_pct : float
        Maximum portfolio percentage for a single position
    """
    
    def __init__(
        self,
        take_profit_pct: float = 0.20,
        stop_loss_pct: float = 0.05,
        cash_at_risk: float = 0.50,
        max_position_pct: float = 0.25
    ):
        """
        Initialize the risk manager.
        
        Parameters:
        -----------
        take_profit_pct : float, default=0.20
            When to take profits (20% gain).
            Higher = more profit potential but fewer filled orders.
            
        stop_loss_pct : float, default=0.05
            When to cut losses (5% loss).
            Lower = smaller losses but more stopped out.
            
        cash_at_risk : float, default=0.50
            How much of available cash to use per trade.
            The original bot uses 0.5 (50%).
            Lower = more conservative.
            
        max_position_pct : float, default=0.25
            Maximum percentage of portfolio in a single position.
            Prevents over-concentration.
        """
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.cash_at_risk = cash_at_risk
        self.max_position_pct = max_position_pct
        
        # Calculate risk/reward ratio
        self.risk_reward_ratio = take_profit_pct / stop_loss_pct
        
        print(f"✓ RiskManager initialized")
        print(f"  Take Profit: +{take_profit_pct:.0%}")
        print(f"  Stop Loss: -{stop_loss_pct:.0%}")
        print(f"  Risk/Reward Ratio: 1:{self.risk_reward_ratio:.1f}")
    
    def calculate_position_size(
        self,
        available_cash: float,
        current_price: float,
        portfolio_value: Optional[float] = None
    ) -> int:
        """
        Calculate how many shares to buy.
        
        Parameters:
        -----------
        available_cash : float
            How much cash is available to trade.
            
        current_price : float
            Current price per share.
            
        portfolio_value : float, optional
            Total portfolio value (for max position check).
            If not provided, uses available_cash.
            
        Returns:
        --------
        int
            Number of shares to buy (whole shares only).
            
        Example:
        --------
        >>> rm = RiskManager(cash_at_risk=0.5)
        >>> rm.calculate_position_size(10000, 400)
        12  # $10,000 × 50% / $400 = 12.5 → 12 shares
        """
        # Calculate basic position based on cash at risk
        cash_to_use = available_cash * self.cash_at_risk
        shares = cash_to_use / current_price
        
        # Apply max position limit if portfolio value provided
        if portfolio_value:
            max_value = portfolio_value * self.max_position_pct
            max_shares = max_value / current_price
            shares = min(shares, max_shares)
        
        # Return whole shares only (can't buy fractional in most cases)
        # Note: Some brokers support fractional shares
        return int(shares)
    
    def calculate_exit_prices(
        self,
        entry_price: float,
        direction: TradeDirection
    ) -> Tuple[float, float]:
        """
        Calculate take-profit and stop-loss prices.
        
        Parameters:
        -----------
        entry_price : float
            The price at which we're entering the trade.
            
        direction : TradeDirection
            LONG (expecting price up) or SHORT (expecting price down).
            
        Returns:
        --------
        Tuple[float, float]
            (take_profit_price, stop_loss_price)
            
        How it works for LONG trades:
        -----------------------------
        - Take Profit: entry_price × (1 + take_profit_pct)
          Example: $400 × 1.20 = $480 (+20%)
          
        - Stop Loss: entry_price × (1 - stop_loss_pct)
          Example: $400 × 0.95 = $380 (-5%)
          
        How it works for SHORT trades:
        ------------------------------
        - Take Profit: entry_price × (1 - take_profit_pct)
          Example: $400 × 0.80 = $320 (-20%, meaning we profit)
          
        - Stop Loss: entry_price × (1 + stop_loss_pct)
          Example: $400 × 1.05 = $420 (+5%, meaning we lose)
        """
        if direction == TradeDirection.LONG:
            take_profit = entry_price * (1 + self.take_profit_pct)
            stop_loss = entry_price * (1 - self.stop_loss_pct)
        else:  # SHORT
            take_profit = entry_price * (1 - self.take_profit_pct)
            stop_loss = entry_price * (1 + self.stop_loss_pct)
        
        # Round to 2 decimal places (standard for stock prices)
        return round(take_profit, 2), round(stop_loss, 2)
    
    def create_bracket_order(
        self,
        symbol: str,
        direction: TradeDirection,
        entry_price: float,
        available_cash: float,
        portfolio_value: Optional[float] = None
    ) -> Optional[BracketOrder]:
        """
        Create a complete bracket order with all parameters.
        
        This is the main method you'll use. It combines:
        - Position sizing
        - Take-profit calculation
        - Stop-loss calculation
        
        Parameters:
        -----------
        symbol : str
            Stock ticker (e.g., "SPY")
            
        direction : TradeDirection
            LONG or SHORT based on sentiment
            
        entry_price : float
            Current price (entry point)
            
        available_cash : float
            Cash available for trading
            
        portfolio_value : float, optional
            Total portfolio value for max position check
            
        Returns:
        --------
        Optional[BracketOrder]
            A bracket order ready to submit, or None if position size is 0.
        """
        # Calculate position size
        quantity = self.calculate_position_size(
            available_cash,
            entry_price,
            portfolio_value
        )
        
        # Can't create order with 0 shares
        if quantity <= 0:
            print(f"⚠️ Position size is 0 - not enough capital")
            return None
        
        # Calculate exit prices
        take_profit, stop_loss = self.calculate_exit_prices(entry_price, direction)
        
        return BracketOrder(
            symbol=symbol,
            quantity=quantity,
            direction=direction,
            entry_price=entry_price,
            take_profit_price=take_profit,
            stop_loss_price=stop_loss,
            risk_reward_ratio=self.risk_reward_ratio
        )
    
    def evaluate_trade(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        direction: TradeDirection
    ) -> dict:
        """
        Evaluate the outcome of a completed trade.
        
        Useful for backtesting and performance analysis.
        
        Parameters:
        -----------
        entry_price : float
            Price at which trade was entered
        exit_price : float
            Price at which trade was exited
        quantity : int
            Number of shares traded
        direction : TradeDirection
            LONG or SHORT
            
        Returns:
        --------
        dict
            Trade evaluation with profit/loss details
        """
        if direction == TradeDirection.LONG:
            pnl = (exit_price - entry_price) * quantity
            pct_change = (exit_price - entry_price) / entry_price
        else:  # SHORT
            pnl = (entry_price - exit_price) * quantity
            pct_change = (entry_price - exit_price) / entry_price
        
        return {
            "profit_loss": round(pnl, 2),
            "percent_change": round(pct_change * 100, 2),
            "is_winner": pnl > 0,
            "hit_take_profit": pct_change >= self.take_profit_pct,
            "hit_stop_loss": pct_change <= -self.stop_loss_pct
        }


class PositionTracker:
    """
    Tracks open positions and prevents over-trading.
    
    This class helps you:
    1. Know if you already have a position in a symbol
    2. Prevent buying when you already own
    3. Track P&L across positions
    
    Attributes:
    -----------
    positions : dict
        Maps symbol → position details
    max_positions : int
        Maximum number of concurrent positions
    """
    
    def __init__(self, max_positions: int = 5):
        """
        Initialize position tracker.
        
        Parameters:
        -----------
        max_positions : int, default=5
            Maximum concurrent positions allowed
        """
        self.positions = {}
        self.max_positions = max_positions
        self.trade_history = []
    
    def can_open_position(self, symbol: str) -> bool:
        """Check if we can open a new position."""
        # Already have this position?
        if symbol in self.positions:
            return False
        
        # At max positions?
        if len(self.positions) >= self.max_positions:
            return False
        
        return True
    
    def open_position(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: int,
        entry_price: float
    ) -> bool:
        """
        Record opening a position.
        
        Returns:
        --------
        bool
            True if position was opened, False if not allowed
        """
        if not self.can_open_position(symbol):
            return False
        
        self.positions[symbol] = {
            "direction": direction,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": None  # Would use datetime.now() in production
        }
        return True
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[dict]:
        """
        Record closing a position.
        
        Returns:
        --------
        Optional[dict]
            Trade result if position existed, None otherwise
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions.pop(symbol)
        
        # Calculate P&L
        if position["direction"] == TradeDirection.LONG:
            pnl = (exit_price - position["entry_price"]) * position["quantity"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["quantity"]
        
        result = {
            "symbol": symbol,
            "direction": position["direction"].value,
            "quantity": position["quantity"],
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "profit_loss": round(pnl, 2)
        }
        
        self.trade_history.append(result)
        return result
    
    def get_summary(self) -> dict:
        """Get summary of trading performance."""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0
            }
        
        winners = sum(1 for t in self.trade_history if t["profit_loss"] > 0)
        total_pnl = sum(t["profit_loss"] for t in self.trade_history)
        
        return {
            "total_trades": len(self.trade_history),
            "winners": winners,
            "losers": len(self.trade_history) - winners,
            "win_rate": winners / len(self.trade_history),
            "total_pnl": round(total_pnl, 2)
        }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING RISK MANAGEMENT MODULE")
    print("="*60 + "\n")
    
    # Create risk manager with bot's parameters
    rm = RiskManager(
        take_profit_pct=0.20,  # +20%
        stop_loss_pct=0.05,    # -5%
        cash_at_risk=0.50      # Use 50% of available cash
    )
    
    # Test scenario: $10,000 account, SPY at $450
    print("\n📊 Scenario: $10,000 account, SPY at $450")
    print("-" * 40)
    
    # Create a LONG bracket order
    order = rm.create_bracket_order(
        symbol="SPY",
        direction=TradeDirection.LONG,
        entry_price=450.00,
        available_cash=10000.00
    )
    
    if order:
        print(order)
        print(f"\nWith {rm.risk_reward_ratio}:1 reward/risk, you only need to")
        print(f"be right {1/(rm.risk_reward_ratio+1):.0%} of the time to break even!")
    
    # Test SHORT order
    print("\n📊 Scenario: Bearish sentiment detected")
    print("-" * 40)
    
    short_order = rm.create_bracket_order(
        symbol="SPY",
        direction=TradeDirection.SHORT,
        entry_price=450.00,
        available_cash=10000.00
    )
    
    if short_order:
        print(short_order)
    
    # Test position tracker
    print("\n📊 Testing Position Tracker")
    print("-" * 40)
    
    tracker = PositionTracker(max_positions=3)
    
    # Open some positions
    tracker.open_position("SPY", TradeDirection.LONG, 10, 450.00)
    tracker.open_position("AAPL", TradeDirection.LONG, 20, 180.00)
    
    print(f"Open positions: {list(tracker.positions.keys())}")
    
    # Close with profit
    result = tracker.close_position("SPY", 540.00)  # +20%
    print(f"Closed SPY: P&L = ${result['profit_loss']}")
    
    # Close with loss
    result = tracker.close_position("AAPL", 171.00)  # -5%
    print(f"Closed AAPL: P&L = ${result['profit_loss']}")
    
    # Get summary
    summary = tracker.get_summary()
    print(f"\nTrading Summary:")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Win rate: {summary['win_rate']:.0%}")
    print(f"  Total P&L: ${summary['total_pnl']}")
