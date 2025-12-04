# Integrated Trading Agent

This is a live trading bot that uses a stacked ensemble of strategies to trade on the Alpaca paper trading platform.

## Strategies

The bot uses the following strategies:

*   **Enhanced Sentiment Trader:** Uses FinBERT to analyze news sentiment.
*   **Pairs Trading:** A statistical arbitrage strategy that finds cointegrated pairs.
*   **Volatility Trading:** A strategy that trades based on volatility regime predictions.

## Architecture

The bot is designed with a clean and simple architecture:

*   `trader.py`: The main entry point for the bot. It contains the main trading loop and connects to the Alpaca API.
*   `strategies/`: This directory contains the implementation of the three trading strategies.
*   `strategies/stacked_ensemble.py`: This script combines the signals from the three base strategies using a meta-learner.

## How to Run

1.  **Set Environment Variables:**
    Set your Alpaca API keys as environment variables:
    ```bash
    export ALPACA_API_KEY="YOUR_API_KEY"
    export ALPACA_API_SECRET="YOUR_API_SECRET"
    ```

2.  **Run the Bot:**
    ```bash
    python trader.py
    ```

    You can also run the bot in dry-run mode to simulate trades without executing them:
    ```bash
    python trader.py --dry-run
    ```
