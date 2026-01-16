# Algorithmic Trading Bot

A trading bot that uses machine learning techniques from Marcos López de Prado's "Advances in Financial Machine Learning". Runs on Alpaca's API for paper/live trading.

## What it does

- Fetches tick data and converts it to imbalance bars (instead of time-based bars like 1min/5min)
- Trains XGBoost models per symbol using triple-barrier labeling
- Uses a meta-model to decide when to actually take trades
- Manages risk with Kelly criterion position sizing, stop-losses, and circuit breakers

## Setup

```bash
# install deps
pip install -r requirements.txt

# set up your alpaca keys in config/.env
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

## Usage

The pipeline goes: backfill ticks → build bars → train models → backtest → live trade

```bash
# 1. backfill historical tick data
python data/backfill_ticks.py

# 2. build imbalance bars from ticks
python scripts/build_bars.py

# 3. train models
python scripts/train_models.py

# 4. run backtest
python scripts/evaluate_backtest.py

# 5. run live (paper trading)
python run_live_trading.py
```

## Project structure

```
├── config/          # settings, symbols list, aws config
├── data/            # tick storage, bar generation, model persistence
├── scripts/         # training, backtesting, bar building
├── strategies/      # main trading bot logic
├── utils/           # ML utilities (triple barrier, CUSUM, purged CV)
└── run_live_trading.py
```

## Key concepts

**Imbalance bars** - bars that form based on order flow imbalance rather than fixed time intervals. Better statistical properties than regular candles.

**Triple-barrier labeling** - labels are based on whether price hits profit target, stop loss, or times out. More realistic than just "price went up/down".

**Meta-labeling** - two-stage approach where a primary model predicts direction, then a meta-model predicts whether the primary model will be correct. Helps with bet sizing.

**Purged K-fold CV** - cross-validation that respects time series structure and prevents data leakage.

## Tech stack

- Python 3.11
- Lumibot (trading framework)
- Alpaca API (broker)
- XGBoost / scikit-learn
- RiskLabAI
- SQLite for tick storage

## Disclaimer

This is for educational purposes. Trading involves risk. Don't use real money without understanding what you're doing.

## Contact

Henry - [@chaboihenry](https://github.com/chaboihenry)
