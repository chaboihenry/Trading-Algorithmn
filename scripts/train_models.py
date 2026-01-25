import argparse
import logging
import sys
import pandas as pd
from pathlib import Path

# Local imports
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.all_symbols import SYMBOLS
from config.logging_config import setup_logging
from strategies.risklabai_bot import RiskLabAIModel
from data.model_storage import ModelStorage

logger = setup_logging(script_name="train_models")

def train_symbol(symbol):
    logger.info(f"============================================================")
    logger.info(f"PROCESSING: {symbol}")
    logger.info(f"============================================================")
    
    # Instantiate the model architecture
    model = RiskLabAIModel(symbol)
    
    # Train the model (Uses Parquet Bars if available)
    result = model.train(symbol)
    
    if result['success']:
        logger.info(f"[{symbol}] Saving Model...")
        
        # Prepare validatable artifact
        model_data = {
            'primary_model': model.primary_model,
            'meta_model': model.meta_model,
            'feature_names': model.feature_names,
            'label_encoder': model.label_encoder,
            'meta_data': {
                'samples': result['n_samples'],
                'features': result['n_features']
            }
        }
        
        # Persist to disk/cloud
        ModelStorage().save_model(symbol, model_data)
        logger.info(f"[{symbol}] Training Complete.")
        return True
    else:
        logger.error(f"[{symbol}] Training Failed: {result.get('reason')}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Symbol to train (e.g. QQQ)")
    parser.add_argument("--all", action="store_true", help="Train ALL symbols in config")
    args = parser.parse_args()
    
    targets = [args.symbol.upper()] if args.symbol else SYMBOLS
    
    for sym in targets:
        try:
            train_symbol(sym)
        except Exception as e:
            logger.error(f"[{sym}] Critical Error: {e}")

if __name__ == "__main__":
    main()