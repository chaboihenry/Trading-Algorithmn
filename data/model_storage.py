import os
import joblib
import logging
from pathlib import Path
from datetime import datetime

# --- PATH SETUP ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
import sys
if project_root not in sys.path:
    sys.path.append(project_root)

from config.aws_config import AWS_REGION, S3_MODEL_BUCKET, is_aws_configured

logger = logging.getLogger(__name__)

class ModelStorage:
    """
    Simple Model Persistence.
    - Saves locally to 'models/'.
    - Uploads to S3 bucket 'models/<SYMBOL>/'.
    """
    def __init__(self, local_dir="models"):
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        self.s3_client = None
        self.bucket = None
        
        if is_aws_configured():
            try:
                import boto3
                self.s3_client = boto3.client('s3', region_name=AWS_REGION)
                self.bucket = S3_MODEL_BUCKET
                logger.info(f"✅ S3 Storage Active: {self.bucket}")
            except Exception as e:
                logger.warning(f"⚠️ S3 Connection Failed: {e}")

    def save_model(self, symbol, model_data, upload_to_s3=True):
        """
        Saves model locally and uploads to S3 folder.
        Format: models/AAPL/AAPL_latest.joblib
        """
        # 1. Local Save
        filename = f"{symbol}_latest.joblib"
        local_path = self.local_dir / filename
        
        try:
            joblib.dump(model_data, local_path)
            logger.info(f"[{symbol}] Saved local: {local_path}")
        except Exception as e:
            logger.error(f"[{symbol}] Local save failed: {e}")
            return False

        # 2. S3 Upload (Clean Folder Structure)
        if upload_to_s3 and self.s3_client:
            s3_key = f"models/{symbol}/{filename}" # Folders by symbol
            try:
                self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
                logger.info(f"[{symbol}] Uploaded to S3: {s3_key}")
                
                # OPTIONAL: Delete local file after upload to save space
                os.remove(local_path) 
                
                return True
            except Exception as e:
                logger.error(f"[{symbol}] S3 upload failed: {e}")
                return False
        
        return True

    def load_model(self, symbol, prefer_s3=True):
        """
        Loads model. Tries local first, then downloads from S3 if missing.
        """
        filename = f"{symbol}_latest.joblib"
        local_path = self.local_dir / filename
        
        # 1. Try S3 Download (if preferred or local missing)
        if prefer_s3 and self.s3_client:
            s3_key = f"models/{symbol}/{filename}"
            try:
                self.s3_client.download_file(self.bucket, s3_key, str(local_path))
                logger.info(f"[{symbol}] Downloaded from S3.")
            except Exception as e:
                # Silent fail if not on S3 (might be a new symbol)
                pass

        # 2. Load Local
        if local_path.exists():
            try:
                return joblib.load(local_path)
            except Exception as e:
                logger.error(f"[{symbol}] Corrupt model file: {e}")
                return None
        
        return None