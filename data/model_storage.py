import os
import io
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
    Cloud-First Model Persistence.
    
    1. Attempts to save directly to S3 (In-Memory).
    2. Falls back to local disk ONLY if S3 fails or is not configured.
    3. Maintains granular version history (e.g., QQQ_model_2026-01-17_1430.joblib).
    """
    def __init__(self, local_dir="models"):
        self.local_dir = Path(local_dir)
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
        else:
            logger.warning("⚠️ AWS Credentials not found. Local storage will be used.")

    def save_model(self, symbol, model_data, upload_to_s3=True):
        """
        Uploads model to S3 directly from memory.
        """
        # Generate filenames with Hour:Minute precision
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        filename_versioned = f"{symbol}_model_{timestamp}.joblib"
        filename_latest = f"{symbol}_latest.joblib"
        
        s3_key_versioned = f"models/{symbol}/{filename_versioned}"
        s3_key_latest = f"models/{symbol}/{filename_latest}"

        # OPTION A: S3 UPLOAD (In-Memory)
        if upload_to_s3 and self.s3_client:
            try:
                # 1. Dump model to bytes buffer
                with io.BytesIO() as buffer:
                    joblib.dump(model_data, buffer)
                    model_bytes = buffer.getvalue()

                # 2. Upload Versioned File (Use fresh stream)
                with io.BytesIO(model_bytes) as stream1:
                    self.s3_client.upload_fileobj(stream1, self.bucket, s3_key_versioned)
                
                # 3. Upload Latest File (Use fresh stream)
                with io.BytesIO(model_bytes) as stream2:
                    self.s3_client.upload_fileobj(stream2, self.bucket, s3_key_latest)

                logger.info(f"[{symbol}] ☁️ Uploaded: {filename_versioned}")
                return True

            except Exception as e:
                logger.error(f"[{symbol}] ❌ S3 Upload Failed: {e}")
                logger.info(f"[{symbol}] Falling back to local disk...")

        # OPTION B: LOCAL SAVE
        self.local_dir.mkdir(parents=True, exist_ok=True)
        local_path = self.local_dir / filename_latest
        
        try:
            joblib.dump(model_data, local_path)
            logger.warning(f"[{symbol}] 💾 Saved locally: {local_path}")
            return True
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Local Save Failed: {e}")
            return False

    def load_model(self, symbol, prefer_s3=True):
        """Loads model (Always tries to get the 'latest' version)."""
        filename = f"{symbol}_latest.joblib"
        s3_key = f"models/{symbol}/{filename}"
        local_path = self.local_dir / filename

        if prefer_s3 and self.s3_client:
            try:
                with io.BytesIO() as buffer:
                    self.s3_client.download_fileobj(self.bucket, s3_key, buffer)
                    buffer.seek(0)
                    model = joblib.load(buffer)
                    logger.info(f"[{symbol}] ☁️ Loaded from S3.")
                    return model
            except: pass 

        if local_path.exists():
            try:
                return joblib.load(local_path)
            except Exception as e:
                logger.error(f"[{symbol}] Corrupt local file: {e}")
                return None
        return None