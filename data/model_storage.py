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
    3. Maintains version history in S3 (e.g., QQQ_model_2026-01-17.joblib).
    """
    def __init__(self, local_dir="models"):
        self.local_dir = Path(local_dir)
        # We only create the local folder if we actually need to use the fallback
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
        # Generate filenames
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename_versioned = f"{symbol}_model_{date_str}.joblib"
        filename_latest = f"{symbol}_latest.joblib"
        
        # S3 Paths
        s3_key_versioned = f"models/{symbol}/{filename_versioned}"
        s3_key_latest = f"models/{symbol}/{filename_latest}"

        # ---------------------------------------------------------
        # OPTION A: S3 UPLOAD (In-Memory) - PREFERRED
        # ---------------------------------------------------------
        if upload_to_s3 and self.s3_client:
            try:
                # 1. Write model to RAM buffer
                with io.BytesIO() as buffer:
                    joblib.dump(model_data, buffer)
                    buffer.seek(0)  # Reset pointer to start of file
                    
                    # 2. Upload Versioned File (History)
                    self.s3_client.upload_fileobj(buffer, self.bucket, s3_key_versioned)
                    
                    # 3. Upload Latest File (Overwrite)
                    buffer.seek(0)  # Reset pointer again
                    self.s3_client.upload_fileobj(buffer, self.bucket, s3_key_latest)

                logger.info(f"[{symbol}] ☁️ Uploaded to S3 (RAM): {s3_key_versioned}")
                return True

            except Exception as e:
                logger.error(f"[{symbol}] ❌ S3 Upload Failed: {e}")
                logger.info(f"[{symbol}] Falling back to local disk...")
                # Proceed to Option B below

        # ---------------------------------------------------------
        # OPTION B: LOCAL SAVE - FALLBACK
        # ---------------------------------------------------------
        self.local_dir.mkdir(parents=True, exist_ok=True)
        local_path = self.local_dir / filename_latest
        
        try:
            joblib.dump(model_data, local_path)
            logger.warning(f"[{symbol}] 💾 Saved locally (Fallback): {local_path}")
            return True
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Local Save Failed: {e}")
            return False

    def load_model(self, symbol, prefer_s3=True):
        """
        Loads model. Tries to stream from S3 first, then checks local.
        """
        filename = f"{symbol}_latest.joblib"
        s3_key = f"models/{symbol}/{filename}"
        local_path = self.local_dir / filename

        # 1. Try S3 Download (In-Memory)
        if prefer_s3 and self.s3_client:
            try:
                with io.BytesIO() as buffer:
                    self.s3_client.download_fileobj(self.bucket, s3_key, buffer)
                    buffer.seek(0)
                    model = joblib.load(buffer)
                    logger.info(f"[{symbol}] ☁️ Loaded from S3 (RAM).")
                    return model
            except Exception as e:
                # If file missing or network error, silently fall through to local
                pass 

        # 2. Load Local
        if local_path.exists():
            try:
                model = joblib.load(local_path)
                logger.info(f"[{symbol}] 💾 Loaded from Local Disk.")
                return model
            except Exception as e:
                logger.error(f"[{symbol}] Corrupt local file: {e}")
                return None
        
        return None