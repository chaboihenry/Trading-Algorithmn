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
    def __init__(self):
        self.root_data_dir = Path(project_root) / "data"
        self.s3_client = None
        self.bucket = None
        
        if is_aws_configured():
            try:
                import boto3
                self.s3_client = boto3.client('s3', region_name=AWS_REGION)
                self.bucket = S3_MODEL_BUCKET
            except Exception: pass

    def save_model(self, symbol, model_data, upload_to_s3=True):
        # 1. Define Local Paths
        symbol_dir = self.root_data_dir / symbol
        models_dir = symbol_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        filename_versioned = f"{symbol}_model_{timestamp}.joblib"
        filename_latest = f"{symbol}_latest.joblib"
        
        local_path_versioned = models_dir / filename_versioned
        local_path_latest = models_dir / filename_latest
        
        # 2. Save Locally
        try:
            joblib.dump(model_data, local_path_versioned)
            joblib.dump(model_data, local_path_latest)
            logger.info(f"[{symbol}] 💾 Saved model to: {local_path_latest}")
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Local Save Failed: {e}")
            return False

        # 3. Upload to S3 (Optional)
        if upload_to_s3 and self.s3_client:
            s3_key = f"{symbol}/models/{filename_latest}"
            try:
                with io.BytesIO() as buffer:
                    joblib.dump(model_data, buffer)
                    buffer.seek(0)
                    self.s3_client.upload_fileobj(buffer, self.bucket, s3_key)
                logger.info(f"[{symbol}] ☁️ Synced to S3.")
            except Exception as e:
                logger.warning(f"[{symbol}] S3 Upload failed: {e}")
        
        return True

    def load_model(self, symbol):
        # 1. Try Local First
        local_path = self.root_data_dir / symbol / "models" / f"{symbol}_latest.joblib"
        if local_path.exists():
            return joblib.load(local_path)
            
        # 2. Try S3
        if self.s3_client:
            s3_key = f"{symbol}/models/{symbol}_latest.joblib"
            try:
                with io.BytesIO() as buffer:
                    self.s3_client.download_fileobj(self.bucket, s3_key, buffer)
                    buffer.seek(0)
                    return joblib.load(buffer)
            except: pass
            
        return None