"""
Model Downloader - Fetches trained models from AWS S3

This allows anyone running the container to download the pre-trained
models without needing to train them locally.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# S3 bucket configuration (configurable via environment)
try:
    from config.aws_config import S3_MODEL_BUCKET, AWS_REGION, is_aws_configured
except Exception:
    S3_MODEL_BUCKET = None
    AWS_REGION = None
    is_aws_configured = lambda: False  # noqa: E731

S3_BUCKET = (
    os.environ.get("MODEL_BUCKET")
    or os.environ.get("S3_MODEL_BUCKET")
    or S3_MODEL_BUCKET
    or "risklabai-models"
)
S3_REGION = (
    os.environ.get("MODEL_REGION")
    or os.environ.get("AWS_REGION")
    or AWS_REGION
    or "us-east-1"
)
MODELS_DIR = Path(os.environ.get("MODELS_PATH", "models"))


def download_models(symbols: list = None, force: bool = False, require_s3: bool = False) -> int:
    """
    Download models from S3 if they don't exist locally.
    
    Args:
        symbols: List of symbols to download models for (None = all)
        force: If True, download even if local file exists
        require_s3: If True, raise if S3 access isn't available
        
    Returns:
        Number of models downloaded
    """
    # Only import boto3 when needed (not everyone has it installed)
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ImportError:
        message = (
            "boto3 not installed. Install with: pip install boto3\n"
            "Skipping model download - using local models only."
        )
        if require_s3:
            raise RuntimeError(message)
        logger.warning(message)
        return 0
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    use_signed = is_aws_configured()
    if use_signed:
        logger.info("Using signed S3 client (AWS credentials detected)")
        s3 = boto3.client("s3", region_name=S3_REGION)
    else:
        logger.info("Using unsigned S3 client (public bucket access)")
        s3 = boto3.client(
            "s3",
            region_name=S3_REGION,
            config=Config(signature_version=UNSIGNED)
        )
    
    downloaded = 0
    
    try:
        # List all objects in the models folder
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="models/")
        
        if 'Contents' not in response:
            logger.warning(f"No models found in s3://{S3_BUCKET}/models/")
            return 0
        
        for obj in response['Contents']:
            key = obj['Key']  # e.g., "models/risklabai_AAPL_models.pkl"
            filename = Path(key).name  # e.g., "risklabai_AAPL_models.pkl"
            
            # Skip if not a .pkl file
            if not filename.endswith('.pkl'):
                continue
            
            # Skip if filtering by symbols
            if symbols:
                # Extract symbol from filename (e.g., "AAPL" from "risklabai_AAPL_models.pkl")
                symbol = filename.replace('risklabai_', '').replace('_models.pkl', '')
                if symbol not in symbols:
                    continue
            
            local_path = MODELS_DIR / filename
            
            # Skip if already exists (unless force=True)
            if local_path.exists() and not force:
                logger.debug(f"Model already exists: {filename}")
                continue
            
            # Download the model
            logger.info(f"Downloading {filename}...")
            s3.download_file(S3_BUCKET, key, str(local_path))
            downloaded += 1
            
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        raise
    
    logger.info(f"Downloaded {downloaded} models from S3")
    return downloaded


def upload_models(symbols: list = None) -> int:
    """
    Upload local models to S3 (for you to update the bucket).
    
    Args:
        symbols: List of symbols to upload (None = all)
        
    Returns:
        Number of models uploaded
    """
    import boto3
    
    s3 = boto3.client('s3', region_name=S3_REGION)
    uploaded = 0
    
    for model_file in MODELS_DIR.glob("risklabai_*_models.pkl"):
        # Filter by symbols if specified
        if symbols:
            symbol = model_file.stem.replace('risklabai_', '').replace('_models', '')
            if symbol not in symbols:
                continue
        
        key = f"models/{model_file.name}"
        logger.info(f"Uploading {model_file.name}...")
        s3.upload_file(str(model_file), S3_BUCKET, key)
        uploaded += 1
    
    logger.info(f"Uploaded {uploaded} models to S3")
    return uploaded


if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload_models()
    else:
        download_models()
