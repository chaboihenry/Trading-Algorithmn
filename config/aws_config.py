import os

# --- Credentials ---
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# --- Storage Settings ---
S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET', 'trading-agent-models')
S3_MODEL_PREFIX = 'models/'

def is_aws_configured() -> bool:
    """Returns True if AWS credentials are present."""
    return bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)

def get_s3_uri(symbol: str, filename: str) -> str:
    """Helper to get the full S3 path for debugging."""
    return f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}{symbol}/{filename}"