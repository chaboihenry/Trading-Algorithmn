"""
AWS Configuration for Model Storage

Environment variables required:
- AWS_ACCESS_KEY_ID: Your AWS access key
- AWS_SECRET_ACCESS_KEY: Your AWS secret key
- AWS_REGION: AWS region (default: us-east-1)
- S3_MODEL_BUCKET: Bucket name for model storage

To set these on Mac/Linux, add to ~/.zshrc or ~/.bashrc:
    export AWS_ACCESS_KEY_ID="your-key-here"
    export AWS_SECRET_ACCESS_KEY="your-secret-here"
    export AWS_REGION="us-east-1"
    export S3_MODEL_BUCKET="trading-agent-models"

Security Best Practice:
- NEVER hardcode credentials in code
- NEVER commit credentials to git
- Use environment variables or AWS IAM roles
"""

import os

# AWS Credentials (read from environment)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# S3 Bucket for model storage
S3_MODEL_BUCKET = os.getenv('S3_MODEL_BUCKET', 'trading-agent-models')

# Model storage paths within the bucket
S3_MODEL_PREFIX = 'models/'  # Folder structure: models/SPY/model_2026-01-10.pkl

def is_aws_configured() -> bool:
    """
    Check if AWS credentials are available.

    Returns:
        True if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set, False otherwise
    """
    return AWS_ACCESS_KEY_ID is not None and AWS_SECRET_ACCESS_KEY is not None

def get_s3_uri(symbol: str, filename: str) -> str:
    """
    Get full S3 URI for a model file.

    Args:
        symbol: Stock symbol (e.g., "SPY")
        filename: Model filename (e.g., "risklabai_SPY_latest.pkl")

    Returns:
        S3 URI string (e.g., "s3://bucket/models/SPY/risklabai_SPY_latest.pkl")
    """
    return f"s3://{S3_MODEL_BUCKET}/{S3_MODEL_PREFIX}{symbol}/{filename}"
