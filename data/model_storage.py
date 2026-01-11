"""
Model Storage with Local + S3 Support

This class manages model persistence using two storage backends:
1. Local filesystem (primary, fast)
2. AWS S3 (backup, shareable, production-ready)

OOP Concepts:
- Encapsulation: All storage logic in one class
- Single Responsibility: This class ONLY handles storage, not training
- Graceful Degradation: Works without AWS, just uses local storage
- Adapter Pattern: Same interface for local and S3 storage

Usage:
    storage = ModelStorage(local_dir="models")

    # Save model (local + S3 if configured)
    storage.save_model("SPY", model_data)

    # Load model (tries local first, then S3)
    model_data = storage.load_model("SPY", version="latest")

    # List all versions
    versions = storage.list_versions("SPY")
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import joblib

logger = logging.getLogger(__name__)


class ModelStorage:
    """
    Handles model persistence with local and cloud storage.

    Think of this as a "smart file manager" that:
    - Saves files locally for speed
    - Backs them up to S3 for safety/sharing
    - Tracks versions so you can roll back

    Attributes:
        local_dir: Where models are stored locally (e.g., "./models")
        s3_client: AWS S3 client (None if AWS not configured)
        bucket: S3 bucket name
    """

    def __init__(self, local_dir: str = "models"):
        """
        Initialize storage with local directory and optional S3.

        Args:
            local_dir: Local directory for model files
        """
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Initialize S3 client if AWS is configured
        self.s3_client = None
        self.bucket = None
        self._init_s3()

    def _init_s3(self):
        """Initialize S3 client if credentials are available."""
        try:
            from config.aws_config import is_aws_configured, S3_MODEL_BUCKET, AWS_REGION

            if not is_aws_configured():
                logger.info("AWS not configured - using local storage only")
                logger.info("To enable S3 backup, set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
                return

            import boto3

            self.s3_client = boto3.client('s3', region_name=AWS_REGION)
            self.bucket = S3_MODEL_BUCKET
            logger.info(f"✓ S3 storage enabled: s3://{self.bucket}")

        except ImportError:
            logger.warning("boto3 not installed - S3 storage disabled")
            logger.info("Install with: pip install boto3")
        except Exception as e:
            logger.warning(f"S3 initialization failed: {e}")
            logger.info("Continuing with local storage only")

    def save_model(
        self,
        symbol: str,
        model_data: Dict[str, Any],
        upload_to_s3: bool = True
    ) -> str:
        """
        Save model locally and optionally to S3.

        Args:
            symbol: Stock symbol (e.g., "SPY")
            model_data: Dictionary containing model objects
            upload_to_s3: Whether to upload to S3 (default True)

        Returns:
            Local filepath where model was saved

        The model_data dictionary should contain:
            - primary_model: The trained primary model
            - meta_model: The trained meta model
            - scaler: StandardScaler instance
            - label_encoder: LabelEncoder instance
            - feature_names: List of feature names
            - important_features: Feature importance data
            - trained_at: Timestamp of training (added automatically)
            - training_metrics: Accuracy, etc.
        """
        # Add metadata
        model_data['trained_at'] = datetime.now().isoformat()
        model_data['symbol'] = symbol

        # Create versioned filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"risklabai_{symbol}_{timestamp}.pkl"
        local_path = self.local_dir / filename

        # Also save as "latest" for easy loading
        latest_filename = f"risklabai_{symbol}_latest.pkl"
        latest_path = self.local_dir / latest_filename

        # Save locally
        try:
            joblib.dump(model_data, local_path)
            joblib.dump(model_data, latest_path)
            logger.info(f"✓ Model saved locally: {local_path}")
        except Exception as e:
            logger.error(f"Failed to save model locally: {e}")
            raise

        # Upload to S3 if configured and requested
        if upload_to_s3 and self.s3_client is not None:
            try:
                self._upload_to_s3(local_path, symbol, filename)
                self._upload_to_s3(latest_path, symbol, latest_filename)
            except Exception as e:
                logger.warning(f"S3 upload failed (local save still succeeded): {e}")

        return str(local_path)

    def _upload_to_s3(self, local_path: Path, symbol: str, filename: str):
        """Upload a file to S3."""
        s3_key = f"models/{symbol}/{filename}"

        try:
            self.s3_client.upload_file(
                str(local_path),
                self.bucket,
                s3_key
            )
            logger.info(f"✓ Uploaded to s3://{self.bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"S3 upload failed for {s3_key}: {e}")
            raise

    def load_model(
        self,
        symbol: str,
        version: str = "latest",
        prefer_s3: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Load model from local storage or S3.

        Args:
            symbol: Stock symbol
            version: "latest" or specific timestamp (e.g., "20260110_143022")
            prefer_s3: If True, download from S3 even if local exists

        Returns:
            Model data dictionary, or None if not found
        """
        if version == "latest":
            filename = f"risklabai_{symbol}_latest.pkl"
        else:
            filename = f"risklabai_{symbol}_{version}.pkl"

        local_path = self.local_dir / filename

        # Try local first (unless prefer_s3)
        if local_path.exists() and not prefer_s3:
            try:
                logger.info(f"Loading model from local: {local_path}")
                return joblib.load(local_path)
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")
                # Fall through to try S3

        # Try S3 if local not found or prefer_s3 or local load failed
        if self.s3_client is not None:
            s3_key = f"models/{symbol}/{filename}"
            downloaded = self._download_from_s3(s3_key, local_path)
            if downloaded:
                try:
                    return joblib.load(local_path)
                except Exception as e:
                    logger.error(f"Failed to load downloaded model: {e}")
                    return None

        # Not found anywhere
        logger.warning(f"Model not found for {symbol} (version: {version})")
        return None

    def _download_from_s3(self, s3_key: str, local_path: Path) -> bool:
        """Download a file from S3. Returns True if successful."""
        try:
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))
            logger.info(f"✓ Downloaded from s3://{self.bucket}/{s3_key}")
            return True
        except self.s3_client.exceptions.NoSuchKey:
            logger.debug(f"S3 object not found: {s3_key}")
            return False
        except Exception as e:
            logger.warning(f"S3 download failed: {e}")
            return False

    def list_versions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        List all available versions for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            List of version dictionaries with keys:
            - version: Version identifier (timestamp)
            - source: "local" or "s3"
            - path: Full path to the model
            - last_modified: ISO timestamp (S3 only)
        """
        versions = []

        # Local versions
        for path in self.local_dir.glob(f"risklabai_{symbol}_*.pkl"):
            if "latest" not in path.name:
                # Extract timestamp from filename
                parts = path.stem.split("_")
                if len(parts) >= 3:
                    version_str = f"{parts[-2]}_{parts[-1]}"
                    versions.append({
                        'version': version_str,
                        'source': 'local',
                        'path': str(path),
                        'last_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
                    })

        # S3 versions (if configured)
        if self.s3_client is not None:
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=f"models/{symbol}/"
                )
                for obj in response.get('Contents', []):
                    key = obj['Key']
                    if "latest" not in key and key.endswith('.pkl'):
                        # Extract version from S3 key
                        filename = Path(key).stem
                        parts = filename.split('_')
                        if len(parts) >= 3:
                            version_str = f"{parts[-2]}_{parts[-1]}"
                            versions.append({
                                'version': version_str,
                                'source': 's3',
                                'path': f"s3://{self.bucket}/{key}",
                                'last_modified': obj['LastModified'].isoformat()
                            })
            except Exception as e:
                logger.warning(f"Failed to list S3 versions: {e}")

        # Sort by version (most recent first)
        return sorted(versions, key=lambda x: x['version'], reverse=True)

    def delete_old_versions(self, symbol: str, keep_count: int = 5):
        """
        Delete old model versions, keeping only the N most recent.

        Args:
            symbol: Stock symbol
            keep_count: Number of versions to keep (default 5)
        """
        versions = self.list_versions(symbol)

        if len(versions) <= keep_count:
            logger.info(f"{symbol}: Only {len(versions)} versions, no cleanup needed")
            return

        # Delete old versions (keep newest keep_count)
        to_delete = versions[keep_count:]

        for version_info in to_delete:
            if version_info['source'] == 'local':
                path = Path(version_info['path'])
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted old local version: {path.name}")

            elif version_info['source'] == 's3' and self.s3_client is not None:
                # Extract S3 key from path
                s3_path = version_info['path']
                if s3_path.startswith('s3://'):
                    key = '/'.join(s3_path.split('/')[3:])  # Remove s3://bucket/
                    try:
                        self.s3_client.delete_object(Bucket=self.bucket, Key=key)
                        logger.info(f"Deleted old S3 version: {key}")
                    except Exception as e:
                        logger.warning(f"Failed to delete S3 object {key}: {e}")

        logger.info(f"{symbol}: Cleaned up {len(to_delete)} old versions")
