#!/usr/bin/env python3
"""
Cleanup Old Model Files
=======================
Removes bloated model files, keeping only the latest 3 versions per strategy.

Models grew from 657KB -> 29MB due to uncapped incremental training.
This script cleans up the mess and frees disk space.

Usage:
    python cleanup_old_models.py [--dry-run]
"""

import os
import re
from pathlib import Path
from collections import defaultdict
import argparse

def get_model_version(filename: str) -> int:
    """Extract version number from model filename"""
    match = re.search(r'_v(\d+)_', filename)
    return int(match.group(1)) if match else -1

def cleanup_old_models(models_dir: Path, keep_latest: int = 3, dry_run: bool = False):
    """
    Remove old model files, keeping only the latest N versions

    Args:
        models_dir: Path to models directory
        keep_latest: Number of latest versions to keep (default 3)
        dry_run: If True, only print what would be deleted
    """
    print("="*80)
    print("MODEL CLEANUP UTILITY")
    print("="*80)
    print(f"\nDirectory: {models_dir}")
    print(f"Policy: Keep latest {keep_latest} versions per strategy")
    print(f"Mode: {'DRY RUN (no files deleted)' if dry_run else 'LIVE (files will be deleted)'}")
    print("="*80)

    # Group files by strategy and file type
    strategy_files = defaultdict(lambda: {'models': [], 'scalers': [], 'features': []})

    for file in models_dir.glob('*'):
        if not file.is_file():
            continue

        filename = file.name

        # Skip non-versioned files (like volatility_xgboost.joblib)
        if '_v' not in filename:
            continue

        # Extract strategy name
        if filename.startswith('SentimentTradingStrategy'):
            strategy = 'SentimentTradingStrategy'
        elif filename.startswith('VolatilityTradingStrategy'):
            strategy = 'VolatilityTradingStrategy'
        elif filename.startswith('PairsTradingStrategy'):
            strategy = 'PairsTradingStrategy'
        else:
            continue

        # Categorize file type
        if 'scaler' in filename:
            file_type = 'scalers'
        elif 'features' in filename:
            file_type = 'features'
        else:
            file_type = 'models'

        version = get_model_version(filename)
        if version >= 0:
            strategy_files[strategy][file_type].append((version, file))

    # Process each strategy
    total_deleted = 0
    total_freed = 0

    for strategy, file_groups in sorted(strategy_files.items()):
        print(f"\n{'='*80}")
        print(f"STRATEGY: {strategy}")
        print('='*80)

        for file_type, files in file_groups.items():
            if not files:
                continue

            # Sort by version (descending)
            files.sort(key=lambda x: x[0], reverse=True)

            print(f"\n{file_type.upper()}:")
            print(f"  Total files: {len(files)}")

            # Keep latest N, delete rest
            files_to_keep = files[:keep_latest]
            files_to_delete = files[keep_latest:]

            print(f"  Keeping latest {len(files_to_keep)} versions:")
            for version, file in files_to_keep:
                size = file.stat().st_size / (1024*1024)  # MB
                print(f"    ✅ v{version} - {file.name} ({size:.1f} MB)")

            if files_to_delete:
                print(f"\n  Deleting {len(files_to_delete)} old versions:")
                for version, file in files_to_delete:
                    size = file.stat().st_size / (1024*1024)  # MB
                    total_freed += file.stat().st_size

                    if dry_run:
                        print(f"    🗑️  v{version} - {file.name} ({size:.1f} MB) [WOULD DELETE]")
                    else:
                        try:
                            file.unlink()
                            print(f"    🗑️  v{version} - {file.name} ({size:.1f} MB) [DELETED]")
                            total_deleted += 1
                        except Exception as e:
                            print(f"    ❌ v{version} - {file.name} [ERROR: {e}]")

    # Summary
    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)

    if dry_run:
        print(f"Would delete: {len([f for files in strategy_files.values() for file_list in files.values() for f in file_list[keep_latest:]])} files")
        print(f"Would free: {total_freed / (1024*1024):.1f} MB")
    else:
        print(f"Deleted: {total_deleted} files")
        print(f"Freed: {total_freed / (1024*1024):.1f} MB")

    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Cleanup old model files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--keep', type=int, default=3,
                       help='Number of latest versions to keep (default: 3)')
    args = parser.parse_args()

    models_dir = Path(__file__).parent / 'models'

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return 1

    cleanup_old_models(models_dir, keep_latest=args.keep, dry_run=args.dry_run)

    if args.dry_run:
        print("\n💡 Run without --dry-run to actually delete files")

    return 0

if __name__ == '__main__':
    exit(main())
