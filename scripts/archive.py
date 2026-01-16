#!/usr/bin/env python3
"""
Data archival script for Fraud Detection System.

Archives old prediction logs and audit data to cold storage,
optimizing database performance and reducing storage costs.

Usage:
    python archive.py --days 30           # Archive data older than 30 days
    python archive.py --days 30 --dry-run # Preview what would be archived
"""
import os
import sys
import argparse
import gzip
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_url():
    """Get database URL from environment."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/fraud_db"
    )


def archive_predictions(cutoff_date: datetime, archive_dir: Path, dry_run: bool = False):
    """Archive old prediction logs to compressed files.
    
    Args:
        cutoff_date: Archive records before this date
        archive_dir: Directory to store archived files
        dry_run: If True, only preview what would be archived
    """
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd
        
        engine = create_engine(get_database_url())
        
        # Count records to archive
        count_query = text("""
            SELECT COUNT(*) FROM prediction_logs
            WHERE created_at < :cutoff
        """)
        
        with engine.connect() as conn:
            result = conn.execute(count_query, {"cutoff": cutoff_date})
            count = result.scalar()
        
        logger.info(f"Found {count} prediction records to archive (before {cutoff_date})")
        
        if dry_run:
            logger.info("[DRY RUN] Would archive these records")
            return count
        
        if count == 0:
            logger.info("No records to archive")
            return 0
        
        # Export data in batches
        batch_size = 10000
        offset = 0
        total_archived = 0
        
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while offset < count:
            select_query = text("""
                SELECT * FROM prediction_logs
                WHERE created_at < :cutoff
                ORDER BY created_at
                LIMIT :limit OFFSET :offset
            """)
            
            df = pd.read_sql(
                select_query,
                engine,
                params={"cutoff": cutoff_date, "limit": batch_size, "offset": offset}
            )
            
            if len(df) == 0:
                break
            
            # Save to compressed JSON
            archive_file = archive_dir / f"predictions_{timestamp}_batch_{offset // batch_size}.json.gz"
            with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
                df.to_json(f, orient='records', date_format='iso')
            
            total_archived += len(df)
            offset += batch_size
            logger.info(f"Archived {total_archived}/{count} records to {archive_file}")
        
        # Delete archived records from database
        delete_query = text("""
            DELETE FROM prediction_logs
            WHERE created_at < :cutoff
        """)
        
        with engine.connect() as conn:
            conn.execute(delete_query, {"cutoff": cutoff_date})
            conn.commit()
        
        logger.info(f"Successfully archived {total_archived} prediction records")
        return total_archived
        
    except ImportError:
        logger.error("Required packages not installed. Run: pip install sqlalchemy pandas psycopg2-binary")
        return 0
    except Exception as e:
        logger.error(f"Archive failed: {e}")
        raise


def archive_audit_logs(cutoff_date: datetime, archive_dir: Path, dry_run: bool = False):
    """Archive old audit logs to compressed files.
    
    Args:
        cutoff_date: Archive records before this date
        archive_dir: Directory to store archived files
        dry_run: If True, only preview what would be archived
    """
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd
        
        engine = create_engine(get_database_url())
        
        # Check if audit_logs table exists
        check_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'audit_logs'
            )
        """)
        
        with engine.connect() as conn:
            result = conn.execute(check_query)
            exists = result.scalar()
        
        if not exists:
            logger.info("No audit_logs table found, skipping")
            return 0
        
        # Count records to archive
        count_query = text("""
            SELECT COUNT(*) FROM audit_logs
            WHERE created_at < :cutoff
        """)
        
        with engine.connect() as conn:
            result = conn.execute(count_query, {"cutoff": cutoff_date})
            count = result.scalar()
        
        logger.info(f"Found {count} audit log records to archive")
        
        if dry_run:
            logger.info("[DRY RUN] Would archive these records")
            return count
        
        if count == 0:
            return 0
        
        # Export all at once (audit logs are smaller)
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        select_query = text("""
            SELECT * FROM audit_logs
            WHERE created_at < :cutoff
            ORDER BY created_at
        """)
        
        df = pd.read_sql(select_query, engine, params={"cutoff": cutoff_date})
        
        archive_file = archive_dir / f"audit_logs_{timestamp}.json.gz"
        with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
            df.to_json(f, orient='records', date_format='iso')
        
        # Delete archived records
        delete_query = text("""
            DELETE FROM audit_logs
            WHERE created_at < :cutoff
        """)
        
        with engine.connect() as conn:
            conn.execute(delete_query, {"cutoff": cutoff_date})
            conn.commit()
        
        logger.info(f"Archived {count} audit log records to {archive_file}")
        return count
        
    except Exception as e:
        logger.error(f"Audit log archive failed: {e}")
        return 0


def cleanup_old_archives(archive_dir: Path, keep_days: int = 90):
    """Remove archive files older than specified days.
    
    Args:
        archive_dir: Directory containing archived files
        keep_days: Keep archives for this many days
    """
    cutoff = datetime.now() - timedelta(days=keep_days)
    removed = 0
    
    for archive_file in archive_dir.glob("*.json.gz"):
        if archive_file.stat().st_mtime < cutoff.timestamp():
            archive_file.unlink()
            removed += 1
            logger.info(f"Removed old archive: {archive_file}")
    
    if removed:
        logger.info(f"Cleaned up {removed} old archive files")


def main():
    parser = argparse.ArgumentParser(
        description="Archive old prediction and audit data"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Archive data older than this many days (default: 30)"
    )
    parser.add_argument(
        "--archive-dir",
        type=str,
        default="./archives",
        help="Directory to store archived files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be archived without making changes"
    )
    parser.add_argument(
        "--cleanup-days",
        type=int,
        default=90,
        help="Remove archive files older than this many days"
    )
    
    args = parser.parse_args()
    
    cutoff_date = datetime.now() - timedelta(days=args.days)
    archive_dir = Path(args.archive_dir)
    
    logger.info(f"Starting archive process (cutoff: {cutoff_date})")
    
    # Archive predictions
    prediction_count = archive_predictions(cutoff_date, archive_dir, args.dry_run)
    
    # Archive audit logs
    audit_count = archive_audit_logs(cutoff_date, archive_dir, args.dry_run)
    
    # Cleanup old archives
    if not args.dry_run:
        cleanup_old_archives(archive_dir, args.cleanup_days)
    
    logger.info(f"Archive complete: {prediction_count} predictions, {audit_count} audit logs")


if __name__ == "__main__":
    main()
