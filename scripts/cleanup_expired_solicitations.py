#!/usr/bin/env python3
"""
Cleanup expired solicitations script for KIP.

This script removes solicitations from the database where the response_date 
has already passed (is before today's date). This helps keep the database 
clean and focused on current opportunities.

Environment variables:
  SUPABASE_DB_URL - Database connection string (required)
  CLEANUP_DRY_RUN - Set to 'true' to preview deletions without actually deleting
  CLEANUP_BATCH_SIZE - Number of records to delete per batch (default: 1000)
  CLEANUP_MAX_AGE_DAYS - Also delete records older than this many days regardless of response_date (default: 365)
"""

import os
import sys
from datetime import datetime, date, timezone, timedelta
from typing import List, Tuple
import sqlalchemy as sa
from sqlalchemy import text, create_engine
import re

# ---------------- Config ----------------
DB_URL = os.getenv("SUPABASE_DB_URL", "sqlite:///app.db")
DRY_RUN = os.getenv("CLEANUP_DRY_RUN", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("CLEANUP_BATCH_SIZE", "1000"))
MAX_AGE_DAYS = int(os.getenv("CLEANUP_MAX_AGE_DAYS", "365"))

# Date patterns for parsing response_date field
DATE_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
DATE_US_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")


def _engine():
    """Create database engine with appropriate settings"""
    kw = {}
    if DB_URL.startswith("postgresql"):
        kw.update(dict(pool_pre_ping=True, pool_size=5, max_overflow=2))
    return create_engine(DB_URL, **kw)


def _parse_date(date_str: str) -> date | None:
    """
    Parse various date formats and return a date object.
    Returns None if the date cannot be parsed or is invalid.
    """
    if not date_str or date_str.lower() in ("none", "n/a", "na", "null", ""):
        return None

    # Try ISO format first (YYYY-MM-DD)
    match = DATE_ISO_RE.search(date_str)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    # Try US format (M/D/YYYY or MM/DD/YYYY)
    match = DATE_US_RE.search(date_str)
    if match:
        try:
            return datetime.strptime(match.group(1), "%m/%d/%Y").date()
        except ValueError:
            pass

    # Try parsing as ISO datetime (with time component)
    if "T" in date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.date()
        except ValueError:
            pass

    return None


def _get_expired_solicitations(conn) -> Tuple[List[str], int]:
    """
    Find solicitations that should be deleted based on response_date and age.
    Returns (list_of_notice_ids, total_count)
    """
    today = date.today()
    max_age_cutoff = today - timedelta(days=MAX_AGE_DAYS)

    # Get all solicitations with their response dates and posted dates
    sql = text("""
        SELECT notice_id, response_date, posted_date, title
        FROM solicitationraw
        ORDER BY posted_date ASC NULLS FIRST
    """)

    result = conn.execute(sql)
    expired_ids = []

    print(f"Today's date: {today}")
    print(f"Max age cutoff: {max_age_cutoff}")
    print("Analyzing solicitations for expiration...")

    total_processed = 0
    expired_by_response_date = 0
    expired_by_age = 0
    unparseable_dates = 0

    for row in result:
        total_processed += 1
        notice_id = row.notice_id
        response_date_str = row.response_date
        posted_date_str = row.posted_date
        title = (row.title or "")[:50]

        should_delete = False
        reason = ""

        # Check if response date has passed
        response_date = _parse_date(response_date_str)
        if response_date and response_date < today:
            should_delete = True
            reason = f"response date {response_date} has passed"
            expired_by_response_date += 1

        # Check if record is too old (regardless of response date)
        if not should_delete:
            posted_date = _parse_date(posted_date_str)
            if posted_date and posted_date < max_age_cutoff:
                should_delete = True
                reason = f"posted date {posted_date} is older than {MAX_AGE_DAYS} days"
                expired_by_age += 1

        # Track unparseable dates for debugging
        if response_date_str and not response_date and response_date_str.lower() not in ("none", "n/a", "na", "null", ""):
            unparseable_dates += 1
            if unparseable_dates <= 5:  # Show first few examples
                print(
                    f"  Warning: Could not parse response_date '{response_date_str}' for {notice_id}")

        if should_delete:
            expired_ids.append(notice_id)
            print(f"  Will delete {notice_id}: {reason} | {title}")

    print(f"\nSummary:")
    print(f"  Total solicitations processed: {total_processed}")
    print(f"  Expired by response date: {expired_by_response_date}")
    print(f"  Expired by age ({MAX_AGE_DAYS}+ days): {expired_by_age}")
    print(f"  Total to delete: {len(expired_ids)}")
    print(f"  Unparseable dates: {unparseable_dates}")

    return expired_ids, len(expired_ids)


def _delete_solicitations_batch(conn, notice_ids: List[str]) -> int:
    """
    Delete a batch of solicitations by notice_id.
    Returns the number of records actually deleted.
    """
    if not notice_ids:
        return 0

    # Create placeholders for the IN clause
    placeholders = ",".join(f":id{i}" for i in range(len(notice_ids)))
    params = {f"id{i}": notice_ids[i] for i in range(len(notice_ids))}

    sql = text(f"""
        DELETE FROM solicitationraw 
        WHERE notice_id IN ({placeholders})
    """)

    result = conn.execute(sql, params)
    return result.rowcount or 0


def main():
    print("=== Cleanup Expired Solicitations ===")
    print(datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y"))
    # Hide credentials
    print(f"Database: {DB_URL.split('@')[-1] if '@' in DB_URL else DB_URL}")
    print(f"Dry run mode: {DRY_RUN}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Cleanup criteria: Delete solicitations where response_date has passed")
    print()

    if not DB_URL:
        print("ERROR: SUPABASE_DB_URL environment variable is required")
        sys.exit(1)

    engine = _engine()

    try:
        with engine.connect() as conn:
            # Verify table exists
            try:
                conn.execute(text("SELECT 1 FROM solicitationraw LIMIT 1"))
            except Exception as e:
                print(f"ERROR: Cannot access solicitationraw table: {e}")
                sys.exit(1)

            # Get database stats before cleanup
            before_count = conn.execute(
                text("SELECT COUNT(*) FROM solicitationraw")).scalar() or 0
            with_response_dates = conn.execute(text(
                "SELECT COUNT(*) FROM solicitationraw WHERE response_date IS NOT NULL AND response_date != 'None' AND response_date != ''")).scalar() or 0
            print(f"Total solicitations before cleanup: {before_count}")
            print(f"Solicitations with response dates: {with_response_dates}")

            # Find expired solicitations
            expired_ids, total_to_delete = _get_expired_solicitations(conn)

            if total_to_delete == 0:
                print(
                    "\nNo expired solicitations found. All response dates are current or missing.")
                return

            print(
                f"\nFound {total_to_delete} solicitations with expired response dates.")

            if DRY_RUN:
                print("\n*** DRY RUN MODE - No actual deletions will be performed ***")
                print("Set CLEANUP_DRY_RUN=false to perform actual cleanup.")
                return

            # Perform deletions in batches
            total_deleted = 0

            with engine.begin() as trans_conn:  # Use transaction for safety
                for i in range(0, len(expired_ids), BATCH_SIZE):
                    batch = expired_ids[i:i + BATCH_SIZE]
                    batch_num = i // BATCH_SIZE + 1

                    print(
                        f"Deleting batch {batch_num} ({len(batch)} records)...")
                    deleted_count = _delete_solicitations_batch(
                        trans_conn, batch)
                    total_deleted += deleted_count

                    print(
                        f"  Deleted {deleted_count} records in batch {batch_num}")

            # Get final count
            after_count = conn.execute(
                text("SELECT COUNT(*) FROM solicitationraw")).scalar() or 0

            print(f"\nCleanup completed successfully!")
            print(f"  Total records deleted: {total_deleted}")
            print(f"  Solicitations before: {before_count}")
            print(f"  Solicitations after: {after_count}")
            print(f"  Expected after: {before_count - total_to_delete}")

            if after_count != (before_count - total_to_delete):
                print(f"  WARNING: Unexpected count difference!")

    except Exception as e:
        print(f"ERROR: Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
