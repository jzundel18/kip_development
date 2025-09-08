#!/usr/bin/env python3
"""
Simple database statistics script for post-cleanup reporting.
"""

import os
import sys
import sqlalchemy as sa
from sqlalchemy import text, create_engine


def main():
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        print("ERROR: SUPABASE_DB_URL environment variable is required")
        sys.exit(1)

    engine = create_engine(db_url, pool_pre_ping=True)

    try:
        with engine.connect() as conn:
            # Total solicitations
            total = conn.execute(
                text("SELECT COUNT(*) FROM solicitationraw")).scalar() or 0

            # Solicitations with response dates
            with_response_date = conn.execute(text("""
                SELECT COUNT(*) FROM solicitationraw 
                WHERE response_date IS NOT NULL 
                AND response_date != 'None' 
                AND response_date != ''
            """)).scalar() or 0

            # Recent solicitations (last 7 days)
            recent = conn.execute(text("""
                SELECT COUNT(*) FROM solicitationraw 
                WHERE posted_date >= CURRENT_DATE - INTERVAL '7 days'
            """)).scalar() or 0

            print(f"Total solicitations: {total}")
            print(f"With response dates: {with_response_date}")
            print(f"Posted in last 7 days: {recent}")

    except Exception as e:
        print(f"Error getting database stats: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
