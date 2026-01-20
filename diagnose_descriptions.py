#!/usr/bin/env python3
"""
Diagnostic script to check if descriptions exist in the database
and what the actual content looks like.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

# Load .env file FIRST
try:
    from dotenv import load_dotenv

    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
    else:
        # Try parent directory
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
            print(f"✓ Loaded environment from {env_file}")
        else:
            print("⚠ No .env file found. Using system environment variables.")
except ImportError:
    print("⚠ python-dotenv not installed. Using system environment variables.")

# Load environment
DB_URL = os.getenv("SUPABASE_DB_URL")

if not DB_URL:
    print("\n" + "=" * 70)
    print("ERROR: SUPABASE_DB_URL environment variable not set")
    print("=" * 70)
    print("\nPlease set the environment variable:")
    print("  export SUPABASE_DB_URL='your_database_url'")
    print("\nOr create a .env file with:")
    print("  SUPABASE_DB_URL=your_database_url")
    print("\n" + "=" * 70)
    sys.exit(1)

print(f"✓ Database URL loaded (length: {len(DB_URL)} chars)")
print(f"  Connection string starts with: {DB_URL[:20]}...")

try:
    engine = sa.create_engine(DB_URL, pool_pre_ping=True)
    print("✓ Database engine created")
except Exception as e:
    print(f"\n❌ Failed to create database engine: {e}")
    sys.exit(1)


def check_descriptions():
    """Check what descriptions look like in the database"""

    # Test connection first
    print("\n" + "=" * 70)
    print("Testing database connection...")
    print("=" * 70)

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            print("✓ Database connection successful!")
    except Exception as e:
        print(f"\n❌ Database connection failed: {e}")
        print("\nPlease check:")
        print("  1. Database URL is correct")
        print("  2. Database is accessible")
        print("  3. Credentials are valid")
        sys.exit(1)

    # Get yesterday's date
    now = datetime.now(timezone.utc)
    yesterday = (now.date() - timedelta(days=1)).strftime("%Y-%m-%d")

    print("\n" + "=" * 70)
    print("DESCRIPTION DIAGNOSTIC TOOL")
    print("=" * 70)
    print(f"\nChecking RESEARCH solicitations from: {yesterday}")

    with engine.connect() as conn:
        # First, check recent dates
        print("\n1. Recent dates in database:")
        print("-" * 70)
        recent_dates = conn.execute(text("""
            SELECT posted_date, COUNT(*) as count
            FROM solicitationraw 
            WHERE posted_date IS NOT NULL 
              AND category = 'research'
            GROUP BY posted_date 
            ORDER BY posted_date DESC 
            LIMIT 5
        """)).fetchall()

        for date_val, count in recent_dates:
            print(f"  {date_val}: {count} research records")

        # Fetch some sample notices from yesterday
        print(f"\n2. Sample RESEARCH notices from {yesterday}:")
        print("-" * 70)

        sql = text("""
            SELECT notice_id, title, description, 
                   LENGTH(description) as desc_length,
                   CASE 
                     WHEN description IS NULL THEN 'NULL'
                     WHEN description = '' THEN 'EMPTY STRING'
                     ELSE 'HAS CONTENT'
                   END as desc_status
            FROM solicitationraw
            WHERE posted_date = :yesterday_date
            AND category = 'research'
            LIMIT 10
        """)

        df = pd.read_sql_query(sql, conn, params={"yesterday_date": yesterday})

        if df.empty:
            print(f"\n  ⚠️  No RESEARCH solicitations found for {yesterday}")
            print("     Try adjusting the date or check if data exists")
        else:
            print(f"\n  Found {len(df)} sample records:\n")

            for idx, row in df.iterrows():
                print(f"  Notice {idx + 1}:")
                print(f"    ID: {row['notice_id']}")
                print(f"    Title: {row['title'][:60]}...")
                print(f"    Description Status: {row['desc_status']}")
                print(f"    Description Length: {row['desc_length']} chars")

                if row['desc_status'] == 'HAS CONTENT':
                    desc_preview = str(row['description'])[:150]
                    print(f"    Preview: {desc_preview}...")
                print()

        # Summary statistics
        print("\n3. Description statistics:")
        print("-" * 70)

        stats_sql = text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN description IS NULL THEN 1 ELSE 0 END) as null_count,
                SUM(CASE WHEN description = '' THEN 1 ELSE 0 END) as empty_count,
                SUM(CASE WHEN description IS NOT NULL AND description != '' THEN 1 ELSE 0 END) as has_content,
                AVG(LENGTH(description)) as avg_length
            FROM solicitationraw
            WHERE posted_date = :yesterday_date
            AND category = 'research'
        """)

        stats = conn.execute(stats_sql, {"yesterday_date": yesterday}).fetchone()

        if stats and stats[0] > 0:
            print(f"  Total records: {stats[0]}")
            print(f"  NULL descriptions: {stats[1]}")
            print(f"  Empty string descriptions: {stats[2]}")
            print(f"  Has content: {stats[3]}")
            print(f"  Average length: {stats[4]:.1f} chars")

            if stats[3] == 0:
                print("\n  ❌ PROBLEM: No descriptions have content!")
                print("     All descriptions are either NULL or empty strings.")
            elif stats[3] < stats[0]:
                print(f"\n  ⚠️  WARNING: Only {stats[3]}/{stats[0]} records have descriptions")
            else:
                print("\n  ✅ All records have descriptions")
        else:
            print("  No data available for statistics")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print("""
If descriptions show as NULL or EMPTY STRING:
  1. Check if the scraper is populating the description field
  2. Verify the database schema has the description column
  3. Check if descriptions exist for other dates
  4. Look at the raw SAM.gov data to see if descriptions are available

If descriptions have content but emails show "No description available":
  1. Run test_daily_digest.py and check the logs carefully
  2. Look for "Notice XXX has description: YYY chars" messages
  3. Check for "Generated summary for notice_id" messages
  4. Verify OpenAI API key is working

If you see "HAS CONTENT" above, the descriptions ARE in the database,
and the issue is in the email generation code.
    """)


if __name__ == "__main__":
    try:
        check_descriptions()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()