#!/usr/bin/env python3
"""
cleanup_url_descriptions.py

Remove URL strings that were incorrectly saved as descriptions.
Sets them back to NULL so they can be properly backfilled.
"""

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(p))

import sqlalchemy as sa
from sqlalchemy import text

DB_URL = os.environ.get("SUPABASE_DB_URL", "sqlite:///app.db")
engine = sa.create_engine(DB_URL, pool_pre_ping=True)

print("=" * 70)
print("🧹 Cleaning Up URL Descriptions")
print("=" * 70)
print()

with engine.begin() as conn:
    # Find descriptions that are URLs
    find_query = text("""
        SELECT notice_id, naics_code, description
        FROM solicitationraw
        WHERE description LIKE 'http%'
        AND naics_code LIKE '5417%'
    """)

    results = conn.execute(find_query).fetchall()

    if not results:
        print("✅ No URL descriptions found - database is clean!")
        sys.exit(0)

    print(f"Found {len(results)} solicitations with URL descriptions:")
    print()

    for i, (notice_id, naics, desc) in enumerate(results[:10], 1):
        print(f"{i}. {notice_id} (NAICS: {naics})")
        print(f"   URL: {desc[:80]}...")
        print()

    if len(results) > 10:
        print(f"   ... and {len(results) - 10} more")
        print()

    # Ask for confirmation
    response = input(f"Clear {len(results)} URL descriptions? (yes/no): ")

    if response.lower() not in ['yes', 'y']:
        print("❌ Cancelled - no changes made")
        sys.exit(0)

    # Clear the URLs
    update_query = text("""
        UPDATE solicitationraw
        SET description = NULL
        WHERE description LIKE 'http%'
        AND naics_code LIKE '5417%'
    """)

    result = conn.execute(update_query)

    print()
    print(f"✅ Cleared {result.rowcount} URL descriptions")
    print("   These solicitations can now be properly backfilled")
    print()
    print("Run backfill_descriptions_simple.py to fetch real descriptions")