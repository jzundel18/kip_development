#!/usr/bin/env python3
"""
check_notice_naics.py

Check the NAICS codes for specific notice IDs to verify they are research codes.

Usage:
    python check_notice_naics.py <notice_id1> <notice_id2> ...
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

# Get notice IDs from command line args
if len(sys.argv) < 2:
    print("Usage: python check_notice_naics.py <notice_id1> <notice_id2> ...")
    print()
    print("Example notice IDs from your output:")
    notice_ids = [
        '68f054d49f544a7faf77a3ac3aa21fa4',
        '000dca8baf6d44c39c90c47d62c47f48',
        'b3bffeeac6534909aaa373375d818f88',
        '69a4c9d2112d468ebdbccb9a9f2ff209',
        '6e1c9f0b12e14294a3e25469bca75d8a',
    ]
    print(f"python check_notice_naics.py {' '.join(notice_ids[:3])}")
    sys.exit(1)

notice_ids = sys.argv[1:]

print("=" * 70)
print("🔍 Checking NAICS Codes for Notice IDs")
print("=" * 70)
print()

with engine.connect() as conn:
    query = text("""
        SELECT 
            notice_id,
            naics_code,
            title,
            CASE 
                WHEN naics_code LIKE '5417%' THEN '🔬 RESEARCH'
                ELSE '❌ NOT RESEARCH'
            END as category,
            CASE 
                WHEN description IS NULL OR description = '' THEN 'Missing'
                ELSE 'Has description'
            END as desc_status
        FROM solicitationraw
        WHERE notice_id = ANY(:notice_ids)
    """)

    results = conn.execute(query, {"notice_ids": notice_ids}).fetchall()

    if not results:
        print("❌ No solicitations found with those notice IDs")
        print()
        print("Possible reasons:")
        print("  1. Notice IDs are from a different environment/database")
        print("  2. Notice IDs were typed incorrectly")
        print("  3. Solicitations were deleted")
        sys.exit(1)

    research_count = 0
    non_research_count = 0

    for notice_id, naics_code, title, category, desc_status in results:
        print(f"{category}  {notice_id}")
        print(f"  NAICS: {naics_code}")
        print(f"  Title: {title[:80]}..." if len(title) > 80 else f"  Title: {title}")
        print(f"  Status: {desc_status}")
        print()

        if 'RESEARCH' in category:
            research_count += 1
        else:
            non_research_count += 1

    print("=" * 70)
    print("Summary:")
    print(f"  Research solicitations (5417xx): {research_count}")
    print(f"  Non-research solicitations: {non_research_count}")
    print("=" * 70)
    print()

    if non_research_count > 0:
        print("⚠️  WARNING: Your backfill script is processing NON-RESEARCH solicitations!")
        print("   This means the NAICS filter is NOT working properly.")
        print()
        print("   Check your fetch_missing() function and ensure it has:")
        print("   AND naics_code LIKE '5417%'")
    else:
        print("✅ All checked solicitations are research codes (5417xx)")
        print()
        print("The '[skip]' messages mean SAM.gov API doesn't have descriptions")
        print("for these research solicitations, which is normal - not all")
        print("solicitations have descriptions available in the API.")