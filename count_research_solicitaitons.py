#!/usr/bin/env python3
"""
count_research_solicitations.py

Show exact counts of research solicitations broken down by whether they have descriptions.
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
print("🔬 Research Solicitations (NAICS 5417xx) Breakdown")
print("=" * 70)
print()

with engine.connect() as conn:
    # Total research solicitations
    total_research = text("""
        SELECT COUNT(*) 
        FROM solicitationraw 
        WHERE naics_code LIKE '5417%'
    """)
    total = conn.execute(total_research).scalar()

    # Research with descriptions
    with_desc = text("""
        SELECT COUNT(*) 
        FROM solicitationraw 
        WHERE naics_code LIKE '5417%'
        AND description IS NOT NULL 
        AND description != ''
    """)
    has_desc = conn.execute(with_desc).scalar()

    # Research without descriptions
    without_desc = text("""
        SELECT COUNT(*) 
        FROM solicitationraw 
        WHERE naics_code LIKE '5417%'
        AND (description IS NULL OR description = '')
    """)
    missing_desc = conn.execute(without_desc).scalar()

    print(f"Total research solicitations (5417xx):        {total:6,}")
    print(
        f"  ✅ With descriptions:                       {has_desc:6,} ({has_desc / total * 100:.1f}%)" if total > 0 else "  ✅ With descriptions:                            0")
    print(
        f"  ❌ Missing descriptions:                    {missing_desc:6,} ({missing_desc / total * 100:.1f}%)" if total > 0 else "  ❌ Missing descriptions:                         0")
    print()

    if missing_desc > 0:
        print("=" * 70)
        print(f"📋 Sample of {min(10, missing_desc)} research solicitations MISSING descriptions:")
        print("=" * 70)

        sample_query = text("""
            SELECT notice_id, naics_code, title, posted_date
            FROM solicitationraw
            WHERE naics_code LIKE '5417%'
            AND (description IS NULL OR description = '')
            ORDER BY pulled_at DESC NULLS LAST
            LIMIT 10
        """)

        results = conn.execute(sample_query).fetchall()
        for i, (notice_id, naics, title, posted) in enumerate(results, 1):
            print(f"{i}. {notice_id}")
            print(f"   NAICS: {naics}")
            print(f"   Posted: {posted}")
            print(f"   Title: {title[:70]}...")
            print()

    print("=" * 70)
    print("💡 What This Means")
    print("=" * 70)
    print()

    if missing_desc == 0:
        print("✅ All research solicitations already have descriptions!")
        print("   Nothing to backfill.")
    elif missing_desc < 50:
        print(f"✅ Only {missing_desc} research solicitations need descriptions.")
        print("   Your backfill script should process exactly this many.")
    else:
        print(f"⚠️  {missing_desc} research solicitations need descriptions.")
        print("   Your backfill script should try to fetch all of these.")
        print()
        print("   If many are showing '[skip]' it means SAM.gov doesn't have")
        print("   descriptions available for them in the API.")
