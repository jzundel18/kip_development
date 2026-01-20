#!/usr/bin/env python3
"""
check_missing_descriptions.py

Quick diagnostic to see how many solicitations are missing descriptions,
broken down by category (research vs non-research).
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
print("🔍 Missing Descriptions Diagnostic")
print("=" * 70)
print()

with engine.connect() as conn:
    # Total missing descriptions
    total_query = text("""
        SELECT COUNT(*) as count
        FROM solicitationraw
        WHERE (description IS NULL OR description = '')
    """)
    total_missing = conn.execute(total_query).scalar()
    print(f"Total solicitations missing descriptions: {total_missing:,}")
    print()

    # Research missing descriptions (5417xx)
    research_query = text("""
        SELECT COUNT(*) as count
        FROM solicitationraw
        WHERE (description IS NULL OR description = '')
        AND naics_code LIKE '5417%'
    """)
    research_missing = conn.execute(research_query).scalar()
    print(f"Research solicitations (5417xx) missing descriptions: {research_missing:,}")
    print()

    # Non-research missing descriptions
    non_research_missing = total_missing - research_missing
    print(f"Non-research solicitations missing descriptions: {non_research_missing:,}")
    print()

    # Percentages
    if total_missing > 0:
        research_pct = (research_missing / total_missing) * 100
        non_research_pct = (non_research_missing / total_missing) * 100
        print(f"Breakdown:")
        print(f"  Research:     {research_pct:5.1f}% ({research_missing:,} solicitations)")
        print(f"  Non-research: {non_research_pct:5.1f}% ({non_research_missing:,} solicitations)")
    print()

    # Show top NAICS codes missing descriptions
    print("=" * 70)
    print("Top 10 NAICS codes with missing descriptions:")
    print("=" * 70)

    top_naics_query = text("""
        SELECT 
            naics_code,
            COUNT(*) as count,
            CASE 
                WHEN naics_code LIKE '5417%' THEN '🔬 RESEARCH'
                ELSE '  '
            END as category
        FROM solicitationraw
        WHERE (description IS NULL OR description = '')
        AND naics_code IS NOT NULL
        GROUP BY naics_code
        ORDER BY count DESC
        LIMIT 10
    """)

    results = conn.execute(top_naics_query).fetchall()

    for naics_code, count, category in results:
        print(f"  {category}  {naics_code:10}  {count:6,} solicitations")

    print()
    print("=" * 70)
    print("💡 RECOMMENDATION")
    print("=" * 70)
    print()

    if research_missing > 0:
        savings_pct = (non_research_missing / total_missing) * 100 if total_missing > 0 else 0
        print(f"By filtering to research NAICS codes only, you will:")
        print(f"  ✅ Process {research_missing:,} research solicitations")
        print(f"  ⏭️  Skip {non_research_missing:,} non-research solicitations")
        print(f"  📉 Reduce API calls by {savings_pct:.0f}%")
        print()
        print("Make sure your backfill_descriptions.py has this SQL query:")
        print()
        print("    WHERE (description IS NULL OR description = '')")
        print("    AND naics_code LIKE '5417%'   <-- THIS LINE IS CRITICAL")
        print()
    else:
        print("✅ All research solicitations already have descriptions!")
        print("   No backfilling needed for research items.")