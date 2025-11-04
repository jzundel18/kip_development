#!/usr/bin/env python3
"""
Classify Company Business Types - FIXED VERSION
Handles case-sensitive column names in PostgreSQL
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import sqlalchemy as sa
from sqlalchemy import text, create_engine
from openai import OpenAI

# Try to load .env file if present
try:
    from dotenv import load_dotenv

    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# Configuration
DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fallback to streamlit secrets
if not DB_URL or not OPENAI_API_KEY:
    try:
        import streamlit as st

        if not DB_URL:
            DB_URL = st.secrets.get("SUPABASE_DB_URL")
        if not OPENAI_API_KEY:
            OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    except:
        pass

DEFAULT_BATCH_SIZE = 50

# NAICS code mappings
GOODS_NAICS_PATTERNS = {
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale Trade", "44": "Retail Trade", "45": "Retail Trade",
}

SERVICES_NAICS_PATTERNS = {
    "48": "Transportation", "49": "Transportation", "51": "Information",
    "52": "Finance", "53": "Real Estate", "54": "Professional Services",
    "55": "Management", "56": "Administrative Services", "61": "Educational Services",
    "62": "Health Care", "71": "Arts/Entertainment", "72": "Accommodation/Food",
    "81": "Other Services",
}

# Keywords for classification
GOODS_KEYWORDS = [
    "manufacturer", "manufacturing", "fabrication", "production", "assembly",
    "machining", "cnc", "parts", "components", "equipment", "products",
    "hardware", "materials", "supplies", "distributor", "wholesale",
    "casting", "molding", "welding", "tooling", "machinery"
]

SERVICES_KEYWORDS = [
    "services", "consulting", "maintenance", "repair", "installation",
    "support", "training", "engineering", "design", "analysis",
    "management", "logistics", "transportation", "it services",
    "software development", "consulting", "professional services",
    "testing", "inspection", "certification", "staffing"
]


def validate_environment():
    """Validate required environment variables"""
    if not DB_URL:
        print("❌ ERROR: SUPABASE_DB_URL environment variable is required")
        print("\nPlease set your environment variables:")
        print("  Create a .env file or export SUPABASE_DB_URL and OPENAI_API_KEY")
        sys.exit(1)

    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY environment variable is required")
        sys.exit(1)


def get_engine():
    """Create database engine"""
    return create_engine(DB_URL, pool_pre_ping=True)


def get_table_columns(conn) -> List[str]:
    """Get actual column names from company_list table"""
    sql = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'company_list'
        ORDER BY ordinal_position
    """)

    result = conn.execute(sql)
    return [row.column_name for row in result]


def fetch_unclassified_companies(conn, limit: int) -> List[Dict]:
    """Fetch companies that need classification"""
    # Get actual column names
    all_columns = get_table_columns(conn)
    print(f"  Available columns: {', '.join(all_columns[:10])}{'...' if len(all_columns) > 10 else ''}")

    # Find NAICS column (case-insensitive search)
    naics_col = None
    other_naics_col = None

    for col in all_columns:
        col_lower = col.lower()
        if col_lower == 'naics':
            naics_col = col
        elif 'other' in col_lower and 'naics' in col_lower:
            other_naics_col = col

    # Build query with actual column names
    if naics_col and other_naics_col:
        sql = text(f"""
            SELECT id, name, description, "{naics_col}" as naics, "{other_naics_col}" as other_naics
            FROM company_list
            WHERE business_type IS NULL
            ORDER BY id
            LIMIT :limit
        """)
    elif naics_col:
        sql = text(f"""
            SELECT id, name, description, "{naics_col}" as naics
            FROM company_list
            WHERE business_type IS NULL
            ORDER BY id
            LIMIT :limit
        """)
    else:
        # No NAICS columns found
        sql = text("""
            SELECT id, name, description
            FROM company_list
            WHERE business_type IS NULL
            ORDER BY id
            LIMIT :limit
        """)

    result = conn.execute(sql, {"limit": limit})
    companies = []

    for row in result:
        companies.append({
            "id": row.id,
            "name": row.name or "",
            "description": row.description or "",
            "naics": getattr(row, 'naics', '') or "",
            "other_naics": getattr(row, 'other_naics', '') or ""
        })

    return companies


def classify_by_naics(naics_code: str) -> Optional[str]:
    """Classify based on NAICS code patterns"""
    if not naics_code:
        return None

    naics_prefix = naics_code[:2]

    if naics_prefix in GOODS_NAICS_PATTERNS:
        return "Goods"
    elif naics_prefix in SERVICES_NAICS_PATTERNS:
        return "Services"

    return None


def classify_by_keywords(text: str) -> Optional[str]:
    """Simple keyword-based classification"""
    if not text:
        return None

    text_lower = text.lower()

    goods_count = sum(1 for kw in GOODS_KEYWORDS if kw in text_lower)
    services_count = sum(1 for kw in SERVICES_KEYWORDS if kw in text_lower)

    if goods_count > services_count * 2:
        return "Goods"
    elif services_count > goods_count * 2:
        return "Services"

    return None


def classify_with_ai(companies: List[Dict], api_key: str) -> Dict[int, str]:
    """Use AI to classify companies in batch"""
    if not companies:
        return {}

    client = OpenAI(api_key=api_key)

    system_prompt = """You are a business classification expert. Classify companies as either:
- "Goods" - if they primarily manufacture, produce, or supply physical products/equipment/parts
- "Services" - if they primarily provide services, consulting, maintenance, software, or expertise
- "Mixed" - if they clearly do both equally (rare)

Be decisive. Most companies lean one way or the other.

Return ONLY valid JSON:
{"classifications": [{"id": 123, "type": "Goods", "confidence": "high"}]}

Valid types: "Goods", "Services", "Mixed"
Valid confidence: "high", "medium", "low"""

    batch_data = []
    for company in companies:
        batch_data.append({
            "id": company["id"],
            "name": company["name"],
            "description": company["description"][:300],
            "naics": company["naics"]
        })

    user_prompt = {
        "companies": batch_data,
        "instructions": "Classify each company. Be decisive. Return JSON only."
    }

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)}
            ],
            temperature=0.1,
            max_tokens=2000,
            timeout=60
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        classifications = {}
        for item in data.get("classifications", []):
            company_id = item.get("id")
            classification_type = item.get("type")
            confidence = item.get("confidence", "medium")

            if classification_type not in ["Goods", "Services", "Mixed"]:
                continue

            if classification_type == "Mixed" or confidence == "low":
                continue

            classifications[company_id] = classification_type

        return classifications

    except Exception as e:
        print(f"  ❌ AI classification failed: {e}")
        return {}


def update_classifications(conn, classifications: Dict[int, str], dry_run: bool = False):
    """Update database with classifications"""
    if not classifications:
        return 0

    if dry_run:
        print(f"\n  DRY RUN: Would update {len(classifications)} companies:")
        for company_id, biz_type in list(classifications.items())[:5]:
            print(f"    Company {company_id} → {biz_type}")
        if len(classifications) > 5:
            print(f"    ... and {len(classifications) - 5} more")
        return len(classifications)

    updated_count = 0

    for company_id, business_type in classifications.items():
        try:
            sql = text("""
                UPDATE company_list
                SET business_type = :btype
                WHERE id = :cid
            """)

            result = conn.execute(sql, {"btype": business_type, "cid": company_id})

            if result.rowcount > 0:
                updated_count += 1

        except Exception as e:
            print(f"  ❌ Error updating company {company_id}: {e}")
            continue

    return updated_count


def classify_companies_batch(companies: List[Dict], api_key: str, dry_run: bool = False) -> Dict[int, str]:
    """Multi-stage classification"""
    classifications = {}
    needs_ai = []

    print(f"\n  Stage 1: NAICS & keyword-based classification...")
    for company in companies:
        # Try NAICS first
        naics_classification = classify_by_naics(company["naics"])
        if naics_classification:
            classifications[company["id"]] = naics_classification
            continue

        # Try keywords
        text = f"{company['name']} {company['description']}"
        keyword_classification = classify_by_keywords(text)
        if keyword_classification:
            classifications[company["id"]] = keyword_classification
            continue

        needs_ai.append(company)

    print(f"    ✓ Classified {len(classifications)} companies by rules")

    if needs_ai:
        print(f"\n  Stage 2: AI classification for {len(needs_ai)} unclear cases...")
        ai_classifications = classify_with_ai(needs_ai, api_key)
        classifications.update(ai_classifications)
        print(f"    ✓ AI classified {len(ai_classifications)} additional companies")

    return classifications


def main():
    parser = argparse.ArgumentParser(description="Classify company business types")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview classifications without updating database")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of companies to process (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--all", action="store_true",
                        help="Process all unclassified companies")

    args = parser.parse_args()

    print("=" * 70)
    print("COMPANY BUSINESS TYPE CLASSIFICATION")
    print("=" * 70)

    validate_environment()

    print(f"\nConfiguration:")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Batch size: {'ALL' if args.all else args.batch_size}")

    engine = get_engine()

    try:
        with engine.connect() as conn:
            # Get schema info
            print(f"\nDetecting database schema...")

            total_unclassified = conn.execute(text("""
                SELECT COUNT(*) FROM company_list WHERE business_type IS NULL
            """)).scalar()

            print(f"Unclassified companies in database: {total_unclassified}")

            if total_unclassified == 0:
                print("✓ All companies are already classified!")
                return

            limit = total_unclassified if args.all else args.batch_size
            companies = fetch_unclassified_companies(conn, limit)

            if not companies:
                print("No companies to process.")
                return

            print(f"\nProcessing {len(companies)} companies...")
            print("-" * 70)

            classifications = classify_companies_batch(companies, OPENAI_API_KEY, args.dry_run)

            if not classifications:
                print("\n❌ No companies could be confidently classified")
                return

            print(f"\n  Successfully classified: {len(classifications)}/{len(companies)} companies")
            print(f"  Left blank (unclear): {len(companies) - len(classifications)} companies")

            goods_count = sum(1 for v in classifications.values() if v == "Goods")
            services_count = sum(1 for v in classifications.values() if v == "Services")

            print(f"\n  Breakdown:")
            print(f"    Goods: {goods_count}")
            print(f"    Services: {services_count}")

            if not args.dry_run:
                with engine.begin() as trans_conn:
                    updated = update_classifications(trans_conn, classifications)
                    print(f"\n✓ Updated {updated} companies in database")
            else:
                print(f"\n  (Dry run - no changes made)")

            with engine.connect() as conn:
                classified = conn.execute(text("""
                    SELECT COUNT(*) FROM company_list WHERE business_type IS NOT NULL
                """)).scalar()
                total = conn.execute(text("SELECT COUNT(*) FROM company_list")).scalar()

                print(f"\nFinal status:")
                print(f"  Classified: {classified}/{total} ({classified * 100 // total if total > 0 else 0}%)")
                print(f"  Remaining: {total - classified}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()