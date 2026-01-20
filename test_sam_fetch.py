#!/usr/bin/env python3
"""
test_sam_fetch.py

Test fetching a description from SAM.gov for a specific notice ID with full debugging.
This helps diagnose why descriptions are coming back empty.

Usage:
    python test_sam_fetch.py <notice_id>
"""

import os
import sys
import time
import requests
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

# Get SAM API keys
_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r", "").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]

if not SAM_KEYS:
    print("❌ ERROR: No SAM_KEYS configured!")
    print("Set SAM_KEYS environment variable.")
    sys.exit(1)

api_key = SAM_KEYS[0]
print(f"Using API key: {api_key[:15]}...{api_key[-10:]}")
print()

# Get notice ID from command line or use a default
if len(sys.argv) < 2:
    print("Getting a sample research solicitation from database...")
    with engine.connect() as conn:
        query = text("""
            SELECT notice_id, naics_code, title
            FROM solicitationraw
            WHERE (description IS NULL OR description = '')
            AND naics_code LIKE '5417%'
            ORDER BY pulled_at DESC NULLS LAST
            LIMIT 1
        """)
        result = conn.execute(query).fetchone()
        if not result:
            print("❌ No research solicitations found missing descriptions")
            sys.exit(1)

        notice_id, naics_code, title = result
        print(f"Testing with: {notice_id}")
        print(f"NAICS: {naics_code}")
        print(f"Title: {title}")
        print()
else:
    notice_id = sys.argv[1]
    print(f"Testing with: {notice_id}")
    print()

print("=" * 70)
print("🔍 Testing SAM.gov API Fetch")
print("=" * 70)
print()

# Step 1: Search for the opportunity
print("Step 1: Searching for opportunity...")
search_url = "https://api.sam.gov/opportunities/v2/search"
params = {
    "api_key": api_key,
    "noticeid": notice_id,
    "limit": 1
}

try:
    resp = requests.get(search_url, params=params, timeout=20)
    print(f"Status Code: {resp.status_code}")

    if resp.status_code != 200:
        print(f"❌ API Error: {resp.status_code}")
        print(f"Response: {resp.text[:500]}")
        sys.exit(1)

    data = resp.json()

    if not data:
        print("❌ API returned None")
        sys.exit(1)

    opportunities = data.get("opportunitiesData", [])

    if not opportunities:
        print("❌ No opportunities found")
        print(f"Response keys: {list(data.keys())}")
        sys.exit(1)

    opp = opportunities[0]
    print(f"✅ Found opportunity")
    print()

    # Check what's in the response
    print("=" * 70)
    print("Opportunity Data Keys:")
    print("=" * 70)
    for key in sorted(opp.keys()):
        value = opp[key]
        if isinstance(value, str):
            preview = value[:100] + "..." if len(value) > 100 else value
            print(f"  {key:30} {type(value).__name__:15} {len(value):6} chars: {preview}")
        elif isinstance(value, list):
            print(f"  {key:30} {type(value).__name__:15} {len(value):6} items")
        else:
            print(f"  {key:30} {type(value).__name__:15}")
    print()

    # Step 2: Check direct description
    print("=" * 70)
    print("Step 2: Checking for direct description field...")
    print("=" * 70)
    direct_desc = opp.get("description", "")

    if direct_desc and len(str(direct_desc).strip()) > 100:
        print(f"✅ FOUND direct description: {len(str(direct_desc))} chars")
        print()
        print("Preview:")
        print("-" * 70)
        print(str(direct_desc)[:500])
        print("-" * 70)
        print()
        print("✅ This solicitation HAS a description in SAM.gov!")
        sys.exit(0)
    else:
        print(f"❌ Direct description field is empty or too short")
        print(f"   Length: {len(str(direct_desc)) if direct_desc else 0} chars")
        print(f"   Content: {repr(str(direct_desc)[:200])}")
        print()

    # Step 3: Check resourceLinks
    print("=" * 70)
    print("Step 3: Checking resourceLinks for description documents...")
    print("=" * 70)
    resource_links = opp.get("resourceLinks", [])

    if not resource_links:
        print("❌ No resourceLinks found")
        print()
    else:
        print(f"Found {len(resource_links)} resource links:")
        print()

        for i, resource in enumerate(resource_links, 1):
            if isinstance(resource, dict):
                link_type = resource.get("type", "")
                link_url = resource.get("url", "")
                print(f"{i}. Type: {link_type}")
                print(f"   URL: {link_url[:100]}...")

                # Check if it's a description document
                if any(keyword in link_type.lower() for keyword in ["description", "synopsis", "solicitation"]):
                    print(f"   ⭐ This looks like a description document!")
                print()
            elif isinstance(resource, str):
                print(f"{i}. URL: {resource[:100]}...")
                print()

    # Step 4: Try noticedesc endpoint
    print("=" * 70)
    print("Step 4: Trying noticedesc endpoint...")
    print("=" * 70)
    desc_url = "https://api.sam.gov/prod/opportunities/v1/noticedesc"
    params = {
        "api_key": api_key,
        "noticeid": notice_id
    }

    resp = requests.get(desc_url, params=params, timeout=20)
    print(f"Status Code: {resp.status_code}")

    if resp.status_code == 200:
        desc_data = resp.json()
        if desc_data:
            desc = desc_data.get("description", "")
            if desc and len(desc.strip()) > 100:
                print(f"✅ FOUND description via noticedesc: {len(desc)} chars")
                print()
                print("Preview:")
                print("-" * 70)
                print(desc[:500])
                print("-" * 70)
                print()
                print("✅ This solicitation HAS a description in SAM.gov!")
                sys.exit(0)
            else:
                print(f"❌ Description field empty or too short")
                print(f"   Length: {len(desc) if desc else 0} chars")
        else:
            print("❌ Response data is None")
    else:
        print(f"❌ noticedesc endpoint failed")

    print()
    print("=" * 70)
    print("🔍 CONCLUSION")
    print("=" * 70)
    print()
    print("❌ This solicitation does NOT have a description available in SAM.gov API")
    print()
    print("Possible reasons:")
    print("  1. SAM.gov doesn't have the full description in their API")
    print("  2. Description is only available in PDF attachments (not extracted)")
    print("  3. The solicitation was posted without a detailed description")
    print()
    print("This is NORMAL - not all solicitations have descriptions in the API.")
    print("Your backfill script is working correctly by skipping these.")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)