import os, sys, time
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(p))

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sqltext

DB_URL = os.environ.get("SUPABASE_DB_URL", "sqlite:///app.db")

_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r", "").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]

if not SAM_KEYS:
    print("❌ ERROR: No SAM_KEYS configured!")
    sys.exit(1)

print(f"SAM_KEYS configured: {len(SAM_KEYS)} key(s)")

BATCH_SIZE = int(os.environ.get("BACKFILL_BATCH", "50"))
WORKERS = int(os.environ.get("BACKFILL_WORKERS", "3"))
REQUEST_TIMEOUT = 30  # Increased timeout
BATCH_DELAY = float(os.environ.get("BATCH_DELAY", "2.0"))  # Reduced delay
MAX_UPDATES = int(os.environ.get("BACKFILL_MAX", "500"))

pg_opts = {}
if DB_URL.startswith("postgresql"):
    pg_opts["connect_args"] = {"connect_timeout": 10}
engine = create_engine(DB_URL, pool_pre_ping=True, **pg_opts)

current_key_idx = 0
rate_limited_keys = set()


def get_next_api_key() -> Optional[str]:
    global current_key_idx
    if not SAM_KEYS or len(rate_limited_keys) >= len(SAM_KEYS):
        return None

    attempts = 0
    while attempts < len(SAM_KEYS):
        key = SAM_KEYS[current_key_idx % len(SAM_KEYS)]
        current_key_idx += 1
        if key not in rate_limited_keys:
            return key
        attempts += 1
    return None


def mark_key_rate_limited(key: str):
    rate_limited_keys.add(key)
    print(f"  [RATE LIMIT] Key {len(rate_limited_keys)}/{len(SAM_KEYS)} exhausted")


def reset_rate_limited_keys():
    global rate_limited_keys
    if rate_limited_keys:
        print(f"  [RESET] Clearing rate limit flags")
        rate_limited_keys = set()


def fetch_sam_description(notice_id: str, debug: bool = False) -> Tuple[str, Optional[str]]:
    """
    Fetch description from SAM.gov using multiple strategies:
    1. Check fullParent field (often has full text)
    2. Check additionalInfoLink
    3. Try archived/active notice endpoints
    """
    api_key = get_next_api_key()
    if not api_key:
        return (notice_id, None)

    try:
        # Strategy 1: Get opportunity and check multiple fields
        search_url = "https://api.sam.gov/opportunities/v2/search"
        params = {
            "api_key": api_key,
            "noticeid": notice_id,
            "limit": 1
        }

        resp = requests.get(search_url, params=params, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 429:
            mark_key_rate_limited(api_key)
            return (notice_id, None)

        if resp.status_code != 200:
            if debug:
                print(f"  [{resp.status_code}] {notice_id}")
            return (notice_id, None)

        data = resp.json()
        if not data or not data.get("opportunitiesData"):
            return (notice_id, None)

        opp = data["opportunitiesData"][0]

        # Try multiple fields that might contain description text
        # Priority order: fullParent > description > additionalInfoLink

        # 1. Check fullParent field (often has complete text)
        full_parent = opp.get("fullParent")
        if full_parent and isinstance(full_parent, dict):
            desc_text = full_parent.get("description", "")
            if desc_text and len(str(desc_text).strip()) > 50:
                if debug:
                    print(f"  [FULLPARENT] {notice_id}: {len(desc_text)} chars")
                return (notice_id, str(desc_text).strip())

        # 2. Check direct description (if it's actual text, not URL)
        description = opp.get("description", "")
        if description and not str(description).startswith("http"):
            if len(str(description).strip()) > 50:
                if debug:
                    print(f"  [DIRECT] {notice_id}: {len(description)} chars")
                return (notice_id, str(description).strip())

        # 3. Check additionalInfoLink
        additional_info = opp.get("additionalInfoLink", "")
        if additional_info and len(str(additional_info).strip()) > 50:
            if not str(additional_info).startswith("http"):
                if debug:
                    print(f"  [ADDINFO] {notice_id}: {len(additional_info)} chars")
                return (notice_id, str(additional_info).strip())

        # 4. Build description from other fields
        synthetic_parts = []

        # Add title if we don't have it
        title = opp.get("title", "")
        if title and len(title) > 20:
            synthetic_parts.append(f"Title: {title}")

        # Add classification info
        naics = opp.get("naicsCode", "")
        if naics:
            synthetic_parts.append(f"NAICS Code: {naics}")

        # Add point of contact info
        poc_name = opp.get("pointOfContact", [{}])[0].get("fullName", "") if opp.get("pointOfContact") else ""
        if poc_name:
            synthetic_parts.append(f"Point of Contact: {poc_name}")

        # Add place of performance
        place = opp.get("placeOfPerformance", {})
        if place:
            city = place.get("city", {}).get("name", "")
            state = place.get("state", {}).get("name", "")
            if city or state:
                synthetic_parts.append(f"Location: {city}, {state}".strip(", "))

        # Add set aside info
        set_aside = opp.get("typeOfSetAsideDescription", "")
        if set_aside:
            synthetic_parts.append(f"Set-Aside: {set_aside}")

        # Add response deadline
        response_date = opp.get("responseDeadLine", "")
        if response_date:
            synthetic_parts.append(f"Response Deadline: {response_date}")

        if len(synthetic_parts) >= 3:  # At least 3 pieces of info
            synthetic_desc = " | ".join(synthetic_parts)
            if debug:
                print(f"  [SYNTHETIC] {notice_id}: {len(synthetic_desc)} chars")
            return (notice_id, synthetic_desc)

        if debug:
            print(f"  [SKIP] {notice_id}: No description available")
        return (notice_id, None)

    except requests.exceptions.Timeout:
        if debug:
            print(f"  [TIMEOUT] {notice_id}")
        return (notice_id, None)
    except Exception as e:
        if debug:
            print(f"  [ERROR] {notice_id}: {e}")
        return (notice_id, None)


def fetch_missing(limit: int) -> pd.DataFrame:
    """
    Fetch solicitations that need descriptions backfilled.
    Orders by most recently pulled first (records with pulled_at dates come first).
    """
    sql = sqltext("""
        SELECT notice_id, title, link, naics_code, pulled_at
        FROM solicitationraw
        WHERE (description IS NULL OR description = '' OR description LIKE 'http%')
        ORDER BY pulled_at DESC NULLS LAST, posted_date DESC NULLS LAST
        LIMIT :lim
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn, params={"lim": limit})


def update_descriptions_batch(updates: list):
    if not updates:
        return

    # Filter out URLs
    valid_updates = [u for u in updates if not u["desc"].startswith("http")]

    if not valid_updates:
        return

    sql = sa.text("UPDATE solicitationraw SET description = :desc WHERE notice_id = :nid")
    with engine.begin() as conn:
        conn.execute(sql, valid_updates)


def main():
    print("=" * 70)
    print("📝 Solicitation Description Backfill v3")
    print("=" * 70)
    print("Backfilling ALL missing descriptions (most recent first)")
    print(f"Config: BATCH={BATCH_SIZE}, WORKERS={WORKERS}, MAX={MAX_UPDATES}")
    print("=" * 70)
    print()

    DEBUG_MODE = os.environ.get("BACKFILL_DEBUG", "false").lower() == "true"

    total_updated = 0
    total_skipped = 0
    batches = 0

    while total_updated < MAX_UPDATES:
        try:
            df = fetch_missing(limit=BATCH_SIZE)
        except Exception as e:
            print(f"❌ DB fetch failed: {e}")
            sys.exit(2)

        if df.empty:
            print(f"✅ No more solicitations need descriptions")
            break

        print(f"\nBatch {batches + 1}: Processing {len(df)} solicitations...")

        # Show pulled_at date range for this batch
        if 'pulled_at' in df.columns and not df['pulled_at'].isna().all():
            pulled_dates = df['pulled_at'].dropna()
            if not pulled_dates.empty:
                print(f"  Pulled dates: {pulled_dates.min()} to {pulled_dates.max()}")

            # Show first 3 records with their pulled_at dates
            print(f"  Sample records:")
            for idx, row in df.head(3).iterrows():
                nid_short = str(row['notice_id'])[:12]
                pulled = str(row.get('pulled_at', 'NULL'))[:19] if pd.notna(row.get('pulled_at')) else 'NULL'
                print(f"    {nid_short}... pulled: {pulled}")

        naics_counts = df['naics_code'].value_counts()
        print(f"  NAICS: {dict(list(naics_counts.items())[:5])}")

        updates = []
        skipped_count = 0

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_to_notice = {
                executor.submit(fetch_sam_description, str(row["notice_id"]), DEBUG_MODE): str(row["notice_id"])
                for _, row in df.iterrows()
            }

            for future in as_completed(future_to_notice):
                if total_updated >= MAX_UPDATES:
                    break

                try:
                    notice_id, description = future.result()

                    if description:
                        updates.append({"desc": description, "nid": notice_id})
                        total_updated += 1

                        # Show preview
                        preview = description[:60] + "..." if len(description) > 60 else description
                        print(f"  ✅ [{total_updated}] {notice_id[:8]}... ({len(description)} chars)")

                        if total_updated <= 3:  # Show first few
                            print(f"      {preview}")
                    else:
                        skipped_count += 1
                        total_skipped += 1

                except Exception as e:
                    notice_id = future_to_notice[future]
                    print(f"  ❌ {notice_id[:8]}... error: {str(e)[:50]}")
                    skipped_count += 1

        print(f"  📊 Batch results: {len(updates)} updated, {skipped_count} skipped")

        if updates:
            try:
                update_descriptions_batch(updates)
                print(f"  💾 Saved to database")
            except Exception as e:
                print(f"  ❌ DB error: {e}")

        batches += 1

        if len(rate_limited_keys) >= len(SAM_KEYS):
            print(f"\n⚠️  All API keys rate limited!")
            print(f"Results: {total_updated} updated, {total_skipped} skipped")
            break

        if total_updated < MAX_UPDATES and not df.empty:
            time.sleep(BATCH_DELAY)
            if batches % 5 == 0:
                reset_rate_limited_keys()

    print()
    print("=" * 70)
    print(f"✅ Complete: {total_updated} updated, {total_skipped} skipped, {batches} batches")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Crashed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)