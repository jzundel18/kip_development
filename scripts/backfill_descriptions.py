import os, sys, time
from pathlib import Path
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Note: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Will use existing environment variables.")

# ---------- Make local modules importable ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(p))

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.sql import text as sqltext

# ---------- Env / secrets ----------
DB_URL = os.environ.get("SUPABASE_DB_URL", "sqlite:///app.db")

# SAM_KEYS: accept comma/newline separated
_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r", "").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]

if not SAM_KEYS:
    print("WARNING: No SAM_KEYS configured! Set SAM_KEYS environment variable.")
    print("Example: SAM_KEYS='key1,key2,key3'")
else:
    print(f"SAM_KEYS configured: {len(SAM_KEYS)} key(s)")
    for i, key in enumerate(SAM_KEYS, 1):
        preview = f"{key[:15]}...{key[-10:]}" if len(key) > 25 else key[:20] + "..."
        print(f"  Key {i}: {preview}")

# Backfill knobs (env overrideable)
BATCH_SIZE = int(os.environ.get("BACKFILL_BATCH", "50"))  # rows per DB batch
WORKERS = int(os.environ.get("BACKFILL_WORKERS", "2"))  # parallel workers
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "20"))  # API timeout - increased from 10 to 20
BATCH_DELAY = float(os.environ.get("BATCH_DELAY", "3.0"))  # seconds to wait between batches
MAX_UPDATES = int(os.environ.get("BACKFILL_MAX", "500"))  # safety cap per run

# --- DB (with timeout for Postgres) ---
pg_opts = {}
if DB_URL.startswith("postgresql"):
    pg_opts["connect_args"] = {"connect_timeout": 10}
engine = create_engine(DB_URL, pool_pre_ping=True, **pg_opts)

# Rotating API key index
current_key_idx = 0
rate_limited_keys = set()  # Track which keys are rate limited


def get_next_api_key() -> Optional[str]:
    """Round-robin through available API keys."""
    global current_key_idx
    if not SAM_KEYS or len(SAM_KEYS) == 0:
        return None

    # If all keys are rate limited, return None
    if len(rate_limited_keys) >= len(SAM_KEYS):
        return None

    # Try to find a non-rate-limited key
    attempts = 0
    while attempts < len(SAM_KEYS):
        key = SAM_KEYS[current_key_idx % len(SAM_KEYS)]
        current_key_idx += 1

        if key not in rate_limited_keys:
            return key

        attempts += 1

    return None


def mark_key_rate_limited(key: str):
    """Mark a key as rate limited."""
    rate_limited_keys.add(key)
    print(
        f"  [RATE LIMIT] Key marked as rate limited: {key[:15]}... ({len(rate_limited_keys)}/{len(SAM_KEYS)} keys exhausted)")


def reset_rate_limited_keys():
    """Reset rate limit tracking (call between batches)."""
    global rate_limited_keys
    if rate_limited_keys:
        print(f"  [RESET] Clearing rate limit flags for {len(rate_limited_keys)} keys")
        rate_limited_keys = set()


def fetch_sam_description(notice_id: str, debug: bool = False) -> Tuple[str, Optional[str]]:
    """
    Fetch the actual full description text from SAM.gov.

    SAM.gov has the full description in the 'resourceLinks' attachments,
    specifically in the 'Description' or 'Combined Synopsis/Solicitation' documents.

    Returns (notice_id, description_text) or (notice_id, None) on failure.
    """
    api_key = get_next_api_key()
    if not api_key:
        print(f"  [NO API KEY] {notice_id}")
        return (notice_id, None)

    max_retries = 3
    retry_delay = 2  # Start with 2 seconds

    for attempt in range(max_retries):
        # Step 1: Get the opportunity details to find the description attachment
        try:
            search_url = f"https://api.sam.gov/opportunities/v2/search"
            params = {
                "api_key": api_key,
                "noticeid": notice_id,
                "limit": 1
            }
            resp = requests.get(search_url, params=params, timeout=REQUEST_TIMEOUT)

            if debug:
                print(f"  [SEARCH DEBUG] {notice_id} - Status: {resp.status_code} (attempt {attempt + 1})")

            if resp.status_code == 429:
                # Rate limited - mark this key and try a different one
                mark_key_rate_limited(api_key)

                if attempt < max_retries - 1:
                    api_key = get_next_api_key()
                    if not api_key:
                        print(f"  [ALL KEYS EXHAUSTED] All API keys are rate limited for {notice_id}")
                        return (notice_id, None)

                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    if debug:
                        print(f"  [429] Rate limited, waiting {wait_time}s and trying different key...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [429] Rate limited after {max_retries} attempts for {notice_id}")
                    return (notice_id, None)

            elif resp.status_code == 403:
                print(f"  [403] Auth failed for {notice_id}")
                return (notice_id, None)

            elif resp.status_code != 200:
                print(f"  [{resp.status_code}] Failed for {notice_id}")
                return (notice_id, None)

            # Success - process the response
            data = resp.json()

            # FIX 1: Check if data is None
            if data is None:
                if debug:
                    print(f"  [SEARCH DEBUG] API returned None data for {notice_id}")
                return (notice_id, None)

            opportunities = data.get("opportunitiesData", [])

            # FIX 2: Ensure opportunities is a list
            if opportunities is None:
                opportunities = []

            if not opportunities or len(opportunities) == 0:
                if debug:
                    print(f"  [SEARCH DEBUG] No opportunities found for {notice_id}")
                return (notice_id, None)

            opp = opportunities[0]

            # FIX 3: Check if opp is None
            if opp is None:
                if debug:
                    print(f"  [SEARCH DEBUG] Opportunity object is None for {notice_id}")
                return (notice_id, None)

            if debug:
                print(f"  [SEARCH DEBUG] Opportunity keys: {list(opp.keys())[:30]}")

            # Step 2: Check if description is directly in the response (sometimes it is)
            direct_desc = opp.get("description", "")

            if debug:
                print(f"  [DIRECT DESC] Raw description type: {type(direct_desc)}")
                print(f"  [DIRECT DESC] Raw description length: {len(str(direct_desc)) if direct_desc else 0}")
                if direct_desc:
                    preview = str(direct_desc)[:200] + "..." if len(str(direct_desc)) > 200 else str(direct_desc)
                    print(f"  [DIRECT DESC] Content preview: {preview}")

            if direct_desc and len(str(direct_desc).strip()) > 100:  # Only if substantial
                if debug:
                    print(f"  [DIRECT DESC] ✓ Using direct description: {len(str(direct_desc))} chars")
                return (notice_id, str(direct_desc).strip())

            # Step 3: Look for resourceLinks (attachments) that contain the description
            resource_links = opp.get("resourceLinks", [])

            # FIX 4: Ensure resource_links is actually a list, not None
            if resource_links is None:
                resource_links = []

            if debug:
                print(f"  [RESOURCES DEBUG] Found {len(resource_links)} resource links")
                if resource_links and len(resource_links) > 0:
                    print(f"  [RESOURCES DEBUG] First resource type: {type(resource_links[0])}")

            # Look for description documents
            description_urls = []
            for resource in resource_links:
                # FIX 5: Check if resource is None
                if resource is None:
                    continue

                # Handle both string URLs and dict objects
                if isinstance(resource, str):
                    # It's just a URL string
                    link_url = resource
                    if debug:
                        print(f"  [RESOURCE] URL: {link_url[:100] if link_url else 'None'}")

                    # Try to fetch if it looks like a description document
                    description_urls.append(link_url)

                elif isinstance(resource, dict):
                    # It's a dict with type and url
                    link_type = resource.get("type", "")
                    link_url = resource.get("url", "")

                    # FIX 6: Ensure link_type is not None
                    if link_type is None:
                        link_type = ""

                    link_type = link_type.lower()

                    if debug:
                        print(f"  [RESOURCE] Type: {link_type}, URL: {link_url[:80] if link_url else 'None'}")

                    # Common description document types
                    if any(keyword in link_type for keyword in ["description", "synopsis", "solicitation", "combined"]):
                        if link_url:  # Only add if URL exists
                            description_urls.append(link_url)

            # Step 4: Fetch the description document
            for desc_url in description_urls[:2]:  # Try first 2 description documents
                if not desc_url:  # Skip if URL is None or empty
                    continue

                try:
                    if debug:
                        print(f"  [FETCHING] Description from: {desc_url[:80]}...")

                    # Small delay before fetching documents
                    time.sleep(0.3)

                    # Fetch the document
                    doc_resp = requests.get(desc_url, timeout=REQUEST_TIMEOUT)

                    if doc_resp.status_code == 200:
                        content_type = doc_resp.headers.get("Content-Type", "").lower()

                        # Handle text/plain or text/html
                        if "text" in content_type or "html" in content_type:
                            description = doc_resp.text.strip()

                            # Clean HTML if needed
                            if "html" in content_type:
                                from html.parser import HTMLParser

                                class HTMLTextExtractor(HTMLParser):
                                    def __init__(self):
                                        super().__init__()
                                        self.text = []

                                    def handle_data(self, data):
                                        self.text.append(data)

                                    def get_text(self):
                                        return ' '.join(self.text)

                                parser = HTMLTextExtractor()
                                parser.feed(description)
                                description = parser.get_text()

                            if len(description) > 100:  # Substantial content
                                if debug:
                                    print(f"  [SUCCESS] Extracted {len(description)} chars from document")
                                return (notice_id, description.strip())

                        elif "pdf" in content_type:
                            if debug:
                                print(f"  [SKIP] PDF document - would need PDF parsing")
                            # Could add PDF parsing here with PyPDF2 if needed

                except Exception as e:
                    if debug:
                        print(f"  [FETCH ERROR] {desc_url[:50]}: {e}")

            # Step 5: Try the notice description endpoint as fallback
            try:
                time.sleep(0.3)  # Small delay
                desc_url = f"https://api.sam.gov/prod/opportunities/v1/noticedesc"
                params = {
                    "api_key": api_key,
                    "noticeid": notice_id
                }
                resp = requests.get(desc_url, params=params, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 200:
                    desc_data = resp.json()

                    # FIX 7: Check if desc_data is None
                    if desc_data is not None:
                        desc = desc_data.get("description", "")

                        if desc and len(desc.strip()) > 100:
                            if debug:
                                print(f"  [NOTICEDESC] Found description: {len(desc)} chars")
                            return (notice_id, desc.strip())
                elif resp.status_code == 429:
                    if debug:
                        print(f"  [NOTICEDESC 429] Rate limited")
            except Exception as e:
                if debug:
                    print(f"  [NOTICEDESC ERROR] {e}")

            # Step 6: Last resort - try getting from the web page URL
            ui_link = opp.get("uiLink") or opp.get("link", "")
            if ui_link and "sam.gov" in ui_link:
                try:
                    if debug:
                        print(f"  [WEB SCRAPE] Attempting to scrape: {ui_link[:80]}")

                    time.sleep(0.3)  # Small delay
                    web_resp = requests.get(ui_link, timeout=REQUEST_TIMEOUT, headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    })

                    if web_resp.status_code == 200:
                        # Look for description in the HTML
                        html_content = web_resp.text

                        # Find description section (common patterns in SAM.gov pages)
                        import re

                        # Try multiple patterns
                        patterns = [
                            r'<div[^>]*class="[^"]*description[^"]*"[^>]*>(.*?)</div>',
                            r'<section[^>]*id="description"[^>]*>(.*?)</section>',
                            r'Description:</strong>\s*(.*?)</(?:div|p|section)>',
                        ]

                        for pattern in patterns:
                            matches = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)
                            if matches:
                                # Clean HTML tags
                                desc = re.sub(r'<[^>]+>', ' ', matches[0])
                                desc = re.sub(r'\s+', ' ', desc).strip()

                                if len(desc) > 100:
                                    if debug:
                                        print(f"  [WEB SCRAPE SUCCESS] Extracted {len(desc)} chars")
                                    return (notice_id, desc)

                except Exception as e:
                    if debug:
                        print(f"  [WEB SCRAPE ERROR] {e}")

            # If we got here, no description found
            if debug:
                print(f"  [NO DESC] Could not find description for {notice_id}")

            return (notice_id, None)

        except requests.exceptions.Timeout as e:
            print(
                f"  [ERROR] {notice_id}: HTTPSConnectionPool(host='api.sam.gov', port=443): Read timed out. (read timeout={REQUEST_TIMEOUT})")
            return (notice_id, None)
        except Exception as e:
            print(f"  [ERROR] {notice_id}: {e}")
            import traceback
            if debug:
                traceback.print_exc()
            return (notice_id, None)


def fetch_missing(limit: int) -> pd.DataFrame:
    """
    Return up to `limit` rows with missing/empty descriptions.
    Ordered by most recently pulled_at first (newest solicitations first).
    """
    sql = sqltext("""
        SELECT notice_id, title, link
        FROM solicitationraw
        WHERE (description IS NULL OR description = '')
        ORDER BY pulled_at DESC NULLS LAST, posted_date DESC NULLS LAST
        LIMIT :lim
    """)
    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn, params={"lim": limit})


def update_descriptions_batch(updates: list):
    """Batch update multiple descriptions at once for efficiency."""
    if not updates:
        return

    sql = sa.text("UPDATE solicitationraw SET description = :desc WHERE notice_id = :nid")
    with engine.begin() as conn:
        conn.execute(sql, updates)


def main():
    print("backfill_descriptions.py: starting…", flush=True)
    print(
        f"Config: BATCH_SIZE={BATCH_SIZE}, WORKERS={WORKERS}, MAX_UPDATES={MAX_UPDATES}, BATCH_DELAY={BATCH_DELAY}s, TIMEOUT={REQUEST_TIMEOUT}s")

    # Enable debug for environment variable
    DEBUG_MODE = os.environ.get("BACKFILL_DEBUG", "false").lower() == "true"

    total_updated = 0
    batches = 0

    while total_updated < MAX_UPDATES:
        try:
            df = fetch_missing(limit=BATCH_SIZE)
        except Exception as e:
            print("DB fetch failed:", repr(e))
            sys.exit(2)

        if df.empty:
            print(f"No rows missing descriptions. Total updated this run: {total_updated}")
            break

        print(f"\nBatch {batches + 1}: processing {len(df)} rows in parallel…")

        # Debug first record in first batch
        if batches == 0 and len(df) > 0 and not DEBUG_MODE:
            print("\n=== DEBUG MODE: Testing first record ===")
            test_notice = str(df.iloc[0]["notice_id"])
            print(f"Testing notice_id: {test_notice}")
            result = fetch_sam_description(test_notice, debug=True)
            print(f"Result: {result[0]}, Description present: {result[1] is not None}")
            if result[1]:
                print(f"Description preview: {result[1][:200]}...")
            print("=== END DEBUG ===\n")

        # Process in parallel using ThreadPoolExecutor
        updates = []
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            # Submit all tasks
            future_to_notice = {
                executor.submit(fetch_sam_description, str(row["notice_id"]), DEBUG_MODE): str(row["notice_id"])
                for _, row in df.iterrows()
            }

            # Collect results as they complete
            for future in as_completed(future_to_notice):
                if total_updated >= MAX_UPDATES:
                    break

                try:
                    notice_id, description = future.result()

                    if description:
                        updates.append({"desc": description, "nid": notice_id})
                        total_updated += 1
                        desc_preview = description[:100] + "..." if len(description) > 100 else description
                        print(f"  [{total_updated}] {notice_id} — fetched ({len(description)} chars): {desc_preview}")
                    else:
                        print(f"  [skip] {notice_id} — no description available")

                except Exception as e:
                    notice_id = future_to_notice[future]
                    print(f"  [error] {notice_id} — {repr(e)}")

        # Batch update all successful fetches
        if updates:
            try:
                update_descriptions_batch(updates)
                print(f"  ✓ Batch committed {len(updates)} updates to DB")

                # Verify first update was saved
                if batches == 0 and len(updates) > 0:
                    verify_sql = sqltext(
                        "SELECT notice_id, LEFT(description, 100) as desc_preview FROM solicitationraw WHERE notice_id = :nid")
                    with engine.connect() as conn:
                        verify_result = pd.read_sql_query(verify_sql, conn, params={"nid": updates[0]["nid"]})
                        if not verify_result.empty:
                            print(f"  ✓ VERIFIED in DB: {verify_result.iloc[0]['desc_preview']}...")
                        else:
                            print(f"  ⚠ WARNING: Could not verify update for {updates[0]['nid']}")

            except Exception as e:
                print(f"  [DB error] Failed to commit batch: {e}")
                import traceback
                traceback.print_exc()

        batches += 1

        # Check if all keys are rate limited
        if len(rate_limited_keys) >= len(SAM_KEYS):
            print(f"\n⚠ All {len(SAM_KEYS)} API keys are rate limited!")
            print(f"Stopping early. {total_updated} descriptions updated so far.")
            print("Please wait for rate limits to reset (typically 1 hour) and run again.")
            break

        # Wait between batches to avoid rate limiting
        # Also reset rate limit tracking (limits reset over time)
        if total_updated < MAX_UPDATES and not df.empty:
            print(f"  Waiting {BATCH_DELAY}s before next batch...")
            time.sleep(BATCH_DELAY)
            # Reset rate limit flags - they may have cleared during the delay
            if batches % 5 == 0:  # Reset every 5 batches
                reset_rate_limited_keys()

    print(f"\nBackfill complete. {total_updated} descriptions updated in {batches} batch(es).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Backfill script crashed:", repr(e))
        sys.exit(1)