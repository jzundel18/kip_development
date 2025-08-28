#!/usr/bin/env python3
"""
Backfill place-of-performance fields for existing solicitation rows using the SAM.gov *detail* API.

What it does:
  - Ensures pop_* columns exist.
  - Finds rows where all pop_* are empty/NULL.
  - For each notice_id, calls the detail endpoint:
        https://sam.gov/api/prod/opps/v2/opportunities/{noticeId}?api_key=...
  - Extracts PoP and updates the row when found.

Run:
  python backfill_pop.py
"""

import os
import sys
import time
import json
import random
import logging
import requests
import sqlalchemy as sa
from sqlalchemy import text

# -----------------------
# Hardcoded config (as requested)
# -----------------------
# You can still override with env vars if you want.
SUPABASE_DB_URL = os.getenv(
    "SUPABASE_DB_URL",
    "postgresql+psycopg2://postgres.ceemspukffoygxazsvix:Moolah123%21%21%21@aws-1-us-west-1.pooler.supabase.com:6543/postgres?sslmode=require"
)

# rotate through these; can also set SAM_KEYS env to override
HARDCODED_SAM_KEYS = [
    "2WWQnuvYVj7cI3qozS5xC0Y2SgGITC7NGWtmqebq",
    "qV4oW8tKis4kinR7oXIBlnTDlJx9Tyf4Se8Tmmmx"
]
SAM_KEYS = [k.strip() for k in os.getenv("SAM_KEYS", "").split(",")
            if k.strip()] or HARDCODED_SAM_KEYS
if not SAM_KEYS:
    print("No SAM API keys configured.", file=sys.stderr)
    sys.exit(1)

SAM_DETAIL_URL_TMPL = "https://api.sam.gov/prod/opportunities/v2/opportunities/{notice_id}"

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# -----------------------
# DB
# -----------------------
engine = sa.create_engine(SUPABASE_DB_URL, pool_pre_ping=True, future=True)


def ensure_pop_columns():
    """Add pop_* columns if missing."""
    with engine.begin() as conn:
        # Create columns if they don't exist (TEXT is fine)
        for col, typ in [
            ("pop_city", "TEXT"),
            ("pop_state", "TEXT"),
            ("pop_zip", "TEXT"),
            ("pop_country", "TEXT"),
            ("pop_raw", "TEXT"),
        ]:
            try:
                conn.execute(
                    text(f'ALTER TABLE solicitationraw ADD COLUMN IF NOT EXISTS "{col}" {typ}'))
            except Exception:
                # Some Postgres versions lack IF NOT EXISTS for columns; try a guarded add
                try:
                    conn.execute(
                        text(f"SELECT {col} FROM solicitationraw LIMIT 1"))
                except Exception:
                    conn.execute(
                        text(f'ALTER TABLE solicitationraw ADD COLUMN "{col}" {typ}'))


def rows_needing_backfill():
    """Return list of notice_ids where all pop_* are empty/NULL."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT notice_id
            FROM solicitationraw
            WHERE COALESCE(NULLIF(TRIM(pop_city), ''), '') = ''
              AND COALESCE(NULLIF(TRIM(pop_state), ''), '') = ''
              AND COALESCE(NULLIF(TRIM(pop_zip), ''), '') = ''
              AND COALESCE(NULLIF(TRIM(pop_country), ''), '') = ''
        """)).fetchall()
    return [r[0] for r in rows if r and r[0]]

# -----------------------
# SAM helpers
# -----------------------


_seen_debug = 0


def sam_get_detail(notice_id: str, api_key: str) -> dict | None:
    """
    Fetch the *detail* record for a notice_id.
    Handles 429 rate limits by raising a sentinel exception.
    """
    global _seen_debug
    url = SAM_DETAIL_URL_TMPL.format(notice_id=notice_id)
    params = {"api_key": api_key}

    r = requests.get(url, params=params, timeout=25)
    if r.status_code == 429:
        raise requests.HTTPError("RATE_LIMIT")
    if r.status_code == 404:
        logging.info("Detail not found (404) for %s", notice_id)
        return None
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        logging.warning("HTTP error for %s: %s (body=%r)",
                        notice_id, e, r.text[:300])
        return None

    data = {}
    try:
        data = r.json() if r.content else {}
    except Exception:
        logging.warning("Non-JSON response for %s: %r",
                        notice_id, r.text[:300])
        return None

    # Show a quick peek at the first couple payloads
    if _seen_debug < 3:
        _seen_debug += 1
        logging.info("Got detail for %s; top-level keys: %s",
                     notice_id, ", ".join(list(data.keys())[:10]))

    return data


def _first_nonempty(*vals) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_pop(detail: dict) -> dict:
    """
    Extract place-of-performance from a detail payload.
    Returns dict with keys: pop_city, pop_state, pop_zip, pop_country, pop_raw
    """
    if not isinstance(detail, dict):
        return blank_pop()

    # Common spot:
    # detail["placeOfPerformance"]["address"] -> {city, state, zip, country}
    pop = detail.get("placeOfPerformance") or {}

    # Some payloads put the raw address fields at the same level
    addr = {}
    if isinstance(pop, dict):
        addr = pop.get("address") if isinstance(
            pop.get("address"), dict) else {}
    if not addr and isinstance(pop, dict):
        # try alternative shapes
        addr = {
            "city": pop.get("city") or pop.get("cityName"),
            "state": pop.get("state") or pop.get("stateCode") or pop.get("stateProvince"),
            "zip": pop.get("zip") or pop.get("zipCode") or pop.get("postalCode"),
            "country": pop.get("country") or pop.get("countryCode") or pop.get("countryName"),
        }

    # Top-level fallbacks (rare)
    if not any(addr.values()):
        addr = detail.get("popAddress") if isinstance(
            detail.get("popAddress"), dict) else addr

    city = _first_nonempty(addr.get("city"), addr.get("cityName"))
    state = _first_nonempty(addr.get("state"), addr.get(
        "stateCode"), addr.get("stateProvince"))
    zipc = _first_nonempty(addr.get("zip"), addr.get(
        "zipCode"), addr.get("postalCode"))
    country = _first_nonempty(addr.get("country"), addr.get(
        "countryCode"), addr.get("countryName"))

    # Some payloads store country code under placeOfPerformance.countryCode
    if not country and isinstance(pop, dict):
        country = _first_nonempty(pop.get("countryCode"), pop.get("country"))

    if not (city or state or zipc or country):
        return blank_pop()

    raw_parts = []
    if city:
        raw_parts.append(city)
    if state:
        raw_parts.append(state)
    raw = ", ".join(raw_parts)
    if zipc:
        raw = (raw + f" {zipc}").strip()
    if country and country.upper() not in ("US", "USA", "UNITED STATES", "UNITED-STATES"):
        raw = (raw + f" ({country})").strip()

    return {
        "pop_city": city,
        "pop_state": state,
        "pop_zip": zipc,
        "pop_country": country,
        "pop_raw": raw,
    }


def blank_pop() -> dict:
    return {"pop_city": "", "pop_state": "", "pop_zip": "", "pop_country": "", "pop_raw": ""}


def update_row(nid: str, pop: dict) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE solicitationraw
            SET pop_city = :city,
                pop_state = :state,
                pop_zip = :zip,
                pop_country = :country,
                pop_raw = :raw
            WHERE notice_id = :nid
        """), {
            "city": pop.get("pop_city", ""),
            "state": pop.get("pop_state", ""),
            "zip": pop.get("pop_zip", ""),
            "country": pop.get("pop_country", ""),
            "raw": pop.get("pop_raw", ""),
            "nid": nid
        })


def key_rotator():
    while True:
        for k in SAM_KEYS:
            yield k

# -----------------------
# Main
# -----------------------


def main():
    logging.info("DB: %s", SUPABASE_DB_URL)
    ensure_pop_columns()

    todo = rows_needing_backfill()
    if not todo:
        logging.info("No rows need backfill.")
        return

    logging.info("Found %d rows with empty PoP.", len(todo))

    rot = key_rotator()
    updated = 0
    skipped = 0
    failures = 0

    # Exponential backoff settings
    base_sleep = 0.2
    max_sleep = 12.0
    backoff = base_sleep

    for i, nid in enumerate(todo, 1):
        api_key = next(rot)

        try:
            detail = sam_get_detail(nid, api_key)
        except requests.HTTPError:
            # 429: back off, rotate key, and retry this nid once
            backoff = min(max_sleep, backoff * 2)  # exponential
            sleep_for = backoff + random.uniform(0, 0.5)
            logging.warning(
                "Rate limited (notice_id=%s). Sleeping %.1fs then continuing…", nid, sleep_for)
            time.sleep(sleep_for)
            # try again with next key
            api_key = next(rot)
            detail = sam_get_detail(nid, api_key)

        if not isinstance(detail, dict) or not detail:
            skipped += 1
            if i % 25 == 0:
                logging.info("Progress %d/%d (updated=%d, skipped=%d, failures=%d)",
                             i, len(todo), updated, skipped, failures)
            # small jitter so we don't hammer
            time.sleep(0.15 + random.uniform(0, 0.1))
            continue

        pop = extract_pop(detail)
        if not any(v.strip() for v in pop.values()):
            # no PoP in detail record
            skipped += 1
            if i % 25 == 0:
                logging.info("Progress %d/%d (updated=%d, skipped=%d, failures=%d)",
                             i, len(todo), updated, skipped, failures)
            time.sleep(0.15 + random.uniform(0, 0.1))
            continue

        try:
            update_row(nid, pop)
            updated += 1
            # reset backoff when successful
            backoff = base_sleep
        except Exception as e:
            failures += 1
            logging.warning("DB update failed (notice_id=%s): %s", nid, e)

        if i % 25 == 0:
            logging.info("Progress %d/%d (updated=%d, skipped=%d, failures=%d)",
                         i, len(todo), updated, skipped, failures)

        # polite pacing
        time.sleep(0.12 + random.uniform(0, 0.15))

    logging.info("Done. Updated=%d, Skipped(no PoP)=%d, Failures=%d",
                 updated, skipped, failures)


if __name__ == "__main__":
    main()
