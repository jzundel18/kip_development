#!/usr/bin/env python3
# backfill_pop.py
# Backfill ONLY Place-of-Performance fields for solicitationraw rows that are missing them.
# - No description fetching
# - Uses SAM.gov v2 detail by noticeid
# - Key rotation + gentle throttling
# - Safe updates, optional dry-run

from __future__ import annotations
import os
import re
import time
import json
import html
import argparse
from typing import Any, Dict, List, Optional, Tuple

import requests
import sqlalchemy as sa
from sqlalchemy.engine import Engine

# --------------------------
# 🔐 CONFIG: Keys & Database
# --------------------------

# Hardcode your SAM keys here (comma-separated list)
SAM_KEYS: List[str] = [
    "2WWQnuvYVj7cI3qozS5xC0Y2SgGITC7NGWtmqebq",
    "qV4oW8tKis4kinR7oXIBlnTDlJx9Tyf4Se8Tmmmx"
]

HARD_LIMIT = 50

# Prefer env var if present; otherwise you can hardcode your Supabase/Postgres URL.
# Example:
# DB_URL = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
DB_URL = (
    "postgresql+psycopg2://postgres.ceemspukffoygxazsvix:Moolah123%21%21%21@aws-1-us-west-1.pooler.supabase.com:6543/postgres?sslmode=require"
)

# HTTP
SAM_SEARCH_URL_V2 = "https://api.sam.gov/prod/opportunities/v2/search"

USER_AGENT = "kip_pop_backfill/1.0"
HTTP_TIMEOUT = 30
SLEEP_BETWEEN_CALLS = 0.25  # seconds (be polite)


# --------------------------
# Small utils
# --------------------------

def _mask_key(k: str) -> str:
    return f"...{k[-4:]}" if k else "(none)"


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _host_from_link(u: str) -> str:
    try:
        from urllib.parse import urlparse
        return (urlparse(u).netloc or "").lower()
    except Exception:
        return ""


def _get_engine() -> Engine:
    # If you want to hardcode Postgres, uncomment and set the line below:
    # db_url = "postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
    db_url = DB_URL
    if db_url.startswith("postgresql+psycopg2://"):
        return sa.create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=2,
            connect_args={"sslmode": "require"},
        )
    return sa.create_engine(db_url, pool_pre_ping=True)

# --------------------------
# SAM.gov detail fetch (v2)
# --------------------------


def _rotate_keys(keys: List[str]):
    while True:
        for k in keys:
            yield k


def _http_get(url: str, params: dict, key: str, timeout: int = HTTP_TIMEOUT) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    return requests.get(url, params={**params, "api_key": key}, headers=headers, timeout=timeout)


def fetch_notice_detail_v2(notice_id: str, api_keys: List[str]) -> Dict[str, Any]:
    """
    Fetch a single record using the v2 search endpoint by noticeid.
    Returns {} on failure.
    """
    if not notice_id or not api_keys:
        return {}
    rot = _rotate_keys(api_keys)
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(SAM_SEARCH_URL_V2, {
                          "noticeid": notice_id, "limit": 1}, key)
            if r.status_code == 429:
                time.sleep(1.0)
                continue
            r.raise_for_status()
            data = r.json() if r.headers.get("Content-Type",
                                             "").startswith("application/json") else {}
            items = data.get("opportunitiesData") or data.get("data") or []
            if isinstance(items, list) and items:
                return items[0]
            return {}
        except Exception:
            time.sleep(0.5)
            continue
    return {}

# --------------------------
# POP extraction
# --------------------------


def _extract_pop_from_obj(obj: dict | None) -> dict:
    """
    Try several common containers/field names to get city/state/zip/country.
    Returns dict with possibly empty strings.
    """
    if not isinstance(obj, dict):
        return {}

    candidates = []
    for k in (
        "placeOfPerformance",
        "place_of_performance",
        "placeOfPerformanceAddress",
        "primaryPlaceOfPerformance",
        "popAddress",
        "placeOfPerformanceLocation",
        "place_of_performance_location",
        "placeOfPerformanceCityState",
    ):
        v = obj.get(k)
        if isinstance(v, dict):
            candidates.append(v)

    # Sometimes appears in lists
    for k in ("addresses", "locations", "placeOfPerformanceAddresses"):
        v = obj.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            candidates.append(v[0])

    # allow using the object itself if it looks like an address
    candidates.append(obj)

    for c in candidates:
        if not isinstance(c, dict):
            continue
        city = c.get("city") or c.get("cityName") or (
            c.get("address") or {}).get("city")
        state = (
            c.get("state") or c.get("stateCode") or (
                c.get("address") or {}).get("state")
            or c.get("stateProvince") or c.get("stateProvinceCode")
        )
        zipc = (
            c.get("zip") or c.get("zipCode") or c.get("postalCode")
            or (c.get("address") or {}).get("postalCode")
        )
        country = (
            c.get("country") or c.get("countryCode") or c.get("countryName")
            or (c.get("address") or {}).get("country")
        )
        if city or state or zipc or country:
            return {
                "pop_city": _s(city).strip(),
                "pop_state": _s(state).strip(),
                "pop_zip": _s(zipc).strip(),
                "pop_country": _s(country).strip(),
            }

    return {}


def _build_pop_raw(pop: dict) -> str:
    parts = []
    if pop.get("pop_city"):
        parts.append(pop["pop_city"])
    if pop.get("pop_state"):
        parts.append(pop["pop_state"])
    raw = ", ".join(parts)
    if pop.get("pop_zip"):
        raw = (raw + f" {pop['pop_zip']}".rstrip()).strip()
    if pop.get("pop_country") and pop["pop_country"].upper() not in ("USA", "US", "UNITED STATES", "UNITED-STATES"):
        raw = (raw + f" ({pop['pop_country']})").strip()
    return raw


def extract_place_of_performance(rec: dict) -> dict:
    """
    Pulls POP from the v2 detail payload (preferred) and, if not found,
    tries some shallow fallbacks on the same object.
    Returns dict with keys: pop_city, pop_state, pop_zip, pop_country, pop_raw
    """
    pop = _extract_pop_from_obj(rec)

    # If still nothing, look for obvious top-level fields (rare)
    if not (pop.get("pop_city") or pop.get("pop_state") or pop.get("pop_zip") or pop.get("pop_country")):
        city = rec.get("city")
        state = rec.get("state") or rec.get("stateCode")
        zipc = rec.get("zip") or rec.get("zipCode") or rec.get("postalCode")
        country = rec.get("country") or rec.get(
            "countryCode") or rec.get("countryName")
        if city or state or zipc or country:
            pop = {
                "pop_city": _s(city).strip(),
                "pop_state": _s(state).strip(),
                "pop_zip": _s(zipc).strip(),
                "pop_country": _s(country).strip(),
            }

    # finalize
    for k in ("pop_city", "pop_state", "pop_zip", "pop_country"):
        pop.setdefault(k, "")
    pop["pop_raw"] = _build_pop_raw(pop)
    return pop

# --------------------------
# DB work
# --------------------------


SQL_SELECT_MISSING = """
SELECT notice_id
FROM solicitationraw
WHERE COALESCE(NULLIF(pop_city, ''), NULLIF(pop_state, ''), NULLIF(pop_zip, ''), NULLIF(pop_country, '')) IS NULL
ORDER BY pulled_at DESC
LIMIT :limit
"""

SQL_UPDATE_POP = """
UPDATE solicitationraw
SET pop_city = :pop_city,
    pop_state = :pop_state,
    pop_zip = :pop_zip,
    pop_country = :pop_country,
    pop_raw = :pop_raw
WHERE notice_id = :notice_id
"""


def pick_targets(conn: sa.engine.Connection, limit: int) -> List[str]:
    rows = conn.execute(sa.text(SQL_SELECT_MISSING), {
                        "limit": int(limit)}).fetchall()
    return [str(r[0]) for r in rows]


def update_pop(conn: sa.engine.Connection, notice_id: str, pop: dict) -> int:
    params = {
        "notice_id": notice_id,
        "pop_city": pop.get("pop_city", ""),
        "pop_state": pop.get("pop_state", ""),
        "pop_zip": pop.get("pop_zip", ""),
        "pop_country": pop.get("pop_country", ""),
        "pop_raw": pop.get("pop_raw", ""),
    }
    res = conn.execute(sa.text(SQL_UPDATE_POP), params)
    return res.rowcount or 0

# --------------------------
# Main
# --------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Backfill ONLY Place-of-Performance (POP) fields for missing rows.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not write to DB; just log what would change")
    args = parser.parse_args()

    if not SAM_KEYS:
        print("ERROR: No SAM keys configured. Edit SAM_KEYS in this file.")
        return

    engine = _get_engine()

    # Validate the table exists
    with engine.connect() as conn:
        try:
            conn.execute(sa.text("SELECT 1 FROM solicitationraw LIMIT 1"))
        except Exception as e:
            print("ERROR: solicitationraw table not found or DB connection failed.")
            print(e)
            return

    # Do updates inside a transaction
    with engine.begin() as conn:
        targets = pick_targets(conn, HARD_LIMIT)
        if not targets:
            print("Nothing to backfill — all rows have POP or no eligible rows found.")
            return

        print(f"Found {len(targets)} notices missing POP. Fetching v2 details…")

        updated = 0
        skipped = 0
        errors = 0

        for i, nid in enumerate(targets, start=1):
            try:
                detail = fetch_notice_detail_v2(nid, SAM_KEYS)
                if not detail:
                    skipped += 1
                    print(
                        f"[{i}/{len(targets)}] {nid}: no detail returned; skipped")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                pop = extract_place_of_performance(detail)

                if not (pop.get("pop_city") or pop.get("pop_state") or pop.get("pop_zip") or pop.get("pop_country")):
                    skipped += 1
                    print(
                        f"[{i}/{len(targets)}] {nid}: no POP fields present; skipped")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                if args.dry_run:
                    print(f"[{i}/{len(targets)}] {nid}: (dry-run) POP -> {pop}")
                else:
                    n = update_pop(conn, nid, pop)
                    updated += n
                    print(f"[{i}/{len(targets)}] {nid}: updated POP -> {pop}")

                time.sleep(SLEEP_BETWEEN_CALLS)

            except Exception as e:
                errors += 1
                print(f"[{i}/{len(targets)}] {nid}: ERROR {e}")
                time.sleep(0.6)

        print(f"Done. Updated={updated}, Skipped={skipped}, Errors={errors}")
        
if __name__ == "__main__":
    main()
