#!/usr/bin/env python3
"""
Backfill ONLY Place-of-Performance (POP) for rows where it's missing.

- Standalone script (no app imports needed).
- Hardcoded SAM keys list below (edit SAM_KEYS).
- Uses v2 search detail by noticeid (fast) and extracts POP.
- Prints the SAM.gov public URL per row for manual verification.
"""

from __future__ import annotations
import os
import time
import json
import argparse
import re
from typing import Any, Dict, List, Optional

import requests
import sqlalchemy as sa
from sqlalchemy import text, create_engine

# ------------- EDIT THESE ----------------
SAM_KEYS = [
    "2WWQnuvYVj7cI3qozS5xC0Y2SgGITC7NGWtmqebq",
    "qV4oW8tKis4kinR7oXIBlnTDlJx9Tyf4Se8Tmmmx"
]
DB_URL = os.getenv("SUPABASE_DB_URL") or os.getenv(
    "DB_URL") or "sqlite:///app.db"
HARD_LIMIT = 20                 # how many rows to process per run
SLEEP_BETWEEN_CALLS = 0.25       # seconds
# -----------------------------------------

SAM_SEARCH_URL_V2 = "https://api.sam.gov/prod/opportunities/v2/search"


def _mask_key(k: str) -> str:
    return f"...{k[-4:]}" if k else "(none)"


def _rotate_keys(keys: List[str]):
    while True:
        for k in keys:
            yield k


def _http_get(url: str, params: dict, key: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "kip_pop_backfill/1.0"}
    qp = dict(params)
    qp["api_key"] = key
    return requests.get(url, params=qp, headers=headers, timeout=timeout)


def fetch_notice_detail_v2(notice_id: str, api_keys: List[str]) -> Dict[str, Any]:
    if not notice_id or not api_keys:
        return {}
    rot = _rotate_keys(api_keys)
    last_err = None
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(SAM_SEARCH_URL_V2, {
                          "noticeid": notice_id, "limit": 1}, key)
            if r.status_code == 429:
                time.sleep(0.8)
                continue
            r.raise_for_status()
            data = r.json() if "application/json" in r.headers.get("Content-Type", "") else {}
            items = data.get("opportunitiesData") or data.get("data") or []
            if isinstance(items, list) and items:
                return items[0]
            return {}
        except Exception as e:
            last_err = e
            time.sleep(0.4)
            continue
    if last_err:
        raise last_err
    return {}


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v).strip()
    try:
        return json.dumps(v, ensure_ascii=False).strip()
    except Exception:
        return str(v).strip()


def extract_place_of_performance(obj: Dict[str, Any]) -> Dict[str, str]:
    """
    Best-effort extraction of POP from a v2 detail record.
    Returns empty strings when not available.
    """
    if not isinstance(obj, dict):
        obj = {}

    def _pick_from_candidate(c: dict) -> Optional[Dict[str, str]]:
        if not isinstance(c, dict):
            return None
        city = c.get("city") or c.get("cityName") or (
            c.get("address") or {}).get("city")
        state = (c.get("state") or c.get("stateCode") or (c.get("address") or {}).get("state")
                 or c.get("stateProvince") or c.get("stateProvinceCode"))
        zipc = c.get("zip") or c.get("zipCode") or c.get(
            "postalCode") or (c.get("address") or {}).get("postalCode")
        country = c.get("country") or c.get("countryCode") or c.get(
            "countryName") or (c.get("address") or {}).get("country")

        if city or state or zipc or country:
            return {
                "pop_city": _s(city),
                "pop_state": _s(state),
                "pop_zip": _s(zipc),
                "pop_country": _s(country),
            }
        return None

    # Possible containers
    containers = [
        "placeOfPerformance", "place_of_performance", "placeOfPerformanceAddress",
        "primaryPlaceOfPerformance", "popAddress", "placeOfPerformanceLocation",
        "place_of_performance_location", "placeOfPerformanceCityState"
    ]
    lists = ["addresses", "locations", "placeOfPerformanceAddresses"]

    for key in containers:
        cand = obj.get(key)
        if isinstance(cand, dict):
            got = _pick_from_candidate(cand)
            if got:
                break
    else:
        for key in lists:
            arr = obj.get(key)
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                got = _pick_from_candidate(arr[0])
                if got:
                    break
        else:
            # sometimes the address fields are at top-level
            got = _pick_from_candidate(obj)

    pop = got or {"pop_city": "", "pop_state": "",
                  "pop_zip": "", "pop_country": ""}

    parts = []
    if pop.get("pop_city"):
        parts.append(pop["pop_city"])
    if pop.get("pop_state"):
        parts.append(pop["pop_state"])
    raw = ", ".join(parts)
    if pop.get("pop_zip"):
        raw = (raw + f" {pop['pop_zip']}".rstrip()).strip()
    if pop.get("pop_country") and pop["pop_country"].upper() not in ("US", "USA", "UNITED STATES", "UNITED-STATES"):
        raw = (raw + f" ({pop['pop_country']})").strip()

    pop["pop_raw"] = raw
    for k in ("pop_city", "pop_state", "pop_zip", "pop_country", "pop_raw"):
        pop.setdefault(k, "")
        if pop[k] is None:
            pop[k] = ""
    return pop


def make_sam_public_url(notice_id: str) -> str:
    nid = (notice_id or "").strip()
    return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"


def _engine():
    kw = {}
    if DB_URL.startswith("postgresql"):
        kw.update(dict(pool_pre_ping=True, pool_size=3, max_overflow=1))
    return create_engine(DB_URL, **kw)


def ensure_table(conn):
    conn.execute(text("SELECT 1 FROM solicitationraw LIMIT 1"))


def pick_targets(conn, limit: int) -> List[str]:
    sql = text("""
        SELECT notice_id
        FROM solicitationraw
        WHERE
          (pop_city IS NULL OR pop_city = '')
          AND (pop_state IS NULL OR pop_state = '')
          AND (pop_zip IS NULL OR pop_zip = '')
          AND (pop_country IS NULL OR pop_country = '')
        ORDER BY pulled_at DESC
        LIMIT :lim
    """)
    rows = conn.execute(sql, {"lim": int(limit)}).fetchall()
    return [str(r[0]) for r in rows]


def update_pop(conn, notice_id: str, pop: Dict[str, str]) -> int:
    sql = text("""
        UPDATE solicitationraw
        SET pop_city=:city, pop_state=:state, pop_zip=:zip, pop_country=:country, pop_raw=:raw
        WHERE notice_id=:nid
    """)
    res = conn.execute(sql, {
        "city": pop.get("pop_city", ""),
        "state": pop.get("pop_state", ""),
        "zip": pop.get("pop_zip", ""),
        "country": pop.get("pop_country", ""),
        "raw": pop.get("pop_raw", ""),
        "nid": notice_id
    })
    return res.rowcount or 0


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Place of Performance (POP) only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log what would change, do not write to DB")
    args = parser.parse_args()

    if not SAM_KEYS:
        print("ERROR: No SAM keys configured in this script. Edit SAM_KEYS at top.")
        return

    engine = _engine()

    # Validate table exists
    try:
        with engine.connect() as conn:
            ensure_table(conn)
    except Exception as e:
        print("ERROR: solicitationraw table not found or DB connection failed.")
        print(e)
        return

    with engine.begin() as conn:
        targets = pick_targets(conn, HARD_LIMIT)
        if not targets:
            print(
                "Nothing to backfill — all rows already have POP (or no eligible rows).")
            return

        print(f"Found {len(targets)} notices missing POP. Fetching v2 details…")

        updated = 0
        skipped = 0
        errors = 0

        for i, nid in enumerate(targets, start=1):
            web = make_sam_public_url(nid)
            try:
                detail = fetch_notice_detail_v2(nid, SAM_KEYS)
                if not detail:
                    skipped += 1
                    print(
                        f"[{i}/{len(targets)}] {nid}: no detail returned; skipped  | {web}")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                pop = extract_place_of_performance(detail)
                if not (pop.get("pop_city") or pop.get("pop_state") or pop.get("pop_zip") or pop.get("pop_country")):
                    skipped += 1
                    print(
                        f"[{i}/{len(targets)}] {nid}: no POP found in detail; skipped | {web}")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                if args.dry_run:
                    print(
                        f"[{i}/{len(targets)}] {nid}: (dry-run) POP -> {pop} | {web}")
                else:
                    n = update_pop(conn, nid, pop)
                    updated += n
                    print(
                        f"[{i}/{len(targets)}] {nid}: updated POP -> {pop} | {web}")

                time.sleep(SLEEP_BETWEEN_CALLS)

            except Exception as e:
                errors += 1
                print(f"[{i}/{len(targets)}] {nid}: ERROR {e} | {web}")
                time.sleep(0.7)

        print(f"Done. Updated={updated}, Skipped={skipped}, Errors={errors}")


if __name__ == "__main__":
    main()
