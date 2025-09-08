#!/usr/bin/env python3
"""
Fast auto-refresh for SAM.gov opportunities:
- Uses v2 search with pagination + key rotation.
- Inserts ONLY brand-new notice_ids.
- Does NOT fetch long descriptions (keeps it fast).
- Always stores a PUBLIC web URL for each notice (no API key required).
- FIXED: Better key rotation and quota handling
"""

from __future__ import annotations
import os
import sys
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import sqlalchemy as sa
from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError

# Import your shared fetch/mapping helpers
import get_relevant_solicitations as gs

# ---------------- Config ----------------
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "200"))
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "5000"))
DAYS_BACK = int(os.getenv("DAYS_BACK", "1"))
DB_URL = os.getenv("SUPABASE_DB_URL") or os.getenv(
    "DB_URL") or "sqlite:///app.db"

# Read keys from env or secrets-like CSV
SAM_KEYS_RAW = os.getenv("SAM_KEYS", "")
SAM_KEYS = [k.strip() for k in SAM_KEYS_RAW.split(",") if k.strip()]

# ---------------- Helpers ----------------
REQUIRED_COLS = [
    "notice_id", "solicitation_number", "title", "notice_type",
    "posted_date", "response_date", "archive_date",
    "naics_code", "set_aside_code", "description", "link",
    "pop_city", "pop_state", "pop_zip", "pop_country", "pop_raw",
]


def _stringify(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v).strip()
    try:
        return json.dumps(v, ensure_ascii=False).strip()
    except Exception:
        return str(v).strip()


def make_sam_public_url(notice_id: str, link: str | None = None) -> str:
    nid = (notice_id or "").strip()
    if not nid:
        return "https://sam.gov/"
    # If caller already handed us a public web URL, keep it; otherwise force public opp URL
    if link and isinstance(link, str) and ("api.sam.gov" not in link):
        return link
    return f"https://sam.gov/opp/{nid}/view"


def _engine():
    # For Postgres (Supabase) we usually want pool_pre_ping; SQLite is fine default.
    kw = {}
    if DB_URL.startswith("postgresql"):
        kw.update(dict(pool_pre_ping=True, pool_size=5, max_overflow=2))
    return create_engine(DB_URL, **kw)


def _ensure_table(conn):
    # Fail fast if table missing; we don't try to create here—use your app migration
    try:
        conn.execute(text("SELECT 1 FROM solicitationraw LIMIT 1"))
    except Exception as e:
        raise RuntimeError(
            "Table 'solicitationraw' not found. Run your app once to migrate/create schema.") from e


def _already_have_ids(conn) -> set[str]:
    # Pull the set of existing notice_ids to skip duplicates quickly
    rows = conn.execute(
        text("SELECT notice_id FROM solicitationraw")).fetchall()
    return {str(r[0]) for r in rows if r and r[0]}


def _insert_rows(conn, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    # Only include columns that exist in your schema
    cols = ["pulled_at"] + REQUIRED_COLS
    sql = text(f"""
        INSERT INTO solicitationraw (
            {", ".join(cols)}
        ) VALUES (
            {", ".join(":"+c for c in cols)}
        )
        ON CONFLICT (notice_id) DO NOTHING
    """)
    conn.execute(sql, rows)
    return len(rows)

# FIXED: Simplified mapping that doesn't call any additional APIs


def map_record_basic_fields_only(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a raw SAM record to our schema fields WITHOUT making any additional API calls.
    This keeps the refresh fast and avoids quota issues.
    """
    def _first_nonempty(obj: Dict[str, Any], *keys: str, default: str = "None") -> str:
        for k in keys:
            if k in obj and obj[k] not in (None, "", []):
                val = obj[k]
                if isinstance(val, (str, int, float, bool)):
                    return str(val).strip()
                elif isinstance(val, dict):
                    # Try common sub-keys
                    for subkey in ("name", "text", "value", "code"):
                        if subkey in val and val[subkey] not in (None, "", []):
                            return str(val[subkey]).strip()
                elif isinstance(val, list) and val:
                    # Take first non-empty item
                    for item in val:
                        if item not in (None, "", []):
                            if isinstance(item, (str, int, float, bool)):
                                return str(item).strip()
                            elif isinstance(item, dict):
                                for subkey in ("name", "text", "value", "code"):
                                    if subkey in item and item[subkey] not in (None, "", []):
                                        return str(item[subkey]).strip()
        return default

    notice_id = _first_nonempty(rec, "noticeId", "id")
    solicitation_number = _first_nonempty(
        rec, "solicitationNumber", "solicitationNo")
    title = _first_nonempty(rec, "title")
    notice_type = _first_nonempty(rec, "noticeType", "type")
    posted_date = _first_nonempty(rec, "postedDate", "publicationDate")
    archive_date = _first_nonempty(rec, "archiveDate")
    naics_code = _first_nonempty(rec, "naicsCode", "naics")
    set_aside_code = _first_nonempty(
        rec, "setAsideCode", "typeOfSetAside", "setAside")

    # Extract response date from various possible fields
    response_date = _first_nonempty(rec, "responseDeadLine", "responseDateTime",
                                    "responseDate", "dueDate", "closeDate")

    # Extract link
    link = "None"
    links = rec.get("links")
    if isinstance(links, list) and links:
        maybe = links[0]
        if isinstance(maybe, dict) and maybe.get("href"):
            link = str(maybe["href"])
    if link == "None":
        link = _first_nonempty(rec, "url", "samLink")

    # Get basic description from the search result itself (no additional API calls)
    description = _first_nonempty(rec, "description", "synopsis")

    # Basic place of performance extraction (no additional API calls)
    pop_city = ""
    pop_state = ""
    pop_zip = ""
    pop_country = ""
    pop_raw = ""

    # Try to extract from embedded place of performance data
    pop_data = rec.get("placeOfPerformance") or rec.get(
        "place_of_performance") or {}
    if isinstance(pop_data, dict):
        pop_city = _first_nonempty(pop_data, "city", "cityName")
        pop_state = _first_nonempty(
            pop_data, "state", "stateCode", "stateProvince")
        pop_zip = _first_nonempty(pop_data, "zip", "zipCode", "postalCode")
        pop_country = _first_nonempty(
            pop_data, "country", "countryCode", "countryName")

        # Build pop_raw
        parts = []
        if pop_city and pop_city != "None":
            parts.append(pop_city)
        if pop_state and pop_state != "None":
            parts.append(pop_state)
        pop_raw = ", ".join(parts)
        if pop_zip and pop_zip != "None":
            pop_raw = (pop_raw + f" {pop_zip}").strip()
        if pop_country and pop_country != "None" and pop_country.upper() not in ("US", "USA", "UNITED STATES"):
            pop_raw = (pop_raw + f" ({pop_country})").strip()

    return {
        "notice_id": _stringify(notice_id),
        "solicitation_number": _stringify(solicitation_number),
        "title": _stringify(title),
        "notice_type": _stringify(notice_type),
        "posted_date": _stringify(posted_date),
        "response_date": _stringify(response_date),
        "archive_date": _stringify(archive_date),
        "naics_code": _stringify(naics_code),
        "set_aside_code": _stringify(set_aside_code),
        "description": _stringify(description),
        "link": _stringify(link),
        "pop_city": _stringify(pop_city) if pop_city != "None" else "",
        "pop_state": _stringify(pop_state) if pop_state != "None" else "",
        "pop_zip": _stringify(pop_zip) if pop_zip != "None" else "",
        "pop_country": _stringify(pop_country) if pop_country != "None" else "",
        "pop_raw": _stringify(pop_raw),
    }

# ---------------- Main ----------------


def main():
    print("=== Auto refresh start ===")
    print(datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y"))
    print(f"SAM_KEYS configured: {len(SAM_KEYS)} key(s)")
    print(f"DAYS_BACK = {DAYS_BACK}")
    print(f"Paging: PAGE_SIZE={PAGE_SIZE}, MAX_RECORDS={MAX_RECORDS}")

    if not SAM_KEYS:
        print("ERROR: No SAM_KEYS provided (env SAM_KEYS). Exiting.")
        sys.exit(1)

    engine = _engine()
    print("auto_refresh.py: engine created")

    try:
        with engine.connect() as conn:
            _ensure_table(conn)
            print("auto_refresh.py: DB ping OK")

            # For logging DB size (optional)
            try:
                before = conn.execute(
                    text("SELECT COUNT(*) FROM solicitationraw")).scalar() or 0
                last_pulled = conn.execute(
                    text("SELECT MAX(pulled_at) FROM solicitationraw")).scalar()
                print(
                    f"DB before: {before} rows; last pulled_at: {last_pulled}")
            except Exception:
                pass

    except Exception as e:
        print("auto_refresh.py: DB check failed")
        print(e)
        sys.exit(1)

    print("auto_refresh.py: entered main()")
    print("Starting auto-refresh job...")

    total_inserted = 0
    total_seen = 0
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        with engine.begin() as conn:
            existing = _already_have_ids(conn)
            rows_to_insert: List[Dict[str, Any]] = []

            print("Fetching solicitations from SAM.gov with pagination…")
            offset = 0

            while True:
                limit = min(PAGE_SIZE, MAX_RECORDS - total_seen)
                if limit <= 0:
                    break

                print(
                    f"  → Page {offset//PAGE_SIZE + 1}: offset={offset}, limit={limit}")

                try:
                    # This uses proper key rotation in _request_sam()
                    raw = gs.get_sam_raw_v3(
                        days_back=DAYS_BACK,
                        limit=limit,
                        api_keys=SAM_KEYS,
                        filters={},       # we want everything in the window
                        offset=offset,
                    )
                except gs.SamQuotaError:
                    print("All SAM.gov keys exhausted (quota). Stopping refresh.")
                    print(
                        f"Partial success: {total_inserted} new records inserted before quota limit.")
                    break
                except gs.SamAuthError:
                    print("All SAM.gov keys failed authentication. Check your keys.")
                    sys.exit(2)
                except Exception as e:
                    print(f"Unexpected error fetching data: {e}")
                    sys.exit(2)

                if not raw:
                    break

                total_seen += len(raw)
                print(
                    f"    fetched {len(raw)} records (cumulative fetched: {total_seen})")

                for r in raw:
                    # FIXED: Use basic mapping that doesn't make additional API calls
                    m = map_record_basic_fields_only(r)
                    nid = (m.get("notice_id") or "").strip()
                    if not nid or nid in existing:
                        continue

                    # Force link to public URL so clicks never need API key
                    public_link = make_sam_public_url(nid, m.get("link"))
                    row = {
                        "pulled_at": now_iso,
                        "notice_id": _stringify(m.get("notice_id")),
                        "solicitation_number": _stringify(m.get("solicitation_number")),
                        "title": _stringify(m.get("title")),
                        "notice_type": _stringify(m.get("notice_type")),
                        "posted_date": _stringify(m.get("posted_date")),
                        "response_date": _stringify(m.get("response_date")),
                        "archive_date": _stringify(m.get("archive_date")),
                        "naics_code": _stringify(m.get("naics_code")),
                        "set_aside_code": _stringify(m.get("set_aside_code")),
                        # Basic description only
                        "description": _stringify(m.get("description")),
                        "link": public_link,
                        "pop_city": _stringify(m.get("pop_city")),
                        "pop_state": _stringify(m.get("pop_state")),
                        "pop_zip": _stringify(m.get("pop_zip")),
                        "pop_country": _stringify(m.get("pop_country")),
                        "pop_raw": _stringify(m.get("pop_raw")),
                    }
                    rows_to_insert.append(row)
                    existing.add(nid)  # avoid repeats in same run

                # Insert batch
                if rows_to_insert:
                    inserted = _insert_rows(conn, rows_to_insert)
                    total_inserted += inserted
                    print(
                        f"    inserted {inserted} new records (total new: {total_inserted})")
                    rows_to_insert.clear()

                # Next page
                offset += PAGE_SIZE
                if total_seen >= MAX_RECORDS:
                    break

        print(f"Auto refresh success: inserted {total_inserted} new notices.")
        if total_seen == 0:
            print("Note: SAM.gov returned no records for the selected window.")

    except Exception as e:
        print(f"Auto refresh failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
