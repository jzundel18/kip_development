# backfill_pop.py
# Standalone script to backfill Place of Performance (POP) fields only.
# - Tries many structured variants in v2 detail
# - Falls back to v1 noticedesc (HTML) + regex
# - Prints SAM.gov link for manual comparison
#
# Usage:
#   python backfill_pop.py
#
# Requirements:
#   pip install sqlalchemy sqlmodel pandas requests

from __future__ import annotations
import os
import re
import time
import html
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import sqlalchemy as sa
from sqlalchemy import text, create_engine

# ========= CONFIG =========
# Put your SAM.gov API keys here (rotation order)
SAM_KEYS: List[str] = [
    "2WWQnuvYVj7cI3qozS5xC0Y2SgGITC7NGWtmqebq",
    "qV4oW8tKis4kinR7oXIBlnTDlJx9Tyf4Se8Tmmmx"
]

# DB connection: use env if present, else local SQLite
DB_URL = "postgresql+psycopg2://postgres.ceemspukffoygxazsvix:Moolah123%21%21%21@aws-1-us-west-1.pooler.supabase.com:6543/postgres?sslmode=require"

# How many blank rows to process this run
MAX_ROWS_TO_BACKFILL: int = 400

# HTTP / SAM endpoints (prod)
SAM_SEARCH_URL_V2 = "https://api.sam.gov/prod/opportunities/v2/search"
SAM_DESC_URL_V1 = "https://api.sam.gov/prod/opportunities/v1/noticedesc"

USER_AGENT = "kip_pop_backfill/1.0"

# ========= Helpers =========


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


def make_sam_public_url(notice_id: str) -> str:
    nid = (notice_id or "").strip()
    return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"


def _rotate_keys(keys: List[str]):
    while True:
        for k in keys:
            yield k


def _http_get(url: str, params: dict, key: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    q = dict(params or {})
    if key:
        q["api_key"] = key
    return requests.get(url, params=q, headers=headers, timeout=timeout)


def fetch_notice_detail_v2(notice_id: str, api_keys: List[str]) -> Dict[str, Any]:
    """
    Fetch a single record via v2 search detail by noticeid.
    Returns {} when exhausted keys / not found / error.
    """
    if not notice_id or not api_keys:
        return {}
    rot = _rotate_keys(api_keys)
    last_error = None
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(SAM_SEARCH_URL_V2, {
                          "noticeid": notice_id, "limit": 1}, key, timeout=35)
            if r.status_code == 429:
                # quota → rotate
                time.sleep(1.0)
                continue
            r.raise_for_status()
            data = r.json() if r.headers.get("Content-Type",
                                             "").startswith("application/json") else {}
            items = data.get("opportunitiesData") or data.get("data") or []
            if isinstance(items, list) and items:
                # some responses put the full object as first element
                return items[0]
            return {}
        except Exception as e:
            last_error = e
            time.sleep(0.5)
            continue
    # Optional: print the last error in debug scenarios
    return {}


def fetch_notice_description_v1(notice_id: str, api_keys: List[str]) -> str:
    """
    Fallback: v1 noticedesc returns HTML/plain text. We strip tags & collapse whitespace.
    """
    if not notice_id or not api_keys:
        return ""
    rot = _rotate_keys(api_keys)
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(SAM_DESC_URL_V1, {
                          "noticeid": notice_id}, key, timeout=35)
            if r.status_code == 429:
                time.sleep(1.0)
                continue
            r.raise_for_status()
            text_html = r.text or ""
            text_html = html.unescape(text_html)
            text_plain = re.sub(r"<[^>]+>", " ", text_html)
            text_plain = re.sub(r"\s+", " ", text_plain).strip()
            return text_plain
        except Exception:
            time.sleep(0.5)
            continue
    return ""

# ---------- POP extraction (structured) ----------


_POP_CONTAINER_KEYS = [
    "placeOfPerformance",
    "place_of_performance",
    "primaryPlaceOfPerformance",
    "placeOfPerformanceAddress",
    "placeOfPerformanceLocation",
    "place_of_performance_location",
    "placeOfPerformanceCityState",
    "popAddress",
    "primaryPlaceOfPerformanceAddress",
    "deliveryDestination",          # sometimes used
]

_POP_LIST_KEYS = [
    "addresses",
    "locations",
    "placeOfPerformanceAddresses",
    "placeOfPerformanceLocations",
]

_CITY_KEYS = ["city", "cityName"]
_STATE_KEYS = ["state", "stateCode", "stateProvince",
               "stateProvinceCode", "stateName"]
_ZIP_KEYS = ["zip", "zipCode", "postalCode", "zip4"]
_COUNTRY_KEYS = ["country", "countryCode", "countryName"]


def _get_first(obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
    """
    Returns the first non-empty string from obj[k], preferring 'name'/'text' if the value is a dict.
    """
    for k in keys:
        v = obj.get(k)
        if v is None or v == "" or v == []:
            continue
        if isinstance(v, dict):
            # prefer 'name' → 'text' → 'value' → 'code'
            for sub in ("name", "text", "value", "code"):
                if v.get(sub):
                    return str(v[sub]).strip()
            continue
        return str(v).strip()
    return None


def _extract_from_address_like(obj: Dict[str, Any]) -> Dict[str, str]:
    """
    Attempt to read city/state/zip/country from:
      - top-level keys (city, state, postalCode, etc.)
      - nested 'address' dict with same keys
    """
    if not isinstance(obj, dict):
        return {}
    # nested 'address' dict is common
    addr = obj.get("address") if isinstance(obj.get("address"), dict) else {}
    city = _get_first(obj, _CITY_KEYS) or _get_first(addr, _CITY_KEYS)
    state = _get_first(obj, _STATE_KEYS) or _get_first(addr, _STATE_KEYS)
    zipc = _get_first(obj, _ZIP_KEYS) or _get_first(addr, _ZIP_KEYS)
    country = _get_first(obj, _COUNTRY_KEYS) or _get_first(addr, _COUNTRY_KEYS)

    out = {}
    if city or state or zipc or country:
        out = {
            "pop_city": _s(city) or "",
            "pop_state": _s(state) or "",
            "pop_zip": _s(zipc) or "",
            "pop_country": _s(country) or "",
        }
    return out


def extract_pop_from_struct(rec: Dict[str, Any]) -> Dict[str, str]:
    """
    Given a v2 record/detailed object, try many shapes to find POP.
    Returns dict with pop_* if found; {} otherwise.
    """
    if not isinstance(rec, dict):
        return {}

    # 1) known containers
    candidates: List[Dict[str, Any]] = []
    for k in _POP_CONTAINER_KEYS:
        v = rec.get(k)
        if isinstance(v, dict):
            candidates.append(v)

    # 2) known lists of address-like objects
    for k in _POP_LIST_KEYS:
        v = rec.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            candidates.append(v[0])

    # 3) office/organization containers can sometimes hold address
    for k in ("organizationAddress", "officeAddress", "contractOffice", "issuingOffice", "agency"):
        v = rec.get(k)
        if isinstance(v, dict):
            candidates.append(v)

    # 4) include rec itself (in case keys are top-level)
    candidates.append(rec)

    # 5) scan each candidate plus one level of nesting
    def scan_one(obj: Dict[str, Any]) -> Dict[str, str]:
        # direct
        got = _extract_from_address_like(obj)
        if got:
            return got
        # one level deep dicts
        for vv in obj.values():
            if isinstance(vv, dict):
                got = _extract_from_address_like(vv)
                if got:
                    return got
        return {}

    for c in candidates:
        if not isinstance(c, dict):
            continue
        out = scan_one(c)
        if out:
            return out

    # 6) in some payloads, location appears as "cityState" / "cityStateZip" strings -> split
    for k in ("cityState", "cityStateZip", "place", "location"):
        s = rec.get(k)
        if isinstance(s, str) and "," in s:
            left, right = s.split(",", 1)
            left = left.strip()
            right = right.strip()
            m = re.match(r"([A-Z]{2})(?:\s+(\d{5}(?:-\d{4})?))?", right)
            if m:
                state = m.group(1)
                zipc = m.group(2) or ""
                return {"pop_city": left, "pop_state": state, "pop_zip": zipc, "pop_country": ""}

    return {}

# ---------- POP extraction (text fallback) ----------


_US_ST = r"(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)"
_RE_CITY_ST = re.compile(
    rf"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*,\s*{_US_ST}\b")
_RE_CITY_ST_ZIP = re.compile(
    rf"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*,\s*{_US_ST}\s+(\d{{5}}(?:-\d{{4}})?)\b", re.IGNORECASE)
_RE_POP_LABEL = re.compile(
    r"place\s+of\s+performance|work\s+location|service\s+location|performance\s+location", re.IGNORECASE)


def extract_pop_from_text(text_in: str) -> Dict[str, str]:
    """
    Look near POP labels first; else global City, ST[ ZIP] match.
    """
    t = (text_in or "").strip()
    if not t:
        return {}

    # Narrow the window around a POP-like label for precision, if present
    m = _RE_POP_LABEL.search(t)
    window = t
    if m:
        start = max(0, m.start() - 160)
        end = min(len(t), m.end() + 160)
        window = t[start:end]

    ms = _RE_CITY_ST_ZIP.search(window) or _RE_CITY_ST_ZIP.search(t)
    if ms:
        city, st, zipc = ms.group(1), ms.group(2), ms.group(3)
        return {"pop_city": city, "pop_state": st, "pop_zip": zipc or "", "pop_country": ""}

    m2 = _RE_CITY_ST.search(window) or _RE_CITY_ST.search(t)
    if m2:
        city, st = m2.group(1), m2.group(2)
        return {"pop_city": city, "pop_state": st, "pop_zip": "", "pop_country": ""}

    return {}


def build_pop_raw(pop: Dict[str, str]) -> str:
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
    return raw

# ========= DB I/O =========


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def select_rows_to_backfill(conn) -> List[Dict[str, Any]]:
    sql = text("""
        SELECT notice_id, title, description
        FROM solicitationraw
        WHERE COALESCE(pop_city,'')='' AND COALESCE(pop_state,'')='' AND
              COALESCE(pop_zip,'')=''  AND COALESCE(pop_country,'')=''
        ORDER BY pulled_at DESC
        LIMIT :lim
    """)
    rows = conn.execute(
        sql, {"lim": int(MAX_ROWS_TO_BACKFILL)}).mappings().all()
    return [dict(r) for r in rows]


def update_pop(conn, notice_id: str, pop: Dict[str, str]) -> int:
    pop = {k: (pop.get(k) or "")
           for k in ("pop_city", "pop_state", "pop_zip", "pop_country")}
    pop_raw = build_pop_raw(pop)
    sql = text("""
        UPDATE solicitationraw
        SET pop_city=:city, pop_state=:state, pop_zip=:zip, pop_country=:country, pop_raw=:raw
        WHERE notice_id=:nid
    """)
    res = conn.execute(sql, {
        "city": pop["pop_city"], "state": pop["pop_state"],
        "zip": pop["pop_zip"], "country": pop["pop_country"],
        "raw": pop_raw, "nid": notice_id
    })
    return res.rowcount or 0

# ========= Runner =========


def backfill_once():
    if not SAM_KEYS:
        print("ERROR: SAM_KEYS is empty. Edit backfill_pop.py and add your keys to SAM_KEYS[].")
        return

    engine = get_engine()

    with engine.connect() as conn:
        # sanity check table exists
        try:
            conn.execute(text("SELECT 1 FROM solicitationraw LIMIT 1"))
        except Exception as e:
            print(
                "ERROR: solicitationraw table not found. Make sure your DB is initialized.")
            print(e)
            return

        rows = select_rows_to_backfill(conn)
        if not rows:
            print(
                "Nothing to backfill. All rows already have POP or no rows match criteria.")
            return

        print(f"Backfilling POP for up to {len(rows)} rows...\n")

        updated = 0
        for i, r in enumerate(rows, 1):
            nid = (r.get("notice_id") or "").strip()
            title = (r.get("title") or "").strip()
            sam_url = make_sam_public_url(nid)

            print(f"{i:>4}. notice_id={nid}  |  {sam_url}")
            if title:
                print(f"      title: {title[:110]}")

            pop: Dict[str, str] = {}

            # 1) v2 detail (structured)
            detail = fetch_notice_detail_v2(nid, SAM_KEYS)
            if detail:
                pop = extract_pop_from_struct(detail)
                if pop:
                    print(f"      v2 detail → POP: {build_pop_raw(pop)}")
                else:
                    print("      v2 detail → no structured POP keys found.")

            # 2) fallback to description (v1 noticedesc) if needed
            if not pop:
                desc = fetch_notice_description_v1(nid, SAM_KEYS)
                if desc:
                    pop = extract_pop_from_text(desc)
                    if pop:
                        print(
                            f"      v1 noticedesc regex → POP: {build_pop_raw(pop)}")
                    else:
                        print(
                            "      v1 noticedesc present but no recognizable POP pattern.")
                else:
                    print("      v1 noticedesc: no detail returned (empty).")

            # 3) update DB if we got anything
            if pop:
                with engine.begin() as tx:
                    cnt = update_pop(tx, nid, pop)
                if cnt:
                    updated += 1
                    print("      ✅ DB updated.")
                else:
                    print("      ⚠️  DB update skipped (notice not found / unchanged).")
            else:
                print("      ⏭️  Skipped (no POP found via API/regex).")

        print(f"\nDone. Updated {updated} row(s).")


if __name__ == "__main__":
    backfill_once()
