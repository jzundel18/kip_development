# get_relevant_solicitations.py
# SAM.gov v2 fetchers with key rotation, clear quota/auth errors, and client-side filters.

from __future__ import annotations
from datetime import date, timedelta
from typing import List, Dict, Any, Optional

import requests
import time
import html
import json
import re

# --- Endpoints (use /prod/) ---
SAM_BASE_URL = "https://api.sam.gov/prod/opportunities/v2/search"
SAM_SEARCH_URL_V2 = "https://api.sam.gov/prod/opportunities/v2/search"
SAM_DESC_URL_V1 = "https://api.sam.gov/prod/opportunities/v1/noticedesc"
# single notice entity
SAM_NOTICE_ENTITY_V2 = "https://api.sam.gov/prod/opportunities/v2/notices/{notice_id}"

# ---- Custom errors for friendly UI messages ----


class SamQuotaError(Exception):
    pass


class SamAuthError(Exception):
    pass


class SamBadRequestError(Exception):
    pass


# --- Date helpers ---
_DATE_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DATE_US_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")


def _safe_str(val: Any) -> str:
    """Normalize SAM.gov values: prefer 'name'/'text' if dicts, else plain string."""
    if val is None:
        return ""
    if isinstance(val, dict):
        for sub in ("name", "text", "value", "code"):
            if val.get(sub):
                return str(val[sub]).strip()
        return ""
    return str(val).strip()
def _normalize_date(val: str) -> str:
    """
    Return YYYY-MM-DD if we can, else 'None' if empty/placeholder, else original string.
    Handles ISO dates (with time) and US MM/DD/YYYY.
    """
    if not val:
        return "None"
    s = str(val).strip()
    if not s or s.lower() in ("none", "n/a", "na"):
        return "None"

    # ISO-like (may include time)
    m = _DATE_ISO_RE.search(s)
    if m:
        return m.group(1)

    # US m/d/yyyy -> yyyy-mm-dd
    m = _DATE_US_RE.search(s)
    if m:
        mm, dd, yyyy = m.group(1).split("/")
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"

    # If contains 'T', try left of T
    if "T" in s:
        left = s.split("T", 1)[0]
        if _DATE_ISO_RE.match(left):
            return left

    return s


def _mmddyyyy(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def _window_days_back(days_back: int) -> tuple[str, str]:
    today = date.today()
    start = today - timedelta(days=max(0, int(days_back)))
    return (_mmddyyyy(start), _mmddyyyy(today))


def _mask_key(k: str) -> str:
    if not k:
        return "(none)"
    return f"...{k[-4:]}"


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _stringify(v) -> str:
    """Return a safe, trimmed string for DB/logging."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v).strip()
    try:
        return json.dumps(v, ensure_ascii=False).strip()
    except Exception:
        return str(v).strip()

# --- HTTP core with key rotation ---


def _request_sam(params: Dict[str, Any], api_keys: List[str]) -> Dict[str, Any]:
    """
    Try each api_key once. If a key hits quota/auth, rotate to the next.
    If all keys fail, raise a clear aggregated error.
    """
    if not api_keys:
        raise ValueError("No SAM.gov API keys provided.")

    errors: list[str] = []

    for key in api_keys:
        try:
            full_params = dict(params)
            full_params["api_key"] = key
            resp = requests.get(SAM_BASE_URL, params=full_params, timeout=30)

            # Try to extract a short message
            txt = ""
            try:
                j = resp.json()
                if isinstance(j, dict):
                    txt = (j.get("message") or j.get("error") or "") or ""
            except Exception:
                pass

            if resp.status_code == 429:
                errors.append(
                    f"{_mask_key(key)} → 429 Too Many Requests (quota). {txt}")
                continue

            if resp.status_code in (401, 403):
                # Sometimes returns 403 for quota issues
                if any(s in (txt or "").lower() for s in ["exceeded", "limit", "quota"]):
                    errors.append(
                        f"{_mask_key(key)} → {resp.status_code} quota/limit. {txt}")
                else:
                    errors.append(
                        f"{_mask_key(key)} → {resp.status_code} auth error. {txt}")
                continue

            if resp.status_code == 400:
                raise SamBadRequestError(
                    f"400 Bad Request from SAM.gov. {txt or resp.text}")

            resp.raise_for_status()
            return resp.json() or {}

        except SamBadRequestError:
            raise
        except requests.RequestException as e:
            errors.append(f"{_mask_key(key)} → network error: {e}")
            time.sleep(0.3)
            continue

    if errors:
        if any(("quota" in e.lower() or "limit" in e.lower() or "429" in e) for e in errors):
            raise SamQuotaError(
                "All SAM.gov keys appear rate-limited / out of daily quota.")
        raise SamAuthError(
            "All SAM.gov keys failed (auth/network). Check keys or network.")
    raise SamAuthError("SAM.gov request failed.")


def get_sam_raw_v3(
    days_back: int,
    limit: int,
    api_keys: List[str],
    filters: Optional[Dict[str, Any]] = None,
    *,
    offset: int = 0,          # paging
) -> List[Dict[str, Any]]:
    filters = filters or {}
    posted_from, posted_to = _window_days_back(days_back)

    params = {
        "limit": int(limit),
        "postedFrom": posted_from,  # MM/dd/YYYY
        "postedTo": posted_to,      # MM/dd/YYYY
        "offset": int(offset),      # SAM v2 accepts 'offset' for paging
    }

    # if caller provides a specific notice_id, pass it through
    if filters.get("notice_id"):
        params["noticeid"] = str(filters["notice_id"]).strip()
        params["limit"] = 1  # fetching a specific record

    data = _request_sam(params, api_keys)
    raw_records = data.get("opportunitiesData") or data.get("data") or []
    if not raw_records:
        return []

    # ---------- filter helper ----------
    def _match(rec: Dict[str, Any]) -> bool:
        # --- basic type/notice filtering ---
        # specific notice_id requested?
        if filters.get("notice_id"):
            rid = str(rec.get("noticeId") or rec.get("id") or "").strip()
            if rid != str(filters["notice_id"]).strip():
                return False

        # normalize the record's notice type once
        nt = str(rec.get("noticeType") or rec.get("type") or "").strip().lower()

        # drop justifications and awards
        if nt == "justification":
            return False
        if "award" in nt:
            return False

        # --- sanitize filters to strings (robust against dicts/None) ---
        nts = [str(t).strip().lower()
            for t in (filters.get("notice_types") or []) if str(t).strip()]
        kws = [str(k).strip().lower()
            for k in (filters.get("keywords_or") or []) if str(k).strip()]
        naics_targets = [str(n).strip()
                        for n in (filters.get("naics") or []) if str(n).strip()]
        sas = [str(s).strip().lower()
            for s in (filters.get("set_asides") or []) if str(s).strip()]
        due_before = filters.get("due_before")

        # notice types (substring ok)
        if nts and not any(t in nt for t in nts):
            return False

        # keyword OR over title+description/synopsis
        if kws:
            title = str(rec.get("title") or "").lower()
            desc = str(rec.get("description") or rec.get("synopsis") or "").lower()
            blob = f"{title} {desc}"
            if not any(k in blob for k in kws):
                return False

        # NAICS exact match (string compare)
        if naics_targets:
            rec_naics = str(rec.get("naicsCode") or rec.get("naics") or "").strip()
            if not rec_naics or rec_naics not in naics_targets:
                return False

        # set-asides (substring ok)
        if sas:
            rec_sa = str(rec.get("setAsideCode")
                        or rec.get("setAside") or "").lower()
            if not rec_sa or not any(sa in rec_sa for sa in sas):
                return False

        # due before (normalize date strings)
        if due_before:
            raw = (
                rec.get("dueDate") or rec.get("responseDueDate") or
                rec.get("closeDate") or rec.get("responseDate") or
                rec.get("responseDateTime") or ""
            )
            resp_norm = _normalize_date(raw)
            if resp_norm != "None" and resp_norm > str(due_before):
                return False

        return True


def get_raw_sam_solicitations(limit: int, api_keys: List[str]) -> List[Dict[str, Any]]:
    """Back-compat alias: today's raw records with limit=N (no extra filters)."""
    return get_sam_raw_v3(days_back=0, limit=limit, api_keys=api_keys, filters={})

# --- helpers used by detail/description ---


def _rotate_keys(keys: List[str]):
    while True:
        for k in keys:
            yield k


def _http_get(url: str, params: dict, key: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "kip_external/1.0"}
    return requests.get(url, params={**params, "api_key": key}, headers=headers, timeout=timeout)


def fetch_notice_detail_v2(notice_id: str, api_keys: List[str]) -> Dict[str, Any]:
    """
    Try to fetch a *single* record using the v2 search endpoint by noticeid.
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


def fetch_notice_entity_v2(notice_id: str, api_keys: List[str]) -> Dict[str, Any]:
    """
    Fetch the *entity* detail for a single notice:
    GET /opportunities/v2/notices/{notice_id}
    This payload contains attributes.placeOfPerformance.address, which is the most
    reliable source for Place of Performance.
    """
    if not notice_id or not api_keys:
        return {}
    url = SAM_NOTICE_ENTITY_V2.format(notice_id=notice_id)
    rot = _rotate_keys(api_keys)
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(url, {}, key)
            if r.status_code == 429:
                time.sleep(1.0)
                continue
            r.raise_for_status()
            return r.json() if r.content else {}
        except Exception:
            time.sleep(0.5)
            continue
    return {}


def fetch_notice_description(notice_id: str, api_keys: List[str]) -> str:
    """
    Get full description text using two strategies:
      1) v2 detail (if it includes description/synopsis/long text)
      2) v1 noticedesc (HTML/plain -> normalized text)
    Returns an empty string if both fail.
    """
    # 1) try v2 detail (search-by-id)
    detail = fetch_notice_detail_v2(notice_id, api_keys)
    for k in ("description", "synopsis", "longDescription", "fullDescription", "additionalInfo"):
        val = detail.get(k)
        if val and str(val).strip():
            return re.sub(r"\s+", " ", str(val)).strip()

    # 1b) try entity payload's attributes.description
    ent = fetch_notice_entity_v2(notice_id, api_keys)
    try:
        attrs = (ent.get("data") or {}).get("attributes") or {}
        val = attrs.get("description")
        if val and str(val).strip():
            return re.sub(r"\s+", " ", str(val)).strip()
    except Exception:
        pass

    # 2) v1 noticedesc
    rot = _rotate_keys(api_keys)
    for _ in range(len(api_keys)):
        key = next(rot)
        try:
            r = _http_get(SAM_DESC_URL_V1, {"noticeid": notice_id}, key)
            if r.status_code == 429:
                time.sleep(1.0)
                continue
            r.raise_for_status()
            text = r.text or ""
            text = html.unescape(text)
            text = re.sub(r"<[^>]+>", " ", text)     # strip tags
            text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
            if text:
                return text
        except Exception:
            time.sleep(0.5)
            continue

    return ""  # No description found

# --- deep search helpers for nested payloads ---


def _deep_find_first(obj, key_set_lower) -> Optional[str]:
    """
    Recursively search dict/list for the first value whose key (case-insensitive)
    is in key_set_lower. Returns string value if found, else None.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k and k.lower() in key_set_lower and v not in (None, "", []):
                return str(v)
        for v in obj.values():
            res = _deep_find_first(v, key_set_lower)
            if res is not None:
                return res
    elif isinstance(obj, list):
        for it in obj:
            res = _deep_find_first(it, key_set_lower)
            if res is not None:
                return res
    return None


def _pick_response_date(rec: dict, detail: dict) -> str:
    """
    Prefer v2 'reponseDeadLine' (typoed in SAM docs/spec), then try common fallbacks.
    Normalize to YYYY-MM-DD when possible.
    """
    # primary per v2 spec
    preferred = rec.get("reponseDeadLine")
    if preferred:
        return _normalize_date(preferred)

    # try detail payload too
    preferred = detail.get("reponseDeadLine") if isinstance(
        detail, dict) else None
    if preferred:
        return _normalize_date(preferred)

    # conservative fallbacks seen in the wild / legacy payloads
    candidates = [
        "responseDeadLine",      # sometimes appears correctly spelled
        "responseDueDate",
        "dueDate",
        "closeDate",
        "responseDate",
        "responseDateTime",
        "offersDueDate",
        "proposalDueDate",
    ]

    for src in (rec, detail if isinstance(detail, dict) else {}):
        for k in candidates:
            val = src.get(k)
            if val not in (None, "", []):
                return _normalize_date(val)

        # shallow nested dicts
        for v in src.values():
            if isinstance(v, dict):
                for k in candidates:
                    if v.get(k):
                        return _normalize_date(v[k])

    return "None"


# Replace your current _first_nonempty with this version (handles lists cleanly)
def _first_nonempty(obj: Dict[str, Any], *keys: str, default: str = "None") -> str:
    """
    Return a plain string. If value is a dict, prefer name/text/value/code.
    If it's a list, take the first non-empty element (recursively flattened).
    Never returns JSON blobs.
    """
    def _flatten(v):
        if v is None:
            return ""
        if isinstance(v, list):
            for it in v:
                s = _flatten(it)
                if s:
                    return s
            return ""
        if isinstance(v, dict):
            for k in ("name", "text", "value", "code"):
                if k in v and v[k] not in (None, "", []):
                    return str(v[k]).strip()
            # last resort: scan nested
            for vv in v.values():
                s = _flatten(vv)
                if s:
                    return s
            return ""
        return str(v).strip()

    for k in keys:
        if k in obj:
            s = _flatten(obj[k])
            if s:
                return s
    return default

# ---------- Place of Performance extraction ----------
# --- add this helper near your other small helpers ---


def _flatten_name_like(val: Any) -> str:
    """
    Return a plain string for SAM.gov PoP fields.
    - dict: prefer 'name' -> 'text' -> 'value' -> 'code'
    - list: use first non-empty element (recursively)
    - scalar: str(val)
    Always returns a stripped string; never JSON.
    """
    if val is None:
        return ""
    # lists: pick first usable
    if isinstance(val, list):
        for it in val:
            s = _flatten_name_like(it)
            if s:
                return s.strip()
        return ""
    # dicts: prefer human-readable keys
    if isinstance(val, dict):
        for k in ("name", "text", "value", "code"):
            if k in val and val[k] not in (None, "", []):
                return str(val[k]).strip()
        # sometimes nested: {"city":{"name":"..."}}
        for v in val.values():
            s = _flatten_name_like(v)
            if s:
                return s.strip()
        return ""
    # scalars
    return str(val).strip()

# ---------- replace your existing _extract_pop_from_notice_entity with this ----------


def _extract_pop_from_notice_entity(detail_json: dict) -> dict:
    """
    Extract PoP from entity payload:
      data.attributes.placeOfPerformance.address.{city,state,country,zip}
    Fields can be strings or objects { code|name }.
    Returns {pop_city, pop_state, pop_zip, pop_country, pop_raw} as plain strings.
    """
    out = {"pop_city": "", "pop_state": "",
           "pop_zip": "", "pop_country": "", "pop_raw": ""}

    try:
        attrs = (detail_json.get("data") or {}).get("attributes") or {}
        addr = (attrs.get("placeOfPerformance") or {}).get("address") or {}

        city_raw = addr.get("city")
        state_raw = addr.get("state")
        country_raw = addr.get("country")
        zip_raw = addr.get("zip") or addr.get(
            "zipCode") or addr.get("postalCode")

        city = _flatten_name_like(city_raw)
        state = _flatten_name_like(state_raw)
        country = _flatten_name_like(country_raw)
        zipc = _flatten_name_like(zip_raw)

        out["pop_city"] = city
        out["pop_state"] = state
        out["pop_zip"] = zipc
        out["pop_country"] = country

        # pretty pop_raw
        bits = []
        if city:
            bits.append(city)
        if state:
            bits.append(state)
        raw = ", ".join(bits)
        if zipc:
            raw = (raw + f" {zipc}".rstrip()).strip()
        if country and country.upper() not in ("USA", "US", "UNITED STATES", "UNITED-STATES"):
            raw = (raw + f" ({country})").strip()
        out["pop_raw"] = raw
    except Exception:
        # leave defaults
        pass

    return out

# ---------- replace your existing _extract_place_of_performance with this ----------


def _extract_place_of_performance(rec: dict, detail: dict | None = None) -> dict:
    """
    Heuristic PoP extraction from search detail / legacy shapes.
    Returns plain strings (never JSON) for all pop_* fields.
    """
    def _from_obj(obj: dict | None) -> dict:
        if not isinstance(obj, dict):
            return {}

        candidates = []
        # common containers
        for k in (
            "placeOfPerformance", "place_of_performance",
            "placeOfPerformanceAddress", "primaryPlaceOfPerformance",
            "popAddress", "placeOfPerformanceLocation",
            "place_of_performance_location", "placeOfPerformanceCityState",
        ):
            v = obj.get(k)
            if isinstance(v, dict):
                candidates.append(v)

        # lists of addresses
        for k in ("addresses", "locations", "placeOfPerformanceAddresses"):
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                candidates.append(v[0])

        # sometimes the object itself is address-shaped
        candidates.append(obj)

        for c in candidates:
            if not isinstance(c, dict):
                continue

            city = _flatten_name_like(
                c.get("city") or c.get("cityName") or (
                    c.get("address") or {}).get("city")
            )
            state = _flatten_name_like(
                c.get("state") or c.get("stateCode") or (
                    c.get("address") or {}).get("state")
                or c.get("stateProvince") or c.get("stateProvinceCode")
            )
            zipc = _flatten_name_like(
                c.get("zip") or c.get("zipCode") or c.get("postalCode")
                or (c.get("address") or {}).get("postalCode")
            )
            country = _flatten_name_like(
                c.get("country") or c.get(
                    "countryCode") or c.get("countryName")
                or (c.get("address") or {}).get("country")
            )

            if city or state or zipc or country:
                # build raw
                parts = []
                if city:
                    parts.append(city)
                if state:
                    parts.append(state)
                raw = ", ".join(parts)
                if zipc:
                    raw = (raw + f" {zipc}".rstrip()).strip()
                if country and country.upper() not in ("USA", "US", "UNITED STATES", "UNITED-STATES"):
                    raw = (raw + f" ({country})").strip()

                return {
                    "pop_city": city,
                    "pop_state": state,
                    "pop_zip": zipc,
                    "pop_country": country,
                    "pop_raw": raw,
                }

        return {}

    pop = _from_obj(rec)
    if not (pop.get("pop_city") or pop.get("pop_state") or pop.get("pop_zip") or pop.get("pop_country")):
        pop = _from_obj(detail or {})

    # ensure keys exist
    for k in ("pop_city", "pop_state", "pop_zip", "pop_country", "pop_raw"):
        pop.setdefault(k, "")

    return pop

# ---------- Main mapper ----------


def map_record_allowed_fields(
    rec: Dict[str, Any],
    *,
    api_keys: Optional[List[str]] = None,
    fetch_desc: bool = True
) -> Dict[str, Any]:
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

    # link (handy reference)
    link = "None"
    links = rec.get("links")
    if isinstance(links, list) and links:
        maybe = links[0]
        if isinstance(maybe, dict) and maybe.get("href"):
            link = str(maybe["href"])
    if link == "None":
        link = _first_nonempty(rec, "url", "samLink")

    # detail (search-by-id) – often has richer text fields
    search_detail: Dict[str, Any] = {}
    # entity detail – authoritative for placeOfPerformance
    entity_detail: Dict[str, Any] = {}

    if fetch_desc and api_keys and notice_id != "None":
        search_detail = fetch_notice_detail_v2(notice_id, api_keys)
        entity_detail = fetch_notice_entity_v2(notice_id, api_keys)

    # response_date: prefer SAM's dueDate (via _pick_response_date)
    response_date = _pick_response_date(rec, search_detail)

    # description: prefer inline; if missing/URL, use detail/entity, then noticedesc
    description = _first_nonempty(rec, "description", "synopsis", default="")

    def _looks_like_placeholder_or_url(t) -> bool:
        if t is None:
            return True
        s = _s(t).strip().lower()
        return s in ("none", "n/a", "na") or s.startswith("http://") or s.startswith("https://")

    if _looks_like_placeholder_or_url(description):
        detail_text = ""
        # try search detail text fields
        for k in ("description", "synopsis", "longDescription", "fullDescription", "additionalInfo"):
            val = (search_detail or {}).get(k)
            if val and str(val).strip():
                detail_text = re.sub(r"\s+", " ", str(val)).strip()
                break
        # try entity attributes.description
        if not detail_text and entity_detail:
            try:
                attrs = (entity_detail.get("data") or {}).get(
                    "attributes") or {}
                val = attrs.get("description")
                if val and str(val).strip():
                    detail_text = re.sub(r"\s+", " ", str(val)).strip()
            except Exception:
                pass
        # try legacy v1 noticedesc
        if not detail_text and fetch_desc and api_keys and notice_id != "None":
            detail_text = fetch_notice_description(notice_id, api_keys)

        description = detail_text if detail_text else ""

    # Final safety: normalize placeholders that may have slipped through
    if description and description.strip().lower() in ("none", "n/a", "na"):
        description = ""

    # --- Place of Performance ---
    pop = {"pop_city": "", "pop_state": "",
           "pop_zip": "", "pop_country": "", "pop_raw": ""}
    if entity_detail:
        pop = _extract_pop_from_notice_entity(entity_detail)
    # fallback to heuristic if entity was empty/absent
    if not any(pop.values()):
        pop = _extract_place_of_performance(rec, search_detail)

    mapped = {
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
    }

    mapped.update(pop)
    return mapped

# get_relevant_solicitations.py


def enrich_notice_fields(notice_id: str, api_keys: list[str]) -> dict:
    """
    Fetches per-notice details once and returns fields to backfill.
    Returns keys: description, pop_city, pop_state, pop_zip, pop_country, pop_raw
    """
    # Prefer the entity payload for PoP (authoritative)
    ent = fetch_notice_entity_v2(notice_id, api_keys)
    pop = _extract_pop_from_notice_entity(ent) if ent else {
        "pop_city": "", "pop_state": "", "pop_zip": "", "pop_country": "", "pop_raw": ""}

    # Description: try search-by-id then entity attributes.description, then v1 noticedesc
    # (re-use your existing function which tries multiple sources)
    desc = fetch_notice_description(notice_id, api_keys)

    return {
        "description": desc or "",
        **pop
    }
