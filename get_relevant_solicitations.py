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
        # enforce notice_id match if supplied
        if filters.get("notice_id"):
            rid = str(rec.get("noticeId") or rec.get("id") or "").strip()
            if rid != str(filters["notice_id"]).strip():
                return False

        r_type_raw = str(rec.get("noticeType") or rec.get("type") or "")
        if r_type_raw.strip().lower() == "justification":
            return False
        # Drop Award Notices
        nt = str(rec.get("noticeType") or rec.get(
            "type") or "").strip().lower()
        if "award" in nt:
            return False

        # (existing filters you already had)
        nts = filters.get("notice_types") or []
        if nts:
            r_type = nt
            if not r_type or not any(t.lower() in r_type for t in nts):
                return False

        kws = [k.strip().lower()
               for k in (filters.get("keywords_or") or []) if k.strip()]
        if kws:
            title = str(rec.get("title") or "").lower()
            desc = str(rec.get("description")
                       or rec.get("synopsis") or "").lower()
            blob = f"{title} {desc}"
            if not any(k in blob for k in kws):
                return False

        naics_targets = [n for n in (filters.get("naics") or []) if n]
        if naics_targets:
            rec_naics = str(rec.get("naicsCode")
                            or rec.get("naics") or "").strip()
            if not rec_naics or rec_naics not in naics_targets:
                return False

        sas = filters.get("set_asides") or []
        if sas:
            rec_sa = str(rec.get("setAsideCode")
                         or rec.get("setAside") or "").lower()
            if not rec_sa or not any(sa.lower() in rec_sa for sa in sas):
                return False

        due_before = filters.get("due_before")
        if due_before:
            raw = (rec.get("dueDate") or rec.get("responseDueDate") or
                   rec.get("closeDate") or rec.get("responseDate") or
                   rec.get("responseDateTime") or "")
            resp_norm = _normalize_date(raw)
            if resp_norm != "None" and resp_norm > str(due_before):
                return False

        return True

    return [r for r in raw_records if _match(r)]


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


def _first_nonempty(obj: Dict[str, Any], *keys: str, default: str = "None") -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, dict):
            # pick common subkeys if dict
            for sub in ("name", "value", "code", "text"):
                if v.get(sub):
                    return str(v[sub])
            continue
        if v is not None and str(v).strip() != "":
            return str(v)
    return default

# ---------- Place of Performance extraction ----------


def _extract_pop_from_notice_entity(detail_json: dict) -> dict:
    """
    Extract PoP from the official entity payload:
    data.attributes.placeOfPerformance.address{ city{name}, state{code|name}, zip, country{code|name} }
    Returns {pop_city, pop_state, pop_zip, pop_country, pop_raw} with empty strings if not found.
    """
    def _sv(x):
        if x is None:
            return ""
        s = str(x).strip()
        return "" if s.lower() in ("none", "n/a", "na") else s

    out = {"pop_city": "", "pop_state": "",
           "pop_zip": "", "pop_country": "", "pop_raw": ""}
    try:
        attrs = (detail_json.get("data") or {}).get("attributes") or {}
        addr = (attrs.get("placeOfPerformance") or {}).get("address") or {}
        # city can be string or object { name }
        city = addr.get("city")
        city_name = (city or {}).get(
            "name") if isinstance(city, dict) else city
        # state can be string or object { code|name }
        state = addr.get("state")
        state_code = (state or {}).get(
            "code") if isinstance(state, dict) else None
        state_name = (state or {}).get(
            "name") if isinstance(state, dict) else None
        # country can be string or object { code|name }
        country = addr.get("country")
        country_code = (country or {}).get(
            "code") if isinstance(country, dict) else None
        country_name = (country or {}).get(
            "name") if isinstance(country, dict) else None
        # zip variants
        zip_code = addr.get("zip") or addr.get(
            "zipCode") or addr.get("postalCode")

        out["pop_city"] = _sv(city_name)
        out["pop_state"] = _sv(state_code or state_name)
        out["pop_zip"] = _sv(zip_code)
        out["pop_country"] = _sv(country_code or country_name)

        bits = []
        if out["pop_city"]:
            bits.append(out["pop_city"])
        if out["pop_state"]:
            bits.append(out["pop_state"])
        raw = ", ".join(bits)
        if out["pop_zip"]:
            raw = (raw + f" {out['pop_zip']}".rstrip()).strip()
        if out["pop_country"] and out["pop_country"].upper() not in ("USA", "US", "UNITED STATES", "UNITED-STATES"):
            raw = (raw + f" ({out['pop_country']})").strip()
        out["pop_raw"] = raw
    except Exception:
        pass
    return out


def _extract_place_of_performance(rec: dict, detail: dict | None = None) -> dict:
    """
    Fallback heuristic when entity payload is unavailable.
    Returns normalized dict: {"pop_city","pop_state","pop_zip","pop_country","pop_raw"}.
    """
    def _from_obj(obj: dict | None) -> dict:
        if not isinstance(obj, dict):
            return {}
        # potential containers
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
        # addresses list
        for k in ("addresses", "locations", "placeOfPerformanceAddresses"):
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                candidates.append(v[0])

        # also allow using the object itself if it already looks like an address
        candidates.append(obj)

        best = {}
        for c in candidates:
            if not isinstance(c, dict):
                continue
            city = c.get("city") or c.get("cityName") or (
                c.get("address") or {}).get("city")
            state = (c.get("state") or c.get("stateCode") or (c.get("address") or {}).get("state") or
                     c.get("stateProvince") or c.get("stateProvinceCode"))
            zipc = c.get("zip") or c.get("zipCode") or c.get(
                "postalCode") or (c.get("address") or {}).get("postalCode")
            country = c.get("country") or c.get("countryCode") or c.get(
                "countryName") or (c.get("address") or {}).get("country")
            # heuristic: if at least state or city is present, accept
            if city or state or zipc or country:
                best = {"pop_city": _s(city),
                        "pop_state": _s(state),
                        "pop_zip": _s(zipc),
                        "pop_country": _s(country)}
                break
        return best

    pop = {}
    pop.update(_from_obj(rec))
    if not (pop.get("pop_city") or pop.get("pop_state") or pop.get("pop_zip") or pop.get("pop_country")):
        pop.update(_from_obj(detail or {}))

    # build raw pretty string
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

    pop["pop_raw"] = raw
    # ensure all keys exist
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
    description = _s(_first_nonempty(
        rec, "description", "synopsis", default=""))

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
