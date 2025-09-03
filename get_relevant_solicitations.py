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
SAM_NOTICE_ENTITY_V2 = "https://api.sam.gov/prod/opportunities/v2/notices/{notice_id}"

# ---- Custom errors ----


class SamQuotaError(Exception):
    ...


class SamAuthError(Exception):
    ...


class SamBadRequestError(Exception):
    ...


# --- Date helpers ---
_DATE_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DATE_US_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")


def _safe_str(val: Any) -> str:
    return _to_str(val)


def _normalize_date(val: Any) -> str:
    if not val:
        return "None"
    s = _safe_str(val)
    if not s or s.lower() in ("none", "n/a", "na"):
        return "None"
    m = _DATE_ISO_RE.search(s)
    if m:
        return m.group(1)
    m = _DATE_US_RE.search(s)
    if m:
        mm, dd, yyyy = m.group(1).split("/")
        return f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}"
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
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v).strip()
    try:
        return json.dumps(v, ensure_ascii=False).strip()
    except Exception:
        return str(v).strip()


def _to_str(val: Any) -> str:
    """Flatten dicts/lists and return a plain string. Never a JSON blob."""
    if val is None:
        return ""
    if isinstance(val, list):
        for it in val:
            s = _to_str(it)
            if s:
                return s
        return ""
    if isinstance(val, dict):
        for k in ("name", "text", "value", "code"):
            if k in val and val[k] not in (None, "", []):
                return str(val[k]).strip()
        for v in val.values():
            s = _to_str(v)
            if s:
                return s
        return ""
    return str(val).strip()


def _lower(val: Any) -> str:
    return _to_str(val).lower()


def _upper(val: Any) -> str:
    return _to_str(val).upper()
# --- HTTP core with key rotation ---


def _request_sam(params: Dict[str, Any], api_keys: List[str]) -> Dict[str, Any]:
    if not api_keys:
        raise ValueError("No SAM.gov API keys provided.")
    errors: list[str] = []
    for key in api_keys:
        try:
            full_params = dict(params)
            full_params["api_key"] = key
            resp = requests.get(SAM_BASE_URL, params=full_params, timeout=30)

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

# -------------------- SEARCH (page) --------------------


def get_sam_raw_v3(
    days_back: int,
    limit: int,
    api_keys: List[str],
    filters: Optional[Dict[str, Any]] = None,
    *,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    filters = filters or {}
    posted_from, posted_to = _window_days_back(days_back)

    params = {
        "limit": int(limit),
        "postedFrom": posted_from,
        "postedTo": posted_to,
        "offset": int(offset),
    }
    if filters.get("notice_id"):
        params["noticeid"] = _safe_str(filters["notice_id"])
        params["limit"] = 1

    data = _request_sam(params, api_keys)
    raw_records = data.get("opportunitiesData") or data.get("data") or []
    if not raw_records:
        return []

    def _match(rec: Dict[str, Any]) -> bool:
        # enforce notice_id
        if filters.get("notice_id"):
            rid = _safe_str(rec.get("noticeId") or rec.get("id"))
            if rid != _safe_str(filters["notice_id"]):
                return False

        nt = _safe_str(rec.get("noticeType") or rec.get("type")).lower()

        # drop types we don't want
        if nt == "justification":
            return False
        if "award" in nt:
            return False

        nts = [_safe_str(t).lower() for t in (
            filters.get("notice_types") or []) if _safe_str(t)]
        kws = [_safe_str(k).lower() for k in (
            filters.get("keywords_or") or []) if _safe_str(k)]
        naics_targets = [_safe_str(n) for n in (
            filters.get("naics") or []) if _safe_str(n)]
        sas = [_safe_str(s).lower() for s in (
            filters.get("set_asides") or []) if _safe_str(s)]

        if nts:
            if not nt or not any(t in nt for t in nts):
                return False

        if kws:
            title = _safe_str(rec.get("title")).lower()
            desc = _safe_str(rec.get("description")
                             or rec.get("synopsis")).lower()
            blob = f"{title} {desc}".strip()
            if not any(k in blob for k in kws):
                return False

        if naics_targets:
            rec_naics = _safe_str(rec.get("naicsCode") or rec.get("naics"))
            if not rec_naics or rec_naics not in naics_targets:
                return False

        if sas:
            rec_sa = _safe_str(rec.get("setAsideCode")
                               or rec.get("setAside")).lower()
            if not rec_sa or not any(sa in rec_sa for sa in sas):
                return False

        due_before = filters.get("due_before")
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

    return [r for r in raw_records if _match(r)]


def get_raw_sam_solicitations(limit: int, api_keys: List[str]) -> List[Dict[str, Any]]:
    return get_sam_raw_v3(days_back=0, limit=limit, api_keys=api_keys, filters={})

# -------------------- DETAIL HELPERS --------------------


def _rotate_keys(keys: List[str]):
    while True:
        for k in keys:
            yield k


def _http_get(url: str, params: dict, key: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": "kip_development/1.0"}
    return requests.get(url, params={**params, "api_key": key}, headers=headers, timeout=timeout)


def fetch_notice_detail_v2(notice_id: str, api_keys: List[str]) -> Dict[str, Any]:
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
    detail = fetch_notice_detail_v2(notice_id, api_keys)
    for k in ("description", "synopsis", "longDescription", "fullDescription", "additionalInfo"):
        val = detail.get(k)
        if _safe_str(val):
            return re.sub(r"\s+", " ", _safe_str(val)).strip()

    ent = fetch_notice_entity_v2(notice_id, api_keys)
    try:
        attrs = (ent.get("data") or {}).get("attributes") or {}
        val = attrs.get("description")
        if _safe_str(val):
            return re.sub(r"\s+", " ", _safe_str(val)).strip()
    except Exception:
        pass

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
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                return text
        except Exception:
            time.sleep(0.5)
            continue

    return ""

# -------------------- UTIL: FIRST/NONEMPTY --------------------


def _first_nonempty(obj: Dict[str, Any], *keys: str, default: str = "None") -> str:
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

# -------------------- POP EXTRACTION --------------------


def _flatten_name_like(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        for it in val:
            s = _flatten_name_like(it)
            if s:
                return s.strip()
        return ""
    if isinstance(val, dict):
        for k in ("name", "text", "value", "code"):
            if k in val and val[k] not in (None, "", []):
                return str(val[k]).strip()
        for v in val.values():
            s = _flatten_name_like(v)
            if s:
                return s.strip()
        return ""
    return str(val).strip()


def _extract_pop_from_notice_entity(detail_json: dict) -> dict:
    out = {"pop_city": "", "pop_state": "",
           "pop_zip": "", "pop_country": "", "pop_raw": ""}
    try:
        attrs = (detail_json.get("data") or {}).get("attributes") or {}
        addr = (attrs.get("placeOfPerformance") or {}).get("address") or {}

        city = _flatten_name_like(addr.get("city"))
        state = _flatten_name_like(addr.get("state"))
        country = _flatten_name_like(addr.get("country"))
        zipc = _flatten_name_like(addr.get("zip") or addr.get(
            "zipCode") or addr.get("postalCode"))

        out["pop_city"] = city
        out["pop_state"] = state
        out["pop_zip"] = zipc
        out["pop_country"] = country

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
        pass
    return out


def _extract_place_of_performance(rec: dict, detail: dict | None = None) -> dict:
    def _from_obj(obj: dict | None) -> dict:
        if not isinstance(obj, dict):
            return {}
        candidates = []
        for k in (
            "placeOfPerformance", "place_of_performance", "placeOfPerformanceAddress",
            "primaryPlaceOfPerformance", "popAddress", "placeOfPerformanceLocation",
            "place_of_performance_location", "placeOfPerformanceCityState",
        ):
            v = obj.get(k)
            if isinstance(v, dict):
                candidates.append(v)
        for k in ("addresses", "locations", "placeOfPerformanceAddresses"):
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                candidates.append(v[0])
        candidates.append(obj)

        best = {}
        for c in candidates:
            if not isinstance(c, dict):
                continue
            city = _flatten_name_like(c.get("city") or c.get(
                "cityName") or (c.get("address") or {}).get("city"))
            state = _flatten_name_like(c.get("state") or c.get("stateCode") or (c.get(
                "address") or {}).get("state") or c.get("stateProvince") or c.get("stateProvinceCode"))
            zipc = _flatten_name_like(c.get("zip") or c.get("zipCode") or c.get(
                "postalCode") or (c.get("address") or {}).get("postalCode"))
            country = _flatten_name_like(c.get("country") or c.get("countryCode") or c.get(
                "countryName") or (c.get("address") or {}).get("country"))

            if city or state or zipc or country:
                best = {"pop_city": _safe_str(city), "pop_state": _safe_str(
                    state), "pop_zip": _safe_str(zipc), "pop_country": _safe_str(country)}
                break
        return best

    pop = _from_obj(rec)
    if not (pop.get("pop_city") or pop.get("pop_state") or pop.get("pop_zip") or pop.get("pop_country")):
        pop = _from_obj(detail or {})

    parts = []
    if pop.get("pop_city"):
        parts.append(pop["pop_city"])
    if pop.get("pop_state"):
        parts.append(pop["pop_state"])
    raw = ", ".join(parts)
    if pop.get("pop_zip"):
        raw = (raw + f" {pop['pop_zip']}".rstrip()).strip()
    if pop.get("pop_country") and _safe_str(pop["pop_country"]).upper() not in {"USA", "US", "UNITED STATES", "UNITED-STATES"}:
        raw = (raw + f" ({pop['pop_country']})").strip()

    return {
        "pop_city":    pop.get("pop_city", "") or "",
        "pop_state":   pop.get("pop_state", "") or "",
        "pop_zip":     pop.get("pop_zip", "") or "",
        "pop_country": pop.get("pop_country", "") or "",
        "pop_raw":     raw,
    }

# -------------------- MAPPING --------------------


def _take_text_field(obj: dict, keys: list[str]) -> str:
    for k in keys:
        if k in obj and obj[k] not in (None, "", []):
            return _safe_str(obj[k])
    return ""


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

    link = "None"
    links = rec.get("links")
    if isinstance(links, list) and links:
        maybe = links[0]
        if isinstance(maybe, dict) and maybe.get("href"):
            link = str(maybe["href"])
    if link == "None":
        link = _first_nonempty(rec, "url", "samLink")

    search_detail: Dict[str, Any] = {}
    entity_detail: Dict[str, Any] = {}
    if fetch_desc and api_keys and notice_id != "None":
        search_detail = fetch_notice_detail_v2(notice_id, api_keys)
        entity_detail = fetch_notice_entity_v2(notice_id, api_keys)

    response_date = _pick_response_date(rec, search_detail)

    description = _take_text_field(rec, ["description", "synopsis"])
    ds = _safe_str(description).lower()
    if (not ds) or ds.startswith(("http://", "https://")) or ds in ("none", "n/a", "na"):
        detail_text = ""
        if search_detail:
            detail_text = _take_text_field(
                search_detail,
                ["description", "synopsis", "longDescription",
                    "fullDescription", "additionalInfo"]
            )
        if not detail_text and entity_detail:
            try:
                attrs = (entity_detail.get("data") or {}).get(
                    "attributes") or {}
                detail_text = _safe_str(attrs.get("description"))
            except Exception:
                pass
        if not detail_text and fetch_desc and api_keys and notice_id != "None":
            detail_text = fetch_notice_description(notice_id, api_keys)
        description = _safe_str(detail_text) or ""
    if _safe_str(description).lower() in ("none", "n/a", "na"):
        description = ""

    pop = {"pop_city": "", "pop_state": "",
           "pop_zip": "", "pop_country": "", "pop_raw": ""}
    if entity_detail:
        pop = _extract_pop_from_notice_entity(entity_detail)
    if not any(_safe_str(v) for v in pop.values()):
        pop = _extract_place_of_performance(rec, search_detail)

    mapped = {
        "notice_id":           _stringify(notice_id),
        "solicitation_number": _stringify(solicitation_number),
        "title":               _stringify(title),
        "notice_type":         _stringify(notice_type),
        "posted_date":         _stringify(posted_date),
        "response_date":       _stringify(response_date),
        "archive_date":        _stringify(archive_date),
        "naics_code":          _stringify(naics_code),
        "set_aside_code":      _stringify(set_aside_code),
        "description":         _stringify(description),
        "link":                _stringify(link),
    }
    mapped.update(pop)
    return mapped

# -------------------- ENRICH --------------------


def enrich_notice_fields(notice_id: str, api_keys: list[str]) -> dict:
    ent = fetch_notice_entity_v2(notice_id, api_keys)
    pop = _extract_pop_from_notice_entity(ent) if ent else {
        "pop_city": "", "pop_state": "", "pop_zip": "", "pop_country": "", "pop_raw": ""
    }
    desc = fetch_notice_description(notice_id, api_keys)
    return {"description": desc or "", **pop}
