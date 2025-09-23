"""
find_relevant_suppliers.py

Vendor discovery helpers for KIP - Updated to use Google Custom Search JSON API.

Goals:
- Given a solicitation (title/description/etc) find real vendors (manufacturers/distributors/system integrators)
- Avoid aggregator/bid repost sites (sam.gov, govtribe, bidnet, etc.)
- Return a clean DataFrame with: name, website, location (best-effort), reason
- Be resilient when fields are missing; never crash if title/description is None

This module exposes two public entry points:

- find_vendors_for_notice(sol: dict, google_api_key: str, google_cx: str, openai_api_key: str, max_google: int = 8, top_n: int = 3, return_debug: bool = False)
    -> DataFrame (and optional debug dict)

- get_suppliers(solicitations: list[dict], our_recommended_suppliers: list[str], our_not_recommended_suppliers: list[str], Max_Google_Results: int, OpenAi_API_Key: str, Google_API_Key: str, Google_CX: str)
    -> Compatibility wrapper returning a list of rows across solicitations, preserving the old signature used elsewhere.

Notes:
- Network calls are done via Google Custom Search JSON API by simple HTTP; we avoid parsing HTML.
- LLM is used only to score and produce a short reason; failures fall back to heuristic reasons.
"""
from __future__ import annotations

import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import requests
import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# -----------------------------
# Domain filters / heuristics
# -----------------------------

AGGREGATOR_DOMAINS: set[str] = {
    "sam.gov", "beta.sam.gov", "govtribe.com", "bidnet.com", "bidnetdirect.com",
    "bidlink.net", "govwin.com", "govology.com", "usaspending.gov", "usaopps.com",
    "grants.gov", "fedconnect.net", "fbo.gov", "highergov.com", "govsearch.com",
    "govdirections.com", "procureport.com", "periscopeholdings.com", "p1.gov",
    "app.box.com", "state.gov", "mil", "army.mil", "navy.mil", "af.mil", "af.mil",
    "defense.gov", "dfars", "codeofsupport.org", "linkedin.com", "facebook.com",
    "twitter.com", "instagram.com", "x.com", "youtube.com", "tiktok.com",
}

AGGREGATOR_KEYWORDS: tuple[str, ...] = (
    "solicitation", "rfq", "rfp", "sources sought", "notice", "sam.gov", "bid", "tender",
    "opportunity", "govtribe", "bidnet", "government", "contract", "award", "posting",
)

LIKELY_VENDOR_KEYWORDS: tuple[str, ...] = (
    "manufacturer", "supplier", "distributor", "oem", "fabrication", "machine shop",
    "cnc", "molding", "turning", "milling", "weld", "assembly", "reseller",
    "parts", "components",
)

# Some public suffixes that are commonly vendors (heuristic; we do *not* outright block .gov)
VENDOR_TLDS: tuple[str, ...] = (".com", ".net", ".co", ".io", ".us", ".biz",
                                ".tech", ".systems", ".solutions", ".engineering", ".industries")


def _netloc(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_aggregator(url: str, title: str, snippet: str) -> bool:
    host = _netloc(url)
    if not host:
        return True
    host_agg = any(h in host for h in AGGREGATOR_DOMAINS)
    kw_agg = any(k in (title + " " + snippet).lower()
                 for k in AGGREGATOR_KEYWORDS)
    return host_agg or kw_agg


def _looks_like_vendor(url: str) -> bool:
    host = _netloc(url)
    if not host:
        return False
    return host.endswith(VENDOR_TLDS)


def _clean_text(x: Optional[str]) -> str:
    if not x:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def _extract_part_tokens(title: str, description: str) -> List[str]:
    """
    Pulls candidate tokens like part numbers / NSNs and meaningful nouns.
    """
    s = f"{title} {description}".upper()
    # NSN-like (e.g., 5305-00-123-4567) or part-ish words with hyphens/digits
    tokens = set(re.findall(r"[A-Z0-9]{3,}(?:-[A-Z0-9]{2,})+", s))
    # add long alphanum words
    tokens |= set(re.findall(r"\b[A-Z0-9]{6,}\b", s))
    # ensure at least some tokens
    out = [t for t in tokens if not t.isdigit()]
    return out[:6]


def _build_queries(title: str, description: str) -> List[str]:
    title = _clean_text(title)
    description = _clean_text(description)

    # Primary query focuses on title with vendor-intent terms
    base = title or description[:80]
    vendor_terms = " manufacturer OR supplier OR distributor OR OEM"
    q1 = f'{base} {vendor_terms}'

    # Secondary query: add extracted tokens
    tokens = _extract_part_tokens(title, description)
    q2 = f'{base} {" ".join(tokens)} {vendor_terms}'.strip()

    # Last chance: narrow to machine shop/manufacturing terms
    q3 = f'{base} ("machine shop" OR fabrication OR "precision machining" OR CNC)'

    # Deduplicate while preserving order
    seen = set()
    out = []
    for q in [q1, q2, q3]:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _google_custom_search(query: str, api_key: str, cx: str, max_results: int = 10) -> Dict[str, Any]:
    """
    Search using Google Custom Search JSON API
    
    Args:
        query: Search query string
        api_key: Google API key
        cx: Custom Search Engine ID
        max_results: Maximum number of results to return
    
    Returns:
        Dict containing search results in Google Custom Search format
    """
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        # Google Custom Search max is 10 per request
        "num": min(10, max_results),
        "safe": "off",
        "lr": "lang_en",  # Prefer English results
    }

    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1", params=params, timeout=25)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        # Return empty structure that matches expected format
        return {"items": []}


def _llm_reason(openai_api_key: str, solicitation_title: str, solicitation_desc: str, vendor_name: str, vendor_url: str, vendor_snippet: str) -> str:
    if not openai_api_key or OpenAI is None:
        return ""
    try:
        client = OpenAI(api_key=openai_api_key)
        sys = "You are a concise sourcing analyst. One sentence. No fluff."
        user = (
            f"Solicitation title:\n{solicitation_title}\n\n"
            f"Solicitation description:\n{(solicitation_desc or '')[:1500]}\n\n"
            f"Vendor candidate:\nName: {vendor_name}\nURL: {vendor_url}\nSnippet: {vendor_snippet}\n\n"
            "In one short sentence, explain why this vendor likely can provide the item/service. "
            "Do not repeat the URL."
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _score_candidate(title: str, description: str, result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Heuristic score combining presence of vendor keywords and overlap with title/part tokens.
    Adapted for Google Custom Search JSON API format.
    """
    # Google Custom Search format: {"title": "...", "link": "...", "snippet": "..."}
    vendor_url = _clean_text(result.get("link"))
    vendor_name = _clean_text(result.get("title"))
    snippet = _clean_text(result.get("snippet"))

    if not vendor_url:
        return 0.0, result

    if _is_aggregator(vendor_url, vendor_name, snippet):
        return 0.0, result
    if not _looks_like_vendor(vendor_url):
        # Allow non-standard TLDs but with strong vendor keywords
        if not any(k in (vendor_name + " " + snippet).lower() for k in LIKELY_VENDOR_KEYWORDS):
            return 0.0, result

    base = (title + " " + description).lower()
    score = 0.0
    for k in LIKELY_VENDOR_KEYWORDS:
        if k in (vendor_name + " " + snippet).lower():
            score += 1.0

    # overlap with tokens
    tokens = [t.lower() for t in _extract_part_tokens(title, description)]
    for t in tokens:
        if t and t in (vendor_name + " " + snippet).lower():
            score += 1.0

    # small bump if domain isn't obviously marketplace (amazon/ebay/alibaba)
    host = _netloc(vendor_url)
    if not any(m in host for m in ("amazon.", "ebay.", "alibaba.", "walmart.", "grainger.", "mcmaster.", "zoro.com")):
        score += 0.5

    return score, {
        "name": vendor_name[:120] or host.split(":")[0],
        "website": vendor_url,
        "location": "",
        "snippet": snippet[:240],
        "host": host,
    }


def _pick_top_candidates(title: str, description: str, google_json: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Extract top candidates from Google Custom Search results.
    Google format: {"items": [{"title": "...", "link": "...", "snippet": "..."}, ...]}
    """
    items = google_json.get("items", [])
    candidates: List[Tuple[float, Dict[str, Any]]] = []

    for item in items:
        sc, payload = _score_candidate(title, description, item)
        if sc > 0:
            candidates.append((sc, payload))

    candidates.sort(key=lambda x: x[0], reverse=True)
    out = []
    for sc, payload in candidates:
        out.append(payload)
        if len(out) >= limit:
            break
    return out


def find_vendors_for_notice(
    sol: Dict[str, Any],
    google_api_key: str,
    google_cx: str,
    openai_api_key: str,
    max_google: int = 8,
    top_n: int = 3,
    return_debug: bool = False,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Primary function used by the Streamlit UI.

    sol must include: title, description (best effort), and optionally naics_code etc.
    Returns a DataFrame with up to top_n vendors and (optionally) a debug dict of raw Google data.
    
    Args:
        sol: Solicitation dictionary with title, description, etc.
        google_api_key: Google API key
        google_cx: Google Custom Search Engine ID  
        openai_api_key: OpenAI API key for generating reasons
        max_google: Maximum results per Google query
        top_n: Number of final vendors to return
        return_debug: Whether to return debug information
    """
    title = _clean_text(str(sol.get("title", "")))
    description = _clean_text(str(sol.get("description", "")))
    if not title and not description:
        df = pd.DataFrame(columns=["name", "website", "location", "reason"])
        return (df, {"error": "empty_solicitation"} if return_debug else None)

    queries = _build_queries(title, description)

    debug: Dict[str, Any] = {"queries": queries, "raw": []}
    accepted: List[Dict[str, Any]] = []

    for qi, q in enumerate(queries, start=1):
        try:
            data = _google_custom_search(
                q, google_api_key, google_cx, max_results=max_google)
            if return_debug:
                debug["raw"].append({"query": q, "hits": len(
                    data.get("items", [])), "data": data})

            picks = _pick_top_candidates(
                title, description, data, limit=top_n * 2)
            # De-dup by domain
            seen = set()
            for p in picks:
                host = p.get("host", "")
                if host in seen:
                    continue
                seen.add(host)
                accepted.append(p)
                if len(accepted) >= top_n:
                    break

            if accepted:
                break  # got enough

        except Exception as e:
            if return_debug:
                debug["raw"].append({"query": q, "error": str(e)})

    # Build DataFrame and add AI reason
    rows: List[Dict[str, Any]] = []
    for p in accepted[:top_n]:
        reason = _llm_reason(openai_api_key, title, description, p.get(
            "name", ""), p.get("website", ""), p.get("snippet", ""))
        rows.append({
            "name": p.get("name", ""),
            "website": p.get("website", ""),
            "location": p.get("location", ""),
            "reason": reason or p.get("snippet", ""),
        })

    df = pd.DataFrame(rows, columns=["name", "website", "location", "reason"])
    return (df, debug if return_debug else None)


# --------------------------------------
# Back-compat: original get_suppliers API
# --------------------------------------
def get_suppliers(
    solicitations: List[Dict[str, Any]],
    our_recommended_suppliers: List[str] | None = None,
    our_not_recommended_suppliers: List[str] | None = None,
    Max_Google_Results: int = 8,
    OpenAi_API_Key: str | None = None,
    Google_API_Key: str | None = None,
    Google_CX: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Returns a flat list of vendor rows across all solicitations to keep old callers working.
    Each row: {"notice_id","name","website","location","reason"}
    
    Updated to use Google Custom Search instead of SerpAPI.
    """
    out: List[Dict[str, Any]] = []
    our_recommended_suppliers = our_recommended_suppliers or []
    our_not_recommended_suppliers = our_not_recommended_suppliers or []

    for sol in solicitations:
        df, _ = find_vendors_for_notice(
            sol=sol,
            google_api_key=Google_API_Key or "",
            google_cx=Google_CX or "",
            openai_api_key=OpenAi_API_Key or "",
            max_google=Max_Google_Results,
            top_n=3,
            return_debug=False,
        )
        nid = str(sol.get("notice_id", ""))
        for _, r in df.iterrows():
            out.append({
                "notice_id": nid,
                "name": r.get("name", ""),
                "website": r.get("website", ""),
                "location": r.get("location", ""),
                "reason": r.get("reason", ""),
            })
    return out
