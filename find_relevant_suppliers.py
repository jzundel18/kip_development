# Enhanced find_relevant_suppliers.py - COMPLETE VERSION WITH STREAMLIT DEBUG
"""
Enhanced vendor discovery with MAXIMUM flexibility and Streamlit debugging.
"""

from __future__ import annotations
import re
import time
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

import requests
import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# Configuration - VERY PERMISSIVE
# -----------------------------

AGGREGATOR_DOMAINS: set[str] = {
    "sam.gov", "beta.sam.gov", "govtribe.com", "fbo.gov"
}

USER_AGENT = "KIP_VendorFinder/1.0"

# -----------------------------
# Utility Functions
# -----------------------------


def _netloc(url: str) -> str:
    """Extract netloc from URL"""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _clean_text(x: Optional[str]) -> str:
    """Clean and normalize text"""
    if not x:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def _company_name_from_url(url: str) -> str:
    """Extract company name from URL"""
    try:
        domain = urlparse(url).netloc.lower()
        for prefix in ['www.', 'm.', 'en.']:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        name = domain.split('.')[0]
        return name.replace('-', ' ').replace('_', ' ').title()
    except:
        return "Company"


def _extract_service_type(title: str, description: str, openai_api_key: str) -> str:
    """Extract what type of service/product is needed"""
    if not openai_api_key or not OpenAI:
        text = f"{title} {description}".lower()
        if any(kw in text for kw in ["machining", "cnc", "fabrication", "manufacturing"]):
            return "manufacturing machining fabrication"
        elif any(kw in text for kw in ["maintenance", "repair", "installation", "service"]):
            return "maintenance repair service"
        elif any(kw in text for kw in ["software", "it", "technology"]):
            return "technology software IT"
        else:
            return "contractors suppliers services"

    try:
        client = OpenAI(api_key=openai_api_key)
        service_prompt = f"""Based on this solicitation, what 3-5 BROAD search terms would find providers?

Title: {title[:200]}
Description: {description[:400]}

Give me simple, broad keywords. Be GENERAL."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": service_prompt}],
            temperature=0.1,
            max_tokens=50,
            timeout=15
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "contractors suppliers services"


def _score_candidate(title: str, description: str, result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """ACCEPT EVERYTHING - minimal filtering"""
    vendor_url = _clean_text(result.get("link"))
    vendor_name = _clean_text(result.get("title"))
    snippet = _clean_text(result.get("snippet"))

    if not vendor_url:
        return 0.0, result

    host = _netloc(vendor_url)

    # ONLY filter these specific aggregator domains
    if host in AGGREGATOR_DOMAINS:
        return 0.0, result

    # ACCEPT EVERYTHING ELSE - give it a high score
    score = 10.0

    return score, {
        "name": vendor_name[:120] if vendor_name else _company_name_from_url(vendor_url),
        "website": vendor_url,
        "location": "",
        "snippet": snippet[:240],
        "host": host,
    }


def _build_search_queries(title: str, description: str, service_type: str, location: dict = None) -> List[str]:
    """Build MORE search queries with BROADER terms"""
    queries = []

    location_str = ""
    use_location = False

    if location and (location.get("city") or location.get("state")):
        city = location.get("city", "").strip()
        state = location.get("state", "").strip()

        if city and state:
            location_str = f"{city} {state}"
            use_location = True
        elif state:
            location_str = state
            use_location = True

    # Extract key terms
    key_terms = []
    for word in (title + " " + description).lower().split():
        if len(word) > 4 and word not in ["government", "federal", "agency"]:
            key_terms.append(word)

    main_terms = " ".join(key_terms[:3]) if key_terms else service_type

    # Build queries
    if use_location:
        queries.append(f"{service_type} {location_str}")
        queries.append(f"{service_type} near {location_str}")
        queries.append(f"{main_terms} {location_str}")
    else:
        queries.append(f"{service_type}")
        queries.append(f"{service_type} USA")
        queries.append(f"{main_terms} suppliers")

    queries.append(f"{service_type} contractors")
    queries.append(f"{service_type} companies")

    # Specific variations
    if "machining" in (title + description).lower():
        loc = f" {location_str}" if use_location else ""
        queries.append(f"machine shop{loc}")
        queries.append(f"CNC machining{loc}")
    elif "maintenance" in (title + description).lower():
        loc = f" {location_str}" if use_location else ""
        queries.append(f"maintenance services{loc}")
    elif "software" in (title + description).lower():
        loc = f" {location_str}" if use_location else ""
        queries.append(f"software development{loc}")

    queries.append(f"{main_terms} providers")
    queries.append(f"contractors {main_terms}")

    return queries[:8]


def _google_custom_search(query: str, api_key: str, cx: str, location: dict = None, max_results: int = 10) -> Dict[str, Any]:
    """Search using Google Custom Search JSON API"""
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": min(10, max_results),
        "safe": "off",
        "lr": "lang_en",
        "filter": "0",
    }

    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=25,
            headers={"User-Agent": USER_AGENT}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {"items": []}


def _generate_ai_reason(openai_api_key: str, solicitation_title: str, solicitation_desc: str,
                        vendor_name: str, vendor_url: str, vendor_snippet: str) -> str:
    """Generate AI reason for vendor recommendation"""
    if not openai_api_key or not OpenAI:
        return f"This vendor appears to offer relevant services."

    try:
        client = OpenAI(api_key=openai_api_key)
        prompt = f"""Why would this vendor be good for this work (1 sentence):

Work: {solicitation_title[:100]}
Vendor: {vendor_name}
About: {vendor_snippet[:150]}

Be positive and brief."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
            timeout=15
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return f"This vendor has relevant capabilities for the work."

# -----------------------------
# Main Search Function
# -----------------------------


def find_vendors_for_notice(
    sol: Dict[str, Any],
    google_api_key: str,
    google_cx: str,
    openai_api_key: str,
    max_google: int = 10,
    top_n: int = 3,
    return_debug: bool = False,
    streamlit_debug: Any = None,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    MAXIMUM FLEXIBILITY vendor search with Streamlit debugging
    """
    def debug_log(msg: str):
        """Log to streamlit if available"""
        if streamlit_debug is not None:
            streamlit_debug.write(msg)

    title = _clean_text(str(sol.get("title", "")))
    description = _clean_text(str(sol.get("description", "")))

    if not title and not description:
        debug_log("❌ No title or description provided")
        df = pd.DataFrame(columns=["name", "website", "location", "reason"])
        return (df, {"error": "empty_solicitation"} if return_debug else None)

    location = {
        "city": _clean_text(str(sol.get("pop_city", ""))),
        "state": _clean_text(str(sol.get("pop_state", "")))
    }

    debug_log(f"🔍 **Searching for vendors:**")
    debug_log(f"Title: {title[:100]}")
    if location.get("city") or location.get("state"):
        debug_log(
            f"Location: {location.get('city', '')} {location.get('state', '')}")
    else:
        debug_log(f"Location: National search")

    service_type = _extract_service_type(title, description, openai_api_key)
    debug_log(f"Service type: {service_type}")

    queries = _build_search_queries(title, description, service_type, location)
    debug_log(f"📋 Generated {len(queries)} search queries")

    debug = {"queries": queries, "location": location,
             "service_type": service_type, "raw": []}
    candidates = []
    seen_domains = set()

    for i, query in enumerate(queries):
        debug_log(f"")
        debug_log(f"**Query {i+1}/{len(queries)}:** `{query}`")

        try:
            search_results = _google_custom_search(
                query, google_api_key, google_cx, location, max_results=max_google
            )

            items = search_results.get("items", [])
            debug_log(f"   → Google returned {len(items)} results")

            if return_debug:
                debug["raw"].append({
                    "query": query,
                    "hits": len(items),
                    "data": search_results
                })

            if not items:
                debug_log(f"   ⚠️ No results from Google")
                continue

            accepted_count = 0
            for j, item in enumerate(items):
                score, candidate = _score_candidate(title, description, item)

                if score > 0:
                    host = candidate.get("host", "")

                    if host and host in seen_domains:
                        continue

                    if host:
                        seen_domains.add(host)

                    candidate["score"] = score
                    candidate["query_used"] = query
                    candidates.append(candidate)
                    accepted_count += 1

            debug_log(f"   ✅ Accepted {accepted_count} candidates")

        except Exception as e:
            debug_log(f"   ❌ Query failed: {str(e)[:100]}")
            if return_debug:
                debug["raw"].append({"query": query, "error": str(e)})
            continue

    debug_log(f"")
    debug_log(f"📊 **Total candidates: {len(candidates)}**")

    if not candidates:
        debug_log(f"")
        debug_log(f"❌ **No candidates found**")
        debug_log(f"Possible reasons:")
        debug_log(f"  1. Google API returned no results")
        debug_log(f"  2. All results were sam.gov/govtribe")
        debug_log(f"  3. API credentials incorrect")
        df = pd.DataFrame(columns=["name", "website", "location", "reason"])
        return (df, debug if return_debug else None)

    debug_log(f"")
    debug_log(f"🎯 **Selecting top {top_n} vendors...**")

    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_candidates = candidates[:min(top_n * 2, len(candidates))]

    rows = []
    for i, candidate in enumerate(top_candidates[:top_n], 1):
        debug_log(
            f"{i}. {candidate.get('name', 'Unknown')[:50]} - {candidate.get('website', '')[:50]}")

        reason = _generate_ai_reason(
            openai_api_key, title, description,
            candidate.get("name", ""), candidate.get("website", ""),
            candidate.get("snippet", "")
        )

        snippet = candidate.get("snippet", "")
        location_match = re.search(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b', snippet)
        extracted_location = ""
        if location_match:
            extracted_location = f"{location_match.group(1)}, {location_match.group(2)}"
        elif location.get("state"):
            extracted_location = f"Serving {location['state']}"

        rows.append({
            "name": candidate.get("name", ""),
            "website": candidate.get("website", ""),
            "location": extracted_location,
            "reason": reason,
        })

    debug_log(f"")
    debug_log(f"✅ **Found {len(rows)} vendors**")

    df = pd.DataFrame(rows, columns=["name", "website", "location", "reason"])
    return (df, debug if return_debug else None)

# -----------------------------
# Service Vendor Search
# -----------------------------


def find_service_vendors_for_opportunity(
    solicitation: dict,
    google_api_key: str,
    google_cx: str,
    openai_api_key: str,
    top_n: int = 3,
    streamlit_debug: Any = None
) -> tuple:
    """Find service vendors with Streamlit debugging"""
    try:
        vendors_df, _ = find_vendors_for_notice(
            sol=solicitation,
            google_api_key=google_api_key,
            google_cx=google_cx,
            openai_api_key=openai_api_key,
            max_google=15,
            top_n=top_n,
            return_debug=False,
            streamlit_debug=streamlit_debug  # <-- ADD THIS LINE
        )

        pop_city = (solicitation.get("pop_city") or "").strip()
        pop_state = (solicitation.get("pop_state") or "").strip()

        if pop_city and pop_state:
            note = f"Searched {pop_city}, {pop_state}"
        elif pop_state:
            note = f"Searched {pop_state}"
        else:
            note = f"National search"

        return vendors_df, note

    except Exception as e:
        if streamlit_debug:
            streamlit_debug.error(f"❌ Error: {e}")
            import traceback
            streamlit_debug.code(traceback.format_exc())
        return None, f"Error: {str(e)[:100]}"

# -----------------------------
# Compatibility
# -----------------------------


def get_suppliers(
    solicitations: List[Dict[str, Any]],
    our_recommended_suppliers: List[str] | None = None,
    our_not_recommended_suppliers: List[str] | None = None,
    Max_Google_Results: int = 10,
    OpenAi_API_Key: str | None = None,
    Google_API_Key: str | None = None,
    Google_CX: str | None = None,
) -> List[Dict[str, Any]]:
    """Compatibility wrapper"""
    results = []

    for sol in solicitations:
        try:
            df, _ = find_vendors_for_notice(
                sol=sol,
                google_api_key=Google_API_Key or "",
                google_cx=Google_CX or "",
                openai_api_key=OpenAi_API_Key or "",
                max_google=Max_Google_Results,
                top_n=3,
                return_debug=False,
            )

            notice_id = str(sol.get("notice_id", ""))
            for _, row in df.iterrows():
                results.append({
                    "notice_id": notice_id,
                    "name": row.get("name", ""),
                    "website": row.get("website", ""),
                    "location": row.get("location", ""),
                    "reason": row.get("reason", ""),
                })
        except Exception:
            continue

    return results
