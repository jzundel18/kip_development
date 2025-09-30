# Enhanced find_relevant_suppliers.py - MUCH MORE PERMISSIVE VERSION
"""
Enhanced vendor discovery with MAXIMUM flexibility to find results.

Key improvements:
1. Much more permissive filtering - accepts almost everything
2. More search queries with broader terms
3. Less strict deduplication
4. Lower scoring thresholds
5. Better fallback mechanisms
"""

from __future__ import annotations
import os
import re
import json
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

# Only exclude obvious aggregators - much shorter list
AGGREGATOR_DOMAINS: set[str] = {
    "sam.gov", "beta.sam.gov", "govtribe.com"
}

# Minimal exclusion keywords
AGGREGATOR_KEYWORDS: tuple[str, ...] = (
    "sam.gov", "govtribe"
)

# Accept almost any TLD
POTENTIAL_VENDOR_TLDS: tuple[str, ...] = (
    ".com", ".net", ".co", ".io", ".us", ".biz", ".org", ".tech", ".systems",
    ".solutions", ".engineering", ".industries", ".services", ".group", ".ai",
    ".info", ".pro", ".me", ".tv", ".cc", ".ws"  # Added more
)

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
    """Extract what type of service/product is needed using AI"""
    if not openai_api_key or not OpenAI:
        # Broader fallback keywords
        text = f"{title} {description}".lower()
        if any(kw in text for kw in ["machining", "cnc", "fabrication", "manufacturing", "machine shop"]):
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

Give me simple, broad keywords like "machining services", "HVAC contractor", "IT support". Be GENERAL, not specific."""

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


def _is_likely_aggregator(url: str, title: str, snippet: str) -> bool:
    """VERY permissive - only filter obvious aggregators"""
    host = _netloc(url)
    if not host:
        return True

    # Only filter explicit aggregator domains
    if any(domain in host for domain in AGGREGATOR_DOMAINS):
        return True

    # Only filter if it's CLEARLY an aggregator site
    combined_text = (title + " " + snippet).lower()
    if "government contracts" in combined_text and "opportunities" in combined_text:
        return True

    return False


def _score_candidate(title: str, description: str, result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """ACCEPT EVERYTHING - minimal filtering"""
    vendor_url = _clean_text(result.get("link"))
    vendor_name = _clean_text(result.get("title"))
    snippet = _clean_text(result.get("snippet"))

    if not vendor_url:
        return 0.0, result

    host = _netloc(vendor_url)

    # ONLY filter these specific aggregator domains
    if host in ["sam.gov", "beta.sam.gov", "govtribe.com", "fbo.gov"]:
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

    # Location handling
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

    # Extract key terms more broadly
    key_terms = []
    for word in (title + " " + description).lower().split():
        if len(word) > 4 and word not in ["government", "federal", "agency"]:
            key_terms.append(word)

    # Use first few key terms
    main_terms = " ".join(key_terms[:3]) if key_terms else service_type

    # Query 1: Service type + location
    if use_location:
        queries.append(f"{service_type} {location_str}")
        queries.append(f"{service_type} near {location_str}")
        queries.append(f"{main_terms} {location_str}")
    else:
        queries.append(f"{service_type}")
        queries.append(f"{service_type} USA")
        queries.append(f"{main_terms} suppliers")

    # Query 2: Just the service type (very broad)
    queries.append(f"{service_type} contractors")
    queries.append(f"{service_type} companies")

    # Query 3: Specific variations based on content
    if "machining" in (title + description).lower():
        loc = f" {location_str}" if use_location else ""
        queries.append(f"machine shop{loc}")
        queries.append(f"CNC machining{loc}")
    elif "maintenance" in (title + description).lower():
        loc = f" {location_str}" if use_location else ""
        queries.append(f"maintenance services{loc}")
        queries.append(f"facility services{loc}")
    elif "software" in (title + description).lower():
        loc = f" {location_str}" if use_location else ""
        queries.append(f"software development{loc}")
        queries.append(f"IT services{loc}")

    # Always add a very broad fallback
    queries.append(f"{main_terms} providers")
    queries.append(f"contractors {main_terms}")

    return queries[:8]  # Try up to 8 different queries!


def _google_custom_search(query: str, api_key: str, cx: str, location: dict = None, max_results: int = 10) -> Dict[str, Any]:
    """Search using Google Custom Search JSON API"""
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": min(10, max_results),
        "safe": "off",
        "lr": "lang_en",
        "filter": "0",  # No duplicate filtering
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
        return f"This vendor appears to offer relevant services based on their profile."

    try:
        client = OpenAI(api_key=openai_api_key)
        prompt = f"""Why would this vendor be good for this work (1 sentence):

Work needed: {solicitation_title[:100]}

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
        return f"This vendor has relevant capabilities for the requested work."

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
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    MAXIMUM FLEXIBILITY vendor search - tries hard to find SOMETHING
    """
    title = _clean_text(str(sol.get("title", "")))
    description = _clean_text(str(sol.get("description", "")))

    if not title and not description:
        df = pd.DataFrame(columns=["name", "website", "location", "reason"])
        return (df, {"error": "empty_solicitation"} if return_debug else None)

    # Extract location
    location = {
        "city": _clean_text(str(sol.get("pop_city", ""))),
        "state": _clean_text(str(sol.get("pop_state", "")))
    }

    # Get service type
    service_type = _extract_service_type(title, description, openai_api_key)

    # Build MANY search queries
    queries = _build_search_queries(title, description, service_type, location)

    debug = {"queries": queries, "location": location,
             "service_type": service_type, "raw": []}
    candidates = []
    seen_domains = set()

    # Execute ALL queries (don't stop early)
    for i, query in enumerate(queries):
        print(f"[DEBUG] Executing query {i+1}/{len(queries)}: {query}")

        try:
            search_results = _google_custom_search(
                query, google_api_key, google_cx, location, max_results=max_google
            )

            items = search_results.get("items", [])
            print(f"[DEBUG] Got {len(items)} results from Google")

            if return_debug:
                debug["raw"].append({
                    "query": query,
                    "hits": len(items),
                    "data": search_results
                })

            # Process ALL results with MINIMAL filtering
            for j, item in enumerate(items):
                score, candidate = _score_candidate(title, description, item)

                print(
                    f"[DEBUG] Result {j+1}: {candidate.get('name', 'N/A')[:50]} - Score: {score}")

                # Accept ANYTHING with a positive score
                if score > 0:
                    host = candidate.get("host", "")

                    # Very lenient deduplication - only skip exact duplicates
                    if host and host in seen_domains:
                        print(f"[DEBUG] Skipping duplicate domain: {host}")
                        continue

                    if host:
                        seen_domains.add(host)

                    candidate["score"] = score
                    candidate["query_used"] = query
                    candidates.append(candidate)
                    print(
                        f"[DEBUG] Added candidate: {candidate.get('name', 'N/A')[:50]}")
                else:
                    print(
                        f"[DEBUG] Rejected (score={score}): {candidate.get('host', 'N/A')}")

        except Exception as e:
            print(f"[DEBUG] Query failed: {e}")
            if return_debug:
                debug["raw"].append({"query": query, "error": str(e)})
            continue

    print(f"[DEBUG] Total candidates collected: {len(candidates)}")

    # If we have NO candidates at all, that's the real problem
    if not candidates:
        df = pd.DataFrame(columns=["name", "website", "location", "reason"])
        return (df, debug if return_debug else None)

    # Sort by score and take top N (more lenient)
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    # Get extra candidates
    top_candidates = candidates[:min(top_n * 2, len(candidates))]

    # Build result DataFrame
    rows = []
    for candidate in top_candidates[:top_n]:
        # Generate AI reason
        reason = _generate_ai_reason(
            openai_api_key, title, description,
            candidate.get("name", ""), candidate.get("website", ""),
            candidate.get("snippet", "")
        )

        # Extract location from snippet
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

    df = pd.DataFrame(rows, columns=["name", "website", "location", "reason"])
    return (df, debug if return_debug else None)

# -----------------------------
# Service Vendor Search
# -----------------------------


def find_service_vendors_for_opportunity(solicitation: dict, google_api_key: str, google_cx: str,
                                         openai_api_key: str, top_n: int = 3) -> tuple:
    """Find service vendors with maximum flexibility"""
    try:
        vendors_df, _ = find_vendors_for_notice(
            sol=solicitation,
            google_api_key=google_api_key,
            google_cx=google_cx,
            openai_api_key=openai_api_key,
            max_google=15,
            top_n=top_n,
            return_debug=False
        )

        # Generate status message
        pop_city = (solicitation.get("pop_city") or "").strip()
        pop_state = (solicitation.get("pop_state") or "").strip()

        if pop_city and pop_state:
            note = f"Searched for providers in {pop_city}, {pop_state}"
        elif pop_state:
            note = f"Searched for providers in {pop_state}"
        else:
            note = "Conducted national search for providers"

        return vendors_df, note

    except Exception as e:
        return None, f"Vendor search error: {str(e)[:100]}"

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
