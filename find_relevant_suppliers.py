# Enhanced find_relevant_suppliers.py
"""
Enhanced vendor discovery for KIP using Google Custom Search JSON API.

Key improvements:
1. Less restrictive filtering - allows more results through
2. Better location handling with fallback to national search
3. Removed all "SERP" references 
4. Smarter query building with location awareness
5. Better error handling and fallbacks
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse

import requests
import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# Configuration
# -----------------------------

# Aggregator domains to avoid (still filter out obvious non-vendors)
AGGREGATOR_DOMAINS: set[str] = {
    "sam.gov", "beta.sam.gov", "govtribe.com", "bidnet.com", "bidnetdirect.com",
    "fedconnect.net", "fbo.gov", "grants.gov", "usaspending.gov",
    "linkedin.com", "facebook.com", "twitter.com", "instagram.com", "x.com"
}

# Keywords that indicate aggregator/bid sites (reduced list)
AGGREGATOR_KEYWORDS: tuple[str, ...] = (
    "sam.gov", "govtribe", "bidnet", "government contracts", "federal opportunities"
)

# Vendor indicators (expanded and more flexible)
VENDOR_INDICATORS: tuple[str, ...] = (
    "manufacturer", "supplier", "distributor", "contractor", "services", "solutions",
    "company", "corporation", "llc", "inc", "industries", "systems", "technology",
    "engineering", "fabrication", "machining", "maintenance", "repair", "installation"
)

# Broader list of TLDs that could be vendors
POTENTIAL_VENDOR_TLDS: tuple[str, ...] = (
    ".com", ".net", ".co", ".io", ".us", ".biz", ".org", ".tech", ".systems",
    ".solutions", ".engineering", ".industries", ".services", ".group"
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
        # Remove common prefixes
        for prefix in ['www.', 'm.', 'en.']:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        # Remove .com, .org, etc and capitalize
        name = domain.split('.')[0]
        return name.replace('-', ' ').replace('_', ' ').title()
    except:
        return "Company"


def _extract_service_type(title: str, description: str, openai_api_key: str) -> str:
    """Extract what type of service/product is needed using AI"""
    if not openai_api_key or not OpenAI:
        # Fallback: basic keyword extraction
        text = f"{title} {description}".lower()
        if any(kw in text for kw in ["machining", "cnc", "fabrication", "manufacturing"]):
            return "manufacturing services"
        elif any(kw in text for kw in ["maintenance", "repair", "installation", "service"]):
            return "service providers"
        elif any(kw in text for kw in ["software", "it", "technology"]):
            return "technology services"
        else:
            return "contractors"

    try:
        client = OpenAI(api_key=openai_api_key)
        service_prompt = f"""Based on this government solicitation, what type of service company or product supplier should I search for? 
        
Title: {title[:200]}
Description: {description[:400]}

Respond with 2-4 search keywords for the type of provider needed (e.g., "HVAC maintenance contractor", "precision machining services", "IT support services"). Be specific and practical."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": service_prompt}],
            temperature=0.1,
            max_tokens=50,
            timeout=15
        )

        return response.choices[0].message.content.strip()
    except Exception:
        return "contractors and suppliers"


def _is_likely_aggregator(url: str, title: str, snippet: str) -> bool:
    """Check if result is likely an aggregator site (less restrictive)"""
    host = _netloc(url)
    if not host:
        return True

    # Only filter out obvious aggregators
    host_agg = any(domain in host for domain in AGGREGATOR_DOMAINS)

    # Only filter if multiple aggregator keywords present
    combined_text = (title + " " + snippet).lower()
    agg_keyword_count = sum(
        1 for k in AGGREGATOR_KEYWORDS if k in combined_text)

    return host_agg or agg_keyword_count >= 2


def _score_candidate(title: str, description: str, result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Score a search result candidate (more permissive scoring)"""
    vendor_url = _clean_text(result.get("link"))
    vendor_name = _clean_text(result.get("title"))
    snippet = _clean_text(result.get("snippet"))

    if not vendor_url:
        return 0.0, result

    # Skip obvious aggregators
    if _is_likely_aggregator(vendor_url, vendor_name, snippet):
        return 0.0, result

    # More permissive vendor detection
    host = _netloc(vendor_url)
    combined_text = (vendor_name + " " + snippet).lower()

    # Start with base score
    score = 1.0  # Give everything a chance

    # Boost for vendor indicators
    for indicator in VENDOR_INDICATORS:
        if indicator in combined_text:
            score += 1.5

    # Boost for having vendor-like domain
    if host.endswith(POTENTIAL_VENDOR_TLDS):
        score += 1.0

    # Boost for relevance to solicitation
    sol_keywords = re.findall(
        r'\b\w{4,}\b', (title + " " + description).lower())
    for keyword in sol_keywords[:10]:  # Check top 10 keywords
        if keyword in combined_text:
            score += 0.5

    # Small penalty for marketplaces (but don't exclude entirely)
    marketplace_domains = ("amazon.", "ebay.", "alibaba.", "walmart.")
    if any(m in host for m in marketplace_domains):
        score *= 0.7  # Reduce score but don't eliminate

    # Penalty for social media/directories
    social_domains = ("facebook.com", "linkedin.com",
                      "yellowpages.com", "yelp.com")
    if any(s in host for s in social_domains):
        score *= 0.3

    return max(score, 0.0), {
        "name": vendor_name[:120] if vendor_name else _company_name_from_url(vendor_url),
        "website": vendor_url,
        "location": "",
        "snippet": snippet[:240],
        "host": host,
    }


def _build_search_queries(title: str, description: str, service_type: str, location: dict = None) -> List[str]:
    """Build multiple search queries with location awareness"""
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

    # Base search terms
    base_terms = title[:100] if title else description[:100]

    # Query 1: Service type + location (if available)
    if use_location:
        queries.append(f"{service_type} {location_str}")
        queries.append(f"{service_type} near {location_str}")
        queries.append(f"{base_terms} contractor {location_str}")
    else:
        # National search queries
        queries.append(f"{service_type} United States")
        queries.append(f"{service_type} nationwide")
        queries.append(f"{base_terms} contractor nationwide")

    # Query 2: More specific based on solicitation content
    if "machining" in (title + description).lower():
        loc_part = f" {location_str}" if use_location else " USA"
        queries.append(f"CNC machining services{loc_part}")
        queries.append(f"precision machining{loc_part}")
    elif "maintenance" in (title + description).lower():
        loc_part = f" {location_str}" if use_location else " nationwide"
        queries.append(f"maintenance services{loc_part}")
    elif "software" in (title + description).lower():
        loc_part = f" {location_str}" if use_location else " USA"
        queries.append(f"software development{loc_part}")

    # Fallback general query
    if not queries:
        queries.append(f"contractors suppliers {service_type}")

    return queries[:3]  # Limit to 3 queries max


def _google_custom_search(query: str, api_key: str, cx: str, location: dict = None, max_results: int = 10) -> Dict[str, Any]:
    """
    Search using Google Custom Search JSON API with enhanced location handling
    """
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": min(10, max_results),  # Google Custom Search max is 10
        "safe": "off",
        "lr": "lang_en",
        "filter": "0",  # Disable duplicate filtering for more results
    }

    # Add location parameters if available
    if location:
        state = location.get("state", "").upper()
        # Use 'gl' parameter for country/state-level geolocation
        if state and len(state) == 2:
            # For US states, we can't use 'gl' directly, but we can bias results
            # The query already includes location, so we rely on that
            pass

        # Could add 'cr' parameter for country restriction if needed
        # params["cr"] = "countryUS"  # Restrict to US results

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
        # Return empty structure on failure
        return {"items": []}


def _generate_ai_reason(openai_api_key: str, solicitation_title: str, solicitation_desc: str,
                        vendor_name: str, vendor_url: str, vendor_snippet: str) -> str:
    """Generate AI reason for vendor recommendation"""
    if not openai_api_key or not OpenAI:
        return f"This vendor appears to offer relevant services based on their profile."

    try:
        client = OpenAI(api_key=openai_api_key)
        prompt = f"""Explain in one short sentence why this vendor would be good for this government solicitation:

Solicitation: {solicitation_title[:150]}
Description: {(solicitation_desc or '')[:400]}

Vendor: {vendor_name}
About: {vendor_snippet[:200]}

Focus on what specific work they could do. Be concise and practical."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
            timeout=15
        )

        return response.choices[0].message.content.strip()
    except Exception:
        return f"This vendor appears to have relevant capabilities for the requested work."

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
    Enhanced vendor search with less restrictive filtering and better location handling
    """
    title = _clean_text(str(sol.get("title", "")))
    description = _clean_text(str(sol.get("description", "")))

    if not title and not description:
        df = pd.DataFrame(columns=["name", "website", "location", "reason"])
        return (df, {"error": "empty_solicitation"} if return_debug else None)

    # Extract place of performance
    location = {
        "city": _clean_text(str(sol.get("pop_city", ""))),
        "state": _clean_text(str(sol.get("pop_state", "")))
    }

    # Determine service type
    service_type = _extract_service_type(title, description, openai_api_key)

    # Build search queries
    queries = _build_search_queries(title, description, service_type, location)

    debug = {"queries": queries, "location": location,
             "service_type": service_type, "raw": []}
    candidates = []
    seen_domains = set()

    # Execute searches
    for i, query in enumerate(queries):
        try:
            search_results = _google_custom_search(
                query, google_api_key, google_cx, location, max_results=max_google
            )

            if return_debug:
                debug["raw"].append({
                    "query": query,
                    "hits": len(search_results.get("items", [])),
                    "data": search_results
                })

            # Process results
            for item in search_results.get("items", []):
                score, candidate = _score_candidate(title, description, item)

                if score > 0:
                    host = candidate.get("host", "")

                    # Allow more results by being less restrictive on duplicates
                    if host in seen_domains:
                        continue

                    seen_domains.add(host)
                    candidate["score"] = score
                    candidate["query_used"] = query
                    candidates.append(candidate)

                    if len(candidates) >= top_n * 3:  # Get extra candidates
                        break

            # If we have enough good candidates, we can stop early
            if len(candidates) >= top_n * 2:
                break

        except Exception as e:
            if return_debug:
                debug["raw"].append({"query": query, "error": str(e)})
            continue

    # Sort candidates by score and take top N
    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    top_candidates = candidates[:top_n]

    # Build result DataFrame
    rows = []
    for candidate in top_candidates:
        # Generate AI reason
        reason = _generate_ai_reason(
            openai_api_key, title, description,
            candidate.get("name", ""), candidate.get("website", ""),
            candidate.get("snippet", "")
        )

        # Try to extract location from snippet
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
# Service Vendor Search (for Internal Use tab)
# -----------------------------


def find_service_vendors_for_opportunity(solicitation: dict, google_api_key: str, google_cx: str,
                                         openai_api_key: str, top_n: int = 3) -> tuple:
    """
    Find service vendors for a solicitation opportunity with enhanced location handling
    """
    try:
        # Extract solicitation details
        title = solicitation.get("title", "")
        description = solicitation.get("description", "")

        # Extract location with better fallback
        pop_city = (solicitation.get("pop_city") or "").strip()
        pop_state = (solicitation.get("pop_state") or "").strip()

        location = {"city": pop_city, "state": pop_state}

        # Use the enhanced search function
        vendors_df, search_note = find_vendors_for_notice(
            sol=solicitation,
            google_api_key=google_api_key,
            google_cx=google_cx,
            openai_api_key=openai_api_key,
            max_google=15,  # Get more candidates
            top_n=top_n,
            return_debug=False
        )

        # Generate appropriate status message
        if pop_city and pop_state:
            note = f"Searching for providers in {pop_city}, {pop_state}"
        elif pop_state:
            note = f"Searching for providers in {pop_state}"
        else:
            note = "No specific location found - conducting national search"

        return vendors_df, note

    except Exception as e:
        return None, f"Vendor search failed: {str(e)[:100]}"

# -----------------------------
# Compatibility Functions
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
    """
    Compatibility wrapper that returns vendor results across all solicitations
    """
    results = []
    our_recommended_suppliers = our_recommended_suppliers or []
    our_not_recommended_suppliers = our_not_recommended_suppliers or []

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
