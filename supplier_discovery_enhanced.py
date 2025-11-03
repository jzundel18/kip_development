"""
Enhanced DoD Supplier Discovery Script with SAM.gov Integration
Combines Google Custom Search with SAM.gov entity data for comprehensive supplier discovery.
"""

import os
import time
import re
from typing import List, Dict, Optional, Set
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
SAM_API_KEY = os.getenv("SAM_API_KEY")  # Get from sam.gov

# DoD-relevant search terms for different categories
SEARCH_CATEGORIES = {
    "aerospace_defense": [
        "aerospace defense contractor",
        "military aircraft parts manufacturer",
        "defense avionics supplier",
        "UAV drone systems manufacturer"
    ],
    "electronics_communications": [
        "military communications equipment supplier",
        "defense electronics manufacturer",
        "tactical radio systems contractor",
        "military GPS equipment supplier"
    ],
    "it_cyber": [
        "defense cybersecurity contractor",
        "military IT services provider",
        "DoD cloud services contractor",
        "defense software development"
    ],
    "logistics_support": [
        "military logistics services",
        "defense supply chain management",
        "military maintenance contractor",
        "defense warehousing services"
    ],
    "manufacturing_equipment": [
        "military vehicle parts manufacturer",
        "defense armor systems supplier",
        "military weapons systems manufacturer",
        "defense test equipment supplier"
    ],
    "services_consulting": [
        "defense consulting services",
        "military training contractor",
        "defense R&D services",
        "military engineering services"
    ]
}


def get_existing_companies() -> Dict[str, Set[str]]:
    """
    Fetch existing companies and return multiple identifier sets for deduplication.

    Returns:
        Dictionary with sets of names, CAGE codes, and DUNS numbers
    """
    try:
        response = supabase.table("company_list").select("name, cage, duns").execute()

        return {
            "names": {c['name'].lower().strip() for c in response.data if c.get('name')},
            "cages": {c['cage'].upper().strip() for c in response.data if c.get('cage')},
            "duns": {c['duns'].strip() for c in response.data if c.get('duns')}
        }
    except Exception as e:
        print(f"Error fetching existing companies: {e}")
        return {"names": set(), "cages": set(), "duns": set()}


def google_custom_search(query: str, num_results: int = 10) -> List[Dict]:
    """Perform a Google Custom Search."""
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(num_results, 10)
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])
    except requests.exceptions.RequestException as e:
        print(f"  Search error: {e}")
        return []


def search_sam_gov_entity(company_name: str) -> Optional[Dict]:
    """
    Search SAM.gov for entity registration data.

    Args:
        company_name: Name of company to search

    Returns:
        Dictionary with SAM.gov entity data if found
    """
    if not SAM_API_KEY:
        return None

    url = "https://api.sam.gov/entity-information/v3/entities"

    params = {
        "api_key": SAM_API_KEY,
        "legalBusinessName": company_name,
        "registrationStatus": "Active",
        "includeSections": "entityRegistration,coreData"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        entities = data.get("entityData", [])
        if entities:
            return entities[0]  # Return first match
        return None
    except requests.exceptions.RequestException as e:
        print(f"    SAM.gov lookup error: {e}")
        return None


def extract_sam_gov_data(entity_data: Dict) -> Dict:
    """
    Extract relevant fields from SAM.gov entity data.

    Args:
        entity_data: Raw SAM.gov entity response

    Returns:
        Dictionary with extracted company data
    """
    core_data = entity_data.get("coreData", {})
    entity_reg = entity_data.get("entityRegistration", {})

    # Extract NAICS codes
    naics_list = core_data.get("naicsCodesList", [])
    primary_naics = None
    other_naics = []

    for naics in naics_list:
        code = naics.get("naicsCode")
        if naics.get("isPrimary"):
            primary_naics = code
        else:
            other_naics.append(code)

    # Extract address for state
    physical_address = core_data.get("physicalAddress", {})
    state = physical_address.get("stateOrProvinceCode")

    # Extract contact info
    email = core_data.get("entityURL")  # Often companies list website, not direct email
    phone = core_data.get("congressionalDistrict")  # Not ideal, but phones often in other fields

    # Extract business types
    business_types = entity_reg.get("businessTypes", {})
    designation_parts = []

    if business_types.get("veteranOwned"):
        designation_parts.append("Veteran-Owned")
    if business_types.get("womanOwned"):
        designation_parts.append("Woman-Owned")
    if business_types.get("smallBusiness"):
        designation_parts.append("Small Business")
    if business_types.get("eightAProgram"):
        designation_parts.append("8(a)")
    if business_types.get("hubZone"):
        designation_parts.append("HUBZone")

    designation = ", ".join(designation_parts) if designation_parts else None

    return {
        "name": core_data.get("legalBusinessName"),
        "cage": core_data.get("cageCode"),
        "duns": core_data.get("dunsNumber"),
        "NAICS": primary_naics,
        "other_NAICS": ", ".join(other_naics) if other_naics else None,
        "state": state,
        "designation": designation,
        "email": None,  # SAM.gov doesn't reliably provide email
        "phone": None,  # SAM.gov doesn't reliably provide phone
    }


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """Extract email and phone from text using regex."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    return {
        "email": emails[0] if emails else None,
        "phone": f"({phones[0][0]}) {phones[0][1]}-{phones[0][2]}" if phones else None
    }


def extract_state_from_text(text: str) -> Optional[str]:
    """Extract US state abbreviation from text."""
    us_states = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    }

    # Look for state abbreviations with word boundaries
    pattern = r'\b(' + '|'.join(us_states) + r')\b'
    matches = re.findall(pattern, text.upper())

    return matches[0] if matches else None


def extract_company_info_from_search(search_result: Dict, use_sam_gov: bool = True) -> Optional[Dict]:
    """
    Extract and enrich company information from search result.

    Args:
        search_result: Google search result
        use_sam_gov: Whether to enhance with SAM.gov data

    Returns:
        Dictionary with company data or None if invalid
    """
    title = search_result.get("title", "")
    snippet = search_result.get("snippet", "")
    link = search_result.get("link", "")

    # Extract company name
    company_name = title.split("-")[0].split("|")[0].strip()

    # Skip if name is too generic or too short
    if len(company_name) < 3 or len(company_name) > 200:
        return None

    generic_terms = ["home", "search", "login", "contact", "about", "services", "products"]
    if any(term == company_name.lower() for term in generic_terms):
        return None

    # Start with basic info from search
    company_info = {
        "name": company_name,
        "description": snippet if snippet else f"Defense contractor found at {link}",
        "source_url": link
    }

    # Extract contact info from snippet
    full_text = f"{title} {snippet}"
    contact_info = extract_contact_info(full_text)
    company_info["email"] = contact_info["email"]
    company_info["phone"] = contact_info["phone"]

    # Extract state
    company_info["state"] = extract_state_from_text(full_text)

    # Determine prime experience
    prime_keywords = ["prime contractor", "prime contract", "contract award", "awarded"]
    company_info["prime_experience"] = "Yes" if any(kw in full_text.lower() for kw in prime_keywords) else "No"

    # Try to enhance with SAM.gov data
    if use_sam_gov and SAM_API_KEY:
        print(f"    Looking up in SAM.gov: {company_name}")
        sam_data = search_sam_gov_entity(company_name)

        if sam_data:
            print(f"    ✓ Found in SAM.gov")
            sam_fields = extract_sam_gov_data(sam_data)

            # Merge SAM.gov data (SAM.gov data takes precedence for official fields)
            for key, value in sam_fields.items():
                if value:  # Only update if SAM.gov has a value
                    company_info[key] = value
        else:
            print(f"    ✗ Not found in SAM.gov")

    # Set any missing fields to None
    required_fields = ["contact", "states_perform_work", "cage", "duns", "NAICS",
                       "other_NAICS", "designation"]
    for field in required_fields:
        if field not in company_info:
            company_info[field] = None

    return company_info


def is_duplicate(company_info: Dict, existing: Dict[str, Set[str]]) -> bool:
    """
    Check if company already exists using multiple identifiers.

    Args:
        company_info: Company data to check
        existing: Dictionary of existing identifiers

    Returns:
        True if duplicate found
    """
    # Check by name
    if company_info.get("name", "").lower().strip() in existing["names"]:
        return True

    # Check by CAGE code
    cage = company_info.get("cage")
    if cage and cage.upper().strip() in existing["cages"]:
        return True

    # Check by DUNS
    duns = company_info.get("duns")
    if duns and duns.strip() in existing["duns"]:
        return True

    return False


def add_company_to_database(company_info: Dict) -> bool:
    """Add company to Supabase database."""
    try:
        # Remove fields not in schema
        insert_data = {k: v for k, v in company_info.items()
                       if k not in ["source_url"]}

        response = supabase.table("company_list").insert(insert_data).execute()

        # Create summary of added info
        fields_added = [k for k, v in insert_data.items() if v is not None]
        print(f"    ✓ Added: {company_info['name']}")
        print(f"      Fields: {', '.join(fields_added)}")

        return True
    except Exception as e:
        print(f"    ✗ Error adding {company_info['name']}: {e}")
        return False


def discover_suppliers(max_companies: int = 50,
                       use_sam_gov: bool = True,
                       search_delay: float = 1.0,
                       sam_delay: float = 0.5):
    """
    Main supplier discovery function.

    Args:
        max_companies: Maximum new companies to add
        use_sam_gov: Whether to enhance data with SAM.gov lookups
        search_delay: Delay between Google searches (seconds)
        sam_delay: Delay between SAM.gov lookups (seconds)
    """
    print("=" * 70)
    print("DOD SUPPLIER DISCOVERY - ENHANCED WITH SAM.GOV INTEGRATION")
    print("=" * 70)

    # Check for required credentials
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        print("ERROR: Google API credentials not found in environment variables")
        return

    if use_sam_gov and not SAM_API_KEY:
        print("WARNING: SAM_API_KEY not found. Running without SAM.gov enrichment.")
        use_sam_gov = False

    print(f"\nConfiguration:")
    print(f"  Max companies: {max_companies}")
    print(f"  SAM.gov enrichment: {use_sam_gov}")
    print(f"  Search delay: {search_delay}s")

    # Load existing companies
    print(f"\nFetching existing companies from database...")
    existing = get_existing_companies()
    print(f"  {len(existing['names'])} companies")
    print(f"  {len(existing['cages'])} CAGE codes")
    print(f"  {len(existing['duns'])} DUNS numbers")

    companies_added = 0
    companies_processed = 0

    print(f"\nStarting supplier discovery...")
    print("-" * 70)

    # Iterate through search categories
    for category, queries in SEARCH_CATEGORIES.items():
        if companies_added >= max_companies:
            break

        print(f"\n[{category.upper()}]")

        for query in queries:
            if companies_added >= max_companies:
                break

            print(f"\n  Searching: '{query}'")
            results = google_custom_search(query, num_results=10)

            if not results:
                print(f"    No results")
                continue

            print(f"    Processing {len(results)} results...")

            for result in results:
                if companies_added >= max_companies:
                    break

                companies_processed += 1

                # Extract and enrich company info
                company_info = extract_company_info_from_search(result, use_sam_gov)

                if not company_info:
                    continue

                # Check for duplicates
                if is_duplicate(company_info, existing):
                    print(f"    ⊘ Duplicate: {company_info['name']}")
                    continue

                # Add to database
                if add_company_to_database(company_info):
                    # Update existing sets to prevent duplicates in same run
                    existing["names"].add(company_info["name"].lower().strip())
                    if company_info.get("cage"):
                        existing["cages"].add(company_info["cage"].upper().strip())
                    if company_info.get("duns"):
                        existing["duns"].add(company_info["duns"].strip())

                    companies_added += 1

                # Rate limiting
                if use_sam_gov:
                    time.sleep(sam_delay)

            time.sleep(search_delay)

    print("\n" + "=" * 70)
    print(f"DISCOVERY COMPLETE")
    print(f"  Results processed: {companies_processed}")
    print(f"  New companies added: {companies_added}")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    MAX_COMPANIES = 50
    USE_SAM_GOV = True  # Set to False if you don't have SAM.gov API key
    SEARCH_DELAY = 1.0  # Seconds between searches
    SAM_DELAY = 0.5  # Seconds between SAM.gov lookups

    discover_suppliers(
        max_companies=MAX_COMPANIES,
        use_sam_gov=USE_SAM_GOV,
        search_delay=SEARCH_DELAY,
        sam_delay=SAM_DELAY
    )