"""
DoD Supplier Discovery Script
Searches for potential DoD suppliers using Google Custom Search API
and adds new companies to the Supabase company_list database.
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

# Google Custom Search API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Search queries targeting DoD suppliers
SEARCH_QUERIES = [
    "DoD contractor defense suppliers",
    "military equipment manufacturer",
    "aerospace defense components supplier",
    "defense electronics manufacturer",
    "military vehicle parts supplier",
    "defense logistics services",
    "military communications equipment",
    "defense cybersecurity services",
    "military ammunition supplier",
    "defense R&D contractor",
    "military training services provider",
    "defense IT services contractor",
    "military aviation parts supplier",
    "defense shipbuilding contractor",
    "military uniform supplier",
    "defense medical equipment supplier",
    "military radar systems manufacturer",
    "defense satellite components",
    "military ground support equipment",
    "defense maintenance services"
]

# Common DoD-related NAICS codes
DOD_NAICS_CODES = {
    "336411": "Aircraft Manufacturing",
    "336412": "Aircraft Engine and Engine Parts Manufacturing",
    "336413": "Other Aircraft Parts and Auxiliary Equipment Manufacturing",
    "334511": "Search, Detection, Navigation, Guidance, Aeronautical, and Nautical System and Instrument Manufacturing",
    "336992": "Military Armored Vehicle, Tank, and Tank Component Manufacturing",
    "332993": "Ammunition (except Small Arms) Manufacturing",
    "332994": "Small Arms Ammunition Manufacturing",
    "541330": "Engineering Services",
    "541512": "Computer Systems Design Services",
    "541519": "Other Computer Related Services",
    "541715": "Research and Development in the Physical, Engineering, and Life Sciences",
    "336611": "Ship Building and Repairing",
    "315210": "Cut and Sew Apparel Contractors",
    "339113": "Surgical Appliance and Supplies Manufacturing",
}


def get_existing_companies() -> Set[str]:
    """Fetch all existing company names from database to avoid duplicates."""
    try:
        response = supabase.table("company_list").select("name").execute()
        # Normalize names to lowercase for comparison
        return {company['name'].lower().strip() for company in response.data}
    except Exception as e:
        print(f"Error fetching existing companies: {e}")
        return set()


def google_custom_search(query: str, num_results: int = 10) -> List[Dict]:
    """
    Perform a Google Custom Search and return results.

    Args:
        query: Search query string
        num_results: Number of results to return (max 10 per request)

    Returns:
        List of search result dictionaries
    """
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(num_results, 10)
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])
    except requests.exceptions.RequestException as e:
        print(f"Error performing search for '{query}': {e}")
        return []


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """
    Extract contact information from text using regex patterns.

    Returns:
        Dictionary with email and phone if found
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'

    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)

    return {
        "email": emails[0] if emails else None,
        "phone": f"({phones[0][0]}) {phones[0][1]}-{phones[0][2]}" if phones else None
    }


def extract_state_from_text(text: str) -> Optional[str]:
    """
    Extract US state abbreviation from text.
    """
    state_pattern = r'\b([A-Z]{2})\b'
    us_states = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    }

    matches = re.findall(state_pattern, text)
    for match in matches:
        if match in us_states:
            return match
    return None


def extract_company_info(search_result: Dict) -> Dict:
    """
    Extract company information from a search result.

    Args:
        search_result: Dictionary containing search result data

    Returns:
        Dictionary with extracted company information
    """
    title = search_result.get("title", "")
    snippet = search_result.get("snippet", "")
    link = search_result.get("link", "")

    # Combine text for analysis
    full_text = f"{title} {snippet}"

    # Extract company name (usually the first part of the title)
    company_name = title.split("-")[0].split("|")[0].strip()

    # Extract contact info
    contact_info = extract_contact_info(full_text)

    # Extract state
    state = extract_state_from_text(full_text)

    # Create description from snippet
    description = snippet if snippet else f"Company found at {link}"

    # Determine if keywords suggest prime contractor experience
    prime_keywords = ["prime contractor", "prime contract", "awarded", "contract vehicle"]
    prime_experience = "Yes" if any(keyword in full_text.lower() for keyword in prime_keywords) else "No"

    # Determine business designation based on keywords
    designation = None
    if "small business" in full_text.lower():
        designation = "Small Business"
    if "veteran" in full_text.lower() or "sdvosb" in full_text.lower():
        designation = "Veteran-Owned" if not designation else f"{designation}, Veteran-Owned"
    if "woman-owned" in full_text.lower() or "wosb" in full_text.lower():
        designation = "Woman-Owned" if not designation else f"{designation}, Woman-Owned"
    if "8(a)" in full_text or "8a" in full_text.lower():
        designation = "8(a)" if not designation else f"{designation}, 8(a)"

    return {
        "name": company_name,
        "description": description,
        "email": contact_info["email"],
        "phone": contact_info["phone"],
        "state": state,
        "contact": None,  # Would need more sophisticated extraction
        "NAICS": None,  # Would need to scrape company website or use additional APIs
        "other_NAICS": None,
        "states_perform_work": None,
        "cage": None,
        "duns": None,
        "designation": designation,
        "prime_experience": prime_experience,
        "source_url": link
    }


def is_valid_company(company_info: Dict) -> bool:
    """
    Validate that company has minimum required information.

    Args:
        company_info: Dictionary with company data

    Returns:
        Boolean indicating if company meets minimum requirements
    """
    # Must have name and description
    if not company_info.get("name") or not company_info.get("description"):
        return False

    # Name should be reasonable length
    if len(company_info["name"]) < 3 or len(company_info["name"]) > 200:
        return False

    # Filter out generic terms that might not be actual companies
    generic_terms = ["home", "search", "login", "contact us", "about", "services"]
    name_lower = company_info["name"].lower()
    if any(term in name_lower for term in generic_terms) and len(name_lower.split()) < 3:
        return False

    return True


def add_company_to_database(company_info: Dict) -> bool:
    """
    Add a new company to the Supabase database.

    Args:
        company_info: Dictionary with company data

    Returns:
        Boolean indicating success
    """
    try:
        # Remove source_url before inserting (not in schema)
        insert_data = {k: v for k, v in company_info.items() if k != "source_url"}

        response = supabase.table("company_list").insert(insert_data).execute()
        print(f"✓ Added: {company_info['name']}")
        return True
    except Exception as e:
        print(f"✗ Error adding {company_info['name']}: {e}")
        return False


def discover_suppliers(max_companies: int = 50, delay: float = 1.0):
    """
    Main function to discover DoD suppliers and add them to the database.

    Args:
        max_companies: Maximum number of new companies to add
        delay: Delay in seconds between API calls to avoid rate limiting
    """
    print("=" * 60)
    print("DoD SUPPLIER DISCOVERY")
    print("=" * 60)

    # Get existing companies to avoid duplicates
    print("\nFetching existing companies from database...")
    existing_companies = get_existing_companies()
    print(f"Found {len(existing_companies)} existing companies in database")

    companies_added = 0
    companies_processed = 0

    print(f"\nStarting supplier discovery (target: {max_companies} new companies)...")
    print("-" * 60)

    for query in SEARCH_QUERIES:
        if companies_added >= max_companies:
            break

        print(f"\nSearching: '{query}'")

        # Perform search
        results = google_custom_search(query, num_results=10)

        if not results:
            print("  No results found")
            continue

        print(f"  Found {len(results)} results")

        # Process each result
        for result in results:
            if companies_added >= max_companies:
                break

            companies_processed += 1

            # Extract company information
            company_info = extract_company_info(result)

            # Validate company info
            if not is_valid_company(company_info):
                print(f"  ⊘ Skipped: {company_info.get('name', 'Invalid')} (insufficient data)")
                continue

            # Check if company already exists
            if company_info["name"].lower().strip() in existing_companies:
                print(f"  ⊘ Skipped: {company_info['name']} (already in database)")
                continue

            # Add to database
            if add_company_to_database(company_info):
                existing_companies.add(company_info["name"].lower().strip())
                companies_added += 1

        # Rate limiting delay
        time.sleep(delay)

    print("\n" + "=" * 60)
    print(f"DISCOVERY COMPLETE")
    print(f"Companies processed: {companies_processed}")
    print(f"New companies added: {companies_added}")
    print("=" * 60)


if __name__ == "__main__":
    # Configuration
    MAX_COMPANIES = 50  # Adjust as needed
    API_DELAY = 1.0  # Seconds between API calls

    discover_suppliers(max_companies=MAX_COMPANIES, delay=API_DELAY)