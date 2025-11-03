#!/usr/bin/env python3
"""
Defense Supplier Web Scraping Script
Uses Google Custom Search API to find defense suppliers and adds them to Supabase.

Required Environment Variables:
- SUPABASE_DB_URL: Your Supabase database connection string
- GOOGLE_API_KEY: Your Google Custom Search API key
- GOOGLE_CX: Your Google Custom Search Engine ID
- OPENAI_API_KEY: OpenAI API key for intelligent data extraction

Usage:
    python defense_supplier_scraper.py [--max-companies 100] [--delay 1.0]
"""

import os
import sys
import time
import re
import json
import argparse
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlparse
from datetime import datetime
import requests
from openai import OpenAI
import sqlalchemy as sa
from sqlalchemy import text, create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("SUPABASE_DB_URL")

# Defense supplier search queries targeting different industries
SEARCH_QUERIES = [
    # Aerospace & Aviation
    "aerospace defense parts manufacturer supplier",
    "aircraft components defense contractor",
    "avionics systems defense supplier",
    "helicopter parts military supplier",
    "UAV drone components manufacturer",

    # Electronics & Communications
    "military electronics components supplier",
    "defense communications equipment manufacturer",
    "tactical radio systems supplier",
    "military radar components manufacturer",
    "defense semiconductor supplier",

    # Weapons & Ammunition
    "military ammunition components supplier",
    "defense weapons systems parts manufacturer",
    "firearms components defense supplier",
    "military ordnance parts manufacturer",

    # Vehicles & Ground Systems
    "military vehicle parts supplier",
    "defense ground systems components",
    "tactical vehicle parts manufacturer",
    "military truck components supplier",
    "armored vehicle parts manufacturer",

    # Naval & Maritime
    "naval defense components supplier",
    "ship systems parts manufacturer",
    "submarine components defense supplier",
    "maritime defense equipment manufacturer",

    # IT & Cybersecurity
    "defense IT services contractor",
    "military cybersecurity services provider",
    "defense software development company",
    "military network security contractor",

    # Manufacturing & Machining
    "precision machining defense contractor",
    "CNC machining military parts",
    "defense metal fabrication supplier",
    "military precision manufacturing",

    # Logistics & Support
    "defense logistics services contractor",
    "military supply chain management",
    "defense maintenance services contractor",
    "military warehousing services provider",

    # Textiles & Equipment
    "military uniform manufacturer supplier",
    "defense tactical gear manufacturer",
    "military body armor supplier",
    "defense protective equipment manufacturer",

    # Testing & Engineering
    "defense testing services contractor",
    "military engineering services provider",
    "defense R&D contractor",
    "military test equipment manufacturer",
]

# Common defense-related NAICS codes
DEFENSE_NAICS_MAP = {
    "336411": "Aircraft Manufacturing",
    "336412": "Aircraft Engine and Engine Parts",
    "336413": "Other Aircraft Parts",
    "334511": "Navigation and Control Instruments",
    "336992": "Military Armored Vehicle Manufacturing",
    "332993": "Ammunition Manufacturing",
    "541330": "Engineering Services",
    "541512": "Computer Systems Design",
    "541715": "R&D in Physical/Engineering Sciences",
    "336611": "Ship Building and Repairing",
    "315210": "Cut and Sew Apparel Manufacturing",
    "339113": "Surgical and Medical Instruments",
    "332117": "Powder Metallurgy Part Manufacturing",
    "332710": "Machine Shops",
    "332812": "Metal Coating and Nonprecious Engraving",
}


class DefenseSupplierScraper:
    """Main scraper class for finding and processing defense suppliers"""

    def __init__(self):
        self.validate_credentials()
        self.engine = create_engine(DB_URL, pool_pre_ping=True)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.existing_companies = self._load_existing_companies()
        self.companies_added = 0
        self.companies_skipped = 0

    def validate_credentials(self):
        """Validate that all required credentials are present"""
        missing = []
        if not GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not GOOGLE_CX:
            missing.append("GOOGLE_CX")
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not DB_URL:
            missing.append("SUPABASE_DB_URL")

        if missing:
            print(f"❌ ERROR: Missing required environment variables: {', '.join(missing)}")
            print("\nPlease set these in your .env file or environment:")
            for var in missing:
                print(f"  {var}=your_value_here")
            sys.exit(1)

    def _load_existing_companies(self) -> Set[str]:
        """Load existing company names to avoid duplicates"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT name FROM company_list"))
                # Normalize names for comparison
                return {row[0].lower().strip() for row in result if row[0]}
        except Exception as e:
            print(f"⚠️  Warning: Could not load existing companies: {e}")
            return set()

    def google_search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform Google Custom Search"""
        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "num": min(num_results, 10)  # Google API max is 10
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            return data.get("items", [])
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Search error for '{query}': {str(e)[:100]}")
            return []

    def extract_company_info_with_ai(self, search_result: Dict) -> Optional[Dict]:
        """
        Use AI to extract comprehensive company information from search result
        and by analyzing the company's website content
        """
        title = search_result.get("title", "")
        snippet = search_result.get("snippet", "")
        url = search_result.get("link", "")

        # Skip non-company results
        if not title or len(title) < 3:
            return None

        # Filter out obvious non-company pages
        exclude_keywords = ["login", "sign in", "register", "cart", "checkout",
                            "search results", "about us", "contact us", "privacy policy"]
        title_lower = title.lower()
        if any(kw in title_lower for kw in exclude_keywords) and len(title.split()) < 4:
            return None

        # Extract basic company name from title
        company_name = title.split("-")[0].split("|")[0].strip()

        # Fetch website content for better data extraction
        website_text = self._fetch_website_content(url)

        # Combine all text for AI analysis
        full_text = f"Title: {title}\n\nSnippet: {snippet}\n\nWebsite Content:\n{website_text[:3000]}"

        # Use AI to extract structured company information
        try:
            system_prompt = """You are a data extraction specialist for defense industry companies. 
Extract company information and return ONLY valid JSON with these exact keys:

{
  "name": "Company legal name",
  "description": "Brief company description (1-2 sentences about what they do)",
  "state": "Two-letter US state code (e.g., CA, TX) or NULL",
  "email": "Contact email or NULL",
  "phone": "Phone number in format (XXX) XXX-XXXX or NULL",
  "contact": "Name of contact person or NULL",
  "states_perform_work": "Comma-separated states or 'All' or NULL",
  "NAICS": "Primary 6-digit NAICS code or NULL",
  "other_NAICS": "Comma-separated other NAICS codes or NULL",
  "cage": "5-character CAGE code or NULL",
  "duns": "9-digit DUNS number or NULL",
  "designation": "Comma-separated: Small Business, Veteran-Owned, Woman-Owned, 8(a), HUBZone, etc. or NULL",
  "prime_experience": "Yes or No or NULL"
}

Guidelines:
- Extract ONLY information that is explicitly stated
- Use NULL (not "None" or empty string) for missing data
- For states_perform_work: Use "All" only if explicitly stated as nationwide/all states
- For designation: Include all that apply, comma-separated
- For prime_experience: "Yes" only if explicitly mentions working with primes like Lockheed Martin, Boeing, Raytheon, Northrop Grumman, General Dynamics, etc.
- Ensure phone numbers are formatted consistently
- NAICS codes must be exactly 6 digits
- CAGE codes must be exactly 5 characters
- DUNS numbers must be exactly 9 digits"""

            user_prompt = f"""Extract company information from this defense supplier:

{full_text}

Return valid JSON only."""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800,
                timeout=30
            )

            company_data = json.loads(response.choices[0].message.content)

            # Validate and clean the data
            company_data = self._validate_company_data(company_data, url)

            return company_data

        except Exception as e:
            print(f"    ⚠️  AI extraction failed for {company_name}: {str(e)[:100]}")
            return None

    def _fetch_website_content(self, url: str) -> str:
        """Fetch and extract text content from website"""
        try:
            # Add user agent to avoid blocks
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()

            # Simple text extraction (you could use BeautifulSoup for better extraction)
            text = response.text

            # Remove HTML tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text[:5000]  # Limit to first 5000 chars

        except Exception:
            return ""

    def _validate_company_data(self, data: Dict, source_url: str) -> Dict:
        """Validate and clean extracted company data"""
        # Ensure all required fields exist
        required_fields = [
            "name", "description", "state", "email", "phone", "contact",
            "states_perform_work", "NAICS", "other_NAICS", "cage", "duns",
            "designation", "prime_experience"
        ]

        for field in required_fields:
            if field not in data:
                data[field] = None

        # Convert empty strings and "NULL" strings to None
        for key, value in data.items():
            if value in ("", "NULL", "null", "N/A", "n/a", "None", "none"):
                data[key] = None
            elif isinstance(value, str):
                data[key] = value.strip()

        # Validate NAICS codes (must be 6 digits)
        if data.get("NAICS"):
            naics = re.sub(r'[^\d]', '', str(data["NAICS"]))
            data["NAICS"] = naics if len(naics) == 6 else None

        if data.get("other_NAICS"):
            # Clean and validate other NAICS codes
            other_naics = data["other_NAICS"].split(",")
            valid_naics = []
            for code in other_naics:
                clean_code = re.sub(r'[^\d]', '', code.strip())
                if len(clean_code) == 6:
                    valid_naics.append(clean_code)
            data["other_NAICS"] = ", ".join(valid_naics) if valid_naics else None

        # Validate CAGE code (must be 5 characters)
        if data.get("cage"):
            cage = re.sub(r'[^\w]', '', str(data["cage"]).upper())
            data["cage"] = cage if len(cage) == 5 else None

        # Validate DUNS (must be 9 digits)
        if data.get("duns"):
            duns = re.sub(r'[^\d]', '', str(data["duns"]))
            data["duns"] = duns if len(duns) == 9 else None

        # Validate state (must be 2 letter code)
        if data.get("state"):
            state = str(data["state"]).upper().strip()
            us_states = {
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
            }
            data["state"] = state if state in us_states else None

        # Validate email
        if data.get("email"):
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, data["email"]):
                data["email"] = None

        # Validate phone (format as (XXX) XXX-XXXX)
        if data.get("phone"):
            phone = re.sub(r'[^\d]', '', str(data["phone"]))
            if len(phone) == 10:
                data["phone"] = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
            elif len(phone) == 11 and phone[0] == '1':
                data["phone"] = f"({phone[1:4]}) {phone[4:7]}-{phone[7:]}"
            else:
                data["phone"] = None

        # Validate prime_experience (must be Yes/No or None)
        if data.get("prime_experience"):
            value = str(data["prime_experience"]).lower().strip()
            if value in ("yes", "y", "true", "1"):
                data["prime_experience"] = "Yes"
            elif value in ("no", "n", "false", "0"):
                data["prime_experience"] = "No"
            else:
                data["prime_experience"] = None

        # Validate states_perform_work
        if data.get("states_perform_work"):
            value = str(data["states_perform_work"]).strip()
            if value.lower() in ("nationwide", "national", "usa", "united states"):
                data["states_perform_work"] = "All"

        # Validate name (must be reasonable length)
        if data.get("name"):
            if len(data["name"]) < 2 or len(data["name"]) > 200:
                return None  # Invalid company name

        return data

    def is_duplicate(self, company_name: str) -> bool:
        """Check if company already exists in database"""
        if not company_name:
            return False
        normalized_name = company_name.lower().strip()
        return normalized_name in self.existing_companies

    def add_company_to_database(self, company_data: Dict) -> bool:
        """Add company to Supabase database"""
        try:
            # Validate that name exists
            if not company_data.get("name"):
                print("    ⊘ Skipped: Missing company name")
                return False

            with self.engine.begin() as conn:
                # Get the next ID value from the sequence
                next_id_result = conn.execute(text(
                    "SELECT COALESCE(MAX(id), 0) + 1 as next_id FROM company_list"
                ))
                next_id = next_id_result.scalar()

                sql = text("""
                    INSERT INTO company_list 
                    (id, name, description, state, email, phone, contact, 
                     states_perform_work, "NAICS", "other_NAICS", cage, duns, 
                     designation, prime_experience)
                    VALUES 
                    (:id, :name, :description, :state, :email, :phone, :contact,
                     :states_perform_work, :NAICS, :other_NAICS, :cage, :duns,
                     :designation, :prime_experience)
                """)

                company_data['id'] = next_id
                conn.execute(sql, company_data)

            # Add to existing companies set
            self.existing_companies.add(company_data["name"].lower().strip())

            # Log what was added
            non_null_fields = [k for k, v in company_data.items() if v is not None]
            print(f"    ✅ Added: {company_data['name']}")
            print(f"       Fields: {', '.join(non_null_fields)}")

            return True

        except Exception as e:
            print(f"    ❌ Database error for {company_data.get('name', 'Unknown')}: {str(e)[:100]}")
            return False

    def run(self, max_companies: int = 100, search_delay: float = 1.0,
            results_per_query: int = 10):
        """
        Main scraping loop

        Args:
            max_companies: Maximum number of companies to add
            search_delay: Delay between searches (seconds)
            results_per_query: Number of results to fetch per search
        """
        print("=" * 80)
        print("DEFENSE SUPPLIER WEB SCRAPER")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Target companies: {max_companies}")
        print(f"  Search delay: {search_delay}s")
        print(f"  Results per query: {results_per_query}")
        print(f"  Existing companies: {len(self.existing_companies)}")
        print(f"  Total search queries: {len(SEARCH_QUERIES)}")

        print(f"\n{'=' * 80}")
        print("Starting scraping process...")
        print(f"{'=' * 80}\n")

        start_time = time.time()

        for query_num, query in enumerate(SEARCH_QUERIES, 1):
            if self.companies_added >= max_companies:
                break

            print(f"\n[Query {query_num}/{len(SEARCH_QUERIES)}] {query}")

            # Perform search
            results = self.google_search(query, num_results=results_per_query)

            if not results:
                print("  No results found")
                continue

            print(f"  Found {len(results)} results, processing...")

            # Process each result
            for result_num, result in enumerate(results, 1):
                if self.companies_added >= max_companies:
                    break

                print(f"\n  [{result_num}/{len(results)}] Analyzing: {result.get('title', 'Unknown')[:60]}...")

                # Extract company information
                company_data = self.extract_company_info_with_ai(result)

                if not company_data or not company_data.get("name"):
                    print("    ⊘ Skipped: Invalid/insufficient data")
                    self.companies_skipped += 1
                    continue

                # Check for duplicates
                if self.is_duplicate(company_data["name"]):
                    print(f"    ⊘ Skipped: {company_data['name']} (already in database)")
                    self.companies_skipped += 1
                    continue

                # Add to database
                if self.add_company_to_database(company_data):
                    self.companies_added += 1
                else:
                    self.companies_skipped += 1

                # Small delay to be polite to APIs
                time.sleep(0.5)

            # Delay between searches
            time.sleep(search_delay)

        # Final report
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 80)
        print("SCRAPING COMPLETE")
        print("=" * 80)
        print(f"\n📊 Results:")
        print(f"  Companies added: {self.companies_added}")
        print(f"  Companies skipped: {self.companies_skipped}")
        print(f"  Total processed: {self.companies_added + self.companies_skipped}")
        print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes")
        print(f"  Average time per company: {elapsed_time / (self.companies_added + self.companies_skipped):.1f}s")
        print("\n" + "=" * 80)


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(
        description="Scrape defense suppliers from the web and add to Supabase database"
    )
    parser.add_argument(
        "--max-companies",
        type=int,
        default=100,
        help="Maximum number of companies to add (default: 100)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between searches in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=10,
        help="Number of results to fetch per query (default: 10, max: 10)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.max_companies < 1:
        print("Error: --max-companies must be at least 1")
        sys.exit(1)

    if args.delay < 0:
        print("Error: --delay must be non-negative")
        sys.exit(1)

    if args.results_per_query < 1 or args.results_per_query > 10:
        print("Error: --results-per-query must be between 1 and 10")
        sys.exit(1)

    # Run scraper
    try:
        scraper = DefenseSupplierScraper()
        scraper.run(
            max_companies=args.max_companies,
            search_delay=args.delay,
            results_per_query=args.results_per_query
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()