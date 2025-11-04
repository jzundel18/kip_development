#!/usr/bin/env python3
"""
Defense Supplier Web Scraping Script - ENHANCED VERSION
Features:
- Better deduplication (checks names, domains, emails)
- Search query rotation to find NEW companies each run
- Persistent state tracking to avoid repeating searches
- Domain-based duplicate detection

Required Environment Variables:
- SUPABASE_DB_URL: Your Supabase database connection string
- GOOGLE_API_KEY: Your Google Custom Search API key
- GOOGLE_CX: Your Google Custom Search Engine ID
- OPENAI_API_KEY: OpenAI API key for intelligent data extraction

Usage:
    python defense_supplier_scraper_enhanced.py [--max-companies 100] [--delay 1.0]
"""

import os
import sys
import time
import re
import json
import argparse
import hashlib
from typing import List, Dict, Optional, Set, Tuple
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
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

# State file to track what we've searched
STATE_FILE = Path(".scraper_state.json")

# Defense supplier search queries - EXPANDED with more variations
SEARCH_QUERY_POOL = [
    # Aerospace & Aviation
    "aerospace defense parts manufacturer supplier",
    "aircraft components defense contractor",
    "avionics systems defense supplier",
    "helicopter parts military supplier",
    "UAV drone components manufacturer",
    "military aircraft maintenance services",
    "aerospace engineering defense contractor",
    "flight systems integration military",

    # Electronics & Communications
    "military electronics components supplier",
    "defense communications equipment manufacturer",
    "tactical radio systems supplier",
    "military radar components manufacturer",
    "defense semiconductor supplier",
    "electronic warfare systems contractor",
    "military signal intelligence equipment",
    "defense satellite communications",

    # Weapons & Ammunition
    "military ammunition components supplier",
    "defense weapons systems parts manufacturer",
    "firearms components defense supplier",
    "military ordnance parts manufacturer",
    "explosive ordnance disposal equipment",
    "munitions handling equipment military",

    # Vehicles & Ground Systems
    "military vehicle parts supplier",
    "defense ground systems components",
    "tactical vehicle parts manufacturer",
    "military truck components supplier",
    "armored vehicle parts manufacturer",
    "combat vehicle systems contractor",
    "military vehicle maintenance services",

    # Naval & Maritime
    "naval defense components supplier",
    "ship systems parts manufacturer",
    "submarine components defense supplier",
    "maritime defense equipment manufacturer",
    "naval propulsion systems contractor",
    "shipboard electronics military",

    # IT & Cybersecurity
    "defense IT services contractor",
    "military cybersecurity services provider",
    "defense software development company",
    "military network security contractor",
    "defense cloud services provider",
    "military data analytics contractor",

    # Manufacturing & Machining
    "precision machining defense contractor",
    "CNC machining military parts",
    "defense metal fabrication supplier",
    "military precision manufacturing",
    "additive manufacturing defense",
    "defense tooling and fixtures",

    # Logistics & Support
    "defense logistics services contractor",
    "military supply chain management",
    "defense maintenance services contractor",
    "military warehousing services provider",
    "defense inventory management",
    "military distribution services",

    # Textiles & Equipment
    "military uniform manufacturer supplier",
    "defense tactical gear manufacturer",
    "military body armor supplier",
    "defense protective equipment manufacturer",
    "combat helmet manufacturer military",
    "tactical load bearing equipment",

    # Testing & Engineering
    "defense testing services contractor",
    "military engineering services provider",
    "defense R&D contractor",
    "military test equipment manufacturer",
    "defense systems integration",
    "military prototype development",

    # Medical & Healthcare
    "military medical equipment supplier",
    "defense healthcare services contractor",
    "combat casualty care equipment",
    "military pharmaceutical supplier",

    # Energy & Power
    "military power systems contractor",
    "defense energy solutions provider",
    "tactical power generation military",
    "defense renewable energy systems",

    # Training & Simulation
    "military training systems contractor",
    "defense simulation equipment supplier",
    "combat training facility services",
    "military virtual reality training",
]

# NAICS codes remain the same
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


class ScraperState:
    """Track scraper state to avoid repeating searches"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                pass

        return {
            "searched_queries": [],
            "last_run": None,
            "total_companies_added": 0,
            "search_history": {}
        }

    def save_state(self):
        """Save state to file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save state: {e}")

    def mark_query_searched(self, query: str, found_count: int):
        """Mark a query as searched"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.state["searched_queries"].append(query_hash)
        self.state["search_history"][query_hash] = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "found_count": found_count
        }

    def is_query_searched(self, query: str) -> bool:
        """Check if query was already searched"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return query_hash in self.state["searched_queries"]

    def get_unsearched_queries(self, query_pool: List[str]) -> List[str]:
        """Get queries that haven't been searched yet"""
        return [q for q in query_pool if not self.is_query_searched(q)]

    def reset(self):
        """Reset state (useful for starting fresh)"""
        self.state = {
            "searched_queries": [],
            "last_run": None,
            "total_companies_added": 0,
            "search_history": {}
        }
        self.save_state()

    def update_stats(self, companies_added: int):
        """Update statistics"""
        self.state["last_run"] = datetime.now().isoformat()
        self.state["total_companies_added"] = self.state.get("total_companies_added", 0) + companies_added


class DefenseSupplierScraper:
    """Main scraper class with enhanced deduplication"""

    def __init__(self, state: ScraperState):
        self.validate_credentials()
        self.engine = create_engine(DB_URL, pool_pre_ping=True)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.existing_companies = self._load_existing_companies()
        self.state = state
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

    def _load_existing_companies(self) -> dict:
        """Load existing company names, domains, and emails to avoid duplicates"""
        existing = {
            "names": set(),
            "domains": set(),
            "emails": set()
        }

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT name, email FROM company_list"))

                for row in result:
                    if row[0]:  # name
                        existing["names"].add(row[0].lower().strip())

                    if row[1]:  # email
                        existing["emails"].add(row[1].lower().strip())
                        # Extract domain from email
                        if "@" in row[1]:
                            domain = row[1].split("@")[1].lower().strip()
                            existing["domains"].add(domain)

                print(f"✅ Loaded {len(existing['names'])} companies, {len(existing['domains'])} domains")

        except Exception as e:
            print(f"⚠️  Warning: Could not load existing companies: {e}")

        return existing

    def google_search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform Google Custom Search"""
        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "num": min(num_results, 10)
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
        """Use AI to extract comprehensive company information"""
        title = search_result.get("title", "")
        snippet = search_result.get("snippet", "")
        url = search_result.get("link", "")

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
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
            response.raise_for_status()

            text = response.text

            # Remove HTML tags
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)

            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            return text[:5000]

        except Exception:
            return ""

    def _validate_company_data(self, data: Dict, source_url: str) -> Dict:
        """Validate and clean extracted company data"""
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
                return None

        return data

    def is_duplicate(self, company_data: Dict) -> Tuple[bool, str]:
        """
        Check if company already exists using multiple methods.
        Returns (is_duplicate, reason)
        """
        if not company_data.get("name"):
            return True, "Missing company name"

        # Check by name
        normalized_name = company_data["name"].lower().strip()
        if normalized_name in self.existing_companies["names"]:
            return True, "Duplicate name"

        # Check by email domain
        if company_data.get("email"):
            email = company_data["email"].lower().strip()

            # Check exact email
            if email in self.existing_companies["emails"]:
                return True, "Duplicate email"

            # Check domain
            if "@" in email:
                domain = email.split("@")[1]
                if domain in self.existing_companies["domains"]:
                    return True, f"Duplicate domain ({domain})"

        return False, "Not duplicate"

    def add_company_to_database(self, company_data: Dict) -> bool:
        """Add company to Supabase database"""
        try:
            if not company_data.get("name"):
                print("    ⊘ Skipped: Missing company name")
                return False

            with self.engine.begin() as conn:
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
            self.existing_companies["names"].add(company_data["name"].lower().strip())

            # Add email and domain
            if company_data.get("email"):
                email = company_data["email"].lower().strip()
                self.existing_companies["emails"].add(email)
                if "@" in email:
                    domain = email.split("@")[1]
                    self.existing_companies["domains"].add(domain)

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
        Main scraping loop with state tracking
        """
        print("=" * 80)
        print("DEFENSE SUPPLIER WEB SCRAPER - ENHANCED")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Target companies: {max_companies}")
        print(f"  Search delay: {search_delay}s")
        print(f"  Results per query: {results_per_query}")
        print(f"  Existing companies: {len(self.existing_companies['names'])}")
        print(f"  Existing domains: {len(self.existing_companies['domains'])}")

        # Get unsearched queries
        unsearched = self.state.get_unsearched_queries(SEARCH_QUERY_POOL)
        print(f"  Available search queries: {len(unsearched)}")
        print(f"  Previously searched: {len(self.state.state['searched_queries'])}")

        if not unsearched:
            print("\n⚠️  All queries have been searched!")
            print("Options:")
            print("  1. Run with --reset flag to start fresh")
            print("  2. Add more search queries to SEARCH_QUERY_POOL")
            return

        print(f"\n{'=' * 80}")
        print("Starting scraping process...")
        print(f"{'=' * 80}\n")

        start_time = time.time()

        for query_num, query in enumerate(unsearched, 1):
            if self.companies_added >= max_companies:
                break

            print(f"\n[Query {query_num}/{len(unsearched)}] {query}")

            # Perform search
            results = self.google_search(query, num_results=results_per_query)

            if not results:
                print("  No results found")
                self.state.mark_query_searched(query, 0)
                continue

            print(f"  Found {len(results)} results, processing...")
            found_in_query = 0

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
                is_dup, reason = self.is_duplicate(company_data)
                if is_dup:
                    print(f"    ⊘ Skipped: {company_data['name']} ({reason})")
                    self.companies_skipped += 1
                    continue

                # Add to database
                if self.add_company_to_database(company_data):
                    self.companies_added += 1
                    found_in_query += 1
                else:
                    self.companies_skipped += 1

                # Small delay to be polite to APIs
                time.sleep(0.5)

            # Mark query as searched
            self.state.mark_query_searched(query, found_in_query)

            # Save state after each query
            self.state.save_state()

            # Delay between searches
            time.sleep(search_delay)

        # Final report
        elapsed_time = time.time() - start_time

        # Update state
        self.state.update_stats(self.companies_added)
        self.state.save_state()

        print("\n" + "=" * 80)
        print("SCRAPING COMPLETE")
        print("=" * 80)
        print(f"\n📊 Results:")
        print(f"  Companies added: {self.companies_added}")
        print(f"  Companies skipped: {self.companies_skipped}")
        print(f"  Total processed: {self.companies_added + self.companies_skipped}")
        print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes")
        print(f"  Average time per company: {elapsed_time / (self.companies_added + self.companies_skipped):.1f}s")
        print(f"\n📈 Overall Progress:")
        print(f"  Total queries searched: {len(self.state.state['searched_queries'])}/{len(SEARCH_QUERY_POOL)}")
        print(f"  Total companies added (all time): {self.state.state.get('total_companies_added', 0)}")
        print(f"  Queries remaining: {len(SEARCH_QUERY_POOL) - len(self.state.state['searched_queries'])}")
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
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset search state and start fresh"
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

    # Initialize state
    state = ScraperState(STATE_FILE)

    if args.reset:
        print("🔄 Resetting search state...")
        state.reset()
        print("✅ State reset complete")

    # Run scraper
    try:
        scraper = DefenseSupplierScraper(state)
        scraper.run(
            max_companies=args.max_companies,
            search_delay=args.delay,
            results_per_query=args.results_per_query
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrupted by user")
        state.save_state()
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        state.save_state()
        sys.exit(1)


if __name__ == "__main__":
    main()