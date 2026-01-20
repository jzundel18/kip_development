#!/usr/bin/env python3
"""
Teaming Opportunity Matcher
Matches technology areas with companies that do similar work and generates
collaboration opportunity descriptions.

Usage:
    python teaming.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from sqlalchemy import create_engine, text
from openai import OpenAI


# Load environment variables directly in the script
def load_env_variables():
    """Load environment variables from .env file."""
    env_paths = [
        Path('.env'),
        Path(__file__).parent / '.env',
        Path.home() / '.kip' / '.env',
    ]

    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        # Remove quotes if present
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value
            return True
    return False


# Load environment variables
load_env_variables()

# Get required environment variables
# Support both DATABASE_URL and SUPABASE_DB_URL
DATABASE_URL = os.getenv('DATABASE_URL') or os.getenv('SUPABASE_DB_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not DATABASE_URL:
    print("Error: DATABASE_URL or SUPABASE_DB_URL environment variable not set")
    print("\nPlease create a .env file in one of these locations:")
    print("  - Current directory: .env")
    print("  - Script directory: .env")
    print("  - Home directory: ~/.kip/.env")
    print("\nWith contents:")
    print("SUPABASE_DB_URL=postgresql://user:password@host:port/database")
    print("# OR")
    print("DATABASE_URL=postgresql://user:password@host:port/database")
    print("OPENAI_API_KEY=sk-...")
    sys.exit(1)

if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Please add it to your .env file")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

print(f"✓ Database URL configured")
print(f"✓ OpenAI API key configured")


def get_database_connection():
    """Create and return a database engine."""
    return create_engine(DATABASE_URL)


def inspect_schema(engine):
    """Inspect and validate database schema."""
    from sqlalchemy import inspect as sqla_inspect
    inspector = sqla_inspect(engine)

    print("\n" + "=" * 80)
    print("DATABASE SCHEMA INSPECTION")
    print("=" * 80)

    tables = inspector.get_table_names()
    print(f"\nFound {len(tables)} tables: {', '.join(tables)}")

    schema_info = {}

    # Check for technology_areas table
    if 'technology_areas' in tables:
        print("\n📋 technology_areas columns:")
        cols = inspector.get_columns('technology_areas')
        schema_info['technology_areas'] = [col['name'] for col in cols]
        for col in cols:
            print(f"  - {col['name']} ({col['type']})")
    else:
        print("\n⚠️  WARNING: 'technology_areas' table not found!")

    # Check for companies table
    if 'companies' in tables:
        print("\n🏢 companies columns:")
        cols = inspector.get_columns('companies')
        schema_info['companies'] = [col['name'] for col in cols]
        for col in cols:
            print(f"  - {col['name']} ({col['type']})")
    else:
        print("\n⚠️  WARNING: 'companies' table not found!")

    print("=" * 80 + "\n")

    return schema_info


def fetch_technology_areas(engine, schema_info):
    """Fetch all technology areas from the database."""
    query = text("""
        SELECT id, technology_name, description, emails
        FROM technology_areas
        ORDER BY technology_name
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = [dict(row._mapping) for row in result]

    print(f"✓ Fetched {len(rows)} technology areas")
    return rows


def fetch_companies(engine, schema_info):
    """Fetch all companies from the database."""
    query = text("""
        SELECT id, name, contact, phone, email, state, duns, cage, 
               "NAICS", designation, states_perform_work, description
        FROM companies
        ORDER BY name
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        rows = [dict(row._mapping) for row in result]

    print(f"✓ Fetched {len(rows)} companies")
    return rows


def generate_match_score_and_rationale(tech_area, company):
    """
    Use AI to score how well a company matches a technology area
    and generate a rationale for teaming.
    """
    # Build technology area description
    tech_desc = f"Name: {tech_area.get('technology_name', 'N/A')}\n"
    if 'description' in tech_area and tech_area['description']:
        tech_desc += f"Description: {tech_area['description']}\n"

    # Build company description
    company_desc = f"Name: {company.get('name', 'N/A')}\n"
    if 'description' in company and company['description']:
        company_desc += f"Description: {company['description']}\n"
    if 'designation' in company and company['designation']:
        company_desc += f"Designation: {company['designation']}\n"
    if 'NAICS' in company and company['NAICS']:
        company_desc += f"Primary NAICS: {company['NAICS']}\n"
    if 'other_naics' in company and company['other_naics']:
        company_desc += f"Other NAICS: {company['other_naics']}\n"
    if 'states_perform_work' in company and company['states_perform_work']:
        company_desc += f"States of Performance: {company['states_perform_work']}\n"
    if 'state' in company and company['state']:
        company_desc += f"Location: {company['state']}\n"

    prompt = f"""You are analyzing potential teaming opportunities between a technology area and a company for federal government contracting.

TECHNOLOGY AREA:
{tech_desc}

COMPANY:
{company_desc}

Task: Analyze this match and provide:
1. A match score from 0-100 (how well does this company's work align with this technology area?)
2. A 2-3 paragraph teaming opportunity description that includes:
   - How the technology area could be applied to the company's work
   - Specific ways we could collaborate on federal contracts
   - Why this partnership would be beneficial
   - How to approach them (what to emphasize, what projects to highlight)
   - Consider their business designation and NAICS codes when relevant

Format your response as:
SCORE: [number]

TEAMING OPPORTUNITY:
[Your 2-3 paragraph description]

Be specific and actionable. Focus on concrete collaboration opportunities in federal contracting."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "You are an expert at identifying business development and teaming opportunities in government contracting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        content = response.choices[0].message.content

        # Parse the response
        lines = content.split('\n')
        score = 0
        opportunity_text = []
        found_score = False
        found_opportunity = False

        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    score = int(line.split(':')[1].strip())
                    found_score = True
                except:
                    pass
            elif 'TEAMING OPPORTUNITY:' in line:
                found_opportunity = True
            elif found_opportunity and line.strip():
                opportunity_text.append(line.strip())

        opportunity = '\n\n'.join(opportunity_text) if opportunity_text else content

        return score, opportunity

    except Exception as e:
        print(
            f"Error generating match for {tech_area.get('technology_name', 'Unknown')} + {company.get('name', 'Unknown')}: {e}")
        return 0, ""


def get_embeddings_batch(texts, model="text-embedding-3-small"):
    """Get embeddings for multiple texts at once (much faster than one-by-one)."""
    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)


def build_company_embeddings_cache(companies):
    """
    Build embeddings for all companies at once (much faster than individually).
    This is done once at the start and reused for all technology areas.
    """
    print("\n" + "=" * 80)
    print("BUILDING COMPANY EMBEDDINGS CACHE (one-time setup)")
    print("=" * 80)
    print(f"Processing {len(companies)} companies in batches...")

    company_texts = []
    for company in companies:
        # Create company description
        company_text = f"{company.get('name', '')} - {company.get('description', '')}"
        if company.get('designation'):
            company_text += f" {company['designation']}"
        if company.get('NAICS'):
            company_text += f" NAICS: {company['NAICS']}"
        company_texts.append(company_text)

    # Process in batches of 100 (API limit is typically 2048)
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(company_texts), batch_size):
        batch = company_texts[i:i + batch_size]
        print(
            f"  Processing batch {i // batch_size + 1}/{(len(company_texts) - 1) // batch_size + 1} ({len(batch)} companies)...")
        embeddings = get_embeddings_batch(batch)
        if embeddings:
            all_embeddings.extend(embeddings)
        else:
            print(f"  Warning: Failed to get embeddings for batch {i // batch_size + 1}")
            all_embeddings.extend([None] * len(batch))

    print(f"✓ Created embeddings for {len(all_embeddings)} companies")
    print("=" * 80 + "\n")

    # Return dictionary mapping company index to embedding
    return {i: emb for i, emb in enumerate(all_embeddings) if emb is not None}


def prefilter_companies_by_embedding(tech_area, companies, company_embeddings_cache, top_k=5):
    """
    Use embeddings to quickly filter down to the most relevant companies.
    Uses pre-computed company embeddings for speed.
    """
    # Create a rich description of the technology area
    tech_text = f"{tech_area.get('technology_name', '')} - {tech_area.get('description', '')}"

    print(f"  Getting embedding for technology area...")
    tech_embeddings = get_embeddings_batch([tech_text])
    if not tech_embeddings:
        return companies[:top_k]  # Fallback to first K companies

    tech_embedding = tech_embeddings[0]

    # Calculate similarity using cached company embeddings
    print(f"  Calculating similarity scores for {len(companies)} companies...")
    company_scores = []

    for i, company in enumerate(companies):
        if i in company_embeddings_cache:
            similarity = cosine_similarity(tech_embedding, company_embeddings_cache[i])
            company_scores.append((company, similarity))

    # Sort by similarity and return top K
    company_scores.sort(key=lambda x: x[1], reverse=True)
    top_companies = [comp for comp, score in company_scores[:top_k]]

    print(f"  ✓ Narrowed down to top {len(top_companies)} most relevant companies")
    return top_companies


def find_top_matches(tech_area, companies, company_embeddings_cache, top_n=2):
    """Find the top N company matches for a technology area."""
    print(f"\nAnalyzing matches for: {tech_area.get('technology_name', 'Unknown')}")

    # First, use embeddings to pre-filter to top 10 most relevant companies
    # This is MUCH faster since we're using cached embeddings
    candidate_companies = prefilter_companies_by_embedding(
        tech_area, companies, company_embeddings_cache, top_k=10
    )

    # Now run detailed AI analysis only on the pre-filtered candidates
    print(f"  Running detailed analysis on {len(candidate_companies)} candidates...")
    matches = []
    for i, company in enumerate(candidate_companies):
        print(f"    Analyzing {i + 1}/{len(candidate_companies)}: {company.get('name', 'Unknown')}...")
        score, opportunity = generate_match_score_and_rationale(tech_area, company)

        if score > 0:  # Only include matches with positive scores
            matches.append({
                'company': company,
                'score': score,
                'opportunity': opportunity
            })

    # Sort by score and return top N
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches[:top_n]


def generate_report(technology_areas, companies):
    """Generate the teaming opportunities report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create outputs directory if it doesn't exist
    output_dir = Path(__file__).parent / 'outputs'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"teaming_{timestamp}.txt"

    # Build company embeddings cache once (instead of recalculating for each tech area)
    company_embeddings_cache = build_company_embeddings_cache(companies)

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEAMING OPPORTUNITIES REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for tech_area in technology_areas:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TECHNOLOGY AREA: {tech_area.get('technology_name', 'N/A')}\n")
            f.write("=" * 80 + "\n")
            if 'description' in tech_area and tech_area['description']:
                f.write(f"Description: {tech_area['description']}\n")
            if 'emails' in tech_area and tech_area['emails']:
                f.write(f"Contact Emails: {tech_area['emails']}\n")
            f.write("\n")

            # Find top matches for this technology area
            matches = find_top_matches(tech_area, companies, company_embeddings_cache, top_n=2)

            if matches:
                for i, match in enumerate(matches, 1):
                    company = match['company']
                    f.write(f"\n{'-' * 80}\n")
                    f.write(f"MATCH #{i}: {company.get('name', 'N/A')} (Score: {match['score']}/100)\n")
                    f.write(f"{'-' * 80}\n\n")

                    # Company details
                    if company.get('contact'):
                        f.write(f"Contact: {company['contact']}\n")
                    if company.get('email'):
                        f.write(f"Email: {company['email']}\n")
                    if company.get('phone'):
                        f.write(f"Phone: {company['phone']}\n")
                    if company.get('state'):
                        f.write(f"Location: {company['state']}\n")
                    if company.get('designation'):
                        f.write(f"Designation: {company['designation']}\n")
                    if company.get('NAICS'):
                        f.write(f"Primary NAICS: {company['NAICS']}\n")
                    if company.get('cage'):
                        f.write(f"CAGE Code: {company['cage']}\n")
                    if company.get('duns'):
                        f.write(f"DUNS: {company['duns']}\n")
                    f.write("\n")

                    if company.get('description'):
                        f.write(f"Company Description:\n{company['description']}\n\n")

                    f.write("TEAMING OPPORTUNITY:\n\n")
                    f.write(match['opportunity'])
                    f.write("\n\n")
            else:
                f.write("No strong matches found for this technology area.\n\n")

    return output_file


def main():
    """Main execution function."""
    print("Starting Teaming Opportunity Matcher...")
    print("=" * 80)

    # Create database connection
    engine = get_database_connection()

    # Inspect database schema
    try:
        schema_info = inspect_schema(engine)
    except Exception as e:
        print(f"Error inspecting database schema: {e}")
        print("Continuing with default schema assumptions...")
        schema_info = {
            'technology_areas': ['id', 'name', 'description', 'keywords'],
            'companies': ['id', 'name', 'description', 'capabilities', 'past_performance']
        }

    # Fetch data
    print("\nFetching technology areas...")
    try:
        technology_areas = fetch_technology_areas(engine, schema_info)
        print(f"Found {len(technology_areas)} technology areas")
    except Exception as e:
        print(f"Error fetching technology areas: {e}")
        sys.exit(1)

    print("\nFetching companies...")
    try:
        companies = fetch_companies(engine, schema_info)
        print(f"Found {len(companies)} companies")
    except Exception as e:
        print(f"Error fetching companies: {e}")
        sys.exit(1)

    if not technology_areas:
        print("Error: No technology areas found in database")
        sys.exit(1)

    if not companies:
        print("Error: No companies found in database")
        sys.exit(1)

    # Generate report
    print("\nGenerating teaming opportunities report...")
    print("This may take several minutes depending on the number of technology areas and companies...\n")

    output_file = generate_report(technology_areas, companies)

    print("\n" + "=" * 80)
    print(f"Report generated successfully: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()