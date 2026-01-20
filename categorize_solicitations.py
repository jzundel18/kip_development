#!/usr/bin/env python3
"""
categorize_solicitations_naics.py

Fast categorization using NAICS codes instead of AI.
Categorizes solicitations as parts/services/research based on NAICS code ranges.

This is 100x faster and free compared to the AI approach!

Usage:
    python categorize_solicitations_naics.py [--batch-size N] [--dry-run]

Options:
    --batch-size N    Process N solicitations at a time (default: 1000)
    --dry-run         Show what would be categorized without making changes
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path
import sqlalchemy as sa
from sqlalchemy import text

# Try to load .env file
try:
    from dotenv import load_dotenv

    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
    else:
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

DB_URL = None
engine = None


def _load_config():
    """Load configuration from environment variables"""
    global DB_URL
    DB_URL = os.getenv("SUPABASE_DB_URL")

    if not DB_URL:
        print("Missing required env var: SUPABASE_DB_URL", file=sys.stderr)
        sys.exit(1)


# NAICS Code Categorization Rules
# Based on official NAICS 2-digit sector codes
NAICS_CATEGORIES = {
    'research': [
        '541711',  # Research and Development in Biotechnology
        '541712',  # Research and Development in Physical, Engineering, Life Sciences (except Biotech)
        '541713',  # Research and Development in Nanotechnology
        '541714',  # Research and Development in Biotechnology (except Nanobiotechnology)
        '541715',  # Research and Development in Physical, Engineering, and Life Sciences
        # Any 5417xx codes are R&D
    ],
    'services': [
        # Professional, Scientific, and Technical Services (54xxxx)
        '541',  # Professional, Scientific, and Technical Services
        '5415',  # Computer Systems Design and Related Services
        '5416',  # Management, Scientific, and Technical Consulting Services
        '5418',  # Advertising, Public Relations, and Related Services
        '5419',  # Other Professional, Scientific, and Technical Services

        # Administrative and Support Services (56xxxx)
        '561',  # Administrative and Support Services
        '562',  # Waste Management and Remediation Services

        # Educational Services (61xxxx)
        '611',  # Educational Services

        # Health Care Services (62xxxx)
        '621',  # Ambulatory Health Care Services
        '622',  # Hospitals
        '623',  # Nursing and Residential Care Facilities
        '624',  # Social Assistance

        # Information Services (51xxxx)
        '511',  # Publishing Industries
        '512',  # Motion Picture and Sound Recording Industries
        '515',  # Broadcasting (except Internet)
        '517',  # Telecommunications
        '518',  # Data Processing, Hosting, and Related Services
        '519',  # Other Information Services

        # Utilities (22xxxx)
        '221',  # Utilities

        # Transportation (48-49xxxx)
        '481',  # Air Transportation
        '482',  # Rail Transportation
        '483',  # Water Transportation
        '484',  # Truck Transportation
        '485',  # Transit and Ground Passenger Transportation
        '486',  # Pipeline Transportation
        '487',  # Scenic and Sightseeing Transportation
        '488',  # Support Activities for Transportation
        '492',  # Couriers and Messengers
        '493',  # Warehousing and Storage

        # Real Estate and Rental (53xxxx)
        '531',  # Real Estate
        '532',  # Rental and Leasing Services
        '533',  # Lessors of Nonfinancial Intangible Assets

        # Other Services (81xxxx)
        '811',  # Repair and Maintenance
        '812',  # Personal and Laundry Services
        '813',  # Religious, Grantmaking, Civic, Professional Organizations
        '814',  # Private Households
    ],
    'parts': [
        # Manufacturing (31-33xxxx)
        '311',  # Food Manufacturing
        '312',  # Beverage and Tobacco Product Manufacturing
        '313',  # Textile Mills
        '314',  # Textile Product Mills
        '315',  # Apparel Manufacturing
        '316',  # Leather and Allied Product Manufacturing
        '321',  # Wood Product Manufacturing
        '322',  # Paper Manufacturing
        '323',  # Printing and Related Support Activities
        '324',  # Petroleum and Coal Products Manufacturing
        '325',  # Chemical Manufacturing
        '326',  # Plastics and Rubber Products Manufacturing
        '327',  # Nonmetallic Mineral Product Manufacturing
        '331',  # Primary Metal Manufacturing
        '332',  # Fabricated Metal Product Manufacturing
        '333',  # Machinery Manufacturing
        '334',  # Computer and Electronic Product Manufacturing
        '335',  # Electrical Equipment, Appliance, and Component Manufacturing
        '336',  # Transportation Equipment Manufacturing
        '337',  # Furniture and Related Product Manufacturing
        '339',  # Miscellaneous Manufacturing

        # Wholesale Trade (42xxxx)
        '423',  # Merchant Wholesalers, Durable Goods
        '424',  # Merchant Wholesalers, Nondurable Goods
        '425',  # Wholesale Electronic Markets and Agents and Brokers

        # Retail Trade (44-45xxxx) - when buying physical goods
        '441',  # Motor Vehicle and Parts Dealers
        '442',  # Furniture and Home Furnishings Stores
        '443',  # Electronics and Appliance Stores
        '444',  # Building Material and Garden Equipment and Supplies Dealers
        '445',  # Food and Beverage Stores
        '446',  # Health and Personal Care Stores
        '447',  # Gasoline Stations
        '448',  # Clothing and Clothing Accessories Stores
        '451',  # Sporting Goods, Hobby, Musical Instrument, and Book Stores
        '452',  # General Merchandise Stores
        '453',  # Miscellaneous Store Retailers
        '454',  # Nonstore Retailers

        # Construction (23xxxx) - when procuring materials
        '236',  # Construction of Buildings (when materials procurement)
        '237',  # Heavy and Civil Engineering Construction (when materials procurement)
        '238',  # Specialty Trade Contractors (when materials procurement)

        # Agriculture, Forestry, Fishing (11xxxx) - physical products
        '111',  # Crop Production
        '112',  # Animal Production and Aquaculture
        '113',  # Forestry and Logging
        '114',  # Fishing, Hunting and Trapping
        '115',  # Support Activities for Agriculture and Forestry

        # Mining (21xxxx) - physical products
        '211',  # Oil and Gas Extraction
        '212',  # Mining (except Oil and Gas)
        '213',  # Support Activities for Mining
    ]
}


def categorize_by_naics(naics_code: str) -> str:
    """
    Categorize a solicitation based on its NAICS code.

    Args:
        naics_code: The NAICS code (can be 2-6 digits)

    Returns:
        Category: 'research', 'services', or 'parts'
    """
    if not naics_code or naics_code.strip() == '':
        return 'services'  # Default for unknown

    # Clean the NAICS code
    naics = naics_code.strip()

    # Check for exact research codes first (5417xx)
    if naics.startswith('5417'):
        return 'research'

    # Check each category by matching prefixes
    for category, prefixes in NAICS_CATEGORIES.items():
        for prefix in prefixes:
            if naics.startswith(prefix):
                return category

    # Default to services if no match
    # (most government contracts are services if not manufacturing/R&D)
    return 'services'


def categorize_batch_naics(conn, batch_size: int = 1000, dry_run: bool = False) -> int:
    """
    Categorize a batch of solicitations using NAICS codes.

    Returns:
        Number of solicitations categorized
    """
    # Fetch uncategorized solicitations with NAICS codes
    try:
        query = text("""
            SELECT notice_id, naics_code
            FROM solicitationraw
            WHERE category IS NULL
            LIMIT :batch_size
        """)

        result = conn.execute(query, {"batch_size": batch_size})
        rows = result.fetchall()

        if not rows:
            return 0

        logging.info(f"Processing batch of {len(rows)} solicitations...")

        # Categorize each one
        updates = []
        category_counts = {'research': 0, 'services': 0, 'parts': 0}

        for notice_id, naics_code in rows:
            category = categorize_by_naics(naics_code)
            updates.append({
                'notice_id': notice_id,
                'category': category
            })
            category_counts[category] += 1

        logging.info(f"  Research: {category_counts['research']}")
        logging.info(f"  Services: {category_counts['services']}")
        logging.info(f"  Parts: {category_counts['parts']}")

        if dry_run:
            logging.info(f"[DRY RUN] Would update {len(updates)} solicitations")
            return len(updates)

        # Batch update
        update_query = text("""
            UPDATE solicitationraw
            SET category = :category
            WHERE notice_id = :notice_id
        """)

        conn.execute(update_query, updates)
        conn.commit()

        logging.info(f"✓ Updated {len(updates)} solicitations")
        return len(updates)

    except Exception as e:
        logging.error(f"Error in batch categorization: {e}")
        conn.rollback()
        return 0


def show_statistics(conn):
    """Show categorization statistics"""
    try:
        # Total by category
        query = text("""
            SELECT 
                category,
                COUNT(*) as count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as percentage
            FROM solicitationraw
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY count DESC
        """)

        result = conn.execute(query)
        rows = result.fetchall()

        logging.info("\n📊 Categorization Statistics:")
        logging.info("-" * 50)
        for row in rows:
            category, count, pct = row
            logging.info(f"  {category:12} {count:6,} ({pct:5.1f}%)")

        # Count uncategorized
        uncategorized_query = text("""
            SELECT COUNT(*) 
            FROM solicitationraw 
            WHERE category IS NULL
        """)
        uncategorized = conn.execute(uncategorized_query).scalar()
        logging.info(f"  {'Uncategorized':12} {uncategorized:6,}")
        logging.info("-" * 50)

        # Show sample NAICS codes per category
        sample_query = text("""
            SELECT category, naics_code, COUNT(*) as cnt
            FROM solicitationraw
            WHERE category IS NOT NULL
            AND naics_code IS NOT NULL
            GROUP BY category, naics_code
            ORDER BY category, cnt DESC
        """)

        result = conn.execute(sample_query)
        rows = result.fetchall()

        logging.info("\n📋 Top NAICS Codes by Category:")
        logging.info("-" * 50)

        current_category = None
        shown_per_category = 0
        for row in rows:
            category, naics, cnt = row

            if category != current_category:
                current_category = category
                shown_per_category = 0
                logging.info(f"\n{category.upper()}:")

            if shown_per_category < 5:  # Show top 5 per category
                logging.info(f"  {naics:8} ({cnt:4,} solicitations)")
                shown_per_category += 1

        logging.info("-" * 50)

    except Exception as e:
        logging.error(f"Error fetching statistics: {e}")


def main():
    global engine

    parser = argparse.ArgumentParser(description="Categorize solicitations using NAICS codes (fast!)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Number of solicitations to process per batch (default: from env or 1000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    # Support environment variable for batch size
    if args.batch_size is None:
        args.batch_size = int(os.getenv("CATEGORIZE_BATCH_SIZE", "1000"))

    _load_config()

    engine = sa.create_engine(DB_URL, pool_pre_ping=True)

    logging.info("=" * 70)
    logging.info("🏷️  NAICS-Based Solicitation Categorization")
    logging.info("=" * 70)
    if args.dry_run:
        logging.info("⚠️  DRY RUN MODE - No changes will be made")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info("Using NAICS code mapping (fast, no AI needed!)")
    logging.info("=" * 70)
    logging.info("")

    total_processed = 0

    with engine.connect() as conn:
        # Show initial statistics
        show_statistics(conn)
        logging.info("")

        # Process all uncategorized solicitations
        while True:
            updated = categorize_batch_naics(conn, args.batch_size, args.dry_run)

            if updated == 0:
                logging.info("✓ No more uncategorized solicitations found")
                break

            total_processed += updated
            logging.info(f"Progress: {total_processed} total processed\n")

        # Show final statistics
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"✓ Categorization complete!")
        logging.info(f"  Total processed: {total_processed}")
        logging.info("=" * 70)
        logging.info("")

        if not args.dry_run:
            show_statistics(conn)


if __name__ == "__main__":
    main()