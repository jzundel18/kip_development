#!/usr/bin/env python3
"""
Fast auto-refresh for SAM.gov opportunities:
- Uses v2 search with pagination + key rotation.
- Inserts ONLY brand-new notice_ids.
- Does NOT fetch long descriptions (keeps it fast).
- Always stores a PUBLIC web URL for each notice (no API key required).
- FIXED: Better key rotation and quota handling
- NEW: Filters out solicitations with past response dates
- FIXED: Leaves description blank if it's an api.sam.gov URL so backfill can fetch real descriptions
- NEW: Filters for R&D NAICS codes only (5417xx series)
- NEW: Comprehensive filtering to exclude DOT, DHHS, healthcare, events, facilities, etc.
"""

from __future__ import annotations
import os
import sys
import time
import json
import re
from datetime import datetime, timezone, date
from typing import Any, Dict, List

import sqlalchemy as sa
from sqlalchemy import text, create_engine
from sqlalchemy.exc import SQLAlchemyError

# Import your shared fetch/mapping helpers
import get_relevant_solicitations as gs

# ---------------- Config ----------------
PAGE_SIZE = int(os.getenv("PAGE_SIZE", "200"))
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "5000"))
from zoneinfo import ZoneInfo
from datetime import datetime


def get_days_back() -> int:
    mst = ZoneInfo("America/Denver")
    now_mst = datetime.now(mst)

    # Debug output
    print(f"DEBUG: now_mst = {now_mst}")
    print(f"DEBUG: weekday() = {now_mst.weekday()} (0=Mon, 1=Tue, ...)")
    print(f"DEBUG: date name = {now_mst.strftime('%A')}")

    if now_mst.weekday() == 0:  # Monday
        return 3
    return 1


# ADD THIS LINE - it's missing from your script
DAYS_BACK = int(os.getenv("DAYS_BACK")) if os.getenv("DAYS_BACK") else get_days_back()
DB_URL = os.getenv("SUPABASE_DB_URL") or os.getenv(
    "DB_URL") or "sqlite:///app.db"

# Read keys from env or secrets-like CSV
SAM_KEYS_RAW = os.getenv("SAM_KEYS", "")
SAM_KEYS = [k.strip() for k in SAM_KEYS_RAW.split(",") if k.strip()]


# ============================================================================
# COMPREHENSIVE EXCLUSION FILTERS - STRICT MODE (same as daily_digest.py)
# ============================================================================

# Agencies to exclude entirely (DOT and DHHS are consistently irrelevant)
EXCLUDED_AGENCY_KEYWORDS = [
    # Department of Transportation
    "department of transportation", "dept of transportation", "dot ", " dot",
    "federal aviation administration", "faa ", " faa",
    "federal highway administration", "fhwa",
    "federal transit administration", "fta ",
    "federal railroad administration", "fra ",
    "federal motor carrier", "fmcsa",
    "national highway traffic safety", "nhtsa",
    "pipeline and hazardous materials",

    # *** NEW: Department of the Interior ***
    "department of the interior", "dept of the interior",
    "department of interior", "dept of interior",
    "interior department",
    "bureau of land management", "blm ",
    "bureau of reclamation",
    "national park service", "nps ",
    "u.s. geological survey", "us geological survey", "usgs",
    "bureau of indian affairs", "bia ",
    "fish and wildlife service", "usfws", "fws ",
    "office of surface mining",
    "minerals management service",
    "bureau of ocean energy management", "boem",
    "bureau of safety and environmental enforcement", "bsee",

    # Department of Health and Human Services
    "department of health and human services", "dept of health and human services",
    "dhhs", "hhs ", " hhs",
    "centers for medicare", "cms ",
    "food and drug administration", "fda ",
    "centers for disease control", "cdc ",
    "national institutes of health", "nih ",
    "health resources and services", "hrsa",
    "substance abuse and mental health", "samhsa",
    "administration for children and families", "acf ",
    "indian health service", "ihs ",

    # Department of Agriculture (nutrition/food programs)
    "usda", "department of agriculture",
    "food and nutrition service", "fns ",
]

# Healthcare and biological/medical keywords (STRICT - expanded)
HEALTHCARE_KEYWORDS = [
    # General healthcare terms
    "healthcare", "health care", "medical", "hospital", "clinical",
    "pharmaceutical", "pharmacy", "nursing", "patient care", "medicare",
    "medicaid", "veterans health", "mental health", "dental",
    "laboratory services", "biomedical", "health services",
    "healthcare services", "medical supplies", "medical equipment",
    "ambulance", "emergency medical", "telemedicine", "telehealth",
    "electronic health record", "ehr ", " ehr", "emr ",
    "health information", "hipaa", "patient data", "clinical trial",
    "drug development", "therapeutic", "diagnostics", "pathology",
    "radiology", "oncology", "cardiology", "pediatric", "geriatric",
    "rehabilitation", "physical therapy", "occupational therapy",
    "behavioral health", "psychiatric", "epidemiology", "public health",

    # Biological data and life sciences (not defense R&D)
    "biological data", "biologic data", "biodata", "bio-data",
    "biological information", "bioinformatics", "genomic data",
    "genetic data", "dna sequencing", "rna sequencing",
    "biological sample", "specimen management", "biospecimen",
    "life sciences data", "biological research data",
    "biological database", "biology data management",

    # Nutrition and food services
    "nutrition", "nutritional", "dietetic", "dietary",
    "food service", "food preparation", "meal service", "catering",
    "cafeteria", "dining facility", "food program",
    "nutrition analysis", "dietary analysis", "meal planning",
    "nutrition support", "nutritional support",
    "food assistance", "feeding program",

    # AI/ML in healthcare context
    "healthcare ai", "medical ai", "clinical ai", "health ai",
    "healthcare machine learning", "medical machine learning",
    "clinical decision support", "medical imaging ai",
    "healthcare analytics", "clinical analytics",
    "patient outcome prediction", "disease prediction",
    "medical diagnosis", "diagnostic ai", "health prediction",
    "precision medicine", "personalized medicine",
    "drug discovery ai", "pharmaceutical ai",
    "healthcare nlp", "clinical nlp", "medical nlp",
    "biomedical informatics", "health informatics",
]

# Events, demonstrations, workshops, and trade shows (STRICT - expanded)
EVENTS_KEYWORDS = [
    # Trade shows and exhibitions
    "trade show", "tradeshow", "exhibition", "expo ",
    "exhibit space", "display setup", "booth ",

    # Conferences and meetings
    "conference support", "conference planning", "conference services",
    "symposium", "convention", "forum support", "seminar support",
    "meeting support", "meeting planning", "meeting services",

    # Workshops and training events
    "workshop", "workshops", "training event", "training session",
    "educational event", "learning event", "instruction session",

    # Event management and support
    "event support", "event management", "event planning",
    "event coordination", "event logistics", "event services",
    "event facilitation", "facilitating event", "facilitate event",
    "event production", "event execution",

    # Demonstrations (non-R&D)
    "demonstration event", "demo event", "public demonstration",
    "product demonstration", "technology demonstration",
    "demonstration project", "demonstration program",
    "pilot demonstration", "showcase event",

    # Outreach and public events
    "outreach event", "community event", "awareness event",
    "promotional event", "marketing event", "publicity event",
    "public affairs event", "media event", "press event",
    "ribbon cutting", "grand opening", "ceremony",
    "open house", "public meeting",
]

# Building installations and facilities (static systems, not R&D)
FACILITIES_KEYWORDS = [
    "building installation", "facility installation",
    "hvac installation", "hvac system", "hvac maintenance",
    "plumbing installation", "electrical installation",
    "fire suppression system", "fire alarm system", "sprinkler system",
    "security system installation", "access control installation",
    "building automation", "building management system",
    "elevator installation", "elevator maintenance",
    "roofing installation", "roofing repair", "roof replacement",
    "flooring installation", "carpet installation",
    "window installation", "door installation",
    "painting services", "interior painting", "exterior painting",
    "janitorial", "custodial", "cleaning services",
    "grounds maintenance", "landscaping", "lawn care",
    "pest control", "extermination",
    "parking lot", "paving", "asphalt",
    "fencing installation", "gate installation",
    "signage installation", "wayfinding",
    "furniture installation", "office furniture",
    "moving services", "relocation services",
    "warehouse space", "storage space",
]

# Paleontological and archaeological research (not relevant to tech R&D)
PALEO_ARCH_KEYWORDS = [
    "paleontological", "paleontology", "fossil",
    "archaeological", "archaeology", "archeological", "archeology",
    "excavation site", "dig site", "artifact",
    "prehistoric", "ancient remains", "cultural resources",
    "historic preservation", "historical preservation",
    "heritage site", "cultural heritage",
]

# Generic IT/data systems support (STRICT - greatly expanded)
GENERIC_IT_KEYWORDS = [
    # Help desk and desktop support
    "help desk", "helpdesk", "service desk",
    "desktop support", "end user support", "user support",
    "technical support", "tech support", "it support",
    "tier 1 support", "tier 2 support", "tier 3 support",
    "customer support", "support services",

    # Hardware and equipment support
    "computer refresh", "pc refresh", "hardware refresh",
    "laptop support", "workstation support",
    "printer support", "print services", "copier",
    "hardware maintenance", "equipment maintenance",

    # Business systems and enterprise IT
    "business system", "business systems", "enterprise system",
    "erp system", "erp support", "erp implementation",
    "sap support", "oracle support", "peoplesoft",
    "financial system", "accounting system", "hr system",
    "payroll system", "timekeeping system",
    "enterprise resource planning",

    # Generic IT services
    "it services", "information technology services",
    "managed services", "msp services", "it outsourcing",
    "it operations", "it maintenance", "it administration",
    "network support", "network maintenance", "network administration",
    "system administration", "systems administration",
    "email support", "email migration", "office 365",
    "microsoft 365", "email services",
    "password reset", "account management", "user provisioning",

    # Generic data management (not R&D data science)
    "data management", "data handling", "data entry",
    "data processing", "data services", "data support",
    "database administration", "database support", "dba services",
    "data migration", "data conversion", "data cleanup",
    "data storage", "data archiving", "records management",
    "document management", "content management",
    "information management", "data governance",
    "data center", "data centre", "hosting services",

    # Software support (not development)
    "software support", "application support", "app support",
    "software maintenance", "application maintenance",
    "software licensing", "license management",
    "software updates", "patch management",
]

# Shuttle and transit demonstrations (DOT-related) - expanded
TRANSIT_DEMO_KEYWORDS = [
    "shuttle demonstration", "shuttle demo", "automated shuttle",
    "autonomous shuttle", "self-driving shuttle", "driverless shuttle",
    "transit demonstration", "bus demonstration",
    "vehicle demonstration", "mobility demonstration",
    "transportation demonstration", "connected vehicle demo",
    "smart city demonstration", "smart transportation",
    "av demonstration", "autonomous vehicle demonstration",
    "transit pilot", "mobility pilot", "shuttle pilot",
    "transit project", "public transit",
]

# Packaging and logistics support (not R&D)
PACKAGING_KEYWORDS = [
    "packaging support", "packaging services", "packaging material",
    "packing services", "packing support", "crating",
    "shipping support", "shipping services", "freight",
    "logistics support", "logistics services",
    "warehousing", "distribution services", "fulfillment",
    "mail services", "mailing services", "postage",
    "courier services", "delivery services",
]

# Sources sought that are typically not actionable
SOURCES_SOUGHT_EXCLUSIONS = [
    "sources sought for event",
    "sources sought for demonstration",
    "market research for event",
    "rfi for event", "rfi for demonstration",
    "sources sought for workshop",
    "sources sought for conference",
]

# Administrative and clerical support (not R&D)
ADMIN_SUPPORT_KEYWORDS = [
    "administrative support", "admin support", "clerical",
    "secretarial", "receptionist", "front desk",
    "office support", "office services", "office management",
    "mailroom", "mail handling", "correspondence",
    "scheduling", "calendar management", "travel arrangement",
    "document preparation", "word processing", "typing",
    "data entry", "filing", "records clerk",
]

# NAICS codes to exclude (in addition to only allowing R&D codes)
EXCLUDED_NAICS_PREFIXES = [
    "621", "622", "623", "624",  # Healthcare
    "485",  # Transit
    "237",  # Heavy construction
    "561720",  # Janitorial services
    "561730",  # Landscaping services
    "561210",  # Facilities support services
    "561110",  # Office administrative services
    "561320",  # Temporary help services
    "493",  # Warehousing and storage
    "488",  # Support activities for transportation
    "722",  # Food services and drinking places
]

# *** NEW: 8(a) set-aside codes to exclude ***
EXCLUDED_SET_ASIDE_CODES = {
    "8A",    # 8(a) Set-Aside
    "8AN",   # 8(a) Sole Source
}

EXCLUDED_SET_ASIDE_KEYWORDS = [
    "8(a) set-aside",
    "8(a) sole source",
    "8a set-aside",
    "8a sole source",
    "set aside 8(a)",
    "set-aside: 8(a)",
]

def _should_exclude_solicitation(
    title: str, description: str, naics_code: str, set_aside_code: str = ""
) -> tuple[bool, str]:
    title_lower = (title or "").lower()
    desc_lower = (description or "").lower()
    combined = f"{title_lower} {desc_lower}"
    naics_str = str(naics_code or "").strip()

    # *** NEW: 8(a) set-aside code check ***
    set_aside_upper = (set_aside_code or "").strip().upper()
    if set_aside_upper in EXCLUDED_SET_ASIDE_CODES:
        return True, f"set_aside_code:{set_aside_upper}"

    for keyword in EXCLUDED_SET_ASIDE_KEYWORDS:
        if keyword in combined:
            return True, f"set_aside_text:{keyword}"

    # Check healthcare keywords (includes biological data, nutrition)
    for keyword in HEALTHCARE_KEYWORDS:
        if keyword in combined:
            return True, f"healthcare:{keyword}"

    # Check events/demonstrations/workshops/trade shows
    for keyword in EVENTS_KEYWORDS:
        if keyword in combined:
            return True, f"events:{keyword}"

    # Check facilities/building installations
    for keyword in FACILITIES_KEYWORDS:
        if keyword in combined:
            return True, f"facilities:{keyword}"

    # Check paleontological/archaeological
    for keyword in PALEO_ARCH_KEYWORDS:
        if keyword in combined:
            return True, f"paleo_arch:{keyword}"

    # Check generic IT/business systems/data management support
    for keyword in GENERIC_IT_KEYWORDS:
        if keyword in combined:
            return True, f"generic_it:{keyword}"

    # Check transit/shuttle demonstrations
    for keyword in TRANSIT_DEMO_KEYWORDS:
        if keyword in combined:
            return True, f"transit_demo:{keyword}"

    # Check packaging and logistics
    for keyword in PACKAGING_KEYWORDS:
        if keyword in combined:
            return True, f"packaging:{keyword}"

    # Check administrative/clerical support
    for keyword in ADMIN_SUPPORT_KEYWORDS:
        if keyword in combined:
            return True, f"admin_support:{keyword}"

    # Check sources sought exclusions
    for keyword in SOURCES_SOUGHT_EXCLUSIONS:
        if keyword in combined:
            return True, f"sources_sought:{keyword}"

    # Check excluded NAICS code prefixes
    for prefix in EXCLUDED_NAICS_PREFIXES:
        if naics_str.startswith(prefix):
            return True, f"naics:{prefix}"

    return False, ""


# ---------------- Date Parsing Helpers ----------------
DATE_ISO_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
DATE_US_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})")


def _parse_response_date(date_str: str) -> date | None:
    """
    Parse various date formats and return a date object.
    Returns None if the date cannot be parsed or is invalid.
    """
    if not date_str or str(date_str).lower() in ("none", "n/a", "na", "null", ""):
        return None

    # Try ISO format first (YYYY-MM-DD)
    match = DATE_ISO_RE.search(str(date_str))
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass

    # Try US format (M/D/YYYY or MM/DD/YYYY)
    match = DATE_US_RE.search(str(date_str))
    if match:
        try:
            return datetime.strptime(match.group(1), "%m/%d/%Y").date()
        except ValueError:
            pass

    # Try parsing as ISO datetime (with time component)
    date_str = str(date_str)
    if "T" in date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.date()
        except ValueError:
            pass

    return None


def _is_response_date_past(response_date_str: str) -> bool:
    """
    Check if the response date has already passed.
    Returns True if the date has passed, False if it's still valid or can't be parsed.
    """
    parsed_date = _parse_response_date(response_date_str)
    if parsed_date is None:
        # If we can't parse the date, don't exclude it (could be valid)
        return False

    today = date.today()
    return parsed_date < today


# ---------------- Existing Helper Functions ----------------
REQUIRED_COLS = [
    "notice_id", "solicitation_number", "title", "notice_type",
    "posted_date", "response_date", "archive_date",
    "naics_code", "set_aside_code", "description", "link",
    "pop_city", "pop_state", "pop_zip", "pop_country", "pop_raw",
]


def _stringify(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v).strip()
    try:
        return json.dumps(v, ensure_ascii=False).strip()
    except Exception:
        return str(v).strip()


def make_sam_public_url(notice_id: str, link: str | None = None) -> str:
    nid = (notice_id or "").strip()
    if not nid:
        return "https://sam.gov/"
    # If caller already handed us a public web URL, keep it; otherwise force public opp URL
    if link and isinstance(link, str) and ("api.sam.gov" not in link):
        return link
    return f"https://sam.gov/opp/{nid}/view"


def _engine():
    # For Postgres (Supabase) we usually want pool_pre_ping; SQLite is fine default.
    kw = {}
    if DB_URL.startswith("postgresql"):
        kw.update(dict(pool_pre_ping=True, pool_size=5, max_overflow=2))
    return create_engine(DB_URL, **kw)


def _ensure_table(conn):
    # Fail fast if table missing; we don't try to create here—use your app migration
    try:
        conn.execute(text("SELECT 1 FROM solicitationraw LIMIT 1"))
    except Exception as e:
        raise RuntimeError(
            "Table 'solicitationraw' not found. Run your app once to migrate/create schema.") from e


def _already_have_ids(conn) -> set[str]:
    # Pull the set of existing notice_ids to skip duplicates quickly
    rows = conn.execute(
        text("SELECT notice_id FROM solicitationraw")).fetchall()
    return {str(r[0]) for r in rows if r and r[0]}


def _insert_rows(conn, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    # Only include columns that exist in your schema
    cols = ["pulled_at"] + REQUIRED_COLS
    sql = text(f"""
        INSERT INTO solicitationraw (
            {", ".join(cols)}
        ) VALUES (
            {", ".join(":" + c for c in cols)}
        )
        ON CONFLICT (notice_id) DO NOTHING
    """)
    conn.execute(sql, rows)
    return len(rows)


# FIXED: Simplified mapping that doesn't call any additional APIs


def map_record_basic_fields_only(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a raw SAM record to our schema fields WITHOUT making any additional API calls.
    This keeps the refresh fast and avoids quota issues.
    """

    def _first_nonempty(obj: Dict[str, Any], *keys: str, default: str = "None") -> str:
        for k in keys:
            if k in obj and obj[k] not in (None, "", []):
                val = obj[k]
                if isinstance(val, (str, int, float, bool)):
                    return str(val).strip()
                elif isinstance(val, dict):
                    # Try common sub-keys
                    for subkey in ("name", "text", "value", "code"):
                        if subkey in val and val[subkey] not in (None, "", []):
                            return str(val[subkey]).strip()
                elif isinstance(val, list) and val:
                    # Take first non-empty item
                    for item in val:
                        if item not in (None, "", []):
                            if isinstance(item, (str, int, float, bool)):
                                return str(item).strip()
                            elif isinstance(item, dict):
                                for subkey in ("name", "text", "value", "code"):
                                    if subkey in item and item[subkey] not in (None, "", []):
                                        return str(item[subkey]).strip()
        return default

    notice_id = _first_nonempty(rec, "noticeId", "id")
    solicitation_number = _first_nonempty(
        rec, "solicitationNumber", "solicitationNo")
    title = _first_nonempty(rec, "title")
    notice_type = _first_nonempty(rec, "noticeType", "type")
    posted_date = _first_nonempty(rec, "postedDate", "publicationDate")
    archive_date = _first_nonempty(rec, "archiveDate")
    naics_code = _first_nonempty(rec, "naicsCode", "naics")
    set_aside_code = _first_nonempty(
        rec, "setAsideCode", "typeOfSetAside", "setAside")

    # Extract response date from various possible fields
    response_date = _first_nonempty(rec, "responseDeadLine", "responseDateTime",
                                    "responseDate", "dueDate", "closeDate")

    # Extract link
    link = "None"
    links = rec.get("links")
    if isinstance(links, list) and links:
        maybe = links[0]
        if isinstance(maybe, dict) and maybe.get("href"):
            link = str(maybe["href"])
    if link == "None":
        link = _first_nonempty(rec, "url", "samLink")

    # Don't extract description from search results - they're usually just API URLs
    # Leave it blank so backfill_descriptions.py can fetch the real description later
    description = ""

    # Basic place of performance extraction (no additional API calls)
    pop_city = ""
    pop_state = ""
    pop_zip = ""
    pop_country = ""
    pop_raw = ""

    # Try to extract from embedded place of performance data
    pop_data = rec.get("placeOfPerformance") or rec.get(
        "place_of_performance") or {}
    if isinstance(pop_data, dict):
        pop_city = _first_nonempty(pop_data, "city", "cityName")
        pop_state = _first_nonempty(
            pop_data, "state", "stateCode", "stateProvince")
        pop_zip = _first_nonempty(pop_data, "zip", "zipCode", "postalCode")
        pop_country = _first_nonempty(
            pop_data, "country", "countryCode", "countryName")

        # Build pop_raw
        parts = []
        if pop_city and pop_city != "None":
            parts.append(pop_city)
        if pop_state and pop_state != "None":
            parts.append(pop_state)
        pop_raw = ", ".join(parts)
        if pop_zip and pop_zip != "None":
            pop_raw = (pop_raw + f" {pop_zip}").strip()
        if pop_country and pop_country != "None" and pop_country.upper() not in ("US", "USA", "UNITED STATES"):
            pop_raw = (pop_raw + f" ({pop_country})").strip()

    return {
        "notice_id": _stringify(notice_id),
        "solicitation_number": _stringify(solicitation_number),
        "title": _stringify(title),
        "notice_type": _stringify(notice_type),
        "posted_date": _stringify(posted_date),
        "response_date": _stringify(response_date),
        "archive_date": _stringify(archive_date),
        "naics_code": _stringify(naics_code),
        "set_aside_code": _stringify(set_aside_code),
        "description": _stringify(description),
        "link": _stringify(link),
        "pop_city": _stringify(pop_city) if pop_city != "None" else "",
        "pop_state": _stringify(pop_state) if pop_state != "None" else "",
        "pop_zip": _stringify(pop_zip) if pop_zip != "None" else "",
        "pop_country": _stringify(pop_country) if pop_country != "None" else "",
        "pop_raw": _stringify(pop_raw),
    }


def _normalize_title(title: str) -> str:
    """Normalize a title for deduplication — strips punctuation/case/whitespace."""
    if not title:
        return ""
    t = re.sub(r"\s+", " ", title.lower().strip())
    t = re.sub(r"[^\w\s]", "", t)
    return t


def _already_have_titles_today(conn) -> set[str]:
    """
    Returns normalized titles already inserted TODAY.
    Catches the same solicitation appearing in multiple feeds with different notice IDs.
    """
    rows = conn.execute(text("""
        SELECT title FROM solicitationraw
        WHERE DATE(pulled_at) = CURRENT_DATE
          AND title IS NOT NULL AND title != ''
    """)).fetchall()
    return {_normalize_title(r[0]) for r in rows if r and r[0]}

# ---------------- Main ----------------


def main():
    print("=== Auto refresh start ===")
    print(datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y"))
    print(f"SAM_KEYS configured: {len(SAM_KEYS)} key(s)")
    print(f"DAYS_BACK = {DAYS_BACK}")
    print(f"Paging: PAGE_SIZE={PAGE_SIZE}, MAX_RECORDS={MAX_RECORDS}")
    print(f"Filtering: Will skip solicitations with past response dates")
    print(f"Filtering: Comprehensive exclusions (DOT, DHHS, healthcare, events, facilities, etc.)")

    if not SAM_KEYS:
        print("ERROR: No SAM_KEYS provided (env SAM_KEYS). Exiting.")
        sys.exit(1)

    engine = _engine()
    print("auto_refresh.py: engine created")

    try:
        with engine.connect() as conn:
            _ensure_table(conn)
            print("auto_refresh.py: DB ping OK")

            # For logging DB size (optional)
            try:
                before = conn.execute(
                    text("SELECT COUNT(*) FROM solicitationraw")).scalar() or 0
                last_pulled = conn.execute(
                    text("SELECT MAX(pulled_at) FROM solicitationraw")).scalar()
                print(
                    f"DB before: {before} rows; last pulled_at: {last_pulled}")
            except Exception:
                pass

    except Exception as e:
        print("auto_refresh.py: DB check failed")
        print(e)
        sys.exit(1)

    print("auto_refresh.py: entered main()")
    print("Starting auto-refresh job...")

    total_inserted = 0
    total_seen = 0
    total_skipped_expired = 0
    total_skipped_excluded = 0
    exclusion_stats = {}  # Track reasons for exclusions
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        with engine.begin() as conn:
            existing = _already_have_ids(conn)
            seen_titles_today = _already_have_titles_today(conn)
            print(f"  Loaded {len(seen_titles_today)} distinct titles already ingested today for dedup")
            rows_to_insert: List[Dict[str, Any]] = []

            print("Fetching solicitations from SAM.gov (will filter for R&D NAICS client-side)…")
            offset = 0

            while True:
                limit = min(PAGE_SIZE, MAX_RECORDS - total_seen)
                if limit <= 0:
                    break

                print(
                    f"  → Page {offset // PAGE_SIZE + 1}: offset={offset}, limit={limit}")

                try:
                    # Fetch ALL solicitations - filter client-side since SAM API filtering isn't working
                    raw = gs.get_sam_raw_v3(
                        days_back=DAYS_BACK,
                        limit=limit,
                        api_keys=SAM_KEYS,
                        filters={},  # No server-side filtering
                        offset=offset,
                    )
                except gs.SamQuotaError:
                    print("All SAM.gov keys exhausted (quota). Stopping refresh.")
                    print(
                        f"Partial success: {total_inserted} new records inserted before quota limit.")
                    break
                except gs.SamAuthError:
                    print("All SAM.gov keys failed authentication. Check your keys.")
                    sys.exit(2)
                except Exception as e:
                    print(f"Unexpected error fetching data: {e}")
                    sys.exit(2)

                if not raw:
                    break

                total_seen += len(raw)
                print(
                    f"    fetched {len(raw)} records (cumulative fetched: {total_seen})")

                for r in raw:
                    # FIXED: Use basic mapping that doesn't make additional API calls
                    m = map_record_basic_fields_only(r)
                    nid = (m.get("notice_id") or "").strip()
                    if not nid or nid in existing:
                        continue

                    naics_code = m.get("naics_code", "")

                    # NEW: Check if response date has passed
                    response_date_str = m.get("response_date", "")
                    if _is_response_date_past(response_date_str):
                        total_skipped_expired += 1
                        print(
                            f"    skipping expired solicitation {nid} (response date: {response_date_str})")
                        continue

                    # NEW: Comprehensive exclusion filtering
                    title = m.get("title", "")
                    description = m.get("description", "")
                    set_aside_code = m.get("set_aside_code", "")
                    should_exclude, reason = _should_exclude_solicitation(
                        title, description, naics_code, set_aside_code
                    )
                    if should_exclude:
                        total_skipped_excluded += 1
                        # Track exclusion reasons
                        reason_category = reason.split(":")[0] if ":" in reason else reason
                        exclusion_stats[reason_category] = exclusion_stats.get(reason_category, 0) + 1
                        continue

                    # Force link to public URL so clicks never need API key
                    public_link = make_sam_public_url(nid, m.get("link"))
                    row = {
                        "pulled_at": now_iso,
                        "notice_id": _stringify(m.get("notice_id")),
                        "solicitation_number": _stringify(m.get("solicitation_number")),
                        "title": _stringify(m.get("title")),
                        "notice_type": _stringify(m.get("notice_type")),
                        "posted_date": _stringify(m.get("posted_date")),
                        "response_date": _stringify(m.get("response_date")),
                        "archive_date": _stringify(m.get("archive_date")),
                        "naics_code": _stringify(m.get("naics_code")),
                        "set_aside_code": _stringify(m.get("set_aside_code")),
                        # Basic description only
                        "description": _stringify(m.get("description")),
                        "link": public_link,
                        "pop_city": _stringify(m.get("pop_city")),
                        "pop_state": _stringify(m.get("pop_state")),
                        "pop_zip": _stringify(m.get("pop_zip")),
                        "pop_country": _stringify(m.get("pop_country")),
                        "pop_raw": _stringify(m.get("pop_raw")),
                    }
                    rows_to_insert.append(row)
                    existing.add(nid)  # avoid repeats in same run

                # Insert batch
                if rows_to_insert:
                    inserted = _insert_rows(conn, rows_to_insert)
                    total_inserted += inserted
                    print(
                        f"    inserted {inserted} new R&D records (total new: {total_inserted})")
                    rows_to_insert.clear()

                # Next page
                offset += PAGE_SIZE
                if total_seen >= MAX_RECORDS:
                    break

        print(f"\n=== Auto refresh complete ===")
        print(f"Inserted {total_inserted} new notices.")
        print(f"Skipped {total_skipped_expired} notices with expired response dates.")
        print(f"Skipped {total_skipped_excluded} notices due to comprehensive exclusion filters:")
        for reason, count in sorted(exclusion_stats.items(), key=lambda x: -x[1]):
            print(f"  - {reason}: {count}")
        if total_seen == 0:
            print("Note: SAM.gov returned no records for the selected window.")

    except Exception as e:
        print(f"Auto refresh failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()