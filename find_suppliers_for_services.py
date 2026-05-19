#!/usr/bin/env python3
"""
find_suppliers_for_services.py

Loops over every solicitation in the database categorized as 'services'
and uses find_relevant_suppliers.find_vendors_for_notice to discover local
suppliers in that solicitation's place of performance.

Usage:
    python find_suppliers_for_services.py [options]

Options:
    --limit N           Max solicitations to process (default: all)
    --state XX          Filter by place-of-performance state (e.g. CA)
    --top-n N           Vendors to return per solicitation (default: 3)
    --max-google N      Google results per query (default: 10)
    --output FILE       Write results JSON here (default: services_suppliers.json)
    --csv FILE          Also write a flat CSV (one row per vendor)
    --dry-run           List matching solicitations but skip supplier search

Environment variables (required):
    SUPABASE_DB_URL     PostgreSQL connection string
    GOOGLE_API_KEY      Google Custom Search API key
    GOOGLE_CX           Google Custom Search Engine ID
    OPENAI_API_KEY      OpenAI API key
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

from find_relevant_suppliers import find_vendors_for_notice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_services_solicitations(db_url: str, state: str | None, limit: int | None) -> list[dict[str, Any]]:
    engine = create_engine(db_url, pool_pre_ping=True)
    sql = """
        SELECT notice_id, title, description, naics_code,
               pop_city, pop_state, response_date, link
        FROM solicitationraw
        WHERE category = 'services'
    """
    params: dict[str, Any] = {}
    if state:
        sql += " AND UPPER(pop_state) = :state"
        params["state"] = state.upper()
    sql += " ORDER BY response_date NULLS LAST"
    if limit:
        sql += " LIMIT :limit"
        params["limit"] = limit

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-google", type=int, default=10)
    parser.add_argument("--output", type=str, default="services_suppliers.json")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_url = os.getenv("SUPABASE_DB_URL")
    google_key = os.getenv("GOOGLE_API_KEY")
    google_cx = os.getenv("GOOGLE_CX")
    openai_key = os.getenv("OPENAI_API_KEY")

    missing = [n for n, v in [
        ("SUPABASE_DB_URL", db_url),
        ("GOOGLE_API_KEY", google_key),
        ("GOOGLE_CX", google_cx),
        ("OPENAI_API_KEY", openai_key),
    ] if not v]
    if missing and not args.dry_run:
        log.error(f"Missing required env vars: {', '.join(missing)}")
        return 1
    if not db_url:
        log.error("SUPABASE_DB_URL required")
        return 1

    sols = load_services_solicitations(db_url, args.state, args.limit)
    log.info(f"Found {len(sols)} services solicitations")

    if args.dry_run:
        for s in sols:
            log.info(f"  {s['notice_id']} | {s.get('pop_state') or '--'} | {(s.get('title') or '')[:80]}")
        return 0

    results: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for i, sol in enumerate(sols, 1):
        title = (sol.get("title") or "")[:80]
        log.info(f"[{i}/{len(sols)}] {sol['notice_id']} — {title}")
        try:
            df, _ = find_vendors_for_notice(
                sol,
                google_api_key=google_key,
                google_cx=google_cx,
                openai_api_key=openai_key,
                max_google=args.max_google,
                top_n=args.top_n,
            )
            vendors = df.to_dict(orient="records") if not df.empty else []
        except Exception as exc:
            log.warning(f"  vendor lookup failed: {exc}")
            vendors = []

        results.append({
            "notice_id": sol["notice_id"],
            "title": sol.get("title"),
            "naics_code": sol.get("naics_code"),
            "pop_city": sol.get("pop_city"),
            "pop_state": sol.get("pop_state"),
            "response_date": sol.get("response_date"),
            "link": sol.get("link"),
            "vendors": vendors,
        })

        for v in vendors:
            csv_rows.append({
                "notice_id": sol["notice_id"],
                "solicitation_title": sol.get("title"),
                "pop_city": sol.get("pop_city"),
                "pop_state": sol.get("pop_state"),
                "vendor_name": v.get("name"),
                "vendor_website": v.get("website"),
                "vendor_location": v.get("location"),
                "reason": v.get("reason"),
            })

        log.info(f"  → {len(vendors)} vendor(s)")
        time.sleep(0.5)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    log.info(f"Wrote {out_path}")

    if args.csv:
        csv_path = Path(args.csv)
        if csv_rows:
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            log.info(f"Wrote {csv_path} ({len(csv_rows)} rows)")
        else:
            log.info("No vendors found — skipping CSV")

    return 0


if __name__ == "__main__":
    sys.exit(main())
