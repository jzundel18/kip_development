#!/usr/bin/env python3
"""
run_services_pipeline.py

End-to-end pipeline:
  1. Pull fresh solicitations from SAM.gov   (run_auto_refresh.py)
  2. Categorize them by NAICS                (categorize_solicitations.py)
  3. Find local suppliers for services-
     category contracts and write a Word doc
     of up to --max-results matches that
     have at least one supplier              (find_suppliers_for_services.py)

Usage:
    python run_services_pipeline.py [options]

Options:
    --skip-refresh        Skip step 1 (use existing DB rows)
    --skip-categorize     Skip step 2
    --state XX            Limit step 3 to a single POP state
    --max-results N       Final cap on matches in the doc (default: 10)
    --scan-cap N          Max solicitations to scan in step 3 (default: 200)
    --top-n N             Suppliers per solicitation (default: 3)
    --posted-on DATE      Pin step 3 to rows posted on YYYY-MM-DD (default: today)
    --min-days-out N      Require N days of runway before due (default: 3)
    --include-closed      Disable the response_date filter entirely
    --output-docx FILE    Word doc output (default: services_suppliers.docx)
    --output-json FILE    Optional JSON sidecar
    --output-csv FILE     Optional CSV sidecar

Environment variables (required):
    SUPABASE_DB_URL, SAM_KEYS, GOOGLE_API_KEY, GOOGLE_CX, OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()


class C:
    HEADER = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def header(msg: str) -> None:
    print(f"\n{C.HEADER}{C.BOLD}{'=' * 70}\n  {msg}\n{'=' * 70}{C.END}\n")


def ok(msg: str) -> None:
    print(f"{C.GREEN}✓ {msg}{C.END}")


def warn(msg: str) -> None:
    print(f"{C.YELLOW}⚠ {msg}{C.END}")


def err(msg: str) -> None:
    print(f"{C.RED}✗ {msg}{C.END}")


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        env_file = SCRIPT_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            ok(f"Loaded environment from {env_file}")
    except ImportError:
        warn("python-dotenv not installed; relying on shell env")


def check_env(required: list[str]) -> bool:
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        err(f"Missing required env vars: {', '.join(missing)}")
        return False
    return True


def run_step(label: str, cmd: list[str]) -> bool:
    header(label)
    print(f"{C.CYAN}$ {' '.join(cmd)}{C.END}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=os.environ.copy())
    elapsed = time.time() - start
    if result.returncode == 0:
        ok(f"{label} done in {elapsed:.1f}s")
        return True
    err(f"{label} failed (exit {result.returncode}) after {elapsed:.1f}s")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--skip-categorize", action="store_true")
    parser.add_argument("--state", type=str, default=None)
    parser.add_argument("--max-results", type=int, default=10,
                        help="Stop after collecting N solicitations with vendor matches (default: 10)")
    parser.add_argument("--scan-cap", type=int, default=200,
                        help="Max solicitations to scan in step 3 (default: 200)")
    parser.add_argument("--posted-on", type=str, default=None,
                        help="Pin step 3 to rows posted on YYYY-MM-DD (default: today)")
    parser.add_argument("--include-closed", action="store_true",
                        help="Include solicitations whose response_date has already passed")
    parser.add_argument("--min-days-out", type=int, default=3,
                        help="Require N days of runway before due (default: 3)")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--output-docx", type=str, default="services_suppliers.docx")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional JSON sidecar")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional CSV sidecar")
    args = parser.parse_args()

    print(f"\n{C.BOLD}{C.HEADER}")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           SERVICES PIPELINE — Solicitations → Local Suppliers        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"{C.END}")

    load_env()

    required = ["SUPABASE_DB_URL", "GOOGLE_API_KEY", "GOOGLE_CX", "OPENAI_API_KEY"]
    if not args.skip_refresh:
        required.append("SAM_KEYS")
    if not check_env(required):
        return 1

    py = sys.executable

    # Step 1 — fetch from SAM.gov
    if args.skip_refresh:
        warn("Skipping SAM.gov refresh")
    else:
        if not run_step("Step 1 — Fetch from SAM.gov", [py, "run_auto_refresh.py"]):
            return 1

    # Step 2 — categorize
    if args.skip_categorize:
        warn("Skipping categorization")
    else:
        if not run_step("Step 2 — Categorize by NAICS", [py, "categorize_solicitations.py"]):
            return 1

    # Step 3 — find suppliers for services
    cmd = [
        py, "find_suppliers_for_services.py",
        "--max-results", str(args.max_results),
        "--scan-cap", str(args.scan_cap),
        "--top-n", str(args.top_n),
        "--min-days-out", str(args.min_days_out),
        "--output", args.output_docx,
    ]
    if args.state:
        cmd += ["--state", args.state]
    if args.posted_on:
        cmd += ["--posted-on", args.posted_on]
    if args.include_closed:
        cmd += ["--include-closed"]
    if args.output_json:
        cmd += ["--json", args.output_json]
    if args.output_csv:
        cmd += ["--csv", args.output_csv]
    if not run_step("Step 3 — Find Local Suppliers (services only)", cmd):
        return 1

    header("Pipeline complete")
    ok(f"Word doc: {args.output_docx}")
    if args.output_json:
        ok(f"JSON:     {args.output_json}")
    if args.output_csv:
        ok(f"CSV:      {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
