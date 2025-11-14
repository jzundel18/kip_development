#!/usr/bin/env python3
"""
run_backfill_descriptions.py
Python script to run the backfill_descriptions.py script with proper environment setup
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


def main():
    print("=== Starting Description Backfill ===")
    print(datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y"))
    print()

    # Load environment variables from .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"Loading environment variables from {env_file.absolute()}")
        load_dotenv(env_file)
    else:
        print("Warning: .env file not found. Using existing environment variables.")

    # Verify required environment variables
    db_url = os.environ.get("SUPABASE_DB_URL")
    sam_keys = os.environ.get("SAM_KEYS")

    if not db_url:
        print("ERROR: SUPABASE_DB_URL not set")
        sys.exit(1)

    if not sam_keys:
        print("ERROR: SAM_KEYS not set")
        sys.exit(1)

    # Set backfill configuration (can be overridden by environment)
    backfill_batch = os.environ.get("BACKFILL_BATCH", "50")
    backfill_max = os.environ.get("BACKFILL_MAX", "500")
    backfill_delay = os.environ.get("BACKFILL_DELAY_SEC", "0.5")

    # Count number of keys
    sam_keys_list = [k.strip() for k in sam_keys.replace("\n", ",").split(",") if k.strip()]

    print("Configuration:")
    print(f"  BACKFILL_BATCH: {backfill_batch}")
    print(f"  BACKFILL_MAX: {backfill_max}")
    print(f"  BACKFILL_DELAY_SEC: {backfill_delay}")
    print(f"  SAM_KEYS: {len(sam_keys_list)} key(s) configured")
    print()

    # Find backfill_descriptions.py in common locations
    script_dir = Path(__file__).resolve().parent
    search_paths = [
        script_dir / "scripts" / "backfill_descriptions.py",
        script_dir / "backfill_descriptions.py",
        Path("scripts/backfill_descriptions.py"),
        Path("backfill_descriptions.py"),
    ]

    backfill_script = None
    for path in search_paths:
        if path.exists():
            backfill_script = path
            break

    if not backfill_script:
        print("ERROR: Could not find backfill_descriptions.py")
        print("Searched in:")
        for path in search_paths:
            print(f"  {path.absolute()}")
        sys.exit(1)

    print(f"Using script: {backfill_script.absolute()}")
    print()
    print("Starting backfill process...")
    print("=" * 60)
    print()

    # Run the backfill script
    try:
        result = subprocess.run(
            [sys.executable, str(backfill_script)],
            env=os.environ.copy(),
            check=False
        )
        exit_code = result.returncode
    except Exception as e:
        print(f"ERROR: Failed to run backfill script: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print(f"Backfill process completed with exit code: {exit_code}")
    print(datetime.now().strftime("%a %b %d %H:%M:%S %Z %Y"))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()