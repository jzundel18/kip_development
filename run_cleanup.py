#!/usr/bin/env python3
"""Simple wrapper to run cleanup_expired_solicitations.py with .env file support"""

import os
import sys
from pathlib import Path

# Load .env file FIRST, before any other imports
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
    else:
        print(f"⚠ .env file not found at {env_file}")
except ImportError:
    print("⚠ python-dotenv not installed. Using system environment variables.")
    print("  Install with: pip install python-dotenv")

# Verify the environment variable was loaded
if os.getenv("SUPABASE_DB_URL"):
    print(f"✓ SUPABASE_DB_URL loaded successfully")
else:
    print("⚠ WARNING: SUPABASE_DB_URL not found in environment")

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

# NOW import the cleanup script (after env is loaded)
from scripts.cleanup_expired_solicitations import main

# Check for dry-run flag
if '--dry-run' in sys.argv:
    os.environ['CLEANUP_DRY_RUN'] = 'true'
    print("🔍 Running in DRY RUN mode")

# Run the script
main()