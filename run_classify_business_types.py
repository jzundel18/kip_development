#!/usr/bin/env python3
"""
Wrapper to run classify_business_types.py with proper environment loading
"""

import sys
from pathlib import Path

# Load .env file FIRST before any other imports
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
    else:
        print(f"⚠ .env file not found at {env_file}")
        print("Trying to use system environment variables...")
except ImportError:
    print("⚠ python-dotenv not installed. Using system environment variables.")
    print("  Install with: pip install python-dotenv")

# Verify critical variables loaded
import os
if not os.getenv("SUPABASE_DB_URL"):
    print("\n❌ ERROR: SUPABASE_DB_URL not found in environment")
    print("\nPlease either:")
    print("  1. Create a .env file with your credentials:")
    print("     SUPABASE_DB_URL=postgresql://...")
    print("     OPENAI_API_KEY=sk-...")
    print()
    print("  2. Or set environment variables:")
    print("     export SUPABASE_DB_URL='postgresql://...'")
    print("     export OPENAI_API_KEY='sk-...'")
    sys.exit(1)

if not os.getenv("OPENAI_API_KEY"):
    print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
    sys.exit(1)

print(f"✓ SUPABASE_DB_URL loaded")
print(f"✓ OPENAI_API_KEY loaded")
print()

# Now import and run the main script
import classify_business_types

if __name__ == "__main__":
    classify_business_types.main()