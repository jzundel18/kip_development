#!/usr/bin/env python3
"""Simple wrapper to run daily_digest.py with .env file support"""

from scripts.daily_digest import main
import sys
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✓ Loaded environment from {env_file}")
except ImportError:
    print("⚠ python-dotenv not installed. Using system environment variables.")
    print("  Install with: pip install python-dotenv")

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

# Run the script
main()
