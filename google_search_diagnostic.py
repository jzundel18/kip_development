#!/usr/bin/env python3
"""
Diagnostic script to test Google Custom Search API configuration.
Reads credentials from .streamlit/secrets.toml (same as main app)

Usage:
    python google_search_diagnostic.py
"""

import os
import sys
import requests
import json
from pathlib import Path


def load_secrets():
    """Load secrets from .streamlit/secrets.toml file"""
    secrets_path = Path(".streamlit/secrets.toml")

    if not secrets_path.exists():
        print(f"❌ ERROR: secrets.toml not found at {secrets_path.absolute()}")
        print("\nSearched in:", secrets_path.absolute())
        print("\nMake sure you're running this script from your project root directory")
        return None, None

    print(f"✅ Found secrets file: {secrets_path.absolute()}")

    # Parse the TOML file manually (simple parsing for key = "value" format)
    google_api_key = None
    google_cx = None

    try:
        with open(secrets_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('GOOGLE_API_KEY'):
                    # Extract value between quotes
                    google_api_key = line.split(
                        '=')[1].strip().strip('"').strip("'")
                elif line.startswith('GOOGLE_CX'):
                    google_cx = line.split(
                        '=')[1].strip().strip('"').strip("'")
    except Exception as e:
        print(f"❌ ERROR reading secrets file: {e}")
        return None, None

    return google_api_key, google_cx


def test_google_search():
    """Test Google Custom Search API"""

    print("=" * 60)
    print("Google Custom Search API Diagnostic Tool")
    print("=" * 60)
    print()

    # Load credentials from secrets.toml
    GOOGLE_API_KEY, GOOGLE_CX = load_secrets()

    if not GOOGLE_API_KEY:
        print("❌ ERROR: GOOGLE_API_KEY not found in .streamlit/secrets.toml")
        print("\nYour .streamlit/secrets.toml should contain:")
        print('GOOGLE_API_KEY = "your_api_key_here"')
        return False

    if not GOOGLE_CX:
        print("❌ ERROR: GOOGLE_CX not found in .streamlit/secrets.toml")
        print("\nYour .streamlit/secrets.toml should contain:")
        print('GOOGLE_CX = "your_search_engine_id_here"')
        return False

    print(f"✅ Google API Key found: {GOOGLE_API_KEY[:20]}...")
    print(f"✅ Google CX found: {GOOGLE_CX}")
    print()

    # Test simple search
    test_query = "machine shop contractors"
    print(f"Testing search: '{test_query}'")
    print("-" * 60)

    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": test_query,
        "num": 10
    }

    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=30
        )

        print(f"Response Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ ERROR: API returned status code {response.status_code}")
            print(f"Response: {response.text[:500]}")

            # Check for common errors
            if response.status_code == 403:
                print("\n💡 This usually means:")
                print("   - API key is invalid")
                print("   - Custom Search API is not enabled")
                print("   - Daily quota exceeded (100 searches/day on free tier)")

            return False

        data = response.json()

        # Check for errors
        if "error" in data:
            print(f"❌ API Error: {json.dumps(data['error'], indent=2)}")

            error_msg = str(data.get('error', {}))
            if 'quotaExceeded' in error_msg or 'rateLimitExceeded' in error_msg:
                print("\n💡 You've hit your daily quota limit (100 searches/day)")

            return False

        # Check for results
        items = data.get("items", [])

        if not items:
            print("⚠️  WARNING: Search returned 0 results")
            print("This might indicate:")
            print("  1. Your Custom Search Engine is not configured to search the web")
            print("  2. Your CSE has restrictions that prevent finding results")
            print("  3. The query was too specific")
            print()
            print("Full response:")
            print(json.dumps(data, indent=2))
            print()
            check_cse_configuration()
            return False

        print(f"✅ SUCCESS: Found {len(items)} results")
        print()
        print("Sample results:")
        print("-" * 60)

        for i, item in enumerate(items[:5], 1):
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            snippet = item.get("snippet", "No snippet")[:100]

            print(f"\n{i}. {title}")
            print(f"   URL: {link}")
            print(f"   Snippet: {snippet}...")

        print()
        print("=" * 60)
        print("✅ Google Custom Search API is working correctly!")
        print()
        print(
            "If vendor search in the app still fails, the issue is in the filtering logic.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_cse_configuration():
    """Provide guidance on CSE configuration"""
    print()
    print("=" * 60)
    print("Google Custom Search Engine Configuration Checklist:")
    print("=" * 60)
    print()
    print("1. Go to: https://programmablesearchengine.google.com/")
    print("2. Select your search engine")
    print("3. Click 'Setup' or 'Edit'")
    print("4. Under 'Basics', verify:")
    print("   ✓ Search the entire web: ENABLED")
    print("   ✓ Image search: OFF (not needed)")
    print("   ✓ Safe search: OFF")
    print()
    print("5. Under 'Sites to search':")
    print("   ✓ Should say 'Search the entire web'")
    print("   ✓ If you have specific sites listed, REMOVE them or enable 'Search entire web'")
    print()
    print("6. Your Search Engine ID (CX) is shown at the top")
    print()


if __name__ == "__main__":
    success = test_google_search()

    if not success:
        print()
        print("💡 TIPS:")
        print("   1. Make sure you're running from the project root directory")
        print("   2. Check that .streamlit/secrets.toml exists and has the correct keys")
        print("   3. Verify your Custom Search Engine is configured to 'Search the entire web'")
        print("   4. Check your daily quota (100 searches/day on free tier)")
        print()
        sys.exit(1)
