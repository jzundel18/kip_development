#!/usr/bin/env python3
"""
Diagnostic script to test Google Custom Search API configuration.
Run this to verify your Google API setup is working correctly.

Usage:
    python google_search_diagnostic.py
"""

import os
import requests
import json


def test_google_search():
    """Test Google Custom Search API"""

    # Get credentials from environment
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CX = os.getenv("GOOGLE_CX")

    if not GOOGLE_API_KEY:
        print("❌ ERROR: GOOGLE_API_KEY environment variable not set")
        return False

    if not GOOGLE_CX:
        print("❌ ERROR: GOOGLE_CX environment variable not set")
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
            return False

        data = response.json()

        # Check for errors
        if "error" in data:
            print(f"❌ API Error: {data['error']}")
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
        print("If vendor search still fails, the issue is likely in the filtering logic.")
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
    print()
    print("=" * 60)
    print("Google Custom Search API Diagnostic Tool")
    print("=" * 60)
    print()

    success = test_google_search()

    if not success:
        check_cse_configuration()
        print()
        print("💡 TIP: The most common issue is that the Custom Search Engine")
        print("   is not configured to 'Search the entire web'")
        print()
