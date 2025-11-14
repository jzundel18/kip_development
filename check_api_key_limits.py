#!/usr/bin/env python3
"""
Check SAM.gov API rate limits for each configured key.
Shows remaining requests and reset times.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Parse SAM_KEYS
_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r", "").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]

if not SAM_KEYS:
    print("ERROR: No SAM_KEYS found in environment!")
    print("Set SAM_KEYS in your .env file")
    exit(1)

print(f"Checking rate limits for {len(SAM_KEYS)} API key(s)...\n")

# Test each key with a simple search
test_notice_id = "test"  # Doesn't need to exist, just checking headers

for i, key in enumerate(SAM_KEYS, 1):
    key_preview = f"{key[:15]}...{key[-10:]}" if len(key) > 25 else key
    print(f"Key {i}: {key_preview}")

    try:
        url = "https://api.sam.gov/opportunities/v2/search"
        params = {
            "api_key": key,
            "noticeid": test_notice_id,
            "limit": 1
        }

        response = requests.get(url, params=params, timeout=10)

        # Check rate limit headers
        headers = response.headers

        print(f"  Status: {response.status_code}")

        # Common rate limit header names
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Rate-Limit-Limit",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset",
            "RateLimit-Limit",
            "RateLimit-Remaining",
            "RateLimit-Reset",
            "Retry-After"
        ]

        found_headers = False
        for header in rate_limit_headers:
            if header in headers:
                found_headers = True
                print(f"  {header}: {headers[header]}")

        if not found_headers:
            print("  No rate limit headers found in response")

        # Check if rate limited
        if response.status_code == 429:
            print("  ⚠ KEY IS CURRENTLY RATE LIMITED!")
            if "Retry-After" in headers:
                print(f"  Retry after: {headers['Retry-After']} seconds")
        elif response.status_code == 403:
            print("  ⚠ AUTHENTICATION FAILED - Key may be invalid")
        elif response.status_code == 200:
            print("  ✓ Key is working")

        print()

    except Exception as e:
        print(f"  ERROR: {e}\n")

print("\n" + "=" * 60)
print("SAM.gov API Rate Limit Information:")
print("=" * 60)
print("Standard limits (per key):")
print("  - 10 requests/second")
print("  - 1,000 requests/day (for some endpoints)")
print("  - Limits typically reset every hour")
print("\nWith 3 keys, you effectively have:")
print("  - 30 requests/second")
print("  - 3,000 requests/day")
print("\nBest practices:")
print("  - Use 2-3 parallel workers max")
print("  - Add 0.5-1 second delay between batches")
print("  - Spread requests across all keys evenly")