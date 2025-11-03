import os
import requests
from dotenv import load_dotenv

# Load environment
load_dotenv('.env')

sam_keys_raw = os.getenv("SAM_KEYS", "")
sam_keys = [k.strip() for k in sam_keys_raw.split(",") if k.strip()]

print(f"Found {len(sam_keys)} SAM.gov API key(s)")

for i, key in enumerate(sam_keys, 1):
    print(f"\n=== Testing Key {i} ===")
    print(f"Key preview: {key[:8]}...{key[-4:]}")

    # Test with a simple search
    url = "https://api.sam.gov/prod/opportunities/v2/search"
    params = {
        "api_key": key,
        "limit": 1,
        "postedFrom": "01/01/2025",
        "postedTo": "01/02/2025"
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✅ KEY WORKS!")
            data = response.json()
            print(f"Response keys: {list(data.keys())}")
        elif response.status_code == 401:
            print("❌ 401 Unauthorized - Invalid API key")
        elif response.status_code == 403:
            print("❌ 403 Forbidden - Check if key is valid and has permissions")
            print(f"Response: {response.text[:200]}")
        elif response.status_code == 429:
            print("⚠️  429 Rate Limited - Key is valid but quota exceeded")
        else:
            print(f"❌ Unexpected status: {response.text[:200]}")

    except Exception as e:
        print(f"❌ Error: {e}")