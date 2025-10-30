# test_env.py
from pathlib import Path
from dotenv import load_dotenv
import os

env_file = Path(__file__).parent / '.env'
print(f"Looking for .env at: {env_file}")
print(f"File exists: {env_file.exists()}")

if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment from {env_file}")
else:
    print(f"✗ .env file not found!")

print("\nEnvironment variables:")
print(
    f"SUPABASE_DB_URL: {'SET' if os.getenv('SUPABASE_DB_URL') else 'NOT SET'}")
print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
print(f"GMAIL_EMAIL: {os.getenv('GMAIL_EMAIL', 'NOT SET')}")
print(f"GMAIL_PASSWORD: {'SET' if os.getenv('GMAIL_PASSWORD') else 'NOT SET'}")
