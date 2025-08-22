import os
import sys
from datetime import datetime
import pytz
import sqlalchemy as sa
from sqlmodel import create_engine
import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError
from app import insert_new_records_only  # reuse your function

# --- Config ---
SAM_KEYS = os.getenv("SAM_KEYS", "").split(",")
DB_URL = os.getenv("SUPABASE_DB_URL")
MAX_RESULTS_REFRESH = int(os.getenv("MAX_RESULTS_REFRESH", "500"))
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/Denver")  # MST/MDT

# Times we want to run (24h clock, Mountain Time)
RUN_HOURS = {3, 12, 19}

def should_run_now() -> bool:
    tz = pytz.timezone(LOCAL_TZ)
    now_local = datetime.now(tz)
    return now_local.hour in RUN_HOURS and now_local.minute == 0

def main():
    if not should_run_now():
        print("⏭ Not a scheduled run hour. Exiting.")
        sys.exit(0)

    print("⏰ Running SAM.gov refresh job...")

    try:
        raw = gs.get_sam_raw_v3(
            days_back=0,
            limit=MAX_RESULTS_REFRESH,
            api_keys=SAM_KEYS,
            filters={}
        )
        n = insert_new_records_only(raw)
        print(f"✅ Inserted {n} new records into solicitationraw")
    except SamQuotaError:
        print("⚠️ SAM.gov quota exceeded.")
    except SamAuthError:
        print("❌ SAM.gov auth/network error. Check keys.")
    except SamBadRequestError as e:
        print(f"❌ Bad request: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()