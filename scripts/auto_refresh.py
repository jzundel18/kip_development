# scripts/auto_refresh.py
import os
import sys
import json
import time
from datetime import datetime
import pytz
import sqlalchemy as sa

# ---- Config pulled from env (GH Actions -> “Secrets and variables”) ----
DB_URL         = os.getenv("SUPABASE_DB_URL")  # e.g., postgresql+psycopg2://.../postgres
SAM_KEYS       = [k.strip() for k in (os.getenv("SAM_KEYS") or "").split(",") if k.strip()]
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOCAL_TZ       = os.getenv("LOCAL_TZ", "America/Denver")   # Mountain Time w/ DST
RUN_HOURS      = {3, 12, 19}                               # local hours to run
FORCE_RUN      = os.getenv("FORCE_RUN", "0") == "1"         # manual override

# ---- Your project imports (ship with your repo) ----
import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError

# Minimal mirror of your insert-only function (no Streamlit needed)
COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code",
    "description","link"
]

def connect_engine():
    if not DB_URL:
        raise RuntimeError("SUPABASE_DB_URL not set in env.")
    eng = sa.create_engine(
        DB_URL,
        pool_pre_ping=True,
        connect_args={"sslmode": "require"} if DB_URL.startswith("postgresql") else {},
    )
    return eng

def insert_new_records_only(engine, records) -> int:
    if not records:
        return 0
    now_iso = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
    for r in records:
        m = gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=True)
        # Skip "Justification" types per your UI rule
        if (m.get("notice_type") or "").strip().lower() == "justification":
            continue
        nid = (m.get("notice_id") or "").strip()
        if not nid:
            continue
        row = {k: (m.get(k) or "") for k in COLS_TO_SAVE}
        row["pulled_at"] = now_iso
        rows.append(row)
    if not rows:
        return 0

    cols = ["pulled_at"] + COLS_TO_SAVE
    sql = sa.text(f"""
        INSERT INTO solicitationraw (
            {", ".join(cols)}
        ) VALUES (
            {", ".join(":"+c for c in cols)}
        )
        ON CONFLICT (notice_id) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def log_now():
    tz = pytz.timezone(LOCAL_TZ)
    now_local = datetime.now(tz)
    now_utc = datetime.utcnow()
    print(f"[auto-refresh] Local ({LOCAL_TZ}) now: {now_local:%Y-%m-%d %H:%M:%S}")
    print(f"[auto-refresh] UTC now: {now_utc:%Y-%m-%d %H:%M:%S}")
    print(f"[auto-refresh] RUN_HOURS (local): {sorted(RUN_HOURS)}")
    return now_local

def should_run_now(now_local) -> bool:
    return now_local.hour in RUN_HOURS and now_local.minute == 0

def main():
    now_local = log_now()
    if not (FORCE_RUN or should_run_now(now_local)):
        print("[auto-refresh] ⏭ Not a scheduled local hour. Exiting with code 0.")
        sys.exit(0)  # IMPORTANT: success -> keeps the workflow green

    if not SAM_KEYS:
        print("[auto-refresh] ❌ SAM_KEYS missing. Set repository secret SAM_KEYS.")
        sys.exit(1)

    try:
        engine = connect_engine()
    except Exception as e:
        print(f"[auto-refresh] ❌ DB connect failed: {e}")
        sys.exit(1)

    try:
        # Adjust limit if you like (you had 500 in the UI)
        raw = gs.get_sam_raw_v3(days_back=0, limit=500, api_keys=SAM_KEYS, filters={})
        n = insert_new_records_only(engine, raw)
        print(f"[auto-refresh] ✅ Inserted (or skipped if dup) rows: {n}")
        sys.exit(0)
    except SamQuotaError:
        print("[auto-refresh] ⚠️ SAM quota exceeded on all keys.")
        sys.exit(0)  # treat as success so the workflow doesn’t look “failed”
    except SamBadRequestError as e:
        print(f"[auto-refresh] ❌ Bad request to SAM.gov: {e}")
        sys.exit(1)
    except SamAuthError:
        print("[auto-refresh] ❌ All SAM.gov keys failed (auth/network).")
        sys.exit(1)
    except Exception as e:
        print(f"[auto-refresh] ❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()