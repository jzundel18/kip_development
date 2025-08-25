# scripts/auto_refresh.py
import os, sys
from pathlib import Path
from datetime import datetime, timezone

# ---------- Make local modules importable ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(p))

import sqlalchemy as sa
from sqlalchemy import create_engine, text as sqltext
import pandas as pd

import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError

# ---------- Time gating (America/Denver) ----------
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

ALLOWED_LOCAL_HOURS = {3, 12, 19}  # 3am, 12pm, 7pm Mountain
FORCE_RUN = str(os.environ.get("FORCE_RUN", "")).strip().lower() in ("1","true","yes","on")

if ZoneInfo is not None:
    now_local = datetime.now(ZoneInfo("America/Denver"))
    if now_local.hour not in ALLOWED_LOCAL_HOURS and not FORCE_RUN:
        print(f"[skip] It's {now_local} MT; not an allowed hour {sorted(ALLOWED_LOCAL_HOURS)}. FORCE_RUN={FORCE_RUN}")
        sys.exit(0)
    elif FORCE_RUN:
        print(f"[bypass] FORCE_RUN set; running at {now_local} MT (hour {now_local.hour}).")
else:
    if not FORCE_RUN:
        print("[skip] ZoneInfo unavailable and FORCE_RUN not set.")
        sys.exit(0)
    else:
        print("[bypass] FORCE_RUN set and ZoneInfo unavailable; running anyway.")

# ---------- Env / secrets ----------
DB_URL = os.environ.get("SUPABASE_DB_URL", "sqlite:///app.db")

# Handle SAM_KEYS as comma OR newline separated
_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r","").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]
print(f"SAM_KEYS configured: {len(SAM_KEYS)} key(s)")

# Days back – default to 1 to cover last 24h reliably
try:
    DAYS_BACK = int(os.environ.get("DAYS_BACK", "1"))
except ValueError:
    DAYS_BACK = 1
print(f"DAYS_BACK = {DAYS_BACK}")

# ---------- DB engine ----------
engine = create_engine(DB_URL, pool_pre_ping=True)

# ---------- Insert helper ----------
COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code",
    "description","link"
]

def insert_new_records_only(records) -> int:
    if not records:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for r in records:
        m = gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=True)
        if (m.get("notice_type") or "").strip().lower() == "justification":
            continue
        nid = (m.get("notice_id") or "").strip()
        if not nid:
            continue
        row = {k: (m.get(k) or "") for k in COLS_TO_SAVE}
        row["pulled_at"] = now_iso
        rows.append(row)

    if not rows:
        print("No new rows to insert after filtering/mapping.")
        return 0

    cols = ["pulled_at"] + COLS_TO_SAVE
    placeholders = ", ".join(":" + c for c in cols)
    sql = sa.text(f"""
        INSERT INTO solicitationraw ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (notice_id) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def db_counts():
    try:
        with engine.connect() as conn:
            total = conn.execute(sqltext("SELECT COUNT(*) FROM solicitationraw")).scalar_one()
            max_pulled = conn.execute(sqltext("SELECT MAX(pulled_at) FROM solicitationraw")).scalar_one()
        return int(total or 0), str(max_pulled or "")
    except Exception as e:
        print("db_counts() failed:", repr(e))
        return None, None

# ---------- Main ----------
def main():
    print("Starting auto-refresh job...")
    total_before, last_pulled = db_counts()
    if total_before is not None:
        print(f"DB before: {total_before} rows; last pulled_at: {last_pulled}")

    try:
        print("Fetching solicitations from SAM.gov...")
        raw = gs.get_sam_raw_v3(
            days_back=DAYS_BACK,
            limit=800,
            api_keys=SAM_KEYS,
            filters={}
        )
        print(f"Fetched {len(raw)} raw records from SAM.gov")

        # Show a couple examples to confirm fields present
        for i, rec in enumerate(raw[:3]):
            try:
                m = gs.map_record_allowed_fields(rec, api_keys=SAM_KEYS, fetch_desc=False)
                print(f" sample[{i}]: notice_id={m.get('notice_id')} title={m.get('title')!r} posted_date={m.get('posted_date')}")
            except Exception as e:
                print(f" sample[{i}] map error:", repr(e))

        n = insert_new_records_only(raw)
        print(f"Inserted (attempted): {n}")

        total_after, last_pulled2 = db_counts()
        if total_after is not None:
            print(f"DB after: {total_after} rows; last pulled_at: {last_pulled2}")

        # Helpful summary when no inserts
        if n == 0:
            if len(raw) == 0:
                print("DEBUG: SAM.gov returned 0 items. Possible reasons:")
                print(" - No new postings within DAYS_BACK window")
                print(" - Filters inside gs.get_sam_raw_v3 are too narrow")
                print(" - Quota or auth errors being swallowed (enable debugging in that module)")
            else:
                print("DEBUG: All fetched items appear to be duplicates (conflict on notice_id) or filtered out (Justification).")

        print("Auto-refresh job completed.")

    except SamQuotaError:
        print("ERROR: SAM.gov quota likely exceeded on all provided keys.")
        sys.exit(2)
    except SamAuthError:
        print("ERROR: SAM.gov auth failed for all keys. Check SAM_KEYS secret.")
        sys.exit(2)
    except SamBadRequestError as e:
        print(f"ERROR: Bad request to SAM.gov: {e}")
        sys.exit(2)
    except Exception as e:
        print("Auto refresh failed:", repr(e))
        sys.exit(1)

if __name__ == "__main__":
    main()