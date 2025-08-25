import os, sys
from pathlib import Path
from datetime import datetime, timezone

# ---------- Make local modules importable ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    sys.path.insert(0, str(p))

import sqlalchemy as sa
from sqlalchemy import create_engine

import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError

# ---------- Time gating (America/Denver) ----------
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

ALLOWED_LOCAL_HOURS = {3, 12, 19}  # 3am, 12pm, 7pm Mountain Time

# Normalize FORCE_RUN from env (accepts 1/true/yes/on)
FORCE_RUN = str(os.environ.get("FORCE_RUN", "")).strip().lower() in ("1", "true", "yes", "on")

if ZoneInfo is not None:
    now_local = datetime.now(ZoneInfo("America/Denver"))
    if now_local.hour not in ALLOWED_LOCAL_HOURS and not FORCE_RUN:
        print(f"[skip] It's {now_local} MT; not an allowed hour {sorted(ALLOWED_LOCAL_HOURS)}. "
              f"FORCE_RUN={os.environ.get('FORCE_RUN')!r}")
        sys.exit(0)
    elif FORCE_RUN:
        print(f"[bypass] FORCE_RUN is set; running anyway at {now_local} MT (hour {now_local.hour}).")
else:
    # If ZoneInfo isn’t available, only skip when FORCE_RUN is NOT set
    if not FORCE_RUN:
        print("[skip] ZoneInfo unavailable and FORCE_RUN not set.")
        sys.exit(0)
    else:
        print("[bypass] FORCE_RUN set and ZoneInfo unavailable; running anyway.")

# ---------- Env / secrets ----------
DB_URL = os.environ.get("SUPABASE_DB_URL", "sqlite:///app.db")
SAM_KEYS = os.environ.get("SAM_KEYS", "")
if SAM_KEYS and isinstance(SAM_KEYS, str):
    SAM_KEYS = [k.strip() for k in SAM_KEYS.split(",") if k.strip()]
elif not isinstance(SAM_KEYS, (list, tuple)):
    SAM_KEYS = []

# ---------- DB engine ----------
engine = create_engine(DB_URL, pool_pre_ping=True)

# ---------- Insert helper (no streamlit deps) ----------
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
        print("No new rows to insert.")
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

# ---------- Main ----------
def main():
    print("Starting auto-refresh job...")

    try:
        print("Fetching solicitations from SAM.gov...")
        raw = gs.get_sam_raw_v3(
            days_back=0,
            limit=50,  # adjust if needed
            api_keys=SAM_KEYS,
            filters={}
        )
        print(f"Fetched {len(raw)} records from SAM.gov")

        n = insert_new_records_only(raw)
        print(f"Inserted (attempted): {n}")

        print("Auto-refresh job completed successfully.")

    except Exception as e:
        # Print a helpful error for Actions logs
        print("Auto refresh failed:", repr(e))
        sys.exit(1)