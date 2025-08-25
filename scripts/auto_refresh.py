# scripts/auto_refresh.py
import os, sys, time
from pathlib import Path
from datetime import datetime, timezone

print("[auto_refresh] bootstrap starting...", flush=True)

# ---------- Make local modules importable ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
print("[auto_refresh] sys.path[0:3] =", sys.path[:3], flush=True)

# Early presence checks
print("[auto_refresh] exists scripts/auto_refresh.py:", (REPO_ROOT / "scripts" / "auto_refresh.py").exists(), flush=True)
print("[auto_refresh] exists get_relevant_solicitations.py:", (REPO_ROOT / "get_relevant_solicitations.py").exists(), flush=True)

import sqlalchemy as sa
from sqlalchemy import create_engine, text as sqltext

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
        print(f"[auto_refresh] [skip] It's {now_local} MT; not an allowed hour {sorted(ALLOWED_LOCAL_HOURS)}. FORCE_RUN={FORCE_RUN}", flush=True)
        sys.exit(0)
    elif FORCE_RUN:
        print(f"[auto_refresh] [bypass] FORCE_RUN set; running at {now_local} MT (hour {now_local.hour}).", flush=True)
else:
    if not FORCE_RUN:
        print("[auto_refresh] [skip] ZoneInfo unavailable and FORCE_RUN not set.", flush=True)
        sys.exit(0)
    else:
        print("[auto_refresh] [bypass] FORCE_RUN set and ZoneInfo unavailable; running anyway.", flush=True)

# ---------- Env / secrets ----------
DB_URL = os.environ.get("SUPABASE_DB_URL", "")
if not DB_URL:
    print("[auto_refresh] ERROR: SUPABASE_DB_URL is empty! Check workflow secrets.", flush=True)
    sys.exit(2)

# Mask DB URL but show host/db for sanity
def mask_db(url: str) -> str:
    # crude mask: keep scheme and host, drop credentials
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        host = u.hostname or "?"
        db   = (u.path or "").lstrip("/") or "?"
        return f"{u.scheme}://***:***@{host}:{u.port or ''}/{db}"
    except Exception:
        return "masked"
print("[auto_refresh] DB_URL (masked):", mask_db(DB_URL), flush=True)

# Handle SAM_KEYS as comma OR newline separated
_raw_keys = os.environ.get("SAM_KEYS", "")
SAM_KEYS = []
if _raw_keys:
    parts = []
    for line in _raw_keys.replace("\r","").split("\n"):
        parts.extend([p.strip() for p in line.split(",")])
    SAM_KEYS = [p for p in parts if p]
print(f"[auto_refresh] SAM_KEYS configured: {len(SAM_KEYS)} key(s)", flush=True)

try:
    DAYS_BACK = int(os.environ.get("DAYS_BACK", "1"))
except ValueError:
    DAYS_BACK = 1
print(f"[auto_refresh] DAYS_BACK = {DAYS_BACK}", flush=True)

# --- DB engine (add connect timeout for Postgres) ---
pg_opts = {}
if DB_URL.startswith("postgresql"):
    pg_opts["connect_args"] = {"connect_timeout": 10}  # seconds

print("[auto_refresh] Creating engine...", flush=True)
engine = create_engine(DB_URL, pool_pre_ping=True, **pg_opts)
print("[auto_refresh] Engine created.", flush=True)

# Quick DB ping
try:
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))
    print("[auto_refresh] DB ping OK.", flush=True)
except Exception as e:
    print("[auto_refresh] DB ping FAILED:", repr(e), flush=True)
    sys.exit(2)

# Defer this import until after PATH/prints so import failures are visible
print("[auto_refresh] Importing get_relevant_solicitations...", flush=True)
import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError
print("[auto_refresh] Imported get_relevant_solicitations.", flush=True)

# ---------- Insert helper ----------
COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code",
    "description","link"
]

def insert_new_records_only(records) -> int:
    if not records:
        print("[auto_refresh] No records passed to insert_new_records_only().", flush=True)
        return 0
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    skipped_just = 0
    for r in records:
        m = gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=True)
        if (m.get("notice_type") or "").strip().lower() == "justification":
            skipped_just += 1
            continue
        nid = (m.get("notice_id") or "").strip()
        if not nid:
            continue
        row = {k: (m.get(k) or "") for k in COLS_TO_SAVE}
        row["pulled_at"] = now_iso
        rows.append(row)

    print(f"[auto_refresh] Mapped {len(rows)} rows (skipped {skipped_just} Justification).", flush=True)
    if not rows:
        print("[auto_refresh] No new rows to insert after mapping.", flush=True)
        return 0

    cols = ["pulled_at"] + COLS_TO_SAVE
    placeholders = ", ".join(":" + c for c in cols)
    sql = sa.text(f"""
        INSERT INTO solicitationraw ({", ".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT (notice_id) DO NOTHING
    """)
    with engine.begin() as conn:
        result = conn.execute(sql, rows)
    # result.rowcount can be -1 for executemany; rely on rows length as "attempted"
    return len(rows)

def db_counts():
    try:
        with engine.connect() as conn:
            total = conn.execute(sqltext("SELECT COUNT(*) FROM solicitationraw")).scalar_one()
            max_pulled = conn.execute(sqltext("SELECT MAX(pulled_at) FROM solicitationraw")).scalar_one()
        return int(total or 0), str(max_pulled or "")
    except Exception as e:
        print("[auto_refresh] db_counts() failed:", repr(e), flush=True)
        return None, None

# ---------- Main ----------
def main():
    print("[auto_refresh] main() starting.", flush=True)
    before_cnt, before_last = db_counts()
    if before_cnt is not None:
        print(f"[auto_refresh] DB before: {before_cnt} rows; last pulled_at: {before_last}", flush=True)

    try:
        print("[auto_refresh] Fetching from SAM.gov...", flush=True)
        t0 = time.time()
        raw = gs.get_sam_raw_v3(
            days_back=DAYS_BACK,
            limit=20,       # keep small for debugging; raise once healthy
            api_keys=SAM_KEYS,
            filters={}
        )
        t1 = time.time()
        print(f"[auto_refresh] SAM returned {len(raw)} items in {t1 - t0:.1f}s.", flush=True)

        # Log first couple mapped samples
        for i, rec in enumerate(raw[:3]):
            try:
                m = gs.map_record_allowed_fields(rec, api_keys=SAM_KEYS, fetch_desc=False)
                print(f"[auto_refresh] sample[{i}]: nid={m.get('notice_id')} title={m.get('title')!r} posted={m.get('posted_date')}", flush=True)
            except Exception as e:
                print(f"[auto_refresh] sample[{i}] map error: {e!r}", flush=True)

        n_attempted = insert_new_records_only(raw)
        print(f"[auto_refresh] Insert attempted for {n_attempted} mapped rows.", flush=True)

        after_cnt, after_last = db_counts()
        if after_cnt is not None:
            print(f"[auto_refresh] DB after: {after_cnt} rows; last pulled_at: {after_last}", flush=True)

        if n_attempted == 0:
            if len(raw) == 0:
                print("[auto_refresh] DEBUG: 0 items from SAM.gov. Possible: empty window, filters in gs.*, or network/quota.", flush=True)
            else:
                print("[auto_refresh] DEBUG: All items deduped (notice_id conflicts) or filtered out (Justification).", flush=True)

        print("[auto_refresh] Completed successfully.", flush=True)

    except SamQuotaError:
        print("[auto_refresh] ERROR: SAM.gov quota exceeded.", flush=True)
        sys.exit(2)
    except SamAuthError:
        print("[auto_refresh] ERROR: SAM.gov auth failed. Check SAM_KEYS.", flush=True)
        sys.exit(2)
    except SamBadRequestError as e:
        print(f"[auto_refresh] ERROR: Bad request to SAM.gov: {e}", flush=True)
        sys.exit(2)
    except Exception as e:
        print("[auto_refresh] Unhandled failure:", repr(e), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()