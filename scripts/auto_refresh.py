# scripts/auto_refresh.py
import os
import sys
import time
import json
import sqlalchemy as sa
import pandas as pd
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, create_engine
from typing import Optional
from sqlalchemy import inspect
from zoneinfo import ZoneInfo

# Reuse your existing helper
import get_relevant_solicitations as gs
from get_relevant_solicitations import SamQuotaError, SamAuthError, SamBadRequestError

# ---------- Settings from env ----------
DB_URL = os.getenv("SUPABASE_DB_URL") or "sqlite:///app.db"
SAM_KEYS = os.getenv("SAM_KEYS", "")
SAM_KEYS = [k.strip() for k in SAM_KEYS.split(",") if k.strip()]
MAX_RESULTS = int(os.getenv("MAX_RESULTS_REFRESH", "500"))
LOCAL_TZ = os.getenv("LOCAL_TZ", "America/New_York")  # set your local tz here
TARGET_HOURS = os.getenv("TARGET_HOURS", "3,12,19")   # local hours to run

# ---------- DB schema minimal (mirror app.py) ----------
from sqlmodel import SQLModel, Field

class SolicitationRaw(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    pulled_at: Optional[str] = Field(default=None, index=True)
    notice_id: str = Field(index=True, nullable=False, unique=True)
    solicitation_number: Optional[str] = None
    title: Optional[str] = None
    notice_type: Optional[str] = None
    posted_date: Optional[str] = Field(default=None, index=True)
    response_date: Optional[str] = Field(default=None, index=True)
    archive_date: Optional[str] = Field(default=None, index=True)
    naics_code: Optional[str] = Field(default=None, index=True)
    set_aside_code: Optional[str] = Field(default=None, index=True)
    description: Optional[str] = None
    link: Optional[str] = None

COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code","description","link"
]

def engine_from_url(url: str):
    if url.startswith("postgresql+psycopg2://"):
        return create_engine(
            url, pool_pre_ping=True, pool_size=5, max_overflow=2,
            connect_args={
                "sslmode": "require",
                "keepalives": 1, "keepalives_idle": 30,
                "keepalives_interval": 10, "keepalives_count": 5,
            },
        )
    return create_engine(url, pool_pre_ping=True)

engine = engine_from_url(DB_URL)
SQLModel.metadata.create_all(engine)

def insert_new_records_only(records) -> int:
    if not records:
        return 0
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for r in records:
        m = gs.map_record_allowed_fields(r, api_keys=SAM_KEYS, fetch_desc=True)
        # Skip notice_type == Justification
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

    sql = sa.text(f"""
        INSERT INTO solicitationraw (pulled_at, {", ".join(COLS_TO_SAVE)})
        VALUES (:pulled_at, {", ".join(":"+c for c in COLS_TO_SAVE)})
        ON CONFLICT (notice_id) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)

def should_run_now() -> bool:
    """Return True if local hour is in TARGET_HOURS (e.g., 3,12,19)."""
    try:
        tz = ZoneInfo(LOCAL_TZ)
    except Exception:
        tz = ZoneInfo("UTC")
    local_now = datetime.now(tz)
    target_hours = {int(h.strip()) for h in TARGET_HOURS.split(",") if h.strip()}
    return local_now.hour in target_hours

def main():
    if not SAM_KEYS:
        print("No SAM_KEYS provided; aborting.", file=sys.stderr)
        sys.exit(1)

    if not should_run_now():
        print("Not a target run time; exiting.")
        return

    try:
        print("Fetching from SAM.gov…")
        raw = gs.get_sam_raw_v3(
            days_back=0,
            limit=int(MAX_RESULTS),
            api_keys=SAM_KEYS,
            filters={}
        )
        n = insert_new_records_only(raw)
        print(f"Inserted {n} new records.")
    except SamQuotaError:
        print("SAM.gov quota likely exceeded on all provided keys.", file=sys.stderr)
    except SamBadRequestError as e:
        print(f"Bad request to SAM.gov: {e}", file=sys.stderr)
    except SamAuthError:
        print("All SAM.gov keys failed (auth/network).", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()