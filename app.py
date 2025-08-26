import os, re, json, bcrypt
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timezone
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text, inspect
from sqlmodel import SQLModel, Field, create_engine
import streamlit as st
import uuid
from openai import OpenAI
import find_relevant_suppliers as fs
import generate_proposal as gp
import get_relevant_solicitations as gs
import secrets as pysecrets
import hashlib
import models
from datetime import timedelta
from streamlit_cookies_manager import EncryptedCookieManager
import warnings
from sqlalchemy.exc import SAWarning
import requests
from urllib.parse import urlparse

warnings.filterwarnings(
    "ignore",
    message="This declarative base already contains a class with the same class name and module name",
    category=SAWarning,
)
SQLModel.metadata.clear()
# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="KIP", layout="wide")

def get_secret(name, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)

# Secure cookies (encrypted in browser)
cookies = EncryptedCookieManager(
    prefix="kip_",
    password=get_secret("COOKIE_PASSWORD", "dev-cookie-secret")  # !! set in secrets
)
if not cookies.ready():
    st.stop()

# --- Simple view router ---
# views: "auth", "main", "account"
if "user" not in st.session_state:
    st.session_state.user = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "view" not in st.session_state:
    st.session_state.view = "main" if st.session_state.user else "auth"

# =========================
# Small helpers
# =========================

def _s(v) -> str:
    """Return a safe string for downstream parsers (handles None/NaN/NaT)."""
    try:
        # catches NaN and NaT
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return "" if v is None else str(v)

AGGREGATOR_HOST_DENYLIST = {
    "sam.gov","beta.sam.gov","govinfo.gov","grants.gov","login.gov",
    "linkedin.com","facebook.com","twitter.com","x.com","instagram.com",
    "youtube.com","bloomberg.com","wikipedia.org","indeed.com","glassdoor.com",
    "dnb.com","opencorporates.com","zoominfo.com","rocketreach.co","crunchbase.com"
}

def _host(u: str) -> str:
    try:
        h = urlparse(u).netloc.lower()
        # strip common prefixes
        for p in ("www.", "m.", "en.", "amp."):
            if h.startswith(p): 
                h = h[len(p):]
        return h
    except Exception:
        return ""

def _companyish_name_from_result(title: str, link: str) -> str:
    """Cheap heuristic: prefer title; fall back to domain-based name."""
    t = (title or "").strip()
    if t:
        # trim long SEO tails
        t = re.split(r"[\|\-–·•»]+", t, maxsplit=1)[0].strip()
    if t:
        return t[:80]
    h = _host(link)
    if not h:
        return "Unknown company"
    base = h.split(":")[0].split(".")
    if len(base) >= 2:
        core = base[-2]  # example: acme from acme.com
    else:
        core = base[0]
    return core.capitalize()[:80]

def _fallback_serpapi_fetch(query: str, serp_key: str, max_results: int = 10) -> list[dict]:
    """
    Minimal SerpAPI call: returns list of {title, link, snippet}.
    We filter obvious aggregator/non-vendor domains.
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "num": max_results,
        "api_key": serp_key,
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json() if r.content else {}
    items = []
    for row in (data.get("organic_results") or []):
        title = (row.get("title") or "").strip()
        link  = (row.get("link") or "").strip()
        if not link:
            continue
        h = _host(link)
        if not h or any(h.endswith(bad) or h == bad for bad in AGGREGATOR_HOST_DENYLIST):
            continue
        items.append({
            "title": title,
            "link": link,
            "snippet": (row.get("snippet") or "").strip(),
            "host": h,
        })
    # de-dupe by host
    seen, uniq = set(), []
    for it in items:
        if it["host"] in seen:
            continue
        seen.add(it["host"])
        uniq.append(it)
    return uniq

def normalize_naics_input(text_in: str) -> list[str]:
    if not text_in:
        return []
    values = re.split(r"[,\s]+", text_in.strip())
    return [v for v in (re.sub(r"[^\d]", "", x) for x in values) if v]

def parse_keywords_or(text_in: str) -> list[str]:
    return [k.strip() for k in text_in.split(",") if k.strip()]

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SERP_API_KEY   = get_secret("SERP_API_KEY")
SAM_KEYS       = get_secret("SAM_KEYS", [])

if isinstance(SAM_KEYS, str):
    SAM_KEYS = [k.strip() for k in SAM_KEYS.split(",") if k.strip()]
elif not isinstance(SAM_KEYS, (list, tuple)):
    SAM_KEYS = []

missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "SERP_API_KEY": SERP_API_KEY,
    "SAM_KEYS": SAM_KEYS,
}.items() if not v]
if missing:
    st.error(f"Missing required secrets: {', '.join(missing)}")
    st.stop()

# =========================
# Database (Supabase or SQLite)
# =========================
DB_URL = st.secrets.get("SUPABASE_DB_URL") or "sqlite:///app.db"

if DB_URL.startswith("postgresql+psycopg2://"):
    engine = create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=2,
        connect_args={
            "sslmode": "require",
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )
else:
    engine = create_engine(DB_URL, pool_pre_ping=True)

# Connectivity check (clean—no host/user printed)
try:
    with engine.connect() as conn:
        ver = conn.execute(sa.text("select version()")).first()
    st.sidebar.success("✅ Connected to database")
    if ver and isinstance(ver, tuple):
        st.sidebar.caption(ver[0])
except Exception as e:
    st.sidebar.error("❌ Database connection failed")
    st.sidebar.exception(e)
    st.stop()

# =========================
# Static schema (only the fields you want)
# =========================
# Remember-me tokens table (per-user, revocable)
try:
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS auth_tokens (
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_auth_tokens_user ON auth_tokens (user_id)
        """))
except Exception as e:
    st.warning(f"Auth token table note: {e}")

# Lightweight migration: unique index on users.email and one-profile-per-user constraint
try:
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON users (email);
        """))
        conn.execute(sa.text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_company_profile_user ON company_profile (user_id);
        """))
except Exception as e:
    st.warning(f"User/profile table migration note: {e}")

def _hash_password(pw: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pw.encode("utf-8"), salt).decode("utf-8")

def _check_password(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), pw_hash.encode("utf-8"))
    except Exception:
        return False

def get_user_by_email(email: str):
    with engine.connect() as conn:
        sql = sa.text("SELECT id, email, password_hash FROM users WHERE email = :e")
        row = conn.execute(sql, {"e": email.strip().lower()}).mappings().first()
        return dict(row) if row else None

def create_user(email: str, password: str) -> Optional[int]:
    email = email.strip().lower()
    pw_hash = _hash_password(password)
    with engine.begin() as conn:
        try:
            sql = sa.text("""
                INSERT INTO users (email, password_hash, created_at)
                VALUES (:email, :ph, :ts)
                RETURNING id
            """)
            new_id = conn.execute(sql, {"email": email, "ph": pw_hash, "ts": datetime.now(timezone.utc).isoformat()}).scalar_one()
            return int(new_id)
        except Exception as e:
            st.error(f"Could not create user: {e}")
            return None

def get_profile(user_id: int) -> Optional[dict]:
    with engine.connect() as conn:
        sql = sa.text("""
            SELECT id, user_id, company_name, description, city, state, created_at, updated_at
            FROM company_profile WHERE user_id = :uid
        """)
        row = conn.execute(sql, {"uid": user_id}).mappings().first()
        return dict(row) if row else None

def upsert_profile(user_id: int, company_name: str, description: str, city: str, state: str) -> None:
    with engine.begin() as conn:
        now = datetime.now(timezone.utc).isoformat()
        # Try update first
        upd = conn.execute(sa.text("""
            UPDATE company_profile
            SET company_name = :cn, description = :d, city = :c, state = :s, updated_at = :ts
            WHERE user_id = :uid
        """), {"cn": company_name, "d": description, "c": city, "s": state, "uid": user_id, "ts": now})
        if upd.rowcount == 0:
            conn.execute(sa.text("""
                INSERT INTO company_profile (user_id, company_name, description, city, state, created_at, updated_at)
                VALUES (:uid, :cn, :d, :c, :s, :ts, :ts)
            """), {"uid": user_id, "cn": company_name, "d": description, "c": city, "s": state, "ts": now})

# Create (or update) tables — run once per session
if "db_initialized" not in st.session_state:
    try:
        SQLModel.metadata.create_all(engine)
    finally:
        st.session_state["db_initialized"] = True

# Lightweight migration: ensure columns & unique index
REQUIRED_COLS = {
    "pulled_at": "TEXT",
    "notice_id": "TEXT",
    "solicitation_number": "TEXT",
    "title": "TEXT",
    "notice_type": "TEXT",
    "posted_date": "TEXT",
    "response_date": "TEXT",
    "archive_date": "TEXT",
    "naics_code": "TEXT",
    "set_aside_code": "TEXT",
    "description": "TEXT",
    "link": "TEXT",
}
try:
    insp = inspect(engine)
    existing_cols = {c["name"] for c in insp.get_columns("solicitationraw")}
    missing_cols = [c for c in REQUIRED_COLS if c not in existing_cols]

    if missing_cols:
        with engine.begin() as conn:
            for col in missing_cols:
                conn.execute(sa.text(f'ALTER TABLE solicitationraw ADD COLUMN "{col}" {REQUIRED_COLS[col]}'))

    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_solicitationraw_notice_id
            ON solicitationraw (notice_id)
        """))
except Exception as e:
    st.warning(f"Migration note: {e}")

# Lightweight migration for Companies (safe if already exists)
try:
    insp = inspect(engine)
    if "company" in [t.lower() for t in insp.get_table_names()]:
        existing_cols = {c["name"] for c in insp.get_columns("company")}
        REQUIRED_COMPANY_COLS = {
            "name": "TEXT",
            "description": "TEXT",
            "city": "TEXT",
            "state": "TEXT",
        }
        missing_cols = [c for c in REQUIRED_COMPANY_COLS if c not in existing_cols]
        if missing_cols:
            with engine.begin() as conn:
                for col in missing_cols:
                    conn.execute(sa.text(f'ALTER TABLE company ADD COLUMN "{col}" {REQUIRED_COMPANY_COLS[col]}'))
except Exception as e:
    st.warning(f"Company table migration note: {e}")

def render_sidebar_header():
    """Sidebar header: company name, signed-in email, and settings button."""
    with st.sidebar:
        st.markdown("---")
        if st.session_state.user:
            prof = st.session_state.profile or {}
            company_name = (prof.get("company_name") or "").strip() or "Your Company"
            st.markdown(f"### {company_name}")
            st.caption(f"Signed in as {st.session_state.user['email']}")
            if st.button("⚙️ Account Settings", key="sb_go_settings", use_container_width=True):
                st.session_state.view = "account"
                st.rerun()
        else:
            st.info("Not signed in")
            if st.button("Log in / Sign up", key="sb_go_login", use_container_width=True):
                st.session_state.view = "auth"
                st.rerun()
        st.markdown("---")

with st.sidebar:
    st.success("✅ API keys loaded from Secrets")
    st.caption("Feed refresh runs automatically (no manual refresh needed).")
    st.markdown("---")

render_sidebar_header()

# =========================
# AI helpers
# =========================
def _hash_token(raw: str) -> str:
    # Hash the token before storing (don’t store raw)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _issue_remember_me_token(user_id: int, days: int = None) -> str:
    days = days or int(get_secret("COOKIE_DAYS", 30))
    raw = pysecrets.token_urlsafe(32)
    tok_hash = _hash_token(raw)
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=days)
    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO auth_tokens (user_id, token_hash, expires_at, created_at)
            VALUES (:uid, :th, :exp, :now)
        """), {
            "uid": user_id,
            "th": tok_hash,
            "exp": exp.isoformat(),
            "now": now.isoformat(),
        })
    return raw  # we return raw to set in cookie

def _validate_remember_me_token(raw: str) -> Optional[int]:
    if not raw:
        return None
    tok_hash = _hash_token(raw)
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT user_id, expires_at
            FROM auth_tokens
            WHERE token_hash = :th
            ORDER BY created_at DESC
            LIMIT 1
        """), {"th": tok_hash}).mappings().first()
    if not row:
        return None
    try:
        exp = datetime.fromisoformat(row["expires_at"])
        # normalize to aware UTC if stored naive
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < datetime.now(timezone.utc):
            return None
    except Exception:
        return None
    return int(row["user_id"])

def _revoke_all_tokens_for_user(user_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM auth_tokens WHERE user_id = :uid"), {"uid": user_id})

# Attempt auto-login from remember-me cookie if not already signed in
if st.session_state.user is None:
    raw_cookie = cookies.get(get_secret("COOKIE_NAME", "kip_auth"))
    uid = _validate_remember_me_token(raw_cookie) if raw_cookie else None
    if uid:
        # load user and profile
        with engine.connect() as conn:
            row = conn.execute(sa.text("SELECT id, email FROM users WHERE id = :uid"),
                               {"uid": uid}).mappings().first()
        if row:
            st.session_state.user = {"id": row["id"], "email": row["email"]}
            st.session_state.profile = get_profile(row["id"])
            st.session_state.view = "main"

def _embed_texts(texts: list[str], api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    X = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2 normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X

@st.cache_data(show_spinner=False, ttl=3600)
def cached_company_embeddings(_companies: pd.DataFrame, api_key: str) -> dict:
    """
    Returns {"df": companies_df, "X": normalized embeddings (np.ndarray)}.
    Cache invalidates if companies data changes (we use df contents as key).
    """
    if _companies.empty:
        return {"df": _companies, "X": np.zeros((0, 1536), dtype=np.float32)}
    texts = _companies["description"].fillna("").astype(str).tolist()
    X = _embed_texts(texts, api_key)
    return {"df": _companies.copy(), "X": X}

def ai_identify_gaps(company_desc: str, solicitation_text: str, api_key: str) -> str:
    """
    Ask the model to identify key capability gaps we'd need to fill to bid solo.
    Returns a short paragraph (1–3 sentences).
    """
    client = OpenAI(api_key=api_key)
    sys = "You are a federal contracting expert. Be concise and specific."
    user = (
        "Company description:\n"
        f"{company_desc}\n\n"
        "Solicitation (title+description):\n"
        f"{solicitation_text[:6000]}\n\n"
        "List the biggest capability gaps this company would need to fill to bid competitively. "
        "Return a short paragraph (no bullets)."
    )
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        st.warning(f"Gap identification unavailable ({e}).")
        return ""

def pick_best_partner_for_gaps(gap_text: str, companies: pd.DataFrame, api_key: str, top_n: int = 1) -> pd.DataFrame:
    """
    Use embeddings to match companies to the gap text. Returns top_n rows from `companies`.
    """
    if companies.empty or not gap_text.strip():
        return companies.head(0)
    # Embed gap_text
    q = _embed_texts([gap_text], api_key)[0]  # normalized
    # Embed companies (cached)
    emb = cached_company_embeddings(companies, api_key)
    dfc, X = emb["df"], emb["X"]
    if X.shape[0] == 0:
        return dfc.head(0)
    sims = X @ q  # cosine similarity
    dfc = dfc.copy()
    dfc["score"] = sims
    return dfc.sort_values("score", ascending=False).head(top_n)

def ai_partner_justification(company_row: dict, solicitation_text: str, gap_text: str, api_key: str) -> dict:
    """
    Returns {"justification": "...", "joint_proposal": "..."} short blurbs.
    """
    client = OpenAI(api_key=api_key)
    sys = (
        "You are a federal contracts strategist. Be concise, concrete, and persuasive. "
        "You MUST reply with a single JSON object only."
    )
    # IMPORTANT: include the word "JSON" and the exact shape
    instructions = (
        'Return ONLY a JSON object of the form: '
        '{"justification":"one short sentence", "joint_proposal":"one short sentence"} '
        '— no markdown, no extra text.'
    )
    user_payload = {
        "partner_company": {
            "name": company_row.get("name",""),
            "capabilities": company_row.get("description",""),
            "location": f'{company_row.get("city","")}, {company_row.get("state","")}'.strip(", "),
        },
        "our_capability_gaps": gap_text,
        "solicitation": solicitation_text[:6000],
        "instructions": instructions,
    }

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                # include "JSON" in the user message content as well
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.2,
        )
        content = (r.choices[0].message.content or "").strip()
        data = json.loads(content or "{}")
        j = str(data.get("justification","")).strip()
        jp = str(data.get("joint_proposal","")).strip()
        if not j and not jp:
            # graceful fallback
            return {"justification": "No justification returned.", "joint_proposal": ""}
        return {"justification": j, "joint_proposal": jp}
    except Exception as e:
        return {"justification": f"Justification unavailable ({e})", "joint_proposal": ""}
    
def ai_downselect_df(company_desc: str, df: pd.DataFrame, api_key: str,
                     threshold: float = 0.20, top_k: int | None = None) -> pd.DataFrame:
    """
    Embedding-based similarity between company_desc and (title + description).
    Keep rows with similarity >= threshold, or top_k if provided.
    """
    if df.empty:
        return df

    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).str.slice(0, 2000).tolist()
    try:
        client = OpenAI(api_key=api_key)
        q = client.embeddings.create(model="text-embedding-3-small", input=[company_desc])
        Xq = np.array(q.data[0].embedding, dtype=np.float32)

        r = client.embeddings.create(model="text-embedding-3-small", input=texts)
        X = np.array([d.embedding for d in r.data], dtype=np.float32)

        Xq_norm = Xq / (np.linalg.norm(Xq) + 1e-9)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sims = X_norm @ Xq_norm

        df = df.copy()
        df["ai_score"] = sims

        if top_k is not None and top_k > 0:
            df = df.sort_values("ai_score", ascending=False).head(int(top_k))
        else:
            df = df[df["ai_score"] >= float(threshold)].sort_values("ai_score", ascending=False)

        return df.reset_index(drop=True)

    except Exception as e:
        st.warning(f"AI downselect unavailable right now ({e}). Falling back to simple keyword filter.")
        kws = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{4,}", company_desc)]
        if not kws:
            return df
        blob = (df["title"].fillna("") + " " + df["description"].fillna("")).str.lower()
        mask = blob.apply(lambda t: any(k in t for k in kws))
        return df[mask].reset_index(drop=True)

def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]

def ai_make_blurbs(
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_items: int = 160,
    chunk_size: int = 40,
) -> dict[str, str]:
    """
    Returns {notice_id: blurb}. Short plain-English summaries of solicitations.
    Batches requests to keep prompts small & reliable.
    """
    if df is None or df.empty:
        return {}

    cols = ["notice_id", "title", "description"]
    use = df[[c for c in cols if c in df.columns]].head(max_items).copy()

    # Prepare items (truncate to keep prompt tight)
    items = []
    for _, r in use.iterrows():
        items.append({
            "notice_id": str(r.get("notice_id", "")),
            "title": (r.get("title") or "")[:200],
            "description": (r.get("description") or "")[:800],  # keep short to avoid token bloat
        })

    client = OpenAI(api_key=api_key)
    out: dict[str, str] = {}

    system_msg = (
        "You are helping a contracts analyst. For each item, write one very short, "
        "plain-English blurb (~8–12 words) summarizing what the solicitation buys/needs. "
        "Avoid agency names, set-aside boilerplate, and extra punctuation."
    )

    for batch in _chunk(items, chunk_size):
        user_msg = {
            "items": batch,
            "format": 'Return JSON: {"blurbs":[{"notice_id":"...","blurb":"..."}]} in the same order.'
        }
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(user_msg)},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            for row in data.get("blurbs", []):
                nid = str(row.get("notice_id", "")).strip()
                blurb = (row.get("blurb") or "").strip()
                if nid and blurb:
                    out[nid] = blurb
        except Exception as e:
            # If a batch fails, skip it but continue with others
            st.warning(f"Could not generate blurbs for one batch ({e}).")
            continue

    return out

# =========================
# DB helpers
# =========================
COLS_TO_SAVE = [
    "notice_id","solicitation_number","title","notice_type",
    "posted_date","response_date","archive_date",
    "naics_code","set_aside_code",
    "description","link"
]

DISPLAY_COLS = [
    "pulled_at",
    "solicitation_number",
    "notice_type",
    "posted_date",
    "response_date",
    "naics_code",
    "set_aside_code",
    "sam_url",   # swapped in for link
]

def insert_new_records_only(records) -> int:
    """
    Maps raw SAM records, adds pulled_at, and inserts only new notice_ids.
    """
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
        # Normalize link to a human-facing URL
        row["link"] = make_sam_public_url(row["notice_id"], row.get("link"))
        rows.append(row)

    if not rows:
        return 0

    sql = sa.text(f"""
        INSERT INTO solicitationraw (
            pulled_at, {", ".join(COLS_TO_SAVE)}
        ) VALUES (
            :pulled_at, {", ".join(":"+c for c in COLS_TO_SAVE)}
        )
        ON CONFLICT (notice_id) DO NOTHING
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)  # bulk insert

    return len(rows)

def query_filtered_df(filters: dict) -> pd.DataFrame:
    # Pull a superset of columns; we'll hide some in the UI
    base_cols = ["pulled_at","notice_id","solicitation_number","title","notice_type",
                 "posted_date","response_date","archive_date",
                 "naics_code","set_aside_code","description","link"]

    with engine.connect() as conn:
        df = pd.read_sql_query(f"SELECT {', '.join(base_cols)} FROM solicitationraw", conn)

    if df.empty:
        return df

    # keyword OR filter
    kws = [k.lower() for k in (filters.get("keywords_or") or []) if k]
    if kws:
        title = df["title"].fillna("")
        desc  = df["description"].fillna("")
        blob = (title + " " + desc).str.lower()
        df = df[blob.apply(lambda t: any(k in t for k in kws))]

    # NAICS filter
    naics = [re.sub(r"[^\d]","", x) for x in (filters.get("naics") or []) if x]
    if naics:
        df = df[df["naics_code"].isin(naics)]

    # set-aside filter
    sas = filters.get("set_asides") or []
    if sas:
        df = df[df["set_aside_code"].fillna("").str.lower().apply(lambda s: any(sa.lower() in s for sa in sas))]

    # notice types
    nts = filters.get("notice_types") or []
    if nts:
        df = df[df["notice_type"].fillna("").str.lower().apply(lambda s: any(nt.lower() in s for nt in nts))]

    # due before
    due_before = filters.get("due_before")
    if due_before:
        dd = pd.to_datetime(df["response_date"], errors="coerce", utc=True)
        df = df[dd.dt.date <= pd.to_datetime(due_before).date()]

    return df.reset_index(drop=True)

def companies_df() -> pd.DataFrame:
    with engine.connect() as conn:
        try:
            return pd.read_sql_query(
                "SELECT id, name, description, city, state FROM company ORDER BY name",
                conn
            )
        except Exception:
            return pd.DataFrame(columns=["id","name","description","city","state"])

def insert_company_row(row: dict) -> None:
    sql = sa.text("""
        INSERT INTO company (name, description, city, state)
        VALUES (:name, :description, :city, :state)
    """)
    row = {k: (row.get(k) or "") for k in ["name","description","city","state"]}
    row["created_at"] = datetime.now(timezone.utc).isoformat()
    with engine.begin() as conn:
        conn.execute(sql, row)

def bulk_insert_companies(df: pd.DataFrame) -> int:
    needed = ["name","description","city","state"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    rows = df[needed].fillna("").to_dict(orient="records")
    for r in rows:
        insert_company_row(r)
    return len(rows)

def render_auth_screen():
    st.title("Welcome to KIP")
    st.caption("Sign in or create an account to continue.")

    c1, c2 = st.columns(2)

    # ---- Login
    with c1:
        st.subheader("Log in")
        le = st.text_input("Email", key="login_email_full")
        lp = st.text_input("Password", type="password", key="login_password_full")
        remember_me = st.checkbox("Remember me for 30 days", value=True)
        if st.button("Log in", key="btn_login_full", use_container_width=True):
            u = get_user_by_email(le or "")
            if not u:
                st.error("No account found with that email.")
            elif not _check_password(lp or "", u["password_hash"]):
                st.error("Invalid password.")
            else:
                st.session_state.user = {"id": u["id"], "email": u["email"]}
                st.session_state.profile = get_profile(u["id"])
                st.session_state.view = "main"
                st.success("Logged in.")

                # If "remember me", issue token + set encrypted cookie
                if remember_me:
                    raw_token = _issue_remember_me_token(u["id"], days=int(get_secret("COOKIE_DAYS", 30)))
                    cookie_name = get_secret("COOKIE_NAME", "kip_auth")
                    cookies[cookie_name] = raw_token
                    cookies.save()

                st.rerun()

    # ---- Sign up
    with c2:
        st.subheader("Sign up")
        se = st.text_input("Email", key="signup_email_full")
        sp = st.text_input("Password", type="password", key="signup_password_full")
        sp2 = st.text_input("Confirm password", type="password", key="signup_password2_full")
        if st.button("Create account", key="btn_signup_full", type="primary", use_container_width=True):
            if not se or not sp:
                st.error("Email and password are required.")
            elif sp != sp2:
                st.error("Passwords do not match.")
            elif get_user_by_email(se):
                st.error("An account with that email already exists.")
            else:
                uid = create_user(se, sp)
                if uid:
                    # create an empty profile so settings page has something to edit
                    upsert_profile(uid, company_name="", description="", city="", state="")
                    st.success("Account created. Please log in on the left.")
                else:
                    st.error("Could not create account. Check server logs.")


def render_account_settings():
    st.title("Account Settings")

    if st.button("Sign out", key="btn_signout_settings"):
        # Revoke tokens and clear cookie
        if st.session_state.user:
            _revoke_all_tokens_for_user(st.session_state.user["id"])
        cookie_name = get_secret("COOKIE_NAME", "kip_auth")
        try:
            del cookies[cookie_name]
        except KeyError:
            pass
        cookies.save()

        st.session_state.user = None
        st.session_state.profile = None
        st.session_state.view = "auth"
        st.rerun()

    if st.session_state.user is None:
        st.info("Please log in first.")
        if st.button("Go to Login / Sign up"):
            st.session_state.view = "auth"
            st.rerun()
        st.stop()

    st.write(f"Signed in as **{st.session_state.user['email']}**")
    st.markdown("---")
    st.subheader("Company Profile")

    prof = st.session_state.profile or {}
    company_name = st.text_input("Company name", value=prof.get("company_name", ""))
    description  = st.text_area("Company description", value=prof.get("description", ""), height=140)
    city         = st.text_input("City", value=prof.get("city", "") or "")
    state        = st.text_input("State", value=prof.get("state", "") or "")

    cols = st.columns([1,1,3])
    with cols[0]:
        if st.button("Save profile", key="btn_save_profile_settings"):
            if not company_name.strip() or not description.strip():
                st.error("Company name and description are required.")
            else:
                upsert_profile(
                    st.session_state.user["id"],
                    company_name.strip(),
                    description.strip(),
                    city.strip(),
                    state.strip()
                )
                st.session_state.profile = get_profile(st.session_state.user["id"])
                st.success("Profile saved.")
    with cols[1]:
        if st.button("Back to app", key="btn_back_to_app"):
            st.session_state.view = "main"
            st.rerun()

def _hide_notice_and_description(df: pd.DataFrame) -> pd.DataFrame:
    # UI should not show these two columns
    return df.drop(columns=[c for c in ["notice_id", "description", "link"] if c in df.columns], errors="ignore")

def make_sam_public_url(notice_id: str, link: str | None = None) -> str:
    """
    Return a human-viewable SAM.gov URL for this notice.
    If the saved link is already a public web URL (not the API), keep it.
    Otherwise build https://sam.gov/opp/<notice_id>/view
    """
    if link and isinstance(link, str) and "api.sam.gov" not in link:
        return link
    nid = (notice_id or "").strip()
    return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"

# ====== ROUTER ======
if st.session_state.view == "auth":
    render_auth_screen()
    st.stop()
elif st.session_state.view == "account":
    render_account_settings()
    st.stop()

# ====== MAIN APP HEADER (only when in "main") ======
st.title("KIP")
st.caption("Don't be jealous that I've been chatting online with babes *all day*.")

colR1, colR2 = st.columns([2,1])
with colR1:
    st.info("Feed updates automatically every hour.")
with colR2:
    try:
        with engine.connect() as conn:
            cnt = pd.read_sql_query("SELECT COUNT(*) AS c FROM solicitationraw", conn)["c"].iloc[0]
        st.metric("Rows in DB", int(cnt))
    except Exception:
        st.metric("Rows in DB", 0)


# =========================
# Session state
# =========================
if "sol_df" not in st.session_state:
    st.session_state.sol_df = None
if "sup_df" not in st.session_state:
    st.session_state.sup_df = None
# cache: per-solicitation vendor suggestions (Internal Use tab)
if "vendor_suggestions" not in st.session_state:
    st.session_state.vendor_suggestions = {}  # { notice_id: DataFrame }
# track which Internal Use expanders are open (per notice)
if "expander_open" not in st.session_state:
    st.session_state.expander_open = {}  # { notice_id: bool }
if "iu_results" not in st.session_state:
    st.session_state.iu_results = None    # {"top_df": DataFrame, "reason_by_id": dict}
if "iu_key_salt" not in st.session_state:
    st.session_state.iu_key_salt = ""     # salt to keep button keys unique per run
if "iu_open_nid" not in st.session_state:
    st.session_state.iu_open_nid = None   # keeps the clicked expander open after rerun
# =========================
# AI ranker (used for the expander section)
# =========================
def ai_rank_solicitations_by_fit(
    df: pd.DataFrame,
    company_desc: str,
    api_key: str,
    top_k: int = 10,
    max_candidates: int = 1000,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    if df is None or df.empty:
        return []

    cols_we_care = [
        "notice_id", "title", "description", "naics_code",
        "set_aside_code", "response_date", "posted_date", "link"
    ]
    df2 = df[[c for c in cols_we_care if c in df.columns]].copy().head(max_candidates)

    items = []
    for _, r in df2.iterrows():
        items.append({
            "notice_id": str(r.get("notice_id", "")),
            "title": str(r.get("title", ""))[:300],
            "description": str(r.get("description", ""))[:1500],
            "naics_code": str(r.get("naics_code", "")),
            "set_aside_code": str(r.get("set_aside_code", "")),
            "response_date": str(r.get("response_date", "")),
            "posted_date": str(r.get("posted_date", "")),
            "link": str(r.get("link", "")),
        })

    system_msg = (
        "You are a contracts analyst. Rank solicitations by how well they match the company description. "
        "Consider title, description, NAICS, set-aside, and due date recency."
    )
    user_msg = {
        "company_description": company_desc,
        "solicitations": items,
        "instructions": (
            f"Return the top {top_k} as JSON: "
            '{"ranked":[{"notice_id":"...","score":0-100,"reason":"..."}]}. '
            "Score reflects strength of fit (higher is better). Keep reasons short and specific."
        ),
    }

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg)},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content

    try:
        data = json.loads(content or "{}")
        ranked = data.get("ranked", [])
    except Exception:
        return []

    keep_ids = set(df2["notice_id"].astype(str).tolist())
    cleaned = []
    for item in ranked:
        nid = str(item.get("notice_id", ""))
        if nid in keep_ids:
            cleaned.append({
                "notice_id": nid,
                "score": float(item.get("score", 0)),
                "reason": str(item.get("reason", "")),
            })

    seen, out = set(), []
    for x in cleaned:
        if x["notice_id"] not in seen:
            seen.add(x["notice_id"])
            out.append(x)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1) Solicitation Match",
    "2) Supplier Suggestions",
    "3) Proposal Draft",
    "4) Partner Matches",
    "5) Internal Use"
])
# ---- Tab 1
with tab1:
    st.header("Filter Solicitations")

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        limit_results = st.number_input("Max results to show", min_value=1, max_value=5000, value=20)
    with colB:
        keywords_raw = st.text_input("Filter keywords (OR, comma-separated)", value="")
    with colC:
        naics_raw = st.text_input("Filter by NAICS (comma-separated)", value="")

    with st.expander("More filters (optional)"):
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            set_asides = st.multiselect("Set-aside code", ["SBA","WOSB","EDWOSB","HUBZone","SDVOSB","8A","SDB"])
        with col2:
            due_before = st.date_input("Due before (optional)", value=None, format="YYYY-MM-DD")
        with col3:
            notice_types = st.multiselect(
                "Notice types",
                ["Solicitation","Combined Synopsis/Solicitation","Sources Sought","Special Notice","SRCSGT","RFI"]
            )
    filters = {
        "keywords_or": parse_keywords_or(keywords_raw),
        "naics": normalize_naics_input(naics_raw),
        "set_asides": set_asides,
        "due_before": (due_before.isoformat() if isinstance(due_before, date) else None),
        "notice_types": notice_types,
    }
    st.subheader("Company profile for matching")
    saved_desc = (st.session_state.get("profile") or {}).get("description", "")
    use_saved = st.checkbox("Use saved company profile", value=bool(saved_desc))
    if use_saved and saved_desc:
        st.info("Using your saved company profile description.")
        company_desc = saved_desc
        # show as read-only preview
        st.text_area("Company description (from Account → Company Profile)", value=saved_desc, height=120, disabled=True)
    else:
        company_desc = st.text_area("Brief company description (temporary)", value="", height=120)

    st.session_state.company_desc = company_desc or ""
    use_ai_downselect = st.checkbox("Use AI to downselect based on description", value=False)
    # Let the user pick how many AI-ranked matches to return
    top_k_select = (
        st.number_input(
            "How many AI-ranked matches?",
            min_value=1, max_value=50, value=5, step=1,
            help="How many solicitations the AI should rank and return."
        )
        if use_ai_downselect else 5
)
    if st.button("Show top results", type="primary", key="btn_show_results"):
        try:
            # 1) Apply manual filters from DB (no SAM calls)
            df = query_filtered_df(filters)

            if df.empty:
                st.warning("No solicitations match your filters. Try adjusting filters or refresh today's feed.")
                st.session_state.sol_df = None
            else:
                # ===== IF AI downselect + company description → Rank Top N =====
                if use_ai_downselect and company_desc.strip():
                    # Pre-trim with embeddings to keep prompt small & fast
                    # Keep the most-similar N items before LLM ranking
                    pretrim = ai_downselect_df(company_desc.strip(), df, OPENAI_API_KEY, top_k=80)

                    if pretrim.empty:
                        st.info("AI pre-filter returned nothing. Showing manually filtered table instead.")
                        show_df = df.head(int(limit_results)) if limit_results else df
                        # Generate very short blurbs only for rows we will display
                        blurbs = ai_make_blurbs(show_df, OPENAI_API_KEY, model="gpt-4o-mini",
                                                max_items=min(150, int(limit_results or 150)))
                        show_df = show_df.copy()
                        show_df["blurb"] = show_df["notice_id"].astype(str).map(blurbs).fillna(show_df["title"].fillna(""))
                        st.session_state.sol_df = show_df
                        st.subheader(f"Solicitations ({len(show_df)})")

                        # Add normalized public SAM.gov URL
                        show_df = show_df.copy()
                        show_df["sam_url"] = show_df.apply(
                            lambda r: make_sam_public_url(str(r.get("notice_id","")), r.get("link","")),
                            axis=1
                        )

                        st.dataframe(_hide_notice_and_description(show_df), use_container_width=True)
                        st.download_button(
                            "Download filtered as CSV",
                            show_df.to_csv(index=False).encode("utf-8"),
                            file_name="sol_list.csv",
                            mime="text/csv"
                        )
                    else:
                        ranked = ai_rank_solicitations_by_fit(
                        df=pretrim,
                        company_desc=company_desc.strip(),
                        api_key=OPENAI_API_KEY,
                        top_k=int(top_k_select),
                        max_candidates=60,
                        model="gpt-4o-mini",
)

                        if not ranked:
                            st.info("AI ranking returned no results; showing the manually filtered table instead.")
                            show_df = df.head(int(limit_results)) if limit_results else df
                            # blurbs only for what we show
                            blurbs = ai_make_blurbs(show_df, OPENAI_API_KEY, model="gpt-4o-mini",
                                                    max_items=min(150, int(limit_results or 150)))
                            show_df = show_df.copy()
                            show_df["blurb"] = show_df["notice_id"].astype(str).map(blurbs).fillna(show_df["title"].fillna(""))
                            st.session_state.sol_df = show_df
                            st.subheader(f"Solicitations ({len(show_df)})")

                            # Add normalized public SAM.gov URL
                            show_df = show_df.copy()
                            show_df["sam_url"] = show_df.apply(
                                lambda r: make_sam_public_url(str(r.get("notice_id","")), r.get("link","")),
                                axis=1
                            )

                            st.dataframe(_hide_notice_and_description(show_df), use_container_width=True)
                            st.download_button(
                                "Download filtered as CSV",
                                show_df.to_csv(index=False).encode("utf-8"),
                                file_name="sol_list.csv",
                                mime="text/csv"
                            )
                        else:
                            # Build ordered dataframe
                            id_order = [x["notice_id"] for x in ranked]
                            preorder = {nid: i for i, nid in enumerate(id_order)}
                            top_df = pretrim[pretrim["notice_id"].astype(str).isin(id_order)].copy()
                            top_df["__order"] = top_df["notice_id"].astype(str).map(preorder)
                            top_df = (
                                top_df.sort_values("__order")
                                    .drop_duplicates(subset=["notice_id"])
                                    .drop(columns="__order")
)
                            # Generate blurbs
                            blurbs = ai_make_blurbs(top_df, OPENAI_API_KEY, model="gpt-4o-mini", max_items=10)
                            top_df["blurb"] = top_df["notice_id"].astype(str).map(blurbs).fillna(top_df["title"].fillna(""))

                            # Show expanders: blurb first, click to see details
                            st.success(f"Top {len(top_df)} matches by company fit:")
                            # quick lookup for reasons and scores
                            reason_by_id = {x["notice_id"]: x.get("reason", "") for x in ranked}
                            score_by_id  = {x["notice_id"]: x.get("score", 0)  for x in ranked}

                            # add a fit_score column so Tab 4 can auto-filter moderate/strong matches
                            top_df["fit_score"] = top_df["notice_id"].astype(str).map(score_by_id).fillna(0).astype(float)

                            for i, row in enumerate(top_df.itertuples(index=False), start=1):
                                hdr = (getattr(row, "blurb", None) or getattr(row, "title", None) or "Untitled")
                                with st.expander(f"{i}. {hdr}"):
                                    st.write(f"**Notice Type:** {getattr(row, 'notice_type', '')}")
                                    st.write(f"**Posted:** {getattr(row, 'posted_date', '')}")
                                    st.write(f"**Response Due:** {getattr(row, 'response_date', '')}")
                                    st.write(f"**NAICS:** {getattr(row, 'naics_code', '')}")
                                    st.write(f"**Set-aside:** {getattr(row, 'set_aside_code', '')}")
                                    link = make_sam_public_url(str(getattr(row, "notice_id", "")), getattr(row, "link", ""))
                                    st.write(f"[Open on SAM.gov]({link})")
                                    reason = reason_by_id.get(str(getattr(row, "notice_id", "")), "")
                                    if reason:
                                        st.markdown("**Why this matched (AI):**")
                                        st.info(reason)

                            # Save the AI-ranked dataframe itself for Tab 4
                            st.session_state.topn_df = top_df.reset_index(drop=True)
                            st.session_state.sol_df = top_df.copy()
                            # Invalidate any old partner matches (Tab 4 will auto-rebuild)
                            st.session_state.partner_matches = None
                            st.session_state.topn_stamp = datetime.now(timezone.utc).isoformat()
                            st.download_button(
                            f"Download Top-{int(top_k_select)} (AI-ranked) as CSV",
                            top_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"top{int(top_k_select)}_ai_ranked.csv",
                            mime="text/csv")

                # ===== NO AI → just show the filtered table with blurbs =====
                else:
                    show_df = df.head(int(limit_results)) if limit_results else df
                    # blurbs only for what we show (kept small)
                    blurbs = ai_make_blurbs(show_df, OPENAI_API_KEY, model="gpt-4o-mini",
                                            max_items=min(150, int(limit_results or 150)))
                    show_df = show_df.copy()
                    show_df["blurb"] = show_df["notice_id"].astype(str).map(blurbs).fillna(show_df["title"].fillna(""))

                    st.session_state.sol_df = show_df
                    st.subheader(f"Solicitations ({len(show_df)})")

                    # Add normalized public SAM.gov URL
                    show_df = show_df.copy()
                    show_df["sam_url"] = show_df.apply(
                        lambda r: make_sam_public_url(str(r.get("notice_id","")), r.get("link","")),
                        axis=1
                    )

                    st.dataframe(_hide_notice_and_description(show_df), use_container_width=True)
                    st.download_button(
                        "Download filtered as CSV",
                        show_df.to_csv(index=False).encode("utf-8"),
                        file_name="sol_list.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.exception(e)

# ---- Tab 2
with tab2:
    st.header("This feature is in development...")
    # st.write("This uses your solicitation rows + Google results (via SerpAPI) to propose suppliers and rough quotes.")
    # our_rec = st.text_input("Favored suppliers (comma-separated)", value="")
    # our_not = st.text_input("Do-not-use suppliers (comma-separated)", value="")
    # max_google = st.number_input("Max Google results per item", min_value=1, max_value=20, value=5)

    # if st.button("Run supplier suggestion", type="primary"):
    #     if st.session_state.sol_df is None:
    #         st.error("Load or fetch solicitations in Tab 1 first.")
    #     else:
    #         sol_dicts = st.session_state.sol_df.to_dict(orient="records")
    #         favored = [x.strip() for x in our_rec.split(",") if x.strip()]
    #         not_favored = [x.strip() for x in our_not.split(",") if x.strip()]
    #         try:
    #             results = fs.get_suppliers(
    #                 solicitations=sol_dicts,
    #                 our_recommended_suppliers=favored,
    #                 our_not_recommended_suppliers=not_favored,
    #                 Max_Google_Results=int(max_google),
    #                 OpenAi_API_Key=OPENAI_API_KEY,
    #                 Serp_API_Key=SERP_API_KEY
    #             )
    #             sup_df = pd.DataFrame(results)
    #             st.session_state.sup_df = sup_df
    #             st.success(f"Generated {len(sup_df)} supplier rows.")
    #         except Exception as e:
    #             st.exception(e)

    # if st.session_state.sup_df is not None:
    #     st.subheader("Supplier suggestions")
    #     st.dataframe(st.session_state.sup_df, use_container_width=True)
    #     st.download_button(
    #         "Download as CSV",
    #         st.session_state.sup_df.to_csv(index=False).encode("utf-8"),
    #         file_name="supplier_suggestions.csv",
    #         mime="text/csv"
        # )

# ---- Tab 3
with tab3:
    st.header("This feature is in development...")
    # st.write("Select one or more supplier-suggestion rows and generate a proposal draft using your templates.")
    # bid_template = st.text_input("Bid template file path (DOCX or TXT)", value="/mnt/data/BID_TEMPLATE.docx")
    # solinfo_template = st.text_input("Solicitation info template (DOCX or TXT)", value="/mnt/data/SOLICITATION_INFO_TEMPLATE.docx")
    # out_dir = st.text_input("Output directory", value="/mnt/data/proposals")

    # uploaded_sup2 = st.file_uploader("Or upload supplier_suggestions.csv here", type=["csv"], key="sup_upload2")
    # if uploaded_sup2 is not None:
    #     try:
    #         df_upload = pd.read_csv(uploaded_sup2)
    #         st.session_state.sup_df = df_upload
    #         st.success(f"Loaded {len(df_upload)} supplier suggestions from upload.")
    #     except Exception as e:
    #         st.error(f"Failed to read CSV: {e}")

    # if st.session_state.sup_df is not None:
    #     st.dataframe(st.session_state.sup_df, use_container_width=True)
    #     idxs = st.multiselect(
    #         "Pick rows to draft",
    #         options=list(range(len(st.session_state.sup_df))),
    #         help="Leave empty to draft all"
    #     )
    #     if st.button("Generate proposal(s)", type="primary"):
    #         os.makedirs(out_dir, exist_ok=True)
    #         try:
    #             df_sel = st.session_state.sup_df.iloc[idxs] if idxs else st.session_state.sup_df
    #             gp.validate_supplier_and_write_proposal(
    #                 df=df_sel,
    #                 output_directory=out_dir,
    #                 Open_AI_API_Key=OPENAI_API_KEY,
    #                 BID_TEMPLATE_FILE=bid_template,
    #                 SOl_INFO_TEMPLATE=solinfo_template
    #             )
    #             st.success(f"Drafted proposals to {out_dir}.")
    #         except Exception as e:
    #             st.exception(e)
# ---- Tab 4
with tab4:
    st.header("Partner Matches (from AI-ranked results)")

    # Need AI-ranked results from Tab 1
    topn = st.session_state.get("topn_df")
    df_companies = companies_df()

    if topn is None or topn.empty:
        st.info("No AI-ranked results available. In Tab 1, run AI ranking to generate matches first.")
    elif df_companies.empty:
        st.info("Your company database is empty. Populate the 'company' table in Supabase with: name, description, city, state.")
    else:
        # Reuse company description from Tab 1 (stored there)
        company_desc_global = (st.session_state.get("company_desc") or "").strip()
        if not company_desc_global:
            st.info("No company description provided in Tab 1. Please enter one there and rerun.")
        else:
            # Auto-compute matches when Top-n changes or cache is empty
            need_recompute = (
                st.session_state.get("partner_matches") is None or
                st.session_state.get("partner_matches_stamp") != st.session_state.get("topn_stamp")
            )

            if need_recompute:
                with st.spinner("Analyzing gaps and selecting partners..."):
                    matches = []
                    for _, row in topn.iterrows():
                        title = str(row.get("title", "")) or "Untitled"
                        blurb = str(row.get("blurb", "")).strip()
                        desc  = str(row.get("description", "")) or ""
                        sol_text = f"{title}\n\n{desc}"

                        # 1) Identify our capability gaps for this solicitation
                        gaps = ai_identify_gaps(company_desc_global, sol_text, OPENAI_API_KEY)

                        # 2) Pick best partner from company DB to fill those gaps
                        best = pick_best_partner_for_gaps(gaps or sol_text, df_companies, OPENAI_API_KEY, top_n=1)
                        if best.empty:
                            matches.append({
                                "title": title,
                                "blurb": blurb,
                                "partner": None,
                                "gaps": gaps,
                                "ai": {"justification": "No suitable partner found.", "joint_proposal": ""}
                            })
                            continue

                        partner = best.iloc[0].to_dict()

                        # 3) Short justification + joint-proposal sketch (JSON-safe)
                        ai = ai_partner_justification(partner, sol_text, gaps, OPENAI_API_KEY)

                        matches.append({
                            "title": title,
                            "blurb": blurb,
                            "partner": partner,
                            "gaps": gaps,
                            "ai": ai
                        })

                # Cache results with a stamp tied to the Top-n
                st.session_state.partner_matches = matches
                st.session_state.partner_matches_stamp = st.session_state.get("topn_stamp")

            # Render cached matches
            matches = st.session_state.get("partner_matches", [])
            if not matches:
                st.info("No partner matches computed yet.")
            else:
                for m in matches:
                    hdr = (m.get("blurb") or m.get("title") or "Untitled").strip()
                    partner_name = (m.get("partner") or {}).get("name", "")
                    exp_title = f"Opportunity: {hdr}"
                    if partner_name:
                        exp_title += f" — Partner: {partner_name}"

                    with st.expander(exp_title):
                        # Partner block
                        if m.get("partner"):
                            p = m["partner"]
                            loc = ", ".join([x for x in [p.get("city",""), p.get("state","")] if x])
                            st.markdown("**Recommended Partner:**")
                            st.write(f"{p.get('name','')}" + (f" — {loc}" if loc else ""))
                        else:
                            st.warning("No suitable partner found for this opportunity.")

                        # Gaps
                        if m.get("gaps"):
                            st.markdown("**Why we need a partner (our capability gaps):**")
                            st.write(m["gaps"])

                        # Why this partner
                        just = (m.get("ai", {}) or {}).get("justification", "")
                        if just:
                            st.markdown("**Why this partner:**")
                            st.info(just)

                        # Joint proposal idea
                        jp = (m.get("ai", {}) or {}).get("joint_proposal", "").strip()
                        if jp:
                            st.markdown("**Targeted joint proposal idea:**")
                            st.write(jp)

# ---- Tab 5
with tab5:
    st.header("Internal Use")

    st.caption(
        "Quick presets that filter/rank solicitations with AI. "
        "Results appear below in relevance order with short blurbs."
    )

    # let internal users choose how many AI-ranked results they want
    internal_top_k = st.number_input(
        "How many AI-ranked matches?",
        min_value=1, max_value=50, value=5, step=1
    )

    # optionally let them cap how many DB rows to consider before AI (keeps it fast)
    max_candidates_cap = st.number_input(
        "Max candidates to consider before AI ranking",
        min_value=20, max_value=1000, value=300, step=20,
        help="We first pre-trim with embeddings, then rank with the LLM."
    )

    c1, c2 = st.columns(2)
    with c1:
        run_machine_shop = st.button("Solicitations for Machine Shop", type="primary", use_container_width=True, key="iu_btn_machine")
    with c2:
        run_services = st.button("Solicitations for Services", type="primary", use_container_width=True, key="iu_btn_services")

    def _ai_vendor_why(vendor_name: str, solicitation_title: str, solicitation_desc: str, api_key: str) -> str:
        """Fallback: 1-sentence reason why this vendor might fit."""
        try:
            client = OpenAI(api_key=api_key)
            sys = "You are a concise sourcing analyst. One sentence. No fluff."
            user = (
                f"Solicitation title:\n{solicitation_title}\n\n"
                f"Solicitation description:\n{(solicitation_desc or '')[:1500]}\n\n"
                f"Vendor: {vendor_name}\n\n"
                "In one short sentence, say why this vendor could likely do the work."
            )
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0.2,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _find_vendors_for_opportunity(sol: dict, max_google: int = 5, top_n: int = 3) -> pd.DataFrame:
        """
        1) Try your original helper (fs.get_suppliers)
        2) If it throws (e.g., 'Parser must be a string...' from dateutil), fallback to a direct SerpAPI query we control.
        Returns DataFrame with columns: name, website, location, reason
        """
        # ---- normalize all inputs to safe strings
        sol_norm = {
            "notice_id":     _s(sol.get("notice_id")),
            "title":         _s(sol.get("title")),
            "description":   _s(sol.get("description")),
            "naics_code":    _s(sol.get("naics_code")),
            "set_aside_code":_s(sol.get("set_aside_code")),
            "response_date": _s(sol.get("response_date")),
            "posted_date":   _s(sol.get("posted_date")),
            "link":          _s(sol.get("link")),
        }

        # guard: need some text to search
        if not sol_norm["title"] and not sol_norm["description"]:
            st.warning("This solicitation has no title/description text to search; skipping vendor lookup.")
            return pd.DataFrame(columns=["name","website","location","reason"])

        # ---- 1) TRY your original helper
        try:
            results = fs.get_suppliers(
                solicitations=[sol_norm],
                our_recommended_suppliers=[],
                our_not_recommended_suppliers=[],
                Max_Google_Results=int(max_google),
                OpenAi_API_Key=OPENAI_API_KEY,
                Serp_API_Key=SERP_API_KEY,
            )
            df = pd.DataFrame(results) if isinstance(results, (list, tuple)) else pd.DataFrame()
            if not df.empty:
                score_col = next((c for c in ["score","ai_score","relevance","confidence"] if c in df.columns), None)
                if score_col:
                    df = df.sort_values(score_col, ascending=False)

                def pick(d, *names, default=""):
                    for n in names:
                        if n in d and pd.notna(d[n]) and str(d[n]).strip():
                            return str(d[n]).strip()
                    return default

                cleaned = []
                for _, r in df.head(top_n).iterrows():
                    rd = r.to_dict()
                    name     = pick(rd, "supplier_name", "name", "vendor", "company")
                    website  = pick(rd, "website", "url", "link")
                    location = ", ".join([x for x in [
                        pick(rd, "city", "supplier_city", "location_city"),
                        pick(rd, "state", "supplier_state", "location_state"),
                    ] if x])
                    reason   = pick(rd, "reason", "why_matched", "justification", "notes")
                    if not reason and name:
                        reason = _ai_vendor_why(
                            vendor_name=name,
                            solicitation_title=sol_norm["title"],
                            solicitation_desc=sol_norm["description"],
                            api_key=OPENAI_API_KEY,
                        )
                    cleaned.append({"name": name, "website": website, "location": location, "reason": reason})
                if cleaned:
                    return pd.DataFrame(cleaned)

        except Exception as e:
            # NOTE: This is the path you're currently hitting
            st.info(f"Primary supplier finder failed; using direct SerpAPI fallback. ({e})")

        # ---- 2) FALLBACK: direct SerpAPI (no fragile date parsing)
        query_bits = [sol_norm["title"]]
        if sol_norm["naics_code"]:
            query_bits.append(f"NAICS {sol_norm['naics_code']}")
        # steer the search a bit toward vendors
        query_bits.append("manufacturer supplier vendor")
        q = " ".join([b for b in query_bits if b]).strip()

        try:
            raw = _fallback_serpapi_fetch(q, SERP_API_KEY, max_results=max(10, top_n*3))
        except Exception as e:
            st.warning(f"SerpAPI fallback failed: {e}")
            return pd.DataFrame(columns=["name","website","location","reason"])

        rows = []
        for it in raw[: max(10, top_n*3)]:  # look at a few; we’ll prune to top_n later
            website = it["link"]
            name = _companyish_name_from_result(it["title"], website)
            # 1-sentence reason
            reason = _ai_vendor_why(
                vendor_name=name,
                solicitation_title=sol_norm["title"],
                solicitation_desc=sol_norm["description"],
                api_key=OPENAI_API_KEY,
            )
            rows.append({"name": name, "website": website, "location": "", "reason": reason})

        # de-dupe by website host, keep first 3
        out, seen = [], set()
        for r in rows:
            h = _host(r["website"])
            if not h or h in seen:
                continue
            seen.add(h)
            out.append(r)
            if len(out) >= top_n:
                break

        return pd.DataFrame(out, columns=["name","website","location","reason"])

    def _compute_internal_results(preset_desc: str, negative_hint: str = "") -> dict | None:
        """Returns {"top_df": DataFrame, "reason_by_id": dict} or None on failure."""
        # pull everything; we'll let AI do the heavy lifting
        df_all = query_filtered_df({
            "keywords_or": [],
            "naics": [],
            "set_asides": [],
            "due_before": None,
            "notice_types": [],
        })
        if df_all.empty:
            st.warning("No solicitations in the database to evaluate.")
            return None

        company_desc_internal = preset_desc.strip()
        if negative_hint.strip():
            company_desc_internal += f"\n\nDo NOT include non-fits: {negative_hint.strip()}"

        pretrim_cap = min(int(max_candidates_cap), max(20, 12 * int(internal_top_k)))
        pretrim = ai_downselect_df(company_desc_internal, df_all, OPENAI_API_KEY, top_k=pretrim_cap)
        if pretrim.empty:
            st.info("AI pre-filter returned nothing.")
            return None

        ranked = ai_rank_solicitations_by_fit(
            df=pretrim,
            company_desc=company_desc_internal,
            api_key=OPENAI_API_KEY,
            top_k=int(internal_top_k),
            max_candidates=min(len(pretrim), 60),
            model="gpt-4o-mini",
        )
        if not ranked:
            st.info("AI ranking returned no results.")
            return None

        # Order per ranked list
        id_order = [x["notice_id"] for x in ranked]
        preorder = {nid: i for i, nid in enumerate(id_order)}
        top_df = pretrim[pretrim["notice_id"].astype(str).isin(id_order)].copy()
        top_df["__order"] = top_df["notice_id"].astype(str).map(preorder)
        top_df = (
            top_df.sort_values("__order")
                .drop_duplicates(subset=["notice_id"])
                .drop(columns="__order")
                .reset_index(drop=True)
        )

        # blurbs + fit_score
        blurbs = ai_make_blurbs(top_df, OPENAI_API_KEY, model="gpt-4o-mini", max_items=int(len(top_df)))
        top_df["blurb"] = top_df["notice_id"].astype(str).map(blurbs).fillna(top_df["title"].fillna(""))
        reason_by_id = {x["notice_id"]: x.get("reason", "") for x in ranked}
        score_by_id  = {x["notice_id"]: x.get("score", 0) for x in ranked}
        top_df["fit_score"] = top_df["notice_id"].astype(str).map(score_by_id).fillna(0).astype(float)

        return {"top_df": top_df, "reason_by_id": reason_by_id}

    def _render_internal_results():
        data = st.session_state.iu_results
        if not data:
            return
        key_salt = st.session_state.iu_key_salt or ""
        top_df = data["top_df"]
        reason_by_id = data["reason_by_id"]

        # Ensure one expander is open by default if none selected yet
        if st.session_state.iu_open_nid is None and len(top_df):
            st.session_state.iu_open_nid = str(top_df.iloc[0]["notice_id"])

        st.success(f"Top {len(top_df)} matches by relevance:")
        for idx, row in enumerate(top_df.itertuples(index=False), start=1):
            hdr = (getattr(row, "blurb", None) or getattr(row, "title", None) or "Untitled")
            nid = str(getattr(row, "notice_id", ""))

            # no key= on expander; keep it open between reruns by NOT toggling a state var here
            expanded = (st.session_state.get("iu_open_nid") == nid)
            with st.expander(f"{idx}. {hdr}", expanded=expanded):
                left, right = st.columns([2, 1])

                with left:
                    st.write(f"**Notice Type:** {getattr(row, 'notice_type', '')}")
                    st.write(f"**Posted:** {getattr(row, 'posted_date', '')}")
                    st.write(f"**Response Due:** {getattr(row, 'response_date', '')}")
                    st.write(f"**NAICS:** {getattr(row, 'naics_code', '')}")
                    st.write(f"**Set-aside:** {getattr(row, 'set_aside_code', '')}")
                    link = make_sam_public_url(str(getattr(row, 'notice_id', '')), getattr(row, 'link', ''))
                    st.write(f"[Open on SAM.gov]({link})")

                    reason = reason_by_id.get(nid, "")
                    if reason:
                        st.markdown("**Why this matched (AI):**")
                        st.info(reason)

                    # vendor button: unique per notice + run salt
                    btn_key = f"iu_find_vendors_{nid}_{idx}_{key_salt}"
                    if st.button("Find 3 potential vendors (SerpAPI)", key=btn_key):
                        sol_dict = {
                            "notice_id":     _s(nid),
                            "title":         _s(getattr(row, "title", "")),
                            "description":   _s(getattr(row, "description", "")),
                            "naics_code":    _s(getattr(row, "naics_code", "")),
                            "set_aside_code":_s(getattr(row, "set_aside_code", "")),
                            "response_date": _s(getattr(row, "response_date", "")),
                            "posted_date":   _s(getattr(row, "posted_date", "")),
                            "link":          _s(getattr(row, "link", "")),
}
                        vendors_df = _find_vendors_for_opportunity(sol_dict, max_google=5, top_n=3)
                        st.session_state.vendor_suggestions[nid] = vendors_df  # cache per-notice
                        st.session_state.iu_open_nid = nid
                        st.rerun()  # ensure right column updates immediately

                with right:
                    vend_df = st.session_state.vendor_suggestions.get(nid)
                    if isinstance(vend_df, pd.DataFrame) and not vend_df.empty:
                        st.markdown("**Vendor candidates**")
                        for j, v in vend_df.iterrows():
                            name = (v.get("name") or "").strip() or "Unnamed vendor"
                            website = (v.get("website") or "").strip()
                            location = (v.get("location") or "").strip()
                            reason_txt = (v.get("reason") or "").strip()

                            if website:
                                st.markdown(f"- **[{name}]({website})**")
                            else:
                                st.markdown(f"- **{name}**")

                            if location:
                                st.caption(location)
                            if reason_txt:
                                st.write(reason_txt)
                    else:
                        st.caption("No vendors yet. Click the button to fetch.")
   # Run the chosen preset
if run_machine_shop:
    st.session_state.iu_key_salt = uuid.uuid4().hex  # new salt for this run
    preset_desc = (
        "We are pursuing solicitations where a MACHINE SHOP would fabricate or machine parts for us. "
        "Strong fits include CNC machining, milling, turning, drilling, precision tolerances, "
        "metal or plastic fabrication, weldments, assemblies, and production of custom components per drawings. "
        "Prefer solicitations with part drawings, specs, materials (e.g., aluminum, steel, titanium), "
        "and tangible manufactured items."
    )
    negative_hint = (
        "Pure services, staffing-only, software-only, consulting, training, janitorial, IT, "
        "or anything that does not involve fabricating or machining a physical part."
    )
    with st.spinner("Finding best-matching solicitations..."):
        data = _compute_internal_results(preset_desc, negative_hint)
    st.session_state.iu_results = data
    st.rerun()

if run_services:
    st.session_state.iu_key_salt = uuid.uuid4().hex  # new salt for this run
    preset_desc = (
        "We are pursuing solicitations where a SERVICES COMPANY performs the work for us. "
        "Strong fits include maintenance, installation, inspection, logistics, training, field services, "
        "operations support, professional services, and other labor-based or outcome-based services "
        "delivered under SOW/Performance Work Statement."
    )
    negative_hint = "Manufacturing-only or pure product buys without a material services component."
    with st.spinner("Finding best-matching solicitations..."):
        data = _compute_internal_results(preset_desc, negative_hint)
    st.session_state.iu_results = data
    st.rerun()

# Always render cached results (so lists persist across reruns, including vendor-button clicks)
_render_internal_results()

# If you want: export download for cached results
if st.session_state.iu_results and isinstance(st.session_state.iu_results.get("top_df"), pd.DataFrame):
    top_df = st.session_state.iu_results["top_df"]
    st.download_button(
        f"Download Internal Use Results (Top-{int(internal_top_k)})",
        top_df.to_csv(index=False).encode("utf-8"),
        file_name=f"internal_top{int(internal_top_k)}.csv",
        mime="text/csv",
    )
st.markdown("---")
st.caption("DB schema is fixed to only the required SAM fields. Refresh inserts brand-new notices only (no updates).")