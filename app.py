# app.py - Cleaned and Organized Version
from typing import List, Dict, Any
import os
import re
import json
import bcrypt
import uuid
import hashlib
import warnings
import csv
import gc
from functools import partial
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import date, datetime, timezone, timedelta
from dataclasses import dataclass
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text, inspect
from sqlmodel import SQLModel, Field, create_engine
from sqlalchemy.exc import SAWarning
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from openai import OpenAI
import requests

# Local imports
import find_relevant_suppliers as fs
import generate_proposal as gp
import get_relevant_solicitations as gs
import secrets as pysecrets

# =========================
# Configuration & Secrets
# =========================
warnings.filterwarnings(
    "ignore", message="This declarative base already contains a class with the same class name", category=SAWarning)
SQLModel.metadata.clear()

st.set_page_config(page_title="KIP", layout="wide")


def get_secret(name, default=None):
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name, default)


# Load secrets
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
SERP_API_KEY = get_secret("SERP_API_KEY")
SAM_KEYS = get_secret("SAM_KEYS", [])

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
# Database Configuration
# =========================
DB_URL = st.secrets.get("SUPABASE_DB_URL") or "sqlite:///app.db"


@st.cache_resource
def get_optimized_engine(db_url: str):
    """Create optimized database engine with connection pooling"""
    if db_url.startswith("postgresql"):
        return create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=5,
            pool_recycle=3600,
            connect_args={
                "sslmode": "require",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
                "application_name": "kip_streamlit"
            },
        )
    else:
        return create_engine(db_url, pool_pre_ping=True)


engine = get_optimized_engine(DB_URL)

# Test connection
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
# Cookie Manager & Session
# =========================
cookies = EncryptedCookieManager(
    prefix="kip_",
    password=get_secret("COOKIE_PASSWORD", "dev-cookie-secret")
)
if not cookies.ready():
    st.stop()

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "view" not in st.session_state:
    st.session_state.view = "main" if st.session_state.user else "auth"
if "vendor_notes" not in st.session_state:
    st.session_state.vendor_notes = {}
if "sol_df" not in st.session_state:
    st.session_state.sol_df = None
if "sup_df" not in st.session_state:
    st.session_state.sup_df = None

# =========================
# Utility Functions
# =========================


def _stringify(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


def _s(v) -> str:
    """Return a safe string for downstream parsers (handles None/NaN/NaT)."""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return "" if v is None else str(v)


def _host(u: str) -> str:
    try:
        h = urlparse(u).netloc.lower()
        for p in ("www.", "m.", "en.", "amp."):
            if h.startswith(p):
                h = h[len(p):]
        return h
    except Exception:
        return ""


def normalize_naics_input(text_in: str) -> list[str]:
    if not text_in:
        return []
    values = re.split(r"[,\s]+", text_in.strip())
    return [v for v in (re.sub(r"[^\d]", "", x) for x in values) if v]


def parse_keywords_or(text_in: str) -> list[str]:
    return [k.strip() for k in text_in.split(",") if k.strip()]


def make_sam_public_url(notice_id: str, link: str | None = None) -> str:
    """Return a human-viewable SAM.gov URL for this notice."""
    if link and isinstance(link, str) and "api.sam.gov" not in link:
        return link
    nid = (notice_id or "").strip()
    return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"

# =========================
# Database Optimization
# =========================


def optimize_database():
    """Add indexes and optimize database for faster queries"""
    try:
        with engine.begin() as conn:
            conn.execute(sa.text("""
                CREATE INDEX IF NOT EXISTS idx_sol_posted_naics 
                ON solicitationraw (posted_date DESC, naics_code)
            """))
            conn.execute(sa.text("""
                CREATE INDEX IF NOT EXISTS idx_sol_response_date 
                ON solicitationraw (response_date) 
                WHERE response_date IS NOT NULL AND response_date != 'None'
            """))
            conn.execute(sa.text("""
                CREATE INDEX IF NOT EXISTS idx_sol_notice_type 
                ON solicitationraw (notice_type) 
                WHERE notice_type IS NOT NULL
            """))
            if engine.url.get_dialect().name == 'postgresql':
                conn.execute(sa.text("""
                    CREATE INDEX IF NOT EXISTS idx_sol_fulltext 
                    ON solicitationraw USING gin(to_tsvector('english', title || ' ' || description))
                """))
    except Exception as e:
        st.warning(f"Database optimization note: {e}")


# Call optimization once per session
if "db_optimized" not in st.session_state:
    check_database_optimization()
    st.session_state.db_optimized = True

    # Optional admin tools in sidebar
    with st.sidebar.expander("⚙️ Admin Tools", expanded=False):
        if st.button("🔧 Run Database Optimization"):
            with st.spinner("Creating database indexes..."):
                optimize_database()
            st.rerun()

# =========================
# Database Schema & Migration
# =========================
# Create auth tokens table
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

# Create unique indexes
try:
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON users (email);"))
        conn.execute(sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_company_profile_user ON company_profile (user_id);"))
except Exception as e:
    st.warning(f"User/profile table migration note: {e}")

# Create tables
if "db_initialized" not in st.session_state:
    try:
        SQLModel.metadata.create_all(engine)
    finally:
        st.session_state["db_initialized"] = True

# Migrate solicitation columns
REQUIRED_COLS = {
    "pulled_at": "TEXT", "notice_id": "TEXT", "solicitation_number": "TEXT",
    "title": "TEXT", "notice_type": "TEXT", "posted_date": "TEXT",
    "response_date": "TEXT", "archive_date": "TEXT", "naics_code": "TEXT",
    "set_aside_code": "TEXT", "description": "TEXT", "link": "TEXT",
    "pop_city": "TEXT", "pop_state": "TEXT", "pop_country": "TEXT",
    "pop_zip": "TEXT", "pop_raw": "TEXT",
}

try:
    insp = inspect(engine)
    existing_cols = {c["name"] for c in insp.get_columns("solicitationraw")}
    missing_cols = [c for c in REQUIRED_COLS if c not in existing_cols]

    if missing_cols:
        with engine.begin() as conn:
            for col in missing_cols:
                conn.execute(
                    sa.text(f'ALTER TABLE solicitationraw ADD COLUMN "{col}" {REQUIRED_COLS[col]}'))

    with engine.begin() as conn:
        conn.execute(sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_solicitationraw_notice_id ON solicitationraw (notice_id)"))
except Exception as e:
    st.warning(f"Migration note: {e}")

# =========================
# Authentication Functions
# =========================


def _hash_password(pw: str) -> str:
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pw.encode("utf-8"), salt).decode("utf-8")


def _check_password(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8"), pw_hash.encode("utf-8"))
    except Exception:
        return False


def _hash_token(raw: str) -> str:
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
        """), {"uid": user_id, "th": tok_hash, "exp": exp.isoformat(), "now": now.isoformat()})
    return raw


def _validate_remember_me_token(raw: str) -> Optional[int]:
    if not raw:
        return None
    tok_hash = _hash_token(raw)
    with engine.connect() as conn:
        row = conn.execute(sa.text("""
            SELECT user_id, expires_at FROM auth_tokens WHERE token_hash = :th
            ORDER BY created_at DESC LIMIT 1
        """), {"th": tok_hash}).mappings().first()
    if not row:
        return None
    try:
        exp = datetime.fromisoformat(row["expires_at"])
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp < datetime.now(timezone.utc):
            return None
    except Exception:
        return None
    return int(row["user_id"])


def _revoke_all_tokens_for_user(user_id: int) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text("DELETE FROM auth_tokens WHERE user_id = :uid"), {
                     "uid": user_id})


def get_user_by_email(email: str):
    with engine.connect() as conn:
        sql = sa.text(
            "SELECT id, email, password_hash FROM users WHERE email = :e")
        row = conn.execute(
            sql, {"e": email.strip().lower()}).mappings().first()
        return dict(row) if row else None


def create_user(email: str, password: str) -> Optional[int]:
    email = email.strip().lower()
    pw_hash = _hash_password(password)
    with engine.begin() as conn:
        try:
            sql = sa.text("""
                INSERT INTO users (email, password_hash, created_at)
                VALUES (:email, :ph, :ts) RETURNING id
            """)
            new_id = conn.execute(sql, {"email": email, "ph": pw_hash, "ts": datetime.now(
                timezone.utc).isoformat()}).scalar_one()
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


# Auto-login from remember-me cookie
if st.session_state.user is None:
    raw_cookie = cookies.get(get_secret("COOKIE_NAME", "kip_auth"))
    uid = _validate_remember_me_token(raw_cookie) if raw_cookie else None
    if uid:
        with engine.connect() as conn:
            row = conn.execute(sa.text("SELECT id, email FROM users WHERE id = :uid"), {
                               "uid": uid}).mappings().first()
        if row:
            st.session_state.user = {"id": row["id"], "email": row["email"]}
            st.session_state.profile = get_profile(row["id"])
            st.session_state.view = "main"

# =========================
# AI & Embedding Functions
# =========================


def _embed_texts(texts: list[str], api_key: str) -> np.ndarray:
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(
        model="text-embedding-3-small", input=texts)
    X = np.array([d.embedding for d in resp.data], dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)  # L2 normalize
    return X


@st.cache_data(show_spinner=False, ttl=3600)
def cached_company_embeddings(_companies: pd.DataFrame, api_key: str) -> dict:
    """Returns {"df": companies_df, "X": normalized embeddings (np.ndarray)}."""
    if _companies.empty:
        return {"df": _companies, "X": np.zeros((0, 1536), dtype=np.float32)}
    texts = _companies["description"].fillna("").astype(str).tolist()
    X = _embed_texts(texts, api_key)
    return {"df": _companies.copy(), "X": X}


def ai_downselect_df(company_desc: str, df: pd.DataFrame, api_key: str,
                     threshold: float = 0.20, top_k: int | None = None) -> pd.DataFrame:
    """Embedding-based similarity between company_desc and (title + description)."""
    if df.empty:
        return df

    texts = (df["title"].fillna("") + " " +
             df["description"].fillna("")).str.slice(0, 2000).tolist()
    try:
        client = OpenAI(api_key=api_key)
        q = client.embeddings.create(
            model="text-embedding-3-small", input=[company_desc])
        Xq = np.array(q.data[0].embedding, dtype=np.float32)

        X_list = []
        batch_size = 500
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            r = client.embeddings.create(
                model="text-embedding-3-small", input=batch)
            X_list.extend([d.embedding for d in r.data])
        X = np.array(X_list, dtype=np.float32)

        Xq_norm = Xq / (np.linalg.norm(Xq) + 1e-9)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        sims = X_norm @ Xq_norm

        df = df.copy()
        df["ai_score"] = sims

        if top_k is not None and top_k > 0:
            df = df.sort_values("ai_score", ascending=False).head(int(top_k))
        else:
            df = df[df["ai_score"] >= float(threshold)].sort_values(
                "ai_score", ascending=False)

        return df.reset_index(drop=True)

    except Exception as e:
        st.warning(
            f"AI downselect unavailable ({e}). Falling back to keyword filter.")
        kws = [w.lower() for w in re.findall(r"[a-zA-Z0-9]{4,}", company_desc)]
        if not kws:
            return df
        blob = (df["title"].fillna("") + " " +
                df["description"].fillna("")).str.lower()
        mask = blob.apply(lambda t: any(k in t for k in kws))
        return df[mask].reset_index(drop=True)


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


def ai_make_blurbs_fast(df: pd.DataFrame, api_key: str, model: str = "gpt-4o-mini",
                        max_items: int = 50, chunk_size: int = 20, timeout_seconds: int = 30) -> dict[str, str]:
    """Faster version with smaller batches and timeout"""
    if df is None or df.empty:
        return {}

    items = []
    for _, r in df.head(max_items).iterrows():
        items.append({
            "notice_id": str(r.get("notice_id", "")),
            "title": (r.get("title") or "")[:150],
            "description": (r.get("description") or "")[:400],
        })

    client = OpenAI(api_key=api_key, timeout=timeout_seconds)
    out = {}
    system_msg = "Write 6-8 word summaries of what each solicitation needs. Be extremely concise and skip agency names."

    for batch in _chunk(items, chunk_size):
        user_msg = {
            "items": batch, "format": 'Return JSON: {"blurbs":[{"notice_id":"...","blurb":"..."}]}'}
        try:
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": json.dumps(user_msg)},
                ],
                temperature=0.1,
                max_tokens=1000,
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            for row in data.get("blurbs", []):
                nid = str(row.get("notice_id", "")).strip()
                blurb = (row.get("blurb") or "").strip()
                if nid and blurb:
                    out[nid] = blurb
        except Exception:
            continue
    return out

# =========================
# AI Matrix Scorer
# =========================


# Enhanced AI Matrix Scorer - Replace the existing AIMatrixScorer class in app.py


@dataclass
class MatrixComponent:
    key: str
    label: str
    weight: float
    description: str
    hints: list[str]
    scoring_method: str = "llm_assessment"


# Enhanced AI Matrix Scorer - Replace the existing AIMatrixScorer class in app.py
# Complete replacement for your AI Matrix Scorer section in app.py
# Replace everything from the @dataclass MatrixComponent line through the ai_score_and_rank_solicitations_by_fit function

from dataclasses import dataclass
from typing import List, Dict, Any
import json
import streamlit as st
from openai import OpenAI

@dataclass
class MatrixComponent:
    key: str
    label: str
    weight: float
    description: str
    hints: list[str]
    scoring_method: str = "llm_assessment"


class AIMatrixScorer:
    """Enhanced LLM-based scorer using complete scoring matrix with 1-10 scale per component"""

    def __init__(self):
        self.components: list[MatrixComponent] = [
            # Technical Capability (40% total)
            MatrixComponent(
                key="tech_core",
                label="Core Services & Capabilities",
                weight=25.0,
                description="How well does the company's primary services and capabilities align with what the solicitation requires?",
                hints=['manufacturing', 'engineering', 'software', 'consulting', 'maintenance',
                       'installation', 'repair', 'testing', 'inspection', 'training'],
                scoring_method="keyword_match_weighted"
            ),
            MatrixComponent(
                key="tech_industry",
                label="Industry Domain Expertise",
                weight=20.0,
                description="Does the company have relevant industry domain knowledge and experience for this type of work?",
                hints=['aerospace', 'defense', 'medical', 'automotive', 'energy',
                       'construction', 'IT', 'cybersecurity', 'telecommunications', 'logistics'],
                scoring_method="keyword_match_weighted"
            ),
            MatrixComponent(
                key="tech_standards",
                label="Technical Standards & Certifications",
                weight=15.0,
                description="Does the company demonstrate compliance with relevant technical standards, certifications, and quality systems?",
                hints=['ISO', 'CMMI', 'FedRAMP', 'NIST', 'ANSI', 'DOD',
                       'FDA', 'FAA', 'security clearance', 'quality management'],
                scoring_method="keyword_match_binary"
            ),

            # Business Qualifications (20% total)
            MatrixComponent(
                key="biz_size",
                label="Business Size & Set-Aside Eligibility",
                weight=10.0,
                description="Does the company qualify for relevant set-aside categories or business size requirements?",
                hints=['small business', '8A', 'WOSB', 'SDVOSB',
                       'HUBZone', 'SDB', 'minority owned', 'veteran owned'],
                scoring_method="set_aside_alignment"
            ),
            MatrixComponent(
                key="biz_performance",
                label="Government Contracting Experience",
                weight=10.0,
                description="Does the company have relevant past performance with government contracts and federal agencies?",
                hints=['GSA', 'contract', 'federal', 'government',
                       'prime contractor', 'subcontractor', 'SEWP', 'CIO-SP3', 'OASIS'],
                scoring_method="keyword_match_weighted"
            ),

            # Geographic & NAICS (15% total)
            MatrixComponent(
                key="geo_location",
                label="Geographic Location Match",
                weight=8.0,
                description="How well does the company's location align with the place of performance or geographic preferences?",
                hints=['state', 'city', 'region',
                       'nationwide', 'remote', 'on-site'],
                scoring_method="location_proximity"
            ),
            MatrixComponent(
                key="naics_alignment",
                label="NAICS Code Alignment",
                weight=7.0,
                description="How well does the company's business align with the solicitation's NAICS code requirements?",
                hints=['NAICS'],
                scoring_method="naics_match"
            ),

            # Financial & Innovation (5% total)
            MatrixComponent(
                key="financial_capacity",
                label="Financial Capacity",
                weight=3.0,
                description="Does the company appear to have sufficient financial strength and capacity for this contract size?",
                hints=['revenue', 'capacity', 'bonding', 'financial strength'],
                scoring_method="financial_capacity"
            ),
            MatrixComponent(
                key="innovation",
                label="Technology Innovation",
                weight=2.0,
                description="Does the company demonstrate advanced technology capabilities or innovation relevant to the solicitation?",
                hints=['AI', 'machine learning', 'automation', 'IoT',
                       'cloud', 'digital transformation', 'emerging technology'],
                scoring_method="keyword_match_binary"
            ),
        ]
        self.total_weight = sum(c.weight for c in self.components)

    def _prompt_for_batch(self, company_profile: dict, items: list[dict]) -> tuple[list[dict], list[dict]]:
        company_view = {
            "company_name": (company_profile.get("company_name") or "").strip(),
            "description": (company_profile.get("description") or "").strip(),
            "city": (company_profile.get("city") or "").strip(),
            "state": (company_profile.get("state") or "").strip(),
        }

        system = (
            "You are a federal contracting analyst. Score each solicitation on ALL 9 components (1-10 scale).\n\n"
            "Components to score:\n"
            "1. tech_core (25%): Core services alignment\n"
            "2. tech_industry (20%): Industry expertise\n"
            "3. tech_standards (15%): Technical standards\n"
            "4. biz_size (10%): Business size fit\n"
            "5. biz_performance (10%): Government experience\n"
            "6. geo_location (8%): Geographic alignment\n"
            "7. naics_alignment (7%): NAICS code match\n"
            "8. financial_capacity (3%): Financial strength\n"
            "9. innovation (2%): Technology innovation\n\n"
            "Return ONLY this exact JSON format:\n"
            '{"results":[{"notice_id":"ABC123","components":[{"key":"tech_core","score":8,"reason":"good alignment"},{"key":"tech_industry","score":7,"reason":"related field"},...all 9 components...],"total_score":75}]}\n\n'
            "Score 1-10 for each component. Keep reasons under 8 words. Include ALL 9 components for each solicitation."
        )

        user_data = {
            "company_description": company_view["description"][:300],
            "company_location": f"{company_view['city']} {company_view['state']}".strip(),
            "solicitations": [{
                "notice_id": str(x.get("notice_id", "")),
                "title": (x.get("title") or "")[:200],
                "description": (x.get("description") or "")[:600],
                "naics_code": str(x.get("naics_code") or ""),
                "set_aside_code": str(x.get("set_aside_code") or ""),
                "location": f"{x.get('pop_city', '')} {x.get('pop_state', '')}".strip()
            } for x in items]
        }

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_data)}
        ]
        return messages, []

    def score_batch(self, items: list[dict], company_profile: dict, api_key: str, model: str = "gpt-4o-mini") -> dict:
        """Enhanced scoring that returns detailed component breakdown"""
        if not items:
            return {}

        client = OpenAI(api_key=api_key)
        messages, _ = self._prompt_for_batch(company_profile, items)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=2000,  # Increased for detailed breakdown
                timeout=45
            )
            content = response.choices[0].message.content or "{}"

            # Clean and parse JSON
            content = content.strip()
            if not content.startswith('{'):
                start = content.find('{')
                end = content.rfind('}') + 1
                if start >= 0 and end > start:
                    content = content[start:end]

            # Fix common JSON issues
            content = self._fix_json_issues(content)
            data = json.loads(content)

            # Process detailed results
            scored_results = {}
            for result in (data.get("results") or []):
                notice_id = str(result.get("notice_id", "")).strip()
                if not notice_id:
                    continue

                components = result.get("components", [])
                if not components:
                    # Fallback to simple scoring if no components
                    score = float(result.get("total_score", 50))
                    final_score = max(0.0, min(100.0, score))
                    breakdown = [{
                        "key": "overall_fit",
                        "label": "Overall Company Fit",
                        "score": int(score/10),
                        "reasoning": result.get("reason", "AI assessment"),
                        "weight": 100.0,
                        "weighted_contribution": final_score
                    }]
                else:
                    # Process detailed component breakdown
                    breakdown = []
                    total_weighted = 0

                    for comp in components:
                        comp_key = comp.get("key", "unknown")
                        comp_score = max(1, min(10, int(comp.get("score", 5))))
                        comp_reason = comp.get("reason", "No reason provided")

                        # Find the weight for this component
                        comp_weight = 5.0  # default
                        for matrix_comp in self.components:
                            if matrix_comp.key == comp_key:
                                comp_weight = matrix_comp.weight
                                break

                        # Calculate weighted contribution
                        weighted_contribution = (
                            comp_score * comp_weight / self.total_weight) * 10
                        total_weighted += weighted_contribution

                        breakdown.append({
                            "key": comp_key,
                            "label": self._get_component_label(comp_key),
                            "score": comp_score,
                            "reasoning": comp_reason,
                            "weight": comp_weight,
                            "weighted_contribution": weighted_contribution
                        })

                    final_score = max(0.0, min(100.0, total_weighted))

                scored_results[notice_id] = {
                    "score": final_score,
                    "breakdown": breakdown
                }

            return scored_results

        except Exception as e:
            st.warning(
                f"AI scoring failed, using detailed fallback: {str(e)[:100]}...")
            return self._detailed_fallback_scoring(items, company_profile)

    def _get_component_label(self, key: str) -> str:
        """Get human-readable label for component key"""
        labels = {
            "tech_core": "Core Services & Capabilities",
            "tech_industry": "Industry Domain Expertise",
            "tech_standards": "Technical Standards & Certifications",
            "biz_size": "Business Size & Set-Aside Eligibility",
            "biz_performance": "Government Contracting Experience",
            "geo_location": "Geographic Location Match",
            "naics_alignment": "NAICS Code Alignment",
            "financial_capacity": "Financial Capacity",
            "innovation": "Technology Innovation"
        }
        return labels.get(key, key.replace("_", " ").title())

    def _detailed_fallback_scoring(self, items: list[dict], company_profile: dict) -> dict:
        """Detailed fallback scoring with component breakdown when AI fails"""
        company_desc = (company_profile.get("description") or "").lower()
        company_city = (company_profile.get("city") or "").lower()
        company_state = (company_profile.get("state") or "").upper()

        fallback_results = {}

        for item in items:
            notice_id = str(item.get("notice_id", ""))
            title = (item.get("title") or "").lower()
            description = (item.get("description") or "").lower()
            naics = str(item.get("naics_code") or "")
            set_aside = (item.get("set_aside_code") or "").lower()
            pop_city = (item.get("pop_city") or "").lower()
            pop_state = (item.get("pop_state") or "").upper()

            # Score each component with heuristics
            breakdown = []

            # Core Services (25% weight)
            core_keywords = ['manufacturing', 'engineering', 'software',
                             'consulting', 'maintenance', 'installation', 'repair', 'testing']
            core_score = 5  # default
            if any(kw in company_desc for kw in core_keywords):
                if any(kw in title + " " + description for kw in core_keywords):
                    core_score = 8
                else:
                    core_score = 6
            breakdown.append({
                "key": "tech_core",
                "label": "Core Services & Capabilities",
                "score": core_score,
                "reasoning": "Keyword matching heuristic",
                "weight": 25.0,
                "weighted_contribution": (core_score * 25.0 / self.total_weight) * 10
            })

            # Industry Expertise (20% weight)
            industry_keywords = ['aerospace', 'defense', 'medical',
                                 'automotive', 'energy', 'construction', 'IT']
            industry_score = 5
            if any(kw in company_desc for kw in industry_keywords):
                if any(kw in title + " " + description for kw in industry_keywords):
                    industry_score = 7
            breakdown.append({
                "key": "tech_industry",
                "label": "Industry Domain Expertise",
                "score": industry_score,
                "reasoning": "Industry keyword matching",
                "weight": 20.0,
                "weighted_contribution": (industry_score * 20.0 / self.total_weight) * 10
            })

            # Technical Standards (15% weight)
            standards_keywords = ['iso', 'cmmi', 'nist',
                                  'ansi', 'certification', 'quality']
            standards_score = 5
            if any(kw in company_desc for kw in standards_keywords) or any(kw in title + " " + description for kw in standards_keywords):
                standards_score = 7
            breakdown.append({
                "key": "tech_standards",
                "label": "Technical Standards & Certifications",
                "score": standards_score,
                "reasoning": "Standards keyword detection",
                "weight": 15.0,
                "weighted_contribution": (standards_score * 15.0 / self.total_weight) * 10
            })

            # Business Size (10% weight)
            size_score = 6  # neutral default
            if set_aside and any(sa in set_aside for sa in ['sba', '8a', 'wosb', 'sdvosb', 'hubzone']):
                size_score = 8  # Good if set-aside matches
            breakdown.append({
                "key": "biz_size",
                "label": "Business Size & Set-Aside Eligibility",
                "score": size_score,
                "reasoning": "Set-aside code evaluation",
                "weight": 10.0,
                "weighted_contribution": (size_score * 10.0 / self.total_weight) * 10
            })

            # Gov Experience (10% weight)
            gov_keywords = ['contract', 'federal',
                            'government', 'gsa', 'prime']
            gov_score = 5
            if any(kw in company_desc for kw in gov_keywords):
                gov_score = 7
            breakdown.append({
                "key": "biz_performance",
                "label": "Government Contracting Experience",
                "score": gov_score,
                "reasoning": "Government experience keywords",
                "weight": 10.0,
                "weighted_contribution": (gov_score * 10.0 / self.total_weight) * 10
            })

            # Geographic Match (8% weight)
            geo_score = 5  # default neutral
            if pop_state == company_state:
                geo_score = 8
            elif pop_state and company_state:
                geo_score = 4  # Different states
            breakdown.append({
                "key": "geo_location",
                "label": "Geographic Location Match",
                "score": geo_score,
                "reasoning": f"Location comparison: {pop_state} vs {company_state}",
                "weight": 8.0,
                "weighted_contribution": (geo_score * 8.0 / self.total_weight) * 10
            })

            # NAICS Alignment (7% weight)
            naics_score = 6  # neutral default
            breakdown.append({
                "key": "naics_alignment",
                "label": "NAICS Code Alignment",
                "score": naics_score,
                "reasoning": f"NAICS: {naics}",
                "weight": 7.0,
                "weighted_contribution": (naics_score * 7.0 / self.total_weight) * 10
            })

            # Financial Capacity (3% weight)
            financial_score = 6  # neutral default
            breakdown.append({
                "key": "financial_capacity",
                "label": "Financial Capacity",
                "score": financial_score,
                "reasoning": "Default assessment",
                "weight": 3.0,
                "weighted_contribution": (financial_score * 3.0 / self.total_weight) * 10
            })

            # Innovation (2% weight)
            innovation_keywords = ['ai', 'machine learning',
                                   'automation', 'iot', 'cloud', 'digital']
            innovation_score = 5
            if any(kw in company_desc for kw in innovation_keywords) or any(kw in title + " " + description for kw in innovation_keywords):
                innovation_score = 7
            breakdown.append({
                "key": "innovation",
                "label": "Technology Innovation",
                "score": innovation_score,
                "reasoning": "Innovation keyword detection",
                "weight": 2.0,
                "weighted_contribution": (innovation_score * 2.0 / self.total_weight) * 10
            })

            # Calculate final score
            total_weighted = sum(comp["weighted_contribution"]
                                 for comp in breakdown)
            final_score = max(0.0, min(100.0, total_weighted))

            fallback_results[notice_id] = {
                "score": final_score,
                "breakdown": breakdown
            }

        return fallback_results

    def _fix_json_issues(self, content: str) -> str:
        """Attempt to fix common JSON formatting issues"""
        import re

        # Remove trailing commas before closing braces/brackets
        content = re.sub(r',(\s*[}\]])', r'\1', content)

        # Basic quote escaping in reasoning fields
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            if '"reason"' in line and line.count('"') > 4:
                # Try to fix unescaped quotes in reason field
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key_part = parts[0]
                        value_part = parts[1].strip()
                        if value_part.startswith('"') and value_part.endswith('"') and value_part.count('"') > 2:
                            # Replace internal quotes with escaped quotes
                            # Remove outer quotes
                            inner_content = value_part[1:-1]
                            inner_content = inner_content.replace(
                                '"', '\\"')  # Escape internal quotes
                            value_part = f'"{inner_content}"'
                            line = key_part + ': ' + value_part
            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

def render_enhanced_score_results(ranked_results: List[Dict]):
    """Enhanced results display with detailed component scoring"""
    if not ranked_results:
        st.info("No results to display")
        return

    st.write(f"**Found {len(ranked_results)} ranked matches**")

    for idx, item in enumerate(ranked_results):
        nid = str(item.get('notice_id', ''))
        title = item.get('title', 'Untitled')
        score = float(item.get('score', 0.0))
        blurb = (item.get('blurb') or "").strip()

        # Color-code the score
        if score >= 80:
            score_color = "🟢"
        elif score >= 60:
            score_color = "🟡"
        else:
            score_color = "🔴"

        header = f"#{idx+1}: {title} — {score_color} Score: {score:.1f}/100"
        with st.expander(header, expanded=(idx == 0)):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Notice ID:** {nid}")
                if blurb:
                    st.write(f"**Summary:** {blurb}")
                st.markdown(f"**[View on SAM.gov]({item.get('link','')})**")

            with col2:
                # Overall score with color coding
                if score >= 80:
                    st.success(f"Overall Fit Score: {score:.1f}/100")
                elif score >= 60:
                    st.warning(f"Overall Fit Score: {score:.1f}/100")
                else:
                    st.error(f"Overall Fit Score: {score:.1f}/100")

            # Detailed component breakdown
            st.subheader("📊 Detailed Scoring Breakdown")
            breakdown = item.get("breakdown") or []

            if breakdown:
                # Create a more visual breakdown
                for comp in breakdown:
                    score_val = comp.get('score', 0)
                    label = comp.get('label', comp.get('key', 'Unknown'))
                    reasoning = comp.get('reasoning', 'No reasoning provided')
                    weight = comp.get('weight', 0)
                    contribution = comp.get('weighted_contribution', 0)

                    # Score bar visualization
                    score_bar = "█" * int(score_val) + \
                        "░" * (10 - int(score_val))

                    # Color code the score
                    if score_val >= 8:
                        score_emoji = "🟢"
                    elif score_val >= 6:
                        score_emoji = "🟡"
                    else:
                        score_emoji = "🔴"

                    with st.container():
                        st.markdown(f"""
                        **{label}** {score_emoji}  
                        Score: {score_val}/10 `{score_bar}` (Weight: {weight}% → +{contribution:.1f} pts)  
                        *{reasoning}*
                        """)
                        st.markdown("---")
            else:
                st.info("No detailed breakdown available")

# Replace the existing ai_matrix_score_solicitations function
def ai_matrix_score_solicitations(df: pd.DataFrame, company_profile: dict, api_key: str,
                                  top_k: int = 10, model: str = "gpt-4o-mini", max_candidates: int = 60) -> list[dict]:
    """Enhanced matrix scoring with complete scoring components and optimized processing"""
    if df is None or df.empty:
        return []

    # Pre-filter with embeddings for speed
    company_desc = (company_profile.get("description") or "").strip()
    if company_desc:
        # Use a more aggressive pre-filter to reduce LLM calls
        pretrim = ai_downselect_df(company_desc, df, api_key, top_k=min(
            max_candidates, max(30, 3 * int(top_k))))
    else:
        pretrim = df.head(max_candidates)

    if pretrim.empty:
        return []

    # Prepare data for scoring
    cols = ["notice_id", "title", "description", "naics_code", "set_aside_code",
            "response_date", "posted_date", "link", "pop_city", "pop_state"]
    use_df = pretrim[[c for c in cols if c in pretrim.columns]].copy()

    # Score in very small batches for reliability
    scorer = AIMatrixScorer()
    results: dict[str, dict] = {}
    batch_size = 1  # Very small batches to avoid JSON issues

    for i in range(0, len(use_df), batch_size):
        batch_df = use_df.iloc[i:i+batch_size]
        batch_items = batch_df.to_dict(orient="records")

        try:
            batch_results = scorer.score_batch(
                batch_items, company_profile=company_profile,
                api_key=api_key, model=model)
            results.update(batch_results)
        except Exception as e:
            st.warning(f"Batch {i//batch_size + 1} scoring failed: {e}")
            continue

    if not results:
        return []

    # Combine results with original data
    enhanced_df = pretrim.copy()
    enhanced_df["__nid"] = enhanced_df["notice_id"].astype(str)
    enhanced_df["__score"] = enhanced_df["__nid"].map(
        lambda nid: results.get(nid, {}).get("score", 0.0))
    enhanced_df["__breakdown"] = enhanced_df["__nid"].map(
        lambda nid: results.get(nid, {}).get("breakdown", []))

    # Sort and limit results
    enhanced_df = enhanced_df.sort_values("__score", ascending=False).head(
        int(top_k)).reset_index(drop=True)

    # Generate blurbs for top results
    blurbs = ai_make_blurbs_fast(
        enhanced_df, api_key, model="gpt-4o-mini",
        max_items=min(20, len(enhanced_df)))

    # Format final output
    final_results = []
    for _, row in enhanced_df.iterrows():
        nid = str(row.get("notice_id", ""))
        final_results.append({
            "notice_id": nid,
            "title": row.get("title") or "Untitled",
            "link": make_sam_public_url(nid, row.get("link", "")),
            "score": float(row.get("__score", 0.0)),
            "breakdown": row.get("__breakdown", []),
            "blurb": blurbs.get(nid, row.get("title", ""))
        })

    return final_results

def ai_score_and_rank_solicitations_by_fit(df: pd.DataFrame, company_desc: str, company_profile: Dict[str, str], api_key: str, top_k: int = 10) -> list[dict]:
    prof = dict(company_profile or {})
    prof["description"] = company_desc or prof.get("description", "") or ""
    return ai_matrix_score_solicitations(df=df, company_profile=prof, api_key=api_key, top_k=int(top_k), model="gpt-4o-mini", max_candidates=60)

# =========================
# Database Query Functions
# =========================


def query_filtered_df_optimized(filters: dict, limit: int = 1000) -> pd.DataFrame:
    """Optimized version with better SQL and limits"""
    where_conditions = []
    params = {}

    where_conditions.append("LOWER(notice_type) != 'justification'")

    # NAICS filter
    naics = [re.sub(r"[^\d]", "", str(x))
             for x in (filters.get("naics") or []) if x]
    if naics:
        # Use tuple format for IN clause with pandas/psycopg2
        naics_tuple = tuple(naics)
        where_conditions.append(f"naics_code IN {naics_tuple}")

    # Date filter
    due_before = filters.get("due_before")
    if due_before:
        where_conditions.append("response_date <= %(due_before)s")
        params["due_before"] = str(due_before)

    # Set-aside filter
    sas = [str(s).lower() for s in (filters.get("set_asides") or []) if s]
    if sas:
        sa_conditions = []
        for i, sa in enumerate(sas):
            sa_conditions.append(
                f"LOWER(set_aside_code) LIKE %(setaside_{i})s")
            params[f"setaside_{i}"] = f"%{sa}%"
        where_conditions.append(f"({' OR '.join(sa_conditions)})")

    # Notice types
    nts = [str(nt).lower() for nt in (filters.get("notice_types") or []) if nt]
    if nts:
        nt_conditions = []
        for i, nt in enumerate(nts):
            nt_conditions.append(f"LOWER(notice_type) LIKE %(noticetype_{i})s")
            params[f"noticetype_{i}"] = f"%{nt}%"
        where_conditions.append(f"({' OR '.join(nt_conditions)})")

    where_clause = " AND ".join(
        where_conditions) if where_conditions else "1=1"

    # Keywords - PostgreSQL full-text search
    kws = [str(k).lower() for k in (filters.get("keywords_or") or []) if k]
    if kws and engine.url.get_dialect().name == 'postgresql':
        keyword_query = " | ".join(kws)
        where_conditions.append(
            "to_tsvector('english', title || ' ' || description) @@ to_tsquery(%(keyword_query)s)")
        params["keyword_query"] = keyword_query
        where_clause = " AND ".join(where_conditions)

    # Use direct integer substitution for LIMIT to avoid parameter binding issues
    limit_safe = min(int(limit), 5000)
    sql = f"""
        SELECT notice_id, solicitation_number, title, notice_type, posted_date, response_date, archive_date,
               naics_code, set_aside_code, description, link, pop_city, pop_state, pop_zip, pop_country, pop_raw
        FROM solicitationraw 
        WHERE {where_clause}
        ORDER BY posted_date DESC NULLS LAST
        LIMIT {limit_safe}
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return df

    # Handle keyword filtering for non-PostgreSQL databases
    if kws and engine.url.get_dialect().name != 'postgresql':
        for c in ["title", "description"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
        blob = (df["title"] + " " + df["description"]).str.lower()
        df = df[blob.apply(lambda t: any(k in t for k in kws))]

    return df.reset_index(drop=True)


def _hide_notice_and_description(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ["notice_id", "description", "link"] if c in df.columns], errors="ignore")

# =========================
# UI Components
# =========================


def render_sidebar_header():
    with st.sidebar:
        st.markdown("---")
        if st.session_state.user:
            prof = st.session_state.profile or {}
            company_name = (prof.get("company_name")
                            or "").strip() or "Your Company"
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


def render_score_results(ranked_results: List[Dict]):
    if not ranked_results:
        st.info("No results to display")
        return

    st.write(f"**Found {len(ranked_results)} ranked matches**")
    for idx, item in enumerate(ranked_results):
        nid = str(item.get('notice_id', ''))
        title = item.get('title', 'Untitled')
        score = float(item.get('score', 0.0))
        blurb = (item.get('blurb') or "").strip()

        header = f"#{idx+1}: {title} — Score: {score:.1f}"
        with st.expander(header, expanded=(idx == 0)):
            left, right = st.columns([2, 1])
            with left:
                st.write(f"**Notice ID:** {nid}")
                if blurb:
                    st.write(f"**Summary:** {blurb}")
                st.markdown(f"**[View on SAM.gov]({item.get('link','')})**")
            with right:
                st.metric("Fit Score", f"{score:.1f}")

            st.subheader("Scoring Rationale")
            b = item.get("breakdown") or []
            for comp in b:
                st.write(
                    f"- {comp['key']}: {comp['score']:.2f} — {comp.get('why','')}")


def render_auth_screen():
    st.title("Welcome to KIP")
    st.caption("Sign in or create an account to continue.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Log in")
        le = st.text_input("Email", key="login_email_full")
        lp = st.text_input("Password", type="password",
                           key="login_password_full")
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

                if remember_me:
                    raw_token = _issue_remember_me_token(
                        u["id"], days=int(get_secret("COOKIE_DAYS", 30)))
                    cookie_name = get_secret("COOKIE_NAME", "kip_auth")
                    cookies[cookie_name] = raw_token
                    cookies.save()
                st.rerun()

    with c2:
        st.subheader("Sign up")
        se = st.text_input("Email", key="signup_email_full")
        sp = st.text_input("Password", type="password",
                           key="signup_password_full")
        sp2 = st.text_input("Confirm password",
                            type="password", key="signup_password2_full")
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
                    upsert_profile(uid, company_name="",
                                   description="", city="", state="")
                    st.success("Account created. Please log in on the left.")
                else:
                    st.error("Could not create account. Check server logs.")


def render_account_settings():
    st.title("Account Settings")

    if st.button("Sign out", key="btn_signout_settings"):
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
    company_name = st.text_input(
        "Company name", value=prof.get("company_name", ""))
    description = st.text_area(
        "Company description", value=prof.get("description", ""), height=140)
    city = st.text_input("City", value=prof.get("city", "") or "")
    state = st.text_input("State", value=prof.get("state", "") or "")

    cols = st.columns([1, 1, 3])
    with cols[0]:
        if st.button("Save profile", key="btn_save_profile_settings"):
            if not company_name.strip() or not description.strip():
                st.error("Company name and description are required.")
            else:
                upsert_profile(st.session_state.user["id"], company_name.strip(
                ), description.strip(), city.strip(), state.strip())
                st.session_state.profile = get_profile(
                    st.session_state.user["id"])
                st.success("Profile saved.")
    with cols[1]:
        if st.button("Back to app", key="btn_back_to_app"):
            st.session_state.view = "main"
            st.rerun()


# =========================
# View Router
# =========================
if st.session_state.view == "auth":
    render_auth_screen()
    st.stop()
elif st.session_state.view == "account":
    render_account_settings()
    st.stop()

# =========================
# Main App
# =========================
st.title("KIP")
st.caption("Don't be jealous that I've been chatting online with babes *all day*.")

with st.sidebar:
    st.success("✅ API keys loaded from Secrets")
    st.caption("Feed refresh runs automatically (no manual refresh needed).")
    st.markdown("---")

render_sidebar_header()

colR1, colR2 = st.columns([2, 1])
with colR1:
    st.info("Feed updates automatically every hour.")
with colR2:
    try:
        with engine.connect() as conn:
            cnt = pd.read_sql_query(
                "SELECT COUNT(*) AS c FROM solicitationraw", conn)["c"].iloc[0]
        st.metric("Rows in DB", int(cnt))
    except Exception:
        st.metric("Rows in DB", 0)

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

with tab1:
    st.header("Filter Solicitations")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        limit_results = st.number_input(
            "Max results to show", min_value=1, max_value=5000, value=20)
    with colB:
        keywords_raw = st.text_input(
            "Filter keywords (OR, comma-separated)", value="")
    with colC:
        naics_raw = st.text_input(
            "Filter by NAICS (comma-separated)", value="")

    with st.expander("More filters (optional)"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            set_asides = st.multiselect(
                "Set-aside code", ["SBA", "WOSB", "EDWOSB", "HUBZone", "SDVOSB", "8A", "SDB"])
        with col2:
            due_before = st.date_input(
                "Due before (optional)", value=None, format="YYYY-MM-DD")
        with col3:
            notice_types = st.multiselect("Notice types", [
                                          "Solicitation", "Combined Synopsis/Solicitation", "Sources Sought", "Special Notice", "SRCSGT", "RFI"])

    filters = {
        "keywords_or": parse_keywords_or(keywords_raw),
        "naics": normalize_naics_input(naics_raw),
        "set_asides": set_asides,
        "due_before": (due_before.isoformat() if isinstance(due_before, date) else None),
        "notice_types": notice_types,
    }

    st.subheader("Company profile for matching")
    saved_desc = (st.session_state.get("profile") or {}).get("description", "")
    use_saved = st.checkbox("Use saved company profile",
                            value=bool(saved_desc))

    if use_saved and saved_desc:
        st.info("Using your saved company profile description.")
        company_desc = saved_desc
        st.text_area("Company description (from Account → Company Profile)",
                     value=saved_desc, height=120, disabled=True)
    else:
        company_desc = st.text_area(
            "Brief company description (temporary)", value="", height=120)

    st.session_state.company_desc = company_desc or ""
    use_ai_downselect = st.checkbox(
        "Use AI to downselect based on description", value=False)
    top_k_select = st.number_input("How many AI-ranked matches?", min_value=1,
                                   max_value=50, value=5, step=1) if use_ai_downselect else 5

    if st.button("Show top results", type="primary", key="btn_show_results"):
        try:
            df = query_filtered_df_optimized(filters, limit=2000)

            if df.empty:
                st.warning("No solicitations match your filters.")
                st.session_state.sol_df = None
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                if use_ai_downselect and company_desc.strip():
                    status_text.text(
                        "🔍 Finding most relevant solicitations...")
                    progress_bar.progress(20)

                    pretrim = ai_downselect_df(company_desc.strip(), df, OPENAI_API_KEY, top_k=min(
                        100, max(20, 8*int(top_k_select))))
                    progress_bar.progress(50)

                    if pretrim.empty:
                        st.info(
                            "AI pre-filter found no matches. Showing manual results.")
                        show_df = df.head(int(limit_results))
                        status_text.text("📝 Generating summaries...")
                        blurbs = ai_make_blurbs_fast(
                            show_df, OPENAI_API_KEY, max_items=20)
                        show_df["blurb"] = show_df["notice_id"].astype(
                            str).map(blurbs).fillna(show_df["title"])
                        progress_bar.progress(100)
                    else:
                        status_text.text("🎯 Ranking by company fit...")
                        progress_bar.progress(70)

                        prof = st.session_state.get('profile', {}) or {}
                        company_profile = {
                            'description': company_desc.strip(),
                            'city': prof.get('city', ''),
                            'state': prof.get('state', ''),
                            'company_name': prof.get('company_name', '')
                        }

                        enhanced_ranked = ai_score_and_rank_solicitations_by_fit(
                            pretrim, company_desc.strip(), company_profile, OPENAI_API_KEY, top_k=int(top_k_select))
                        progress_bar.progress(90)

                        if enhanced_ranked:
                            id_order = [x["notice_id"]
                                        for x in enhanced_ranked]
                            top_df = pretrim[pretrim["notice_id"].astype(
                                str).isin(id_order)].copy()
                            preorder = {nid: i for i,
                                        nid in enumerate(id_order)}
                            top_df["_order"] = top_df["notice_id"].astype(
                                str).map(preorder)
                            top_df = top_df.sort_values(
                                "_order").drop(columns=["_order"])

                            status_text.text("📝 Generating summaries...")
                            blurbs = ai_make_blurbs_fast(
                                top_df, OPENAI_API_KEY, max_items=int(top_k_select))
                            top_df["blurb"] = top_df["notice_id"].astype(
                                str).map(blurbs)

                            show_df = top_df
                            progress_bar.progress(100)

                            st.success(
                                f"Top {len(top_df)} matches by company fit:")
                            render_enhanced_score_results(enhanced_ranked)

                            st.session_state.topn_df = top_df.reset_index(
                                drop=True)
                            st.session_state.sol_df = top_df.copy()
                            st.session_state.enhanced_ranked = enhanced_ranked
                        else:
                            st.info("Enhanced ranking found no results.")
                            show_df = df.head(int(limit_results))
                            blurbs = ai_make_blurbs_fast(
                                show_df, OPENAI_API_KEY, max_items=20)
                            show_df["blurb"] = show_df["notice_id"].astype(
                                str).map(blurbs)

                    progress_bar.empty()
                    status_text.empty()
                else:
                    show_df = df.head(int(limit_results))
                    if len(show_df) > 0:
                        with st.spinner("Generating summaries..."):
                            blurbs = ai_make_blurbs_fast(
                                show_df, OPENAI_API_KEY, max_items=min(50, len(show_df)))
                            show_df["blurb"] = show_df["notice_id"].astype(
                                str).map(blurbs).fillna(show_df["title"])

                if 'show_df' in locals():
                    st.session_state.sol_df = show_df
                    show_df["sam_url"] = show_df.apply(lambda r: make_sam_public_url(
                        str(r.get("notice_id", "")), r.get("link", "")), axis=1)

                    if not (use_ai_downselect and company_desc.strip() and 'enhanced_ranked' in locals() and enhanced_ranked):
                        st.subheader(f"Solicitations ({len(show_df)})")
                        st.dataframe(_hide_notice_and_description(
                            show_df), use_container_width=True)

                    st.download_button("Download results as CSV", show_df.to_csv(index=False).encode(
                        "utf-8"), file_name="solicitation_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error processing request: {e}")
            st.exception(e)

with tab2:
    st.header("This feature is in development...")

with tab3:
    st.header("This feature is in development...")

with tab4:
    st.header("This feature is in development...")

with tab5:
    st.header("Internal Use")
    st.caption("Quick presets that filter/rank solicitations with AI. Results appear below in relevance order with short blurbs.")

    # Configuration
    internal_top_k = st.number_input(
        "How many AI-ranked matches?", min_value=1, max_value=50, value=5, step=1, key="internal_top_k")
    max_candidates_cap = st.number_input("Max candidates to consider before AI ranking", min_value=20, max_value=1000, value=300, step=20,
                                         help="We first pre-trim with embeddings, then rank with the LLM.", key="internal_max_candidates")

    # Preset buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        run_machine_shop = st.button("Solicitations for Machine Shop",
                                     type="primary", use_container_width=True, key="iu_btn_machine")
    with c2:
        run_services = st.button("Solicitations for Services", type="primary",
                                 use_container_width=True, key="iu_btn_services")
    with c3:
        run_research = st.button(
            "R&D Solicitations", type="primary", use_container_width=True, key="iu_btn_research")

    def compute_internal_results(preset_desc: str, negative_hint: str = "", research_only: bool = False) -> dict | None:
        """Returns {"top_df": DataFrame, "reason_by_id": dict} or None on failure."""
        # Get all solicitations
        df_all = query_filtered_df_optimized({"keywords_or": [], "naics": [], "set_asides": [
        ], "due_before": None, "notice_types": []})
        if df_all.empty:
            st.warning("No solicitations in the database to evaluate.")
            return None

        # Optional: restrict to research-type before AI ranking
        if research_only:
            rd_naics_prefixes = ("5417",)  # R&D NAICS codes
            naics_mask = df_all["naics_code"].fillna("").astype(
                str).str.startswith(rd_naics_prefixes)

            text = (df_all["title"].astype(str) + " " +
                    df_all["description"].astype(str)).str.lower()
            kw_any = ["research", "r&d", "development", "sbir", "sttr", "prototype", "baa", "technology demonstration",
                    "feasibility study", "innovative", "scientific", "laboratory", "experimentation"]
            kw_mask = text.apply(lambda t: any(k in t for k in kw_any))

            nt = df_all["notice_type"].fillna("").str.lower()
            nt_mask = nt.str.contains("baa") | nt.str.contains(
                "sources sought") | nt.str.contains("rfi") | nt.str.contains("special notice")

            df_all = df_all[naics_mask | kw_mask |
                            nt_mask].reset_index(drop=True)
            if df_all.empty:
                st.info("No likely research-type opportunities found.")
                return None

        # Build company description for AI
        base_desc = (st.session_state.get("company_desc") or "").strip()
        company_desc_internal = preset_desc.strip()
        if base_desc:
            company_desc_internal = base_desc + "\n\n" + company_desc_internal
        if negative_hint.strip():
            company_desc_internal += f"\n\nDo NOT include non-fits: {negative_hint.strip()}"

        # Pre-trim with embeddings
        pretrim_cap = min(int(max_candidates_cap),
                        max(20, 12 * int(internal_top_k)))
        pretrim = ai_downselect_df(
            company_desc_internal, df_all, OPENAI_API_KEY, top_k=pretrim_cap)
        if pretrim.empty:
            st.info("AI pre-filter returned nothing.")
            return None

        # AI ranking - FIXED: Use the correct function name
        prof = st.session_state.get('profile', {}) or {}
        company_profile = {
            'description': company_desc_internal,
            'city': prof.get('city', ''),
            'state': prof.get('state', ''),
            'company_name': prof.get('company_name', '')
        }

        ranked = ai_matrix_score_solicitations(
            df=pretrim,
            company_profile=company_profile,
            api_key=OPENAI_API_KEY,
            top_k=int(internal_top_k),
            model="gpt-4o-mini",
            max_candidates=min(len(pretrim), 60)
        )

        if not ranked:
            st.info("AI ranking returned no results.")
            return None

        # Order by ranking
        id_order = [x["notice_id"] for x in ranked]
        preorder = {nid: i for i, nid in enumerate(id_order)}
        top_df = pretrim[pretrim["notice_id"].astype(
            str).isin(id_order)].copy()
        top_df["__order"] = top_df["notice_id"].astype(str).map(preorder)
        top_df = top_df.sort_values("__order").drop_duplicates(
            subset=["notice_id"]).drop(columns="__order").reset_index(drop=True)

        # Add blurbs and scores
        blurbs = ai_make_blurbs_fast(
            top_df, OPENAI_API_KEY, model="gpt-4o-mini", max_items=len(top_df))
        top_df["blurb"] = top_df["notice_id"].astype(str).map(
            blurbs).fillna(top_df["title"].fillna(""))

        # Extract reasons from the ranked results
        reason_by_id = {}
        for item in ranked:
            nid = item.get("notice_id", "")
            # Try to get a summary reason from the breakdown
            breakdown = item.get("breakdown", [])
            if breakdown:
                # Get the top scoring components as the reason
                top_components = sorted(
                    breakdown, key=lambda x: x.get("score", 0), reverse=True)[:3]
                reasons = [f"{comp.get('label', comp.get('key', ''))} ({comp.get('score', 0)}/10)"
                        for comp in top_components]
                reason_by_id[nid] = f"Top matches: {', '.join(reasons)}"
            else:
                reason_by_id[nid] = item.get("blurb", "AI assessment")

        # Add fit scores from the ranked results
        score_by_id = {x["notice_id"]: x.get("score", 0) for x in ranked}
        top_df["fit_score"] = top_df["notice_id"].astype(
            str).map(score_by_id).fillna(0).astype(float)

    return {"top_df": top_df, "reason_by_id": reason_by_id}
    def render_internal_results():
        """Render the results with vendor finding functionality."""
        data = st.session_state.get('iu_results')
        if not data:
            return

        key_salt = st.session_state.iu_key_salt or ""
        top_df = data["top_df"]
        reason_by_id = data["reason_by_id"]

        # Ensure one expander is open by default
        if st.session_state.iu_open_nid is None and len(top_df):
            st.session_state.iu_open_nid = str(top_df.iloc[0]["notice_id"])

        st.success(f"Top {len(top_df)} matches by relevance:")

        for idx, row in enumerate(top_df.itertuples(index=False), start=1):
            hdr = (getattr(row, "blurb", None) or getattr(
                row, "title", None) or "Untitled")
            nid = str(getattr(row, "notice_id", ""))

            expanded = (st.session_state.get("iu_open_nid") == nid)
            with st.expander(f"{idx}. {hdr}", expanded=expanded):
                left, right = st.columns([2, 1])

                with left:
                    st.write(
                        f"**Notice Type:** {getattr(row, 'notice_type', '')}")
                    st.write(f"**Posted:** {getattr(row, 'posted_date', '')}")
                    st.write(
                        f"**Response Due:** {getattr(row, 'response_date', '')}")
                    st.write(f"**NAICS:** {getattr(row, 'naics_code', '')}")
                    st.write(
                        f"**Set-aside:** {getattr(row, 'set_aside_code', '')}")

                    link = make_sam_public_url(
                        str(getattr(row, 'notice_id', '')), getattr(row, 'link', ''))
                    st.write(f"[Open on SAM.gov]({link})")

                    reason = reason_by_id.get(nid, "")
                    if reason:
                        st.markdown("**Why this matched (AI):**")
                        st.info(reason)

                    # Branch based on mode
                    if st.session_state.get("iu_mode") == "rd":
                        # Research mode - show research direction
                        direction = ai_research_direction(
                            getattr(row, "title", ""), getattr(row, "description", ""), OPENAI_API_KEY)
                        st.markdown("**Proposed Research Direction:**")
                        st.write(direction)
                    else:
                        # Services/Machine shop mode - show vendor finder
                        btn_label = "Find 3 potential vendors (SerpAPI)" if st.session_state.get(
                            "iu_mode") == "machine" else "Find 3 local service providers"
                        btn_key = f"iu_find_vendors_{nid}_{idx}_{key_salt}"

                        if st.button(btn_label, key=btn_key):
                            sol_dict = {
                                "notice_id": nid,
                                "title": getattr(row, "title", ""),
                                "description": getattr(row, "description", ""),
                                "naics_code": getattr(row, "naics_code", ""),
                                "set_aside_code": getattr(row, "set_aside_code", ""),
                                "response_date": getattr(row, "response_date", ""),
                                "posted_date": getattr(row, "posted_date", ""),
                                "link": getattr(row, "link", ""),
                            }

                            # Extract locality
                            locality = {"city": _s(getattr(row, "pop_city", "")), "state": _s(
                                getattr(row, "pop_state", ""))}
                            if not _has_locality(locality):
                                locality = _extract_locality(
                                    f"{getattr(row, 'title', '')}\n{getattr(row, 'description', '')}") or {}

                            # Set status message
                            if _has_locality(locality):
                                where = ", ".join(
                                    [x for x in [locality.get("city", ""), locality.get("state", "")] if x])
                                st.session_state.vendor_notes[
                                    nid] = f"Place of performance: {where}"
                                st.session_state.vendor_errors.pop(nid, None)
                            else:
                                st.session_state.vendor_notes[nid] = "No place of performance specified. Conducting national search."

                            # Find vendors
                            vendors_df, note = find_service_vendors_for_opportunity(
                                sol_dict, OPENAI_API_KEY, SERP_API_KEY, top_n=3)

                            if vendors_df is None or vendors_df.empty:
                                loc_msg = ""
                                if _has_locality(locality):
                                    where = ", ".join([x for x in [locality.get("city", ""), locality.get(
                                        "state", "")] if x]) or locality.get("state", "")
                                    loc_msg = f" for the specified locality ({where})"
                                st.session_state.vendor_errors[
                                    nid] = f"No service providers found{loc_msg}."
                            else:
                                st.session_state.vendor_errors.pop(nid, None)

                            st.session_state.vendor_suggestions[nid] = vendors_df
                            st.rerun()

                with right:
                    # Display status messages and vendor results
                    note_msg = st.session_state.vendor_notes.get(nid)
                    if note_msg:
                        st.caption(note_msg)

                    err_msg = st.session_state.vendor_errors.get(nid)
                    if err_msg:
                        st.info(err_msg)

                    vend_df = st.session_state.vendor_suggestions.get(nid)
                    if isinstance(vend_df, pd.DataFrame) and not vend_df.empty:
                        st.markdown("**Vendor candidates**")
                        for j, v in vend_df.iterrows():
                            raw_name = (v.get("name") or "").strip()
                            website = (v.get("website") or "").strip()
                            location = (v.get("location") or "").strip()
                            reason_txt = (v.get("reason") or "").strip()

                            display_name = raw_name or _company_name_from_url(
                                website) or "Unnamed Vendor"
                            if website:
                                st.markdown(
                                    f"- **[{display_name}]({website})**")
                            else:
                                st.markdown(f"- **{display_name}**")
                            if location:
                                st.caption(location)
                            if reason_txt:
                                st.write(reason_txt)
                    else:
                        if not err_msg:
                            st.caption(
                                "No vendors yet. Click the button to fetch.")

    # Handle preset button clicks
    if run_machine_shop:
        st.session_state.iu_key_salt = uuid.uuid4().hex
        st.session_state.iu_mode = "machine"
        preset_desc = ("We are pursuing solicitations where a MACHINE SHOP would fabricate or machine parts for us. "
                       "Strong fits include CNC machining, milling, turning, drilling, precision tolerances, "
                       "metal or plastic fabrication, weldments, assemblies, and production of custom components per drawings. "
                       "Prefer solicitations with part drawings, specs, materials (aluminum, steel, titanium), and tangible manufactured items.")
        negative_hint = ("Pure services, staffing-only, software-only, consulting, training, janitorial, IT, "
                         "or anything that does not involve fabricating or machining a physical part.")
        with st.spinner("Finding best-matching solicitations..."):
            data = compute_internal_results(preset_desc, negative_hint)
        st.session_state.iu_results = data
        st.rerun()

    if run_services:
        st.session_state.iu_key_salt = uuid.uuid4().hex
        st.session_state.iu_mode = "services"
        preset_desc = ("We are pursuing solicitations where a SERVICES COMPANY performs the work for us. "
                       "Strong fits include maintenance, installation, inspection, logistics, training, field services, "
                       "operations support, professional services, and other labor-based or outcome-based services "
                       "delivered under SOW/Performance Work Statement.")
        negative_hint = "Manufacturing-only or pure product buys without a material services component."
        with st.spinner("Finding best-matching solicitations..."):
            data = compute_internal_results(preset_desc, negative_hint)
        st.session_state.iu_results = data
        st.rerun()

    if run_research:
        st.session_state.iu_key_salt = uuid.uuid4().hex
        st.session_state.iu_mode = "rd"
        preset_desc = ("We are pursuing research and development (R&D) opportunities aligned with our capabilities. "
                       "Strong fits include applied research, technology maturation, prototyping, experimentation, "
                       "testing and evaluation, studies, and early-stage development tasks.")
        negative_hint = (
            "Commodity/product-only buys, routine MRO, janitorial, IT support, or other non-research services.")
        with st.spinner("Finding best-matching research solicitations..."):
            data = compute_internal_results(
                preset_desc, negative_hint, research_only=True)
        st.session_state.iu_results = data
        st.rerun()

    # Only render results if they exist
    if 'iu_results' in st.session_state and st.session_state.iu_results:
        render_internal_results()

    # Export option
    if st.session_state.get('iu_results') and isinstance(st.session_state.iu_results.get("top_df"), pd.DataFrame):
        top_df = st.session_state.iu_results["top_df"]
        st.download_button(f"Download Internal Use Results (Top-{int(internal_top_k)})",
                           top_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"internal_top{int(internal_top_k)}.csv",
                           mime="text/csv")

    st.markdown("---")
    st.caption(
        "DB schema is fixed to only the required SAM fields. Refresh inserts brand-new notices only (no updates).")
