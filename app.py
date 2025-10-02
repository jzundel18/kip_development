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
from typing import Dict
from enhanced_matching import EnhancedMatcher

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
import generate_proposal as gp
import get_relevant_solicitations as gs
import secrets

from scoring import AIMatrixScorer, ai_matrix_score_solicitations, ai_score_and_rank_solicitations_by_fit
# =========================
# Performance Optimization Functions
# =========================


@st.cache_data(show_spinner=False, ttl=86400)  # 24 hour cache
def get_cached_embeddings(text_hashes: list[str]) -> dict[str, np.ndarray]:
    """Fetch cached embeddings from database"""
    if not text_hashes:
        return {}

    try:
        with engine.connect() as conn:
            if engine.url.get_dialect().name == 'postgresql':
                # PostgreSQL: Use ANY() with array parameter
                from sqlalchemy import text
                sql = text("""
                    SELECT notice_id, embedding, text_hash 
                    FROM solicitation_embeddings 
                    WHERE text_hash = ANY(:hashes)
                """)
                # Execute directly and fetch to DataFrame
                result = conn.execute(sql, {"hashes": text_hashes})
                rows = result.fetchall()
                df = pd.DataFrame(
                    rows, columns=['notice_id', 'embedding', 'text_hash'])
            else:
                # SQLite: Chunk into smaller batches
                chunk_size = 500
                dfs = []
                for i in range(0, len(text_hashes), chunk_size):
                    chunk = text_hashes[i:i+chunk_size]
                    params = {}
                    placeholders = []
                    for j, hash_val in enumerate(chunk):
                        param_name = f"hash_{j}"
                        placeholders.append(f":{param_name}")
                        params[param_name] = hash_val

                    placeholders_str = ", ".join(placeholders)
                    sql_chunk = f"SELECT notice_id, embedding, text_hash FROM solicitation_embeddings WHERE text_hash IN ({placeholders_str})"
                    df_chunk = pd.read_sql_query(
                        sql_chunk, conn, params=params)
                    dfs.append(df_chunk)

                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
                    columns=['notice_id', 'embedding', 'text_hash'])

        result = {}
        for _, row in df.iterrows():
            try:
                embedding_data = json.loads(row['embedding'])
                result[row['text_hash']] = np.array(
                    embedding_data, dtype=np.float32)
            except Exception:
                continue

        return result

    except Exception as e:
        # Table doesn't exist or other error - return empty dict
        # This allows the system to compute embeddings fresh
        return {}

def store_embeddings_batch(embeddings_data: list[dict]):
    """Store multiple embeddings efficiently"""
    if not embeddings_data:
        return

    with engine.begin() as conn:
        conn.execute(sa.text("""
            INSERT INTO solicitation_embeddings (notice_id, embedding, text_hash, created_at)
            VALUES (:notice_id, :embedding, :text_hash, :created_at)
            ON CONFLICT (notice_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                text_hash = EXCLUDED.text_hash,
                created_at = EXCLUDED.created_at
        """), embeddings_data)


def query_filtered_df_optimized(filters: dict, limit: int = 1000) -> pd.DataFrame:
    """Optimized version with better SQL and limits"""
    where_conditions = []
    params = {}

    where_conditions.append("LOWER(notice_type) != 'justification'")

    # NAICS filter - FIXED
    naics = [re.sub(r"[^\d]", "", str(x))
             for x in (filters.get("naics") or []) if x]
    if naics:
        # Create individual placeholders for each NAICS code
        naics_placeholders = []
        for i, naics_code in enumerate(naics):
            param_name = f"naics_{i}"
            naics_placeholders.append(f"%(naics_{i})s")
            params[param_name] = naics_code
        where_conditions.append(f"naics_code IN ({', '.join(naics_placeholders)})")

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

# Add these functions after your authentication functions in app.py


def save_document(user_id: int, filename: str, file_content: bytes,
                  file_type: str, description: str = "", tags: str = "",
                  notice_id: str = None) -> Optional[int]:
    """Save a document to the database"""
    now = datetime.now(timezone.utc).isoformat()
    file_size = len(file_content)

    with engine.begin() as conn:
        try:
            sql = sa.text("""
                INSERT INTO documents 
                (user_id, filename, file_type, file_size, content, description, tags, notice_id, uploaded_at)
                VALUES (:uid, :fname, :ftype, :fsize, :content, :desc, :tags, :nid, :ts)
                RETURNING id
            """)
            doc_id = conn.execute(sql, {
                "uid": user_id,
                "fname": filename,
                "ftype": file_type,
                "fsize": file_size,
                "content": file_content,
                "desc": description,
                "tags": tags,
                "nid": notice_id,
                "ts": now
            }).scalar_one()
            return int(doc_id)
        except Exception as e:
            st.error(f"Could not save document: {e}")
            return None


def get_user_documents(user_id: int) -> pd.DataFrame:
    """Get all documents for a user (without content for list view)"""
    with engine.connect() as conn:
        sql = sa.text("""
            SELECT id, filename, file_type, file_size, description, tags, 
                   notice_id, uploaded_at
            FROM documents
            WHERE user_id = :uid
            ORDER BY uploaded_at DESC
        """)
        return pd.read_sql_query(sql, conn, params={"uid": user_id})


def get_document_content(doc_id: int, user_id: int) -> Optional[tuple]:
    """Get document content (filename, file_type, content)"""
    with engine.connect() as conn:
        sql = sa.text("""
            SELECT filename, file_type, content
            FROM documents
            WHERE id = :did AND user_id = :uid
        """)
        row = conn.execute(
            sql, {"did": doc_id, "uid": user_id}).mappings().first()
        if row:
            return (row["filename"], row["file_type"], bytes(row["content"]))
        return None


def delete_document(doc_id: int, user_id: int) -> bool:
    """Delete a document"""
    with engine.begin() as conn:
        sql = sa.text(
            "DELETE FROM documents WHERE id = :did AND user_id = :uid")
        result = conn.execute(sql, {"did": doc_id, "uid": user_id})
        return result.rowcount > 0


def update_document_metadata(doc_id: int, user_id: int, description: str = None,
                             tags: str = None, notice_id: str = None) -> bool:
    """Update document metadata"""
    updates = []
    params = {"did": doc_id, "uid": user_id}

    if description is not None:
        updates.append("description = :desc")
        params["desc"] = description
    if tags is not None:
        updates.append("tags = :tags")
        params["tags"] = tags
    if notice_id is not None:
        updates.append("notice_id = :nid")
        params["nid"] = notice_id

    if not updates:
        return False

    with engine.begin() as conn:
        sql = sa.text(f"""
            UPDATE documents
            SET {", ".join(updates)}
            WHERE id = :did AND user_id = :uid
        """)
        result = conn.execute(sql, params)
        return result.rowcount > 0

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


def ai_downselect_df_optimized(company_desc: str, df: pd.DataFrame, api_key: str,
                               threshold: float = 0.20, top_k: int | None = None) -> pd.DataFrame:
    """Optimized version using cached embeddings"""
    if df.empty:
        return df

    # Prepare texts and compute hashes
    texts = (df["title"].fillna("") + " " +
             df["description"].fillna("")).str.slice(0, 2000)
    text_hashes = []
    hash_to_idx = {}

    for idx, text in enumerate(texts):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        text_hashes.append(text_hash)
        hash_to_idx[text_hash] = idx

    # Get cached embeddings
    cached = get_cached_embeddings(text_hashes)

    # Identify missing embeddings
    missing_hashes = [h for h in text_hashes if h not in cached]
    missing_texts = [texts.iloc[hash_to_idx[h]] for h in missing_hashes]

    # Compute missing embeddings in batch
    if missing_texts:
        try:
            client = OpenAI(api_key=api_key)

            # Company query embedding
            q = client.embeddings.create(
                model="text-embedding-3-small", input=[company_desc])
            Xq = np.array(q.data[0].embedding, dtype=np.float32)
            Xq_norm = Xq / (np.linalg.norm(Xq) + 1e-9)

            # Batch compute missing embeddings
            X_list = []
            batch_size = 500
            for i in range(0, len(missing_texts), batch_size):
                batch = missing_texts[i:i+batch_size]
                r = client.embeddings.create(
                    model="text-embedding-3-small", input=batch)
                X_list.extend([d.embedding for d in r.data])

            # Store new embeddings
            embeddings_to_store = []
            for i, text_hash in enumerate(missing_hashes):
                embedding = np.array(X_list[i], dtype=np.float32)
                embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-9)
                cached[text_hash] = embedding_norm

                embeddings_to_store.append({
                    "notice_id": str(df.iloc[hash_to_idx[text_hash]]["notice_id"]),
                    "embedding": json.dumps(embedding_norm.tolist()),
                    "text_hash": text_hash,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

            # Store in background (non-blocking)
            if embeddings_to_store:
                store_embeddings_batch(embeddings_to_store)

        except Exception as e:
            st.warning(
                f"AI downselect failed ({e}). Using simple keyword filter.")
            # Fallback to keyword matching
            kws = [w.lower() for w in re.findall(
                r"[a-zA-Z0-9]{4,}", company_desc)]
            if not kws:
                return df
            blob = (df["title"].fillna("") + " " +
                    df["description"].fillna("")).str.lower()
            mask = blob.apply(lambda t: any(k in t for k in kws))
            return df[mask].reset_index(drop=True)
    else:
        # All embeddings cached, just get company embedding
        client = OpenAI(api_key=api_key)
        q = client.embeddings.create(
            model="text-embedding-3-small", input=[company_desc])
        Xq_norm = np.array(q.data[0].embedding, dtype=np.float32)
        Xq_norm = Xq_norm / (np.linalg.norm(Xq_norm) + 1e-9)

    # Compute similarities using cached embeddings
    X = np.array([cached[h] for h in text_hashes])
    sims = X @ Xq_norm

    df_result = df.copy()
    df_result["ai_score"] = sims

    if top_k is not None and top_k > 0:
        df_result = df_result.sort_values(
            "ai_score", ascending=False).head(int(top_k))
    else:
        df_result = df_result[df_result["ai_score"] >= float(
            threshold)].sort_values("ai_score", ascending=False)

    return df_result.reset_index(drop=True)


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
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GOOGLE_CX = get_secret("GOOGLE_CX")
SAM_KEYS = get_secret("SAM_KEYS", [])

if isinstance(SAM_KEYS, str):
    SAM_KEYS = [k.strip() for k in SAM_KEYS.split(",") if k.strip()]
elif not isinstance(SAM_KEYS, (list, tuple)):
    SAM_KEYS = []

missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "GOOGLE_API_KEY": GOOGLE_API_KEY,  # Updated
    "GOOGLE_CX": GOOGLE_CX,  # New
    "SAM_KEYS": SAM_KEYS,
}.items() if not v]

if missing:
    st.error(f"Missing required secrets: {', '.join(missing)}")
    st.stop()

# =========================
# Database Configuration
# =========================
DB_URL = st.secrets.get("SUPABASE_DB_URL") or "sqlite:///app.db"

# Add these lines to your session state initialization section (around line 75-85)
# Find where you have the other session state initializations and add these:

# ADD THESE NEW LINES:
if "iu_open_nid" not in st.session_state:
    st.session_state.iu_open_nid = None
if "iu_key_salt" not in st.session_state:
    st.session_state.iu_key_salt = ""
if "iu_mode" not in st.session_state:
    st.session_state.iu_mode = ""
if "iu_results" not in st.session_state:
    st.session_state.iu_results = None
if "vendor_suggestions" not in st.session_state:
    st.session_state.vendor_suggestions = {}
if "vendor_errors" not in st.session_state:
    st.session_state.vendor_errors = {}


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
# Add these to your session state initialization
if "topn_df" not in st.session_state:
    st.session_state.topn_df = None
if "partner_matches" not in st.session_state:
    st.session_state.partner_matches = None
if "partner_matches_stamp" not in st.session_state:
    st.session_state.partner_matches_stamp = None
if "topn_stamp" not in st.session_state:
    st.session_state.topn_stamp = None

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
    except (AttributeError, TypeError):  # Handle when pd.isna fails
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
    optimize_database()
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

# Create solicitation_embeddings table
try:
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS solicitation_embeddings (
                notice_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_hash 
            ON solicitation_embeddings (text_hash)
        """))
except Exception as e:
    st.warning(f"Embedding table creation note: {e}")

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

# Create documents table
try:
    with engine.begin() as conn:
        # For PostgreSQL
        if engine.url.get_dialect().name == 'postgresql':
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content BYTEA NOT NULL,
                    description TEXT,
                    tags TEXT,
                    notice_id TEXT,
                    uploaded_at TEXT NOT NULL
                )
            """))
        else:
            # For SQLite
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content BLOB NOT NULL,
                    description TEXT,
                    tags TEXT,
                    notice_id TEXT,
                    uploaded_at TEXT NOT NULL
                )
            """))

        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id)
        """))
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_documents_notice ON documents (notice_id)
        """))
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_documents_uploaded ON documents (uploaded_at)
        """))
except Exception as e:
    st.warning(f"Document table creation note: {e}")

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

# Company table migration (add after your existing migrations)
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
        missing_cols = [
            c for c in REQUIRED_COMPANY_COLS if c not in existing_cols]
        if missing_cols:
            with engine.begin() as conn:
                for col in missing_cols:
                    conn.execute(
                        sa.text(f'ALTER TABLE company ADD COLUMN "{col}" {REQUIRED_COMPANY_COLS[col]}'))
except Exception as e:
    st.warning(f"Company table migration note: {e}")


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
    raw = secrets.token_urlsafe(32)
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
# Partner Matching Functions
# =========================

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
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
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
            "name": company_row.get("name", ""),
            "capabilities": company_row.get("description", ""),
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
        j = str(data.get("justification", "")).strip()
        jp = str(data.get("joint_proposal", "")).strip()
        if not j and not jp:
            # graceful fallback
            return {"justification": "No justification returned.", "joint_proposal": ""}
        return {"justification": j, "joint_proposal": jp}
    except Exception as e:
        return {"justification": f"Justification unavailable ({e})", "joint_proposal": ""}


def companies_df() -> pd.DataFrame:
    """Get companies from the database"""
    with engine.connect() as conn:
        try:
            return pd.read_sql_query(
                "SELECT id, name, description, city, state FROM company ORDER BY name",
                conn
            )
        except Exception:
            return pd.DataFrame(columns=["id", "name", "description", "city", "state"])


def bulk_insert_companies(df: pd.DataFrame) -> int:
    """Insert multiple companies into the database"""
    needed = ["name", "description", "city", "state"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""

    with engine.begin() as conn:
        rows = df[needed].fillna("").to_dict(orient="records")
        for row in rows:
            conn.execute(sa.text("""
                INSERT INTO company (name, description, city, state)
                VALUES (:name, :description, :city, :state)
            """), {k: (row.get(k) or "") for k in needed})
    return len(df)


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
        fit_score = getattr(row, "fit_score", 75)

        expanded = (st.session_state.get("iu_open_nid") == nid)
        with st.expander(f"{idx}. {hdr}", expanded=expanded):
            left, right = st.columns([2, 1])

            with left:
                st.write(f"**Notice Type:** {getattr(row, 'notice_type', '')}")
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
                    btn_label = "Find 3 potential vendors" if st.session_state.get(
                        "iu_mode") == "machine" else "Find 3 local service providers"
                    btn_key = f"iu_find_vendors_{nid}_{idx}_{key_salt}"

                if st.button(btn_label, key=btn_key):
                    try:
                        sol_dict = {
                            "notice_id": nid,
                            "title": _s(getattr(row, "title", "")),
                            "description": _s(getattr(row, "description", "")),
                            "naics_code": _s(getattr(row, "naics_code", "")),
                            "set_aside_code": _s(getattr(row, "set_aside_code", "")),
                            "response_date": _s(getattr(row, "response_date", "")),
                            "posted_date": _s(getattr(row, "posted_date", "")),
                            "link": _s(getattr(row, "link", "")),
                            "pop_city": _s(getattr(row, "pop_city", "")),
                            "pop_state": _s(getattr(row, "pop_state", "")),
                        }
                    except Exception as e:
                        st.error(f"Error creating solicitation data: {e}")
                        continue

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

                    # CREATE DEBUG CONTAINER - THIS IS THE KEY CHANGE
                    with st.expander("🔍 **Vendor Search Debug Log**", expanded=True):
                        debug_container = st.container()
                    
                    # Find vendors - PASS THE DEBUG CONTAINER
                    try:
                        vendors_df, note = find_service_vendors_for_opportunity(
                            sol_dict, 
                            GOOGLE_API_KEY, 
                            GOOGLE_CX, 
                            OPENAI_API_KEY, 
                            top_n=3,
                            streamlit_debug=debug_container
                        )

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

                    except Exception as e:
                        debug_container.error(f"❌ Exception during vendor search: {e}")
                        import traceback
                        debug_container.code(traceback.format_exc())
                        
                        st.session_state.vendor_errors[
                            nid] = f"Error finding vendors: {str(e)[:100]}"
                        st.session_state.vendor_suggestions[nid] = pd.DataFrame()
                    
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


def render_enhanced_score_results(results: List[Dict]):
    """Enhanced results display for new matching system"""
    if not results:
        st.info("No matches found")
        return

    for idx, result in enumerate(results):
        score = result.get("score", 0)
        title = result.get("title", "Untitled")
        reasoning = result.get("reasoning", "")
        notice_id = result.get("notice_id", "")

        # Color-code scores
        if score >= 80:
            score_color = "🟢"
        elif score >= 60:
            score_color = "🟡"
        else:
            score_color = "🔴"

        with st.expander(f"#{idx+1}: {title} — {score_color} {score:.1f}/100", expanded=(idx == 0)):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(
                    f"**[View on SAM.gov]({result.get('link', '#')})**")
                st.write(
                    f"**NAICS:** {result.get('naics_code', 'Not specified')}")
                st.write(
                    f"**Set-aside:** {result.get('set_aside_code', 'Not specified')}")
                st.write(
                    f"**Location:** {result.get('pop_location', 'Not specified')}")
                st.write(
                    f"**Response Due:** {result.get('response_date', 'Not specified')}")

                if reasoning:
                    st.markdown("**Match Analysis:**")
                    st.info(reasoning)

            with col2:
                # Score visualization
                if score >= 80:
                    st.success(f"Score: {score:.1f}/100")
                elif score >= 60:
                    st.warning(f"Score: {score:.1f}/100")
                else:
                    st.error(f"Score: {score:.1f}/100")

                # Component breakdown if available
                components = result.get("component_scores", {})
                if components:
                    st.markdown("**Top Factors:**")
                    # Show top 3 components
                    comp_items = list(components.items())[:3]
                    for comp_name, comp_data in comp_items:
                        if isinstance(comp_data, dict):
                            comp_score = comp_data.get("score", 0)
                            st.caption(f"{comp_name}: {comp_score}/10")
                        else:
                            st.caption(f"{comp_name}: {comp_data}")


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
# Enhanced Vendor Search Functions
# =========================


def find_service_vendors_for_opportunity(
    solicitation: dict,
    google_api_key: str,
    google_cx: str,
    openai_api_key: str,
    top_n: int = 3,
    streamlit_debug: Any = None
) -> tuple:
    """Find service vendors for a solicitation using Google Custom Search API with Streamlit debugging"""

    def debug_log(msg: str):
        """Log to streamlit if available"""
        if streamlit_debug is not None:
            streamlit_debug.write(msg)

    try:
        # Extract what type of service is needed
        title = solicitation.get("title", "")
        description = solicitation.get("description", "")
        naics = solicitation.get("naics_code", "")

        debug_log("🔍 **Starting vendor search**")
        debug_log(f"**Title:** {title[:100]}")

        # Extract location
        pop_city = (solicitation.get("pop_city") or "").strip()
        pop_state = (solicitation.get("pop_state") or "").strip()

        if pop_city and pop_state:
            debug_log(f"**Location:** {pop_city}, {pop_state}")
        elif pop_state:
            debug_log(f"**Location:** {pop_state}")
        else:
            debug_log(f"**Location:** National search")

        # Build search query for the type of service needed
        client = OpenAI(api_key=openai_api_key)

        debug_log("")
        debug_log("**Step 1: Analyzing service type needed...**")

        service_prompt = f"""Based on this government solicitation, what type of service company should I search for? 
        
Title: {title[:200]}
Description: {description[:400]}

Respond with 2-4 search keywords for the type of service provider needed (e.g., "HVAC maintenance contractor", "IT support services", "janitorial services"). Be specific and practical."""

        service_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": service_prompt}],
            temperature=0.1,
            max_tokens=50,
            timeout=15
        )

        service_type = service_response.choices[0].message.content.strip()
        debug_log(f"✅ Service type identified: **{service_type}**")

        # Build location-aware search query
        debug_log("")
        debug_log("**Step 2: Building search queries...**")

        if pop_city and pop_state:
            location_query = f"{pop_city} {pop_state}"
            search_query = f"{service_type} {location_query}"
            search_note = f"Searching for providers in {pop_city}, {pop_state}"
        elif pop_state:
            location_query = pop_state
            search_query = f"{service_type} {pop_state}"
            search_note = f"Searching for providers in {pop_state}"
        else:
            search_query = f"{service_type} contractors United States"
            search_note = "No specific location found - conducting national search"

        debug_log(f"Primary search query: `{search_query}`")

        # Search using Google Custom Search API
        debug_log("")
        debug_log("**Step 3: Searching Google...**")

        google_params = {
            "key": google_api_key,
            "cx": google_cx,
            "q": search_query,
            "num": min(10, 15),
            "safe": "off",
            "lr": "lang_en",
            "filter": "0",
        }

        response = requests.get(
            "https://www.googleapis.com/customsearch/v1", params=google_params, timeout=15)

        if response.status_code != 200:
            debug_log(f"❌ **Search API error:** {response.status_code}")
            return None, f"Search API error: {response.status_code}"

        data = response.json()
        search_results = data.get("items", [])

        debug_log(f"✅ Google returned **{len(search_results)} results**")

        if not search_results:
            debug_log("❌ **No search results found**")
            return None, "No search results found"

        # Enhanced filtering
        debug_log("")
        debug_log("**Step 4: Filtering and scoring candidates...**")

        vendors = []
        seen_domains = set()

        # Domains to avoid
        skip_domains = ['sam.gov', 'govtribe.com', 'facebook.com', 'linkedin.com',
                        'indeed.com', 'glassdoor.com', 'wikipedia.org']

        accepted_count = 0
        skipped_count = 0

        for result in search_results:
            title_text = result.get("title", "")
            snippet = result.get("snippet", "")
            link = result.get("link", "")

            if not link.startswith('http'):
                skipped_count += 1
                continue

            try:
                domain = urlparse(link).netloc.lower()
                if domain in seen_domains:
                    skipped_count += 1
                    continue
                seen_domains.add(domain)
            except:
                skipped_count += 1
                continue

            if any(skip_domain in link.lower() for skip_domain in skip_domains):
                skipped_count += 1
                continue

            company_name = _company_name_from_url(link)
            if title_text and len(title_text) < 100:
                company_name = title_text.split('|')[0].split('-')[0].strip()

            location_text = ""
            location_match = re.search(
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2})\b', snippet)
            if location_match:
                location_text = f"{location_match.group(1)}, {location_match.group(2)}"
            elif pop_state:
                location_text = f"Serving {pop_state}"

            try:
                reason_prompt = f"""Why would "{company_name}" be a good vendor for this work: {title[:100]}?
                
Based on: {snippet[:200]}

Give a 1 sentence reason focusing on their relevant capabilities."""

                reason_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": reason_prompt}],
                    temperature=0.1,
                    max_tokens=50,
                    timeout=10
                )

                reason = reason_response.choices[0].message.content.strip()
            except:
                reason = "Appears to offer relevant services for this opportunity."

            vendors.append({
                "name": company_name,
                "website": link,
                "location": location_text,
                "reason": reason,
                "snippet": snippet[:150]
            })

            accepted_count += 1

            if len(vendors) >= top_n:
                break

        debug_log(
            f"✅ Accepted **{accepted_count}** candidates (skipped {skipped_count})")

        if not vendors:
            debug_log("")
            debug_log("❌ **No relevant service providers found after filtering**")
            debug_log("**Possible reasons:**")
            debug_log("  • All results were aggregator sites (sam.gov, govtribe)")
            debug_log("  • Results were job boards or social media")
            debug_log("  • Search query too specific for available vendors")
            return None, "No relevant service providers found after filtering"

        debug_log("")
        debug_log(f"**Step 5: Final vendor selection (Top {top_n}):**")

        vendors_df = pd.DataFrame(vendors[:top_n], columns=[
                                  "name", "website", "location", "reason"])

        for i, row in vendors_df.iterrows():
            debug_log(f"**{i+1}. {row['name']}**")
            debug_log(f"   Website: {row['website'][:60]}")
            if row['location']:
                debug_log(f"   Location: {row['location']}")
            debug_log(f"   Why: {row['reason']}")
            debug_log("")

        debug_log(f"✅ **Search complete!** Found {len(vendors_df)} vendors")

        return vendors_df, search_note

    except Exception as e:
        if streamlit_debug:
            streamlit_debug.error(f"❌ **Error during vendor search:** {e}")
            import traceback
            streamlit_debug.code(traceback.format_exc())
        return None, f"Vendor search failed: {str(e)[:100]}"

def _has_locality(locality: dict) -> bool:
    """Check if locality has meaningful location data"""
    city = (locality.get("city") or "").strip()
    state = (locality.get("state") or "").strip()
    return bool(city or state)


def _extract_locality(text: str) -> dict:
    """Extract city/state from text using simple regex patterns"""

    # Common patterns for city, state
    patterns = [
        r'(\w+(?:\s+\w+)*),\s*([A-Z]{2})',  # "City Name, ST"
        r'(\w+(?:\s+\w+)*)\s+([A-Z]{2})\s',  # "City Name ST "
        # Proper case cities
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s*([A-Z]{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return {
                "city": match.group(1).strip(),
                "state": match.group(2).strip()
            }

    # Just look for state codes
    state_match = re.search(r'\b([A-Z]{2})\b', text)
    if state_match:
        return {
            "city": "",
            "state": state_match.group(1)
        }

    return {}


def _company_name_from_url(url: str) -> str:
    """Extract company name from URL"""
    try:
        domain = urlparse(url).netloc.lower()
        # Remove common prefixes
        for prefix in ['www.', 'm.', 'en.']:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        # Remove .com, .org, etc and capitalize
        name = domain.split('.')[0]
        return name.replace('-', ' ').replace('_', ' ').title()
    except:
        return "Company"


def ai_research_direction(title: str, description: str, api_key: str) -> str:
    """Generate research direction for R&D opportunities"""
    try:
        client = OpenAI(api_key=api_key)

        prompt = f"""Based on this R&D solicitation, suggest a specific research direction or approach:

Title: {title[:200]}
Description: {description[:400]}

Provide 2-3 sentences describing a promising research approach or technology solution. Be specific and innovative."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
            timeout=15
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Consider leveraging cutting-edge technology and innovative methodologies to address the stated research objectives."

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
    st.caption("Google Custom Search configured for vendor discovery")
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1) Solicitation Match",
    "2) Supplier Suggestions",
    "3) Proposal Draft",
    "4) Partner Matches",
    "5) Internal Use",
    "6) Documents"
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
                        "🔍 Stage 1: Pre-filtering solicitations...")
                    progress_bar.progress(20)

                    # Build enhanced company profile
                    prof = st.session_state.get('profile', {}) or {}
                    company_profile_dict = {
                        'company_name': prof.get('company_name', ''),
                        'description': company_desc.strip(),
                        'city': prof.get('city', ''),
                        'state': prof.get('state', '')
                    }

                    # Initialize enhanced matcher
                    matcher = EnhancedMatcher(api_key=OPENAI_API_KEY)

                    status_text.text("🎯 Stage 2: Finding semantic matches...")
                    progress_bar.progress(50)

                    status_text.text("📊 Stage 3: Detailed scoring analysis...")
                    progress_bar.progress(80)

                    # Run complete matching pipeline
                    enhanced_ranked = matcher.match_solicitations(
                        solicitations=df,
                        raw_company_profile=company_profile_dict,
                        prefilter_limit=200,
                        embedding_limit=50,
                        final_limit=int(top_k_select)
                    )

                    progress_bar.progress(100)

                    if enhanced_ranked:
                        # Convert results to dataframe format for compatibility
                        id_order = [x["notice_id"] for x in enhanced_ranked]
                        top_df = df[df["notice_id"].astype(
                            str).isin(id_order)].copy()

                        # Maintain result order
                        preorder = {nid: i for i, nid in enumerate(id_order)}
                        top_df["_order"] = top_df["notice_id"].astype(
                            str).map(preorder)
                        top_df = top_df.sort_values(
                            "_order").drop(columns=["_order"])

                        # Add scores from matching results
                        score_map = {x["notice_id"]: x["score"]
                                     for x in enhanced_ranked}
                        top_df["fit_score"] = top_df["notice_id"].astype(
                            str).map(score_map)

                        # Generate blurbs
                        blurbs = ai_make_blurbs_fast(
                            top_df, OPENAI_API_KEY, max_items=int(top_k_select))
                        top_df["blurb"] = top_df["notice_id"].astype(
                            str).map(blurbs).fillna(top_df["title"])

                        show_df = top_df

                        st.success(
                            f"Top {len(top_df)} matches by enhanced company fit:")
                        render_enhanced_score_results(
                            enhanced_ranked)  # Use new display function

                        st.session_state.topn_df = top_df.reset_index(
                            drop=True)
                        st.session_state.sol_df = top_df.copy()
                        st.session_state.enhanced_ranked = enhanced_ranked
                    else:
                        st.info(
                            "Enhanced matching found no results. Showing manual results.")
                        show_df = df.head(int(limit_results))
                        blurbs = ai_make_blurbs_fast(
                            show_df, OPENAI_API_KEY, max_items=20)
                        show_df["blurb"] = show_df["notice_id"].astype(
                            str).map(blurbs).fillna(show_df["title"])

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
    st.header("Partner Matches (from AI-ranked results)")

    # Need AI-ranked results from Tab 1
    topn = st.session_state.get("topn_df")
    df_companies = companies_df()

    if topn is None or topn.empty:
        st.info(
            "No AI-ranked results available. In Tab 1, run AI ranking to generate matches first.")
    elif df_companies.empty:
        st.info("Your company database is empty. Upload company data below or populate the 'company' table in your database.")

        # Option to upload company data
        uploaded_companies = st.file_uploader("Upload company database (CSV)", type=[
            "csv"], key="company_upload")
        if uploaded_companies is not None:
            try:
                companies_upload_df = pd.read_csv(uploaded_companies)
                if "name" in companies_upload_df.columns and "description" in companies_upload_df.columns:
                    inserted = bulk_insert_companies(companies_upload_df)
                    st.success(
                        f"Uploaded {inserted} companies to the database.")
                    st.rerun()
                else:
                    st.error(
                        "CSV must have at least 'name' and 'description' columns.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
    else:
        # Reuse company description from Tab 1 (stored there)
        company_desc_global = (st.session_state.get(
            "company_desc") or "").strip()
        if not company_desc_global:
            st.info(
                "No company description provided in Tab 1. Please enter one there and rerun.")
        else:
            # Auto-compute matches when Top-n changes or cache is empty
            need_recompute = (
                st.session_state.get("partner_matches") is None or
                st.session_state.get(
                    "partner_matches_stamp") != st.session_state.get("topn_stamp")
            )

            if need_recompute:
                with st.spinner("Analyzing gaps and selecting partners..."):
                    matches = []
                    for _, row in topn.iterrows():
                        title = str(row.get("title", "")) or "Untitled"
                        blurb = str(row.get("blurb", "")).strip()
                        desc = str(row.get("description", "")) or ""
                        sol_text = f"{title}\n\n{desc}"

                        # 1) Identify our capability gaps for this solicitation
                        gaps = ai_identify_gaps(
                            company_desc_global, sol_text, OPENAI_API_KEY)

                        # 2) Pick best partner from company DB to fill those gaps
                        best = pick_best_partner_for_gaps(
                            gaps or sol_text, df_companies, OPENAI_API_KEY, top_n=1)
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
                        ai = ai_partner_justification(
                            partner, sol_text, gaps, OPENAI_API_KEY)

                        matches.append({
                            "title": title,
                            "blurb": blurb,
                            "partner": partner,
                            "gaps": gaps,
                            "ai": ai
                        })

                # Cache results with a stamp tied to the Top-n
                st.session_state.partner_matches = matches
                st.session_state.partner_matches_stamp = st.session_state.get(
                    "topn_stamp")

            # Render cached matches
            matches = st.session_state.get("partner_matches", [])
            if not matches:
                st.info("No partner matches computed yet.")
            else:
                for m in matches:
                    hdr = (m.get("blurb") or m.get(
                        "title") or "Untitled").strip()
                    partner_name = (m.get("partner") or {}).get("name", "")
                    exp_title = f"Opportunity: {hdr}"
                    if partner_name:
                        exp_title += f" — Partner: {partner_name}"

                    with st.expander(exp_title):
                        # Partner block
                        if m.get("partner"):
                            p = m["partner"]
                            loc = ", ".join(
                                [x for x in [p.get("city", ""), p.get("state", "")] if x])
                            st.markdown("**Recommended Partner:**")
                            st.write(f"{p.get('name','')}" +
                                     (f" — {loc}" if loc else ""))
                        else:
                            st.warning(
                                "No suitable partner found for this opportunity.")

                        # Gaps
                        if m.get("gaps"):
                            st.markdown(
                                "**Why we need a partner (our capability gaps):**")
                            st.write(m["gaps"])

                        # Why this partner
                        just = (m.get("ai", {}) or {}).get("justification", "")
                        if just:
                            st.markdown("**Why this partner:**")
                            st.info(just)

                        # Joint proposal idea
                        jp = (m.get("ai", {}) or {}).get(
                            "joint_proposal", "").strip()
                        if jp:
                            st.markdown("**Targeted joint proposal idea:**")
                            st.write(jp)

# Internal Use helper functions


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
    pretrim = ai_downselect_df_optimized(
        company_desc_internal, df_all, OPENAI_API_KEY, top_k=pretrim_cap)

    if pretrim.empty:
        st.info("AI pre-filter returned nothing.")
        return None

    # For Internal Use, we want simpler reasons, not matrix scoring
    # Use a lightweight ranking approach instead
    ranked_simple = []

    # Determine company type for reasoning
    if research_only:
        company_type = "research and development company"
    elif "machine" in st.session_state.get("iu_mode", ""):
        company_type = "machine shop and manufacturing company"
    else:
        company_type = "services company"

    for idx, row in pretrim.head(int(internal_top_k)).iterrows():
        title = row.get("title", "")
        description = row.get("description", "")
        notice_id = str(row.get("notice_id", ""))

        # Generate simple match reason
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            reason_prompt = f"""Explain in 1-2 sentences why this government solicitation would be a good match for a {company_type}:

Title: {title[:200]}
Description: {description[:500]}

Focus on what specific work the {company_type} would actually do. Be concise and practical."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": reason_prompt}],
                temperature=0.3,
                max_tokens=150,
                timeout=15
            )

            reason = response.choices[0].message.content.strip()
        except Exception as e:
            reason = f"This solicitation appears to need {company_type} capabilities."

        ranked_simple.append({
            "notice_id": notice_id,
            "score": 75,  # Default good score for internal use
            "reason": reason 
        })

    if not ranked_simple:
        st.info("No results generated.")
        return None

    # Order by original pretrim order (already relevance-sorted by embeddings)
    id_order = [x["notice_id"] for x in ranked_simple]
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

    # Extract simple reasons
    reason_by_id = {x["notice_id"]: x["reason"] for x in ranked_simple}

    # Add fit scores
    score_by_id = {x["notice_id"]: x["score"] for x in ranked_simple}
    top_df["fit_score"] = top_df["notice_id"].astype(
        str).map(score_by_id).fillna(75).astype(float)

    return {"top_df": top_df, "reason_by_id": reason_by_id}


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


with tab6:
    st.header("📁 Document Management")
    
    if st.session_state.user is None:
        st.info("Please log in to manage documents.")
        st.stop()
    
    user_id = st.session_state.user["id"]
    
    # Upload section
    st.subheader("Upload Document")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "doc", "txt", "csv", "xlsx", "xls", "png", "jpg", "jpeg"],
            key="doc_uploader"
        )
    
    with col2:
        link_to_solicitation = st.checkbox("Link to solicitation", key="link_sol")
    
    description = st.text_input("Description (optional)", key="doc_desc")
    tags = st.text_input("Tags (comma-separated, optional)", key="doc_tags", 
                         placeholder="e.g., proposal, template, reference")
    
    notice_id_link = None
    if link_to_solicitation:
        notice_id_link = st.text_input("Notice ID", key="doc_notice_id")
    
    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.caption(f"File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 10:
            st.warning("⚠️ File is larger than 10 MB. Consider compressing it.")
        
        if st.button("💾 Save Document", type="primary"):
            file_content = uploaded_file.getvalue()
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            doc_id = save_document(
                user_id=user_id,
                filename=uploaded_file.name,
                file_content=file_content,
                file_type=file_type,
                description=description,
                tags=tags,
                notice_id=notice_id_link
            )
            
            if doc_id:
                st.success(f"✅ Document '{uploaded_file.name}' saved successfully!")
                st.rerun()
            else:
                st.error("Failed to save document. Please try again.")
    
    st.markdown("---")
    
    # Document list section
    st.subheader("My Documents")
    
    docs_df = get_user_documents(user_id)
    
    if docs_df.empty:
        st.info("No documents uploaded yet. Upload your first document above!")
    else:
        # Add search/filter
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_term = st.text_input("🔍 Search documents", key="doc_search")
        with col2:
            file_type_filter = st.multiselect(
                "Filter by type",
                options=docs_df["file_type"].unique().tolist(),
                key="doc_type_filter"
            )
        with col3:
            sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Name"], key="doc_sort")
        
        # Apply filters
        filtered_df = docs_df.copy()
        
        if search_term:
            search_mask = (
                filtered_df["filename"].str.contains(search_term, case=False, na=False) |
                filtered_df["description"].fillna("").str.contains(search_term, case=False, na=False) |
                filtered_df["tags"].fillna("").str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        if file_type_filter:
            filtered_df = filtered_df[filtered_df["file_type"].isin(file_type_filter)]
        
        # Apply sorting
        if sort_by == "Newest":
            filtered_df = filtered_df.sort_values("uploaded_at", ascending=False)
        elif sort_by == "Oldest":
            filtered_df = filtered_df.sort_values("uploaded_at", ascending=True)
        else:  # Name
            filtered_df = filtered_df.sort_values("filename")
        
        st.caption(f"Showing {len(filtered_df)} of {len(docs_df)} documents")
        
        # Display documents
        for idx, row in filtered_df.iterrows():
            with st.expander(f"📄 {row['filename']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Type:** {row['file_type'].upper()}")
                    st.write(f"**Size:** {row['file_size'] / 1024:.1f} KB")
                    st.write(f"**Uploaded:** {row['uploaded_at'][:10]}")
                    
                    if row['description']:
                        st.write(f"**Description:** {row['description']}")
                    
                    if row['tags']:
                        tags_list = [t.strip() for t in row['tags'].split(",")]
                        st.write("**Tags:** " + ", ".join([f"`{t}`" for t in tags_list]))
                    
                    if row['notice_id']:
                        notice_url = make_sam_public_url(row['notice_id'])
                        st.write(f"**Linked to:** [{row['notice_id']}]({notice_url})")
                
                with col2:
                    # Download button
                    doc_content = get_document_content(row['id'], user_id)
                    if doc_content:
                        filename, file_type, content = doc_content
                        st.download_button(
                            label="📥 Download",
                            data=content,
                            file_name=filename,
                            mime=f"application/{file_type}",
                            key=f"download_{row['id']}"
                        )
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{row['id']}", type="secondary"):
                        if delete_document(row['id'], user_id):
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document.")
                
                # Edit metadata
                with st.form(key=f"edit_form_{row['id']}"):
                    st.write("**Edit Metadata**")
                    new_desc = st.text_input("Description", value=row['description'] or "", key=f"new_desc_{row['id']}")
                    new_tags = st.text_input("Tags", value=row['tags'] or "", key=f"new_tags_{row['id']}")
                    new_notice = st.text_input("Notice ID", value=row['notice_id'] or "", key=f"new_notice_{row['id']}")
                    
                    if st.form_submit_button("💾 Update"):
                        if update_document_metadata(row['id'], user_id, new_desc, new_tags, new_notice or None):
                            st.success("Metadata updated!")
                            st.rerun()
try:
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE IF NOT EXISTS solicitation_embeddings (
                notice_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """))
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_hash 
            ON solicitation_embeddings (text_hash)
        """))
except Exception as e:
    st.warning(f"Embedding table creation note: {e}")

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
Add the documents table code RIGHT AFTER the auth tokens table block:
python# Create auth tokens table
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

# ============= ADD THIS BLOCK HERE =============
# Create documents table
try:
    with engine.begin() as conn:
        # For PostgreSQL
        if engine.url.get_dialect().name == 'postgresql':
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content BYTEA NOT NULL,
                    description TEXT,
                    tags TEXT,
                    notice_id TEXT,
                    uploaded_at TEXT NOT NULL
                )
            """))
        else:
            # For SQLite
            conn.execute(sa.text("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content BLOB NOT NULL,
                    description TEXT,
                    tags TEXT,
                    notice_id TEXT,
                    uploaded_at TEXT NOT NULL
                )
            """))
        
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id)
        """))
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_documents_notice ON documents (notice_id)
        """))
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_documents_uploaded ON documents (uploaded_at)
        """))
except Exception as e:
    st.warning(f"Document table creation note: {e}")
# ============= END OF NEW BLOCK =============

# Create unique indexes
try:
    with engine.begin() as conn:
        conn.execute(
            sa.text("CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON users (email);"))
        conn.execute(sa.text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_company_profile_user ON company_profile (user_id);"))
except Exception as e:
    st.warning(f"User/profile table migration note: {e}")