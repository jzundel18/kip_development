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
               naics_code, set_aside_code, description, link, pop_state, pop_zip, pop_country, pop_raw
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
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def load_secrets():
    """Load and validate secrets - only required in production"""
    openai_key = get_secret("OPENAI_API_KEY")
    google_key = get_secret("GOOGLE_API_KEY")
    google_cx = get_secret("GOOGLE_CX")
    sam_keys_raw = get_secret("SAM_KEYS", "")

    # Parse SAM_KEYS
    sam_keys = []
    if isinstance(sam_keys_raw, str):
        sam_keys = [k.strip() for k in sam_keys_raw.split(",") if k.strip()]
    elif isinstance(sam_keys_raw, (list, tuple)):
        sam_keys = sam_keys_raw

    # Check if we're in production (has secrets) or local dev (no secrets)
    missing = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not google_key:
        missing.append("GOOGLE_API_KEY")
    if not google_cx:
        missing.append("GOOGLE_CX")
    if not sam_keys:
        missing.append("SAM_KEYS")

    # Only error if we're clearly trying to run in production
    # (Check if we're on Streamlit Cloud by looking for any configured secrets)
    try:
        has_any_secrets = len(st.secrets) > 0
    except:
        has_any_secrets = False

    if missing and has_any_secrets:
        # We're on Streamlit Cloud but missing some secrets
        st.error(f"❌ Missing required secrets: {', '.join(missing)}")
        st.info(
            "📝 Add these in: **⚙️ Settings → Secrets** in your Streamlit Cloud dashboard")
        st.code("""
# Example secrets format:
OPENAI_API_KEY = "sk-proj-..."
GOOGLE_API_KEY = "AIza..."
GOOGLE_CX = "your-cx-id"
SAM_KEYS = "key1,key2"
        """, language="toml")
        st.stop()
    elif missing and not has_any_secrets:
        # Running locally with no secrets - show helpful message
        st.warning("⚠️ Running in local mode without secrets")
        st.info(
            "This app requires secrets to function. Deploy to Streamlit Cloud and add secrets there.")
        st.stop()

    return openai_key, google_key, google_cx, sam_keys


# =========================
# Database Configuration
# =========================
# =========================
# Database Configuration - SUPABASE ONLY
# =========================
DB_URL = st.secrets.get("SUPABASE_DB_URL")

if not DB_URL:
    st.error("❌ SUPABASE_DB_URL not configured")
    st.info("Add SUPABASE_DB_URL to your Streamlit Cloud secrets")
    st.stop()

if not DB_URL.startswith("postgresql"):
    st.error("❌ Only PostgreSQL (Supabase) databases are supported")
    st.info("This app requires a PostgreSQL connection string")
    st.stop()


@st.cache_resource
def get_optimized_engine(db_url: str):
    """Create optimized database engine with connection pooling for Supabase"""
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


engine = get_optimized_engine(DB_URL)

# Test Supabase connection
try:
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1")).first()
        ver = conn.execute(sa.text("SELECT version()")).first()
    st.sidebar.success("✅ Connected to Supabase")
    if ver and isinstance(ver, tuple):
        # Show just "PostgreSQL 15.x" instead of full version string
        version_short = ver[0].split(',')[0] if ver[0] else "PostgreSQL"
        st.sidebar.caption(version_short)
except Exception as e:
    st.sidebar.error("❌ Supabase connection failed")
    st.sidebar.caption("Check your SUPABASE_DB_URL secret")
    st.sidebar.exception(e)
    st.stop()

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

# Add NAICS indexes for company_list
try:
    with engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE INDEX IF NOT EXISTS idx_company_naics 
            ON company_list (NAICS)
        """))
except Exception as e:
    st.warning(f"NAICS index creation note: {e}")

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
    "pop_state": "TEXT", "pop_country": "TEXT",
    "pop_zip": "TEXT", "pop_raw": "TEXT",
}

# Migrate company_list table
try:
    insp = inspect(engine)
    if "company_list" in [t.lower() for t in insp.get_table_names()]:
        existing_cols = {c["name"] for c in insp.get_columns("company_list")}
        REQUIRED_COMPANY_COLS = {
            "name": "TEXT",
            "description": "TEXT",
            "state": "TEXT",
            "states_perform_work": "TEXT",
            "NAICS": "TEXT",
            "other_NAICS": "TEXT",
            "email": "TEXT",
            "phone": "TEXT",
            "contact": "TEXT",
        }
        missing_cols = [
            c for c in REQUIRED_COMPANY_COLS if c not in existing_cols]
        if missing_cols:
            with engine.begin() as conn:
                for col in missing_cols:
                    conn.execute(
                        sa.text(f'ALTER TABLE company_list ADD COLUMN "{col}" {REQUIRED_COMPANY_COLS[col]}'))
                st.info(
                    f"Added {len(missing_cols)} columns to company_list table")
except Exception as e:
    st.warning(f"Company_list table migration note: {e}")

# Company table migration (add after your existing migrations)
try:
    insp = inspect(engine)
    if "company" in [t.lower() for t in insp.get_table_names()]:
        existing_cols = {c["name"] for c in insp.get_columns("company")}
        REQUIRED_COMPANY_COLS = {
            "name": "TEXT",
            "description": "TEXT",
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

def get_profile(user_id: int) -> Optional[dict]:
    with engine.connect() as conn:
        sql = sa.text("""
            SELECT id, user_id, company_name, description, state, created_at, updated_at
            FROM company_profile WHERE user_id = :uid
        """)
        row = conn.execute(sql, {"uid": user_id}).mappings().first()
        return dict(row) if row else None


def upsert_profile(user_id: int, company_name: str, description: str, state: str) -> None:
    with engine.begin() as conn:
        now = datetime.now(timezone.utc).isoformat()
        upd = conn.execute(sa.text("""
            UPDATE company_profile
            SET company_name = :cn, description = :d, state = :s, updated_at = :ts
            WHERE user_id = :uid
        """), {"cn": company_name, "d": description, "s": state, "uid": user_id, "ts": now})
        if upd.rowcount == 0:
            conn.execute(sa.text("""
                INSERT INTO company_profile (user_id, company_name, description, state, created_at, updated_at)
                VALUES (:uid, :cn, :d, :s, :ts, :ts)
            """), {"uid": user_id, "cn": company_name, "d": description, "s": state, "ts": now})

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
            "location": f'{company_row.get("state", "")}',        },
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
                "SELECT name, description, state, states_perform_work, NAICS, other_NAICS, email, phone, contact FROM company_list ORDER BY name",
                conn
            )
        except Exception:
            return pd.DataFrame(columns=["name", "description", "state", "states_perform_work", "NAICS", "other_NAICS", "email", "phone", "contact"])


def bulk_insert_companies(df: pd.DataFrame) -> int:
    """Insert multiple companies into the database"""
    needed = ["name", "description", "state", "states_perform_work", "NAICS", "other_NAICS", "email", "phone", "contact"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""

    with engine.begin() as conn:
        rows = df[needed].fillna("").to_dict(orient="records")
        for row in rows:
            conn.execute(sa.text("""
                INSERT INTO company_list (name, description, state, states_perform_work, NAICS, other_NAICS, email, phone, contact)
                VALUES (:name, :description, :state, :states_perform_work, :NAICS, :other_NAICS, :email, :phone, :contact)
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

                # Around line 1590, replace the vendor finding logic with:

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
                            "pop_state": _s(getattr(row, "pop_state", "")),
                        }
                    except Exception as e:
                        st.error(f"Error creating solicitation data: {e}")
                        continue

                    locality = {"state": _s(getattr(row, "pop_state", ""))}
                    
                    pop_state = locality.get("state", "").strip()
                    sol_text = f"{sol_dict['title']}\n{sol_dict['description']}"
                    
                    # STEP 1: Check internal database first
                    sol_naics = sol_dict.get('naics_code', '')
                    internal_vendors = search_company_database(
                        sol_text, 
                        pop_state, 
                        OPENAI_API_KEY,
                        solicitation_naics=sol_naics
                        )

                    if not internal_vendors.empty:
                        st.session_state.vendor_notes[nid] = f"Found {len(internal_vendors)} matches (NAICS-filtered)"
                    else:
                        # STEP 2: Fall back to Google search
                        st.session_state.vendor_notes[nid] = "No internal matches found - searching external vendors"
                        
                        try:
                            vendors_df, note = find_service_vendors_for_opportunity(
                                sol_dict, 
                                GOOGLE_API_KEY, 
                                GOOGLE_CX, 
                                OPENAI_API_KEY, 
                                top_n=3
                            )
                            
                            if vendors_df is None or vendors_df.empty:
                                st.session_state.vendor_errors[nid] = "No vendors found in database or external search"
                            else:
                                st.session_state.vendor_suggestions[nid] = vendors_df
                                st.session_state.vendor_errors.pop(nid, None)
                                
                        except Exception as e:
                            st.session_state.vendor_errors[nid] = f"Error finding vendors: {str(e)[:100]}"
                    
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
                        
                        # Get contact info
                        email = (v.get("email") or "").strip()
                        phone = (v.get("phone") or "").strip()
                        contact = (v.get("contact") or "").strip()

                        display_name = raw_name or _company_name_from_url(website) or "Unnamed Vendor"
                        
                        if website:
                            st.markdown(f"- **[{display_name}]({website})**")
                        else:
                            st.markdown(f"- **{display_name}**")
                        
                        if location:
                            st.caption(f"📍 {location}")
                        
                        # Display contact information
                        contact_info = []
                        if contact:
                            contact_info.append(f"Contact: {contact}")
                        if phone:
                            contact_info.append(f"📞 {phone}")
                        if email:
                            contact_info.append(f"✉️ {email}")
                        
                        if contact_info:
                            st.caption(" | ".join(contact_info))
                        
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
               naics_code, set_aside_code, description, link, pop_state, pop_zip, pop_country, pop_raw
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


def naics_prefilter_companies(df_companies: pd.DataFrame, solicitation_naics: str) -> pd.DataFrame:
    """
    Pre-filter companies by NAICS code match before doing expensive semantic search.
    Matches on both primary NAICS and other_NAICS fields.
    """
    if not solicitation_naics or not solicitation_naics.strip():
        return df_companies

    # Clean NAICS code (remove non-digits)
    sol_naics = re.sub(r'[^\d]', '', str(solicitation_naics)).strip()
    if not sol_naics:
        return df_companies

    # Get first 2 digits for flexible matching
    sol_naics_2 = sol_naics[:2] if len(sol_naics) >= 2 else sol_naics

    def matches_naics(row):
        """Check if company NAICS matches solicitation NAICS"""
        # Check primary NAICS
        primary = re.sub(r'[^\d]', '', str(row.get('NAICS', ''))).strip()
        if primary:
            if primary == sol_naics or primary.startswith(sol_naics_2):
                return True

        # Check other_NAICS (comma-separated list)
        other = str(row.get('other_NAICS', ''))
        if other and other != 'nan':
            for naics_code in other.split(','):
                naics_clean = re.sub(r'[^\d]', '', naics_code.strip())
                if naics_clean:
                    if naics_clean == sol_naics or naics_clean.startswith(sol_naics_2):
                        return True

        return False

    # Apply filter
    filtered = df_companies[df_companies.apply(matches_naics, axis=1)]

    return filtered if not filtered.empty else df_companies


def search_company_database(solicitation_desc: str, pop_state: str, api_key: str, solicitation_naics: str = "") -> pd.DataFrame:
    """
    ENHANCED: Search internal company database with NAICS pre-filtering and full contact info
    """
    try:
        with engine.connect() as conn:
            # Get all companies with ALL fields including contact info
            df = pd.read_sql_query(
                "SELECT name, description, state, states_perform_work, NAICS, other_NAICS, email, phone, contact FROM company_list",
                conn
            )

        if df.empty:
            return pd.DataFrame(columns=["name", "website", "location", "reason", "email", "phone", "contact"])

        # STEP 1: NAICS PRE-FILTER (if NAICS provided)
        if solicitation_naics:
            df_naics_filtered = naics_prefilter_companies(
                df, solicitation_naics)
            if not df_naics_filtered.empty:
                df = df_naics_filtered

        # STEP 2: Check for national scope
        is_national = not pop_state or pop_state.strip() == ""

        # STEP 3: Filter by state if not national
        if not is_national:
            df = df[
                (df["state"].str.upper() == pop_state.upper()) |
                (df["states_perform_work"].fillna(
                    "").str.upper().str.contains(pop_state.upper()))
            ]

        if df.empty:
            return pd.DataFrame(columns=["name", "website", "location", "reason", "email", "phone", "contact"])

        # STEP 4: Score companies by description match using embeddings
        client = OpenAI(api_key=api_key)

        sol_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[solicitation_desc[:500]]
        )
        sol_vector = np.array(sol_resp.data[0].embedding, dtype=np.float32)
        sol_vector = sol_vector / (np.linalg.norm(sol_vector) + 1e-9)

        company_descs = df["description"].fillna("").tolist()
        comp_resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=company_descs
        )
        comp_vectors = np.array(
            [d.embedding for d in comp_resp.data], dtype=np.float32)
        comp_vectors = comp_vectors / \
            (np.linalg.norm(comp_vectors, axis=1, keepdims=True) + 1e-9)

        similarities = comp_vectors @ sol_vector
        df["score"] = similarities
        df = df[df["score"] > 0.3].sort_values(
            "score", ascending=False).head(3)

        # STEP 5: Format results WITH CONTACT INFO
        results = []
        for _, row in df.iterrows():
            location_parts = []
            if row.get('state'):
                location_parts.append(row['state'])
            if row.get('states_perform_work'):
                states_work = row['states_perform_work']
                if states_work and states_work != row.get('state'):
                    location_parts.append(f"Also serves: {states_work}")

            location = " | ".join(
                location_parts) if location_parts else "Location not specified"

            # Build NAICS display
            naics_display = ""
            if row.get('NAICS'):
                naics_display = f"NAICS: {row['NAICS']}"
                if row.get('other_NAICS') and str(row['other_NAICS']) != 'nan':
                    naics_display += f" (Also: {row['other_NAICS']})"

            results.append({
                "name": row["name"],
                "website": row.get("email", ""),
                "location": location,
                "reason": f"Internal database match. {naics_display}",
                "email": row.get("email", ""),
                "phone": row.get("phone", ""),
                "contact": row.get("contact", "")
            })

        return pd.DataFrame(results)

    except Exception as e:
        return pd.DataFrame(columns=["name", "website", "location", "reason", "email", "phone", "contact"])

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
        pop_state = (solicitation.get("pop_state") or "").strip()

        if pop_state:
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

        # Build location-aware search query with fallbacks
        debug_log("")
        debug_log("**Step 2: Building search queries...**")

        base_query = service_type.strip()

        # Try multiple query variations
        search_queries = []

        if pop_state:
            search_queries.append(f"{base_query} {pop_state}")
            search_queries.append(f"{base_query} contractors {pop_state}")
            search_note = f"Searching for providers in {pop_state}"
        else:
            search_queries.append(f"{base_query} contractors")
            search_queries.append(f"{base_query} companies")
            search_note = "No specific location found - conducting national search"

        # Always add a generic fallback
        search_queries.append(f"contractors {base_query}")

        debug_log(f"Will try {len(search_queries)} search variations")

        # Try each query until we get results
        search_results = []
        successful_query = None

        for search_query in search_queries:
            debug_log(f"Trying query: `{search_query}`")

            google_params = {
                "key": google_api_key,
                "cx": google_cx,
                "q": search_query,
                "num": 10,
                "safe": "off",
                "lr": "lang_en",
                "filter": "0",
            }

            response = requests.get(
                "https://www.googleapis.com/customsearch/v1", params=google_params, timeout=15)

            if response.status_code != 200:
                debug_log(f"   ⚠️ API error: {response.status_code}")
                continue

            data = response.json()
            items = data.get("items", [])

            debug_log(f"   → Returned {len(items)} results")

            if items:
                search_results = items
                successful_query = search_query
                break

        if not search_results:
            debug_log("❌ **No search results found from any query variation**")
            return None, "No search results found"

        debug_log("")
        debug_log(f"✅ Using results from: `{successful_query}`")
        debug_log(f"✅ Google returned **{len(search_results)} results**")

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

            # Check domain (not full URL) for skip domains
            if any(skip_domain in domain for skip_domain in skip_domains):
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
    state = (locality.get("state") or "").strip()
    return bool(state)


def _extract_locality(text: str) -> dict:
    """Extract state from text using simple regex patterns"""

    # Common patterns for state
    patterns = [
        r'(\w+(?:\s+\w+)*),\s*([A-Z]{2})',  # "Name, ST"
        r'(\w+(?:\s+\w+)*)\s+([A-Z]{2})\s',  # "Name ST "
        # Proper case cities
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),?\s*([A-Z]{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return {
                "state": match.group(1).strip()
            }

    # Just look for state codes
    state_match = re.search(r'\b([A-Z]{2})\b', text)
    if state_match:
        return {
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


if st.session_state.view == "account":
    st.title("Company Settings")
    
    if st.button("← Back to App", key="btn_back_to_app"):
        st.session_state.view = "main"
        st.rerun()
    
    prof = st.session_state.profile or {}
    company_name = st.text_input("Company name", value=prof.get("company_name", ""))
    description = st.text_area("Company description", value=prof.get("description", ""), height=140)
    state = st.text_input("State (2-letter code)", value=prof.get("state", "") or "")
    
    if st.button("💾 Save Profile", key="btn_save_profile_settings", type="primary"):
        if not company_name.strip() or not description.strip():
            st.error("Company name and description are required.")
        else:
            upsert_profile(st.session_state.user["id"], company_name.strip(), description.strip(), state.strip())
            st.session_state.profile = get_profile(st.session_state.user["id"])
            st.success("Profile saved!")
            st.rerun()
    st.stop()

# Minimal session state for app functionality only
if "sol_df" not in st.session_state:
    st.session_state.sol_df = None
if "topn_df" not in st.session_state:
    st.session_state.topn_df = None
if "vendor_suggestions" not in st.session_state:
    st.session_state.vendor_suggestions = {}
if "vendor_errors" not in st.session_state:
    st.session_state.vendor_errors = {}
if "vendor_notes" not in st.session_state:
    st.session_state.vendor_notes = {}
if "iu_results" not in st.session_state:
    st.session_state.iu_results = None
if "iu_key_salt" not in st.session_state:
    st.session_state.iu_key_salt = ""
if "iu_mode" not in st.session_state:
    st.session_state.iu_mode = ""
if "db_optimized" not in st.session_state:
    st.session_state.db_optimized = False
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
    saved_desc = ""
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
    st.header("📝 Proposal Draft Generator")
    st.caption("Generate structured DoD proposal outlines based on solicitation requirements")
    
    # Check if we have solicitations to work with
    if st.session_state.sol_df is None or st.session_state.sol_df.empty:
        st.info("👈 First, go to Tab 1 to find and filter solicitations")
        st.stop()
    
    # Solicitation selection
    st.subheader("1. Select Solicitation")
    
    # Create dropdown with solicitation titles
    sol_options = {}
    for _, row in st.session_state.sol_df.iterrows():
        notice_id = str(row.get("notice_id", ""))
        title = row.get("title", "Untitled")[:80]
        sol_number = row.get("solicitation_number", "")
        display_text = f"{title} ({sol_number})" if sol_number else title
        sol_options[display_text] = notice_id
    
    selected_display = st.selectbox(
        "Choose a solicitation",
        options=list(sol_options.keys()),
        key="proposal_sol_select"
    )
    
    selected_notice_id = sol_options[selected_display]
    selected_sol = st.session_state.sol_df[
        st.session_state.sol_df["notice_id"].astype(str) == selected_notice_id
    ].iloc[0]
    
    # Display solicitation details
    with st.expander("📋 Solicitation Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Title:** {selected_sol.get('title', 'N/A')}")
            st.write(f"**Solicitation #:** {selected_sol.get('solicitation_number', 'N/A')}")
            st.write(f"**Notice Type:** {selected_sol.get('notice_type', 'N/A')}")
            st.write(f"**NAICS:** {selected_sol.get('naics_code', 'N/A')}")
        with col2:
            st.write(f"**Posted:** {selected_sol.get('posted_date', 'N/A')}")
            st.write(f"**Response Due:** {selected_sol.get('response_date', 'N/A')}")
            st.write(f"**Set-Aside:** {selected_sol.get('set_aside_code', 'N/A')}")
            link = make_sam_public_url(selected_notice_id, selected_sol.get('link', ''))
            st.markdown(f"**[View on SAM.gov]({link})**")
    
    # Proposal configuration
    st.subheader("2. Proposal Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        proposal_type = st.selectbox(
            "Proposal Type",
            ["Technical & Cost Proposal", "Technical Proposal Only", "Cost Proposal Only", "Statement of Qualifications"],
            key="proposal_type"
        )
    with col2:
        page_limit = st.number_input(
            "Target Page Count",
            min_value=1,
            max_value=100,
            value=15,
            help="Approximate number of pages for the proposal",
            key="proposal_pages"
        )
    
    # Company information (from profile)
    use_profile = st.checkbox("Use saved company profile", value=True, key="proposal_use_profile")
    
    if use_profile and st.session_state.profile:
        prof = st.session_state.profile
        company_name = prof.get("company_name", "")
        company_desc = prof.get("description", "")
        company_state = prof.get("state", "")

        st.info(f"Using profile: **{company_name}**")
    else:
        st.warning(
            "⚠️ No company profile found. Go to Account Settings to create one.")
        company_name = st.text_input("Company Name", key="proposal_company_name")
        company_desc = st.text_area(
            "Company Description", height=100, key="proposal_company_desc")
        company_state = st.text_input("State", key="proposal_company_state")
    
    # Additional inputs
    with st.expander("⚙️ Additional Details (Optional)"):
        key_personnel = st.text_area(
            "Key Personnel (one per line)",
            placeholder="Jane Smith - Program Manager\nJohn Doe - Technical Lead",
            key="proposal_key_personnel",
            height=100
        )
        
        past_performance = st.text_area(
            "Relevant Past Performance",
            placeholder="List 2-3 relevant past projects...",
            key="proposal_past_performance",
            height=100
        )
        
        facilities = st.text_input(
            "Facilities/Capabilities",
            placeholder="e.g., ISO 9001 certified facility, 50,000 sq ft manufacturing space",
            key="proposal_facilities"
        )
    
    # Generate button
    st.subheader("3. Generate Proposal Outline")
    
    if st.button("🚀 Generate Proposal Outline", type="primary", use_container_width=True):
        if not company_name or not company_desc:
            st.error("Company name and description are required")
            st.stop()
        
        with st.spinner("Generating comprehensive proposal outline... This may take 30-60 seconds..."):
            try:
                # Prepare solicitation context
                sol_context = {
                    "title": str(selected_sol.get("title", "")),
                    "description": str(selected_sol.get("description", ""))[:2000],
                    "solicitation_number": str(selected_sol.get("solicitation_number", "")),
                    "naics_code": str(selected_sol.get("naics_code", "")),
                    "set_aside_code": str(selected_sol.get("set_aside_code", "")),
                    "response_date": str(selected_sol.get("response_date", "")),
                    "notice_type": str(selected_sol.get("notice_type", ""))
                }
                
                # Prepare company context
                # Prepare company context
                company_context = {
                    "name": company_name,
                    "description": company_desc[:800],
                    "location": company_state,
                    "key_personnel": [p.strip() for p in key_personnel.split("\n") if p.strip()],
                    "past_performance": past_performance.strip(),
                    "facilities": facilities.strip()
                }
                
                # Generate proposal outline using OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY)
                
                system_prompt = f"""You are an expert DoD proposal writer. Generate a comprehensive {proposal_type} outline 
with specific, actionable content for each section. The proposal should be approximately {page_limit} pages.

Create a detailed outline that includes:
1. Executive Summary with key win themes
2. Technical Approach with specific methodologies
3. Management Approach with organizational structure
4. Past Performance with relevant examples
5. Personnel qualifications
6. Facilities and capabilities
7. Cost/pricing strategy (if applicable)

Make the outline specific to this opportunity - not generic. Include actual content suggestions, not just section headers."""

                user_prompt = f"""Create a detailed proposal outline for this solicitation:

**SOLICITATION:**
Title: {sol_context['title']}
Number: {sol_context['solicitation_number']}
Type: {sol_context['notice_type']}
NAICS: {sol_context['naics_code']}
Set-Aside: {sol_context['set_aside_code']}

Requirements: {sol_context['description'][:1500]}

**COMPANY:**
Name: {company_context['name']}
Capabilities: {company_context['description'][:600]}
Location: {company_context['location']}
{"Key Personnel: " + ", ".join(company_context['key_personnel'][:3]) if company_context['key_personnel'] else ""}
{"Facilities: " + company_context['facilities'] if company_context['facilities'] else ""}

**REQUIREMENTS:**
- Proposal Type: {proposal_type}
- Target Length: {page_limit} pages
- Focus on DoD/Federal contracting best practices
- Include win themes and discriminators
- Provide specific content guidance for each section

Generate a comprehensive outline with detailed guidance for each section."""

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=4000
                )
                
                proposal_outline = response.choices[0].message.content
                
                # Store in session state
                st.session_state.proposal_outline = proposal_outline
                st.session_state.proposal_metadata = {
                    "solicitation": sol_context,
                    "company": company_context,
                    "type": proposal_type,
                    "pages": page_limit,
                    "generated_at": datetime.now().isoformat()
                }
                
                st.success("✅ Proposal outline generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating proposal: {e}")
                st.exception(e)
    
    # Display generated outline
    if "proposal_outline" in st.session_state and st.session_state.proposal_outline:
        st.markdown("---")
        st.subheader("📄 Generated Proposal Outline")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download as markdown
            st.download_button(
                label="📥 Download Markdown",
                data=st.session_state.proposal_outline,
                file_name=f"proposal_outline_{selected_sol.get('solicitation_number', 'draft')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col2:
            # Download as text
            st.download_button(
                label="📥 Download Text",
                data=st.session_state.proposal_outline,
                file_name=f"proposal_outline_{selected_sol.get('solicitation_number', 'draft')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            # Copy to clipboard button (display only)
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.info("Select the text below and use Ctrl+C (Cmd+C on Mac) to copy")
        
        # Display the outline
        st.markdown(st.session_state.proposal_outline)
        
        # Metadata
        if "proposal_metadata" in st.session_state:
            with st.expander("ℹ️ Proposal Metadata"):
                meta = st.session_state.proposal_metadata
                st.json(meta)

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
                                [x for x in [p.get("state", "")] if x])
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