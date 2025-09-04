# models.py
import pickle
import hashlib
from typing import Optional
from sqlmodel import SQLModel, Field
import OpenAI

class SolicitationRaw(SQLModel, table=True):
    __tablename__ = "solicitationraw"
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

class User(SQLModel, table=True):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True, nullable=False)
    password_hash: str = Field(nullable=False)
    created_at: Optional[str] = Field(default=None)

class CompanyProfile(SQLModel, table=True):
    __tablename__ = "company_profile"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, nullable=False)
    company_name: str = Field(nullable=False)
    description: str = Field(nullable=False)
    city: Optional[str] = None
    state: Optional[str] = None
    created_at: Optional[str] = Field(default=None)
    updated_at: Optional[str] = Field(default=None)

class Company(SQLModel, table=True):
    __tablename__ = "company"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    description: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = Field(default=None, index=True)

# Add to models.py


class SolicitationEmbedding(SQLModel, table=True):
    __tablename__ = "solicitation_embeddings"
    __table_args__ = {"extend_existing": True}

    notice_id: str = Field(primary_key=True, index=True)
    embedding: str = Field()  # JSON-encoded numpy array
    # Hash of title+description to detect changes
    text_hash: str = Field(index=True)
    created_at: str = Field(default=None)


# Add to app.py - optimized embedding functions


@st.cache_data(show_spinner=False, ttl=86400)  # 24 hour cache
def get_cached_embeddings(text_hashes: list[str]) -> dict[str, np.ndarray]:
    """Fetch cached embeddings from database"""
    if not text_hashes:
        return {}

    with engine.connect() as conn:
        placeholders = ",".join("?" if engine.url.get_dialect(
        ).name == "sqlite" else "%s" for _ in text_hashes)
        df = pd.read_sql_query(
            f"SELECT notice_id, embedding, text_hash FROM solicitation_embeddings WHERE text_hash IN ({placeholders})",
            conn,
            params=text_hashes
        )

    result = {}
    for _, row in df.iterrows():
        try:
            # Deserialize numpy array from JSON
            embedding_data = json.loads(row['embedding'])
            result[row['text_hash']] = np.array(
                embedding_data, dtype=np.float32)
        except Exception:
            continue

    return result


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
