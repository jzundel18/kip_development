#!/usr/bin/env python3
"""
Daily email digest:
- For each subscriber, fetch notices posted yesterday (UTC) from solicitationraw
- Score against their company description (embeddings)
- Email up to N top matches (or "no close match today")

Required env vars (set as GitHub Actions repo secrets):
  SUPABASE_DB_URL     -> e.g., postgresql+psycopg2://... (same as your app)
  OPENAI_API_KEY      -> for embeddings and blurbs
  SENDGRID_API_KEY    -> for sending email
  FROM_EMAIL          -> verified sender, e.g. alerts@yourdomain.com
  APP_BASE_URL        -> optional, used for UI links (e.g., https://yourapp.example)
Optional:
  DIGEST_MAX_RESULTS       -> default 5
  DIGEST_MIN_SIMILARITY    -> default 0.35 (0..1 cosine with normalized vectors)
"""

import os
import sys
import math
import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text

# deps: openai (>=1.x), sendgrid
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

# ---------- Config ----------
DB_URL = os.getenv("SUPABASE_DB_URL", "sqlite:///app.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")

MAX_RESULTS = int(os.getenv("DIGEST_MAX_RESULTS", "5"))
MIN_SIM = float(os.getenv("DIGEST_MIN_SIMILARITY", "0.35"))

if not (DB_URL and OPENAI_API_KEY and SENDGRID_API_KEY and FROM_EMAIL):
    print("Missing required env vars. Need SUPABASE_DB_URL, OPENAI_API_KEY, SENDGRID_API_KEY, FROM_EMAIL.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

engine = sa.create_engine(DB_URL, pool_pre_ping=True)
oai = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Helpers ----------


def _iso_utc_floor_day(dt: datetime) -> datetime:
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def _yesterday_utc_window():
    now = datetime.now(timezone.utc)
    end = _iso_utc_floor_day(now)                # today 00:00 UTC
    start = end - timedelta(days=1)              # yesterday 00:00 UTC
    return start, end


def _fetch_subscribers(conn) -> pd.DataFrame:
    """
    Expect a table `digest_subscribers`:
      id SERIAL PK
      user_id INT NOT NULL
      email TEXT NOT NULL
      is_active BOOL DEFAULT TRUE

    And we will try to pull company description from `company_profile` (description column).
    """
    # Minimal, resilient fetch:
    try:
        df = pd.read_sql_query(
            """
            SELECT s.user_id,
                   s.email,
                   COALESCE(cp.description, '') AS company_description
            FROM digest_subscribers s
            LEFT JOIN company_profile cp ON cp.user_id = s.user_id
            WHERE COALESCE(s.is_active, TRUE) = TRUE
            """,
            conn
        )
    except Exception as e:
        logging.error("Error reading subscribers/company_profile: %s", e)
        return pd.DataFrame(columns=["user_id", "email", "company_description"])
    return df.fillna("")


def _fetch_yesterday_notices(conn, start_iso: str, end_iso: str) -> pd.DataFrame:
    """
    solicitationraw.posted_date is TEXT but your app usually stores ISO-8601.
    Lexicographic string compare works for ISO timestamps, so we filter in SQL.
    """
    cols = [
        "notice_id", "title", "description", "naics_code", "set_aside_code",
        "posted_date", "response_date", "link"
    ]
    try:
        df = pd.read_sql_query(
            f"""
            SELECT {", ".join(cols)}
            FROM solicitationraw
            WHERE posted_date >= :start AND posted_date < :end
            """,
            conn,
            params={"start": start_iso, "end": end_iso},
        )
    except Exception as e:
        logging.error("Error reading solicitationraw: %s", e)
        return pd.DataFrame(columns=cols)
    return df.fillna("")


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings (float32)."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    r = oai.embeddings.create(model="text-embedding-3-small", input=texts)
    X = np.array([d.embedding for d in r.data], dtype=np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def _score_matches(company_desc: str, notices: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine similarity between company_desc and (title + description).
    Return df with 'sim' column, sorted desc.
    """
    if notices.empty or not company_desc.strip():
        d = notices.copy()
        d["sim"] = 0.0
        return d

    # Query vector
    Xq = _embed_texts([company_desc.strip()])[0]

    # Candidate vectors
    texts = (notices["title"].fillna("") + " " +
             notices["description"].fillna("")).str.slice(0, 2000).tolist()
    X = _embed_texts(texts)

    sims = X @ Xq  # cosine with normalized vectors
    out = notices.copy()
    out["sim"] = sims
    out.sort_values("sim", ascending=False, inplace=True)
    return out


def _make_blurbs(rows: list[dict]) -> dict[str, str]:
    """Short (~10 words) blurbs with OpenAI; keep it tiny and robust."""
    if not rows:
        return {}
    items = []
    for r in rows[:20]:
        items.append({"id": str(r.get("notice_id", "")), "title": r.get(
            "title", "")[:200], "description": r.get("description", "")[:600]})

    sys = "You are concise. For each item, write a ~10-word plain-English blurb of what the solicitation needs. Return JSON."
    user = {"items": items, "format": {
        "blurbs": [{"id": "string", "blurb": "string"}]}}
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": sys}, {
                "role": "user", "content": json.dumps(user)}],
            temperature=0.2,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        return {str(d.get("id", "")): str(d.get("blurb", "")).strip() for d in data.get("blurbs", [])}
    except Exception as e:
        logging.warning(
            "Blurb generation failed; sending without blurbs (%s).", e)
        return {}


def _sam_public_url(notice_id: str, link: str | None) -> str:
    if link and "api.sam.gov" not in link:
        return link
    nid = (notice_id or "").strip()
    return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"


def _send_email(to_email: str, subject: str, html_body: str, text_body: str = ""):
    sg = SendGridAPIClient(SENDGRID_API_KEY)
    mail = Mail(
        from_email=Email(FROM_EMAIL),
        to_emails=[To(to_email)],
        subject=subject,
        html_content=Content("text/html", html_body),
    )
    if text_body:
        mail.add_content(Content("text/plain", text_body))
    try:
        resp = sg.send(mail)
        logging.info("Sent to %s (status %s)", to_email, resp.status_code)
    except Exception as e:
        logging.error("SendGrid error to %s: %s", to_email, e)


def _render_email(subscriber_email: str, company_desc: str, picks: pd.DataFrame) -> tuple[str, str]:
    """
    Return (subject, html). If no picks, we say 'no close matches today'.
    """
    if picks.empty:
        subject = "KIP Daily: no close matches today"
        html = f"""
        <p>Hi,</p>
        <p>No close matches were found for your company description yesterday.</p>
        <p>You’ll receive up to {MAX_RESULTS} new matches when there are good fits.</p>
        """
        return subject, html

    # optional blurbs
    blurbs = _make_blurbs(picks.to_dict(orient="records"))
    rows_html = []
    for _, r in picks.iterrows():
        nid = str(r.get("notice_id", ""))
        title = (r.get("title", "") or "Untitled").strip()
        sim = float(r.get("sim", 0.0))
        url = _sam_public_url(nid, r.get("link", ""))
        blurb = blurbs.get(nid, "")
        rows_html.append(f"""
          <li style="margin-bottom:10px">
            <div><a href="{url}"><strong>{title}</strong></a></div>
            {'<div>'+blurb+'</div>' if blurb else ''}
            <div style="color:#777">Similarity: {sim:.2f}</div>
          </li>
        """)

    subject = "KIP Daily: top matches from yesterday"
    header_link = f'{APP_BASE_URL}' if APP_BASE_URL else "#"
    html = f"""
    <p>Hi,</p>
    <p>Here are up to {MAX_RESULTS} matches from yesterday that best fit your profile:</p>
    <ul>
      {''.join(rows_html)}
    </ul>
    <p><a href="{header_link}">Open KIP</a> to filter, view details, or update your profile.</p>
    """
    return subject, html

# ---------- Main ----------


def main():
    start, end = _yesterday_utc_window()
    start_iso = start.isoformat()
    end_iso = end.isoformat()
    logging.info("Window: %s to %s (UTC yesterday)", start_iso, end_iso)

    with engine.connect() as conn:
        subs = _fetch_subscribers(conn)
        if subs.empty:
            logging.info("No active subscribers. Exiting.")
            return

        notices = _fetch_yesterday_notices(conn, start_iso, end_iso)
        if notices.empty:
            logging.info(
                "No notices posted yesterday. Sending 'no matches' to all.")
            for _, s in subs.iterrows():
                subject, html = _render_email(
                    s["email"], s["company_description"], pd.DataFrame())
                _send_email(s["email"], subject, html)
            return

        # Precompute embeddings for notices once to speed up (if many subs)
        base_texts = (notices["title"].fillna(
            "") + " " + notices["description"].fillna("")).str.slice(0, 2000).tolist()
        X = _embed_texts(base_texts)

        for _, s in subs.iterrows():
            email = s["email"].strip()
            desc = (s["company_description"] or "").strip()

            if not email:
                continue

            if not desc:
                logging.info(
                    "Subscriber %s has no company description; sending 'no matches'.", email)
                subject, html = _render_email(email, desc, pd.DataFrame())
                _send_email(email, subject, html)
                continue

            # score for this subscriber using precomputed X
            Xq = _embed_texts([desc])[0]
            sims = X @ Xq
            df = notices.copy()
            df["sim"] = sims

            # keep only strong enough matches
            df = df[df["sim"] >= MIN_SIM].sort_values(
                "sim", ascending=False).head(MAX_RESULTS).reset_index(drop=True)

            subject, html = _render_email(email, desc, df)
            _send_email(email, subject, html)

    logging.info("Done.")


if __name__ == "__main__":
    main()
