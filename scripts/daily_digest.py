#!/usr/bin/env python3
"""
Daily email digest with two-stage filtering for efficiency:
Stage 1: Fast embedding-based pre-filter (cheap)
Stage 2: Detailed matrix scoring on top candidates only (expensive but limited)

Required env vars:
  SUPABASE_DB_URL, OPENAI_API_KEY, SENDGRID_API_KEY, FROM_EMAIL
Optional:
  APP_BASE_URL, DIGEST_MAX_RESULTS, DIGEST_MIN_SCORE
  DIGEST_PREFILTER_CANDIDATES (new - controls stage 1 output)
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import text
from openai import OpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content

from scoring import AIMatrixScorer, ai_matrix_score_solicitations

# ---------- Config ----------
DB_URL = os.getenv("SUPABASE_DB_URL", "sqlite:///app.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")

MAX_RESULTS = int(os.getenv("DIGEST_MAX_RESULTS", "5"))
MIN_SCORE = float(os.getenv("DIGEST_MIN_SCORE", "60"))

# NEW: Controls how many candidates pass stage 1 filtering
PREFILTER_CANDIDATES = int(os.getenv("DIGEST_PREFILTER_CANDIDATES", "50"))

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

# ---------- Helper Functions ----------

def _yesterday_utc_window():
    """Return start and end dates as YYYY-MM-DD strings"""
    now = datetime.now(timezone.utc)
    end_date = now.date()
    start_date = end_date - timedelta(days=1)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def _fetch_subscribers(conn) -> pd.DataFrame:
    """Fetch active subscribers with complete company profile info"""
    try:
        df = pd.read_sql_query(
            """
            SELECT s.user_id,
                   s.email,
                   COALESCE(cp.description, '') AS company_description,
                   COALESCE(cp.company_name, '') AS company_name,
                   COALESCE(cp.city, '') AS city,
                   COALESCE(cp.state, '') AS state
            FROM digest_subscribers s
            LEFT JOIN company_profile cp ON cp.user_id = s.user_id
            WHERE COALESCE(s.is_enabled, TRUE) = TRUE
            """,
            conn
        )
    except Exception as e:
        logging.error("Error reading subscribers/company_profile: %s", e)
        return pd.DataFrame(columns=["user_id", "email", "company_description", "company_name", "city", "state"])
    return df.fillna("")

def _fetch_yesterday_notices(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch notices posted on the start_date (yesterday)"""
    cols = [
        "notice_id", "title", "description", "naics_code", "set_aside_code",
        "posted_date", "response_date", "link", "pop_city", "pop_state"
    ]
    try:
        sql = text(f"""
            SELECT {", ".join(cols)}
            FROM solicitationraw
            WHERE posted_date = :yesterday_date
        """)
        df = pd.read_sql_query(sql, conn, params={"yesterday_date": start_date})
        logging.info(f"Found {len(df)} notices posted on {start_date}")
    except Exception as e:
        logging.error("Error reading solicitationraw: %s", e)
        return pd.DataFrame(columns=cols)
    return df.fillna("")

def _stage1_embedding_filter(notices: pd.DataFrame, company_desc: str, top_k: int = 50) -> pd.DataFrame:
    """Stage 1: Fast embedding-based filtering"""
    if notices.empty or not company_desc.strip():
        return notices.head(top_k)  # Fallback without filtering
    
    try:
        logging.info(f"Stage 1: Filtering {len(notices)} notices to top {top_k} using embeddings")
        
        # Create text representations for comparison
        notice_texts = (notices["title"].fillna("") + " " + notices["description"].fillna("")).str.slice(0, 2000).tolist()
        
        # Get company embedding
        company_response = oai.embeddings.create(
            model="text-embedding-3-small", 
            input=[company_desc]
        )
        company_vector = np.array(company_response.data[0].embedding, dtype=np.float32)
        company_vector = company_vector / (np.linalg.norm(company_vector) + 1e-9)
        
        # Get notice embeddings in batches
        notice_vectors = []
        batch_size = 500
        
        for i in range(0, len(notice_texts), batch_size):
            batch = notice_texts[i:i+batch_size]
            batch_response = oai.embeddings.create(
                model="text-embedding-3-small", 
                input=batch
            )
            batch_vectors = [d.embedding for d in batch_response.data]
            notice_vectors.extend(batch_vectors)
        
        # Calculate similarities
        notice_matrix = np.array(notice_vectors, dtype=np.float32)
        notice_matrix = notice_matrix / (np.linalg.norm(notice_matrix, axis=1, keepdims=True) + 1e-9)
        similarities = notice_matrix @ company_vector
        
        # Sort by similarity and take top k
        notices_with_scores = notices.copy()
        notices_with_scores["similarity_score"] = similarities
        top_notices = notices_with_scores.sort_values("similarity_score", ascending=False).head(top_k)
        
        logging.info(f"Stage 1 complete: {len(top_notices)} candidates selected (similarity range: {top_notices['similarity_score'].min():.3f} - {top_notices['similarity_score'].max():.3f})")
        
        return top_notices.drop(columns=["similarity_score"])
        
    except Exception as e:
        logging.warning(f"Stage 1 embedding filter failed: {e}. Using first {top_k} notices as fallback.")
        return notices.head(top_k)

def _stage2_detailed_scoring(notices: pd.DataFrame, subscriber: dict) -> pd.DataFrame:
    """Stage 2: Detailed matrix scoring on pre-filtered candidates"""
    if notices.empty:
        return notices.copy()

    logging.info(f"Stage 2: Detailed scoring of {len(notices)} candidates")
    
    # Build company profile
    company_profile = {
        "description": subscriber.get("company_description", "").strip(),
        "company_name": subscriber.get("company_name", "").strip(),
        "city": subscriber.get("city", "").strip(),
        "state": subscriber.get("state", "").strip()
    }

    if not company_profile["description"]:
        # No company description - return with default scores
        df = notices.copy()
        df["score"] = 50.0
        df["overall_reason"] = "No company description available for detailed scoring"
        return df

    # Use the matrix scoring from scoring.py
    try:
        results = ai_matrix_score_solicitations(
            df=notices,
            company_profile=company_profile,
            api_key=OPENAI_API_KEY,
            top_k=len(notices),  # Score all candidates
            model="gpt-4o-mini"
        )

        # Convert results to DataFrame format
        scores_map = {r["notice_id"]: r["score"] for r in results}
        reasons_map = {r["notice_id"]: r.get("overall_reason", "") for r in results}

        df = notices.copy()
        df["score"] = df["notice_id"].astype(str).map(scores_map).fillna(0.0)
        df["overall_reason"] = df["notice_id"].astype(str).map(reasons_map).fillna("")

        logging.info(f"Stage 2 complete: Scored {len(df)} notices (score range: {df['score'].min():.1f} - {df['score'].max():.1f})")
        
        return df.sort_values("score", ascending=False)
        
    except Exception as e:
        logging.error(f"Stage 2 detailed scoring failed: {e}")
        # Fallback with default scores
        df = notices.copy()
        df["score"] = 50.0
        df["overall_reason"] = f"Detailed scoring failed: {str(e)[:100]}"
        return df

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
    """Render email with matrix scores and reasons"""
    if picks.empty:
        subject = "KIP Daily: no close matches today"
        html = f"""
        <p>Hi,</p>
        <p>No close matches (score ≥ {MIN_SCORE}) were found for your company description yesterday.</p>
        <p>You'll receive up to {MAX_RESULTS} new matches when there are good fits.</p>
        """
        return subject, html

    rows_html = []
    for _, r in picks.iterrows():
        nid = str(r.get("notice_id", ""))
        title = (r.get("title", "") or "Untitled").strip()
        score = float(r.get("score", 0.0))
        reason = r.get("overall_reason", "AI assessment of match quality")
        url = _sam_public_url(nid, r.get("link", ""))

        # Color code the score
        if score >= 80:
            score_color = "#28a745"  # green
            score_icon = "🟢"
        elif score >= 60:
            score_color = "#ffc107"  # yellow
            score_icon = "🟡"
        else:
            score_color = "#dc3545"  # red
            score_icon = "🔴"

        rows_html.append(f"""
          <li style="margin-bottom:15px; border-left: 3px solid {score_color}; padding-left: 10px;">
            <div><a href="{url}" style="color: #0066cc; text-decoration: none;"><strong>{title}</strong></a></div>
            <div style="color: {score_color}; font-weight: bold; margin: 5px 0;">
              {score_icon} Match Score: {score:.1f}/100
            </div>
            {f'<div style="margin: 5px 0; color: #555;">{reason}</div>' if reason else ''}
          </li>
        """)

    subject = f"KIP Daily: {len(rows_html)} top matches from yesterday"
    header_link = f'{APP_BASE_URL}' if APP_BASE_URL else "#"
    html = f"""
    <p>Hi,</p>
    <p>Here are your top-scoring opportunities from yesterday (minimum score: {MIN_SCORE}):</p>
    <ul style="list-style-type: none; padding-left: 0;">
      {''.join(rows_html)}
    </ul>
    <p><a href="{header_link}" style="color: #0066cc;">Open KIP</a> to view full details and manage preferences.</p>
    """
    return subject, html

# ---------- Main ----------

def main():
    yesterday_date, today_date = _yesterday_utc_window()
    logging.info("Processing notices posted on: %s", yesterday_date)
    logging.info("Two-stage filtering: Stage 1 → %d candidates, Stage 2 → detailed scoring", PREFILTER_CANDIDATES)

    with engine.connect() as conn:
        # Debug: show what dates we have
        try:
            recent_dates = conn.execute(text("""
                SELECT posted_date, COUNT(*) 
                FROM solicitationraw 
                WHERE posted_date IS NOT NULL 
                GROUP BY posted_date 
                ORDER BY posted_date DESC 
                LIMIT 5
            """)).fetchall()
            logging.info("Recent posted_dates in database:")
            for date_val, count in recent_dates:
                logging.info("  %s: %d records", date_val, count)
        except Exception as e:
            logging.warning("Could not fetch recent dates: %s", e)

        subs = _fetch_subscribers(conn)
        if subs.empty:
            logging.info("No active subscribers. Exiting.")
            return

        notices = _fetch_yesterday_notices(conn, yesterday_date, today_date)
        if notices.empty:
            logging.info("No notices posted on %s. Sending 'no matches' to all subscribers.", yesterday_date)
            for _, s in subs.iterrows():
                subject, html = _render_email(s["email"], s["company_description"], pd.DataFrame())
                _send_email(s["email"], subject, html)
            return

        logging.info("Found %d notices from %s, processing %d subscribers...", len(notices), yesterday_date, len(subs))

        for _, s in subs.iterrows():
            email = s["email"].strip()
            desc = (s["company_description"] or "").strip()

            if not email:
                continue

            if not desc:
                logging.info("Subscriber %s has no company description; sending 'no matches'.", email)
                subject, html = _render_email(email, desc, pd.DataFrame())
                _send_email(email, subject, html)
                continue

            logging.info(f"Processing subscriber: {email}")
            
            # STAGE 1: Fast embedding-based pre-filter
            stage1_candidates = _stage1_embedding_filter(notices, desc, top_k=PREFILTER_CANDIDATES)
            
            if stage1_candidates.empty:
                logging.info(f"Stage 1 returned no candidates for {email}")
                subject, html = _render_email(email, desc, pd.DataFrame())
                _send_email(email, subject, html)
                continue
            
            # STAGE 2: Detailed scoring on filtered candidates
            scored_notices = _stage2_detailed_scoring(stage1_candidates, s.to_dict())

            # Filter by minimum score and limit results
            final_matches = scored_notices[scored_notices["score"] >= MIN_SCORE].head(MAX_RESULTS)
            logging.info(f"Final results for {email}: {len(final_matches)} matches above {MIN_SCORE}")

            subject, html = _render_email(email, desc, final_matches)
            _send_email(email, subject, html)

    logging.info("Daily digest complete.")

if __name__ == "__main__":
    main()