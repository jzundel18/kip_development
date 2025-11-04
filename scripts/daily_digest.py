#!/usr/bin/env python3
"""
Daily email digest with two-stage filtering for efficiency:
Stage 1: Fast embedding-based pre-filter (cheap)
Stage 2: Detailed matrix scoring on top candidates only (expensive but limited)

Required env vars:
  SUPABASE_DB_URL, OPENAI_API_KEY, GMAIL_EMAIL, GMAIL_PASSWORD
Optional:
  APP_BASE_URL, DIGEST_MAX_RESULTS, DIGEST_MIN_SCORE
  DIGEST_PREFILTER_CANDIDATES (controls stage 1 output)
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
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from scoring import AIMatrixScorer, ai_matrix_score_solicitations

# Module-level config (will be set when module loads)
DB_URL = None
OPENAI_API_KEY = None
GMAIL_EMAIL = None
GMAIL_PASSWORD = None
FROM_EMAIL = None
APP_BASE_URL = None
MAX_RESULTS = 5
MIN_SCORE = 60.0
PREFILTER_CANDIDATES = 25


def _load_config():
    """Load configuration from environment variables"""
    global DB_URL, OPENAI_API_KEY, GMAIL_EMAIL, GMAIL_PASSWORD, FROM_EMAIL, APP_BASE_URL
    global MAX_RESULTS, MIN_SCORE, PREFILTER_CANDIDATES

    DB_URL = os.getenv("SUPABASE_DB_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GMAIL_EMAIL = os.getenv("GMAIL_EMAIL", "kipmatchemail@gmail.com")
    GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD", "kenaidefense!")
    FROM_EMAIL = os.getenv("FROM_EMAIL") or GMAIL_EMAIL
    APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")

    MAX_RESULTS = int(os.getenv("DIGEST_MAX_RESULTS", "5"))
    MIN_SCORE = float(os.getenv("DIGEST_MIN_SCORE", "60"))
    PREFILTER_CANDIDATES = int(os.getenv("DIGEST_PREFILTER_CANDIDATES", "25"))

    if not (DB_URL and OPENAI_API_KEY and GMAIL_EMAIL and GMAIL_PASSWORD):
        print("Missing required env vars. Need SUPABASE_DB_URL, OPENAI_API_KEY, GMAIL_EMAIL, GMAIL_PASSWORD.",
              file=sys.stderr)
        sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

# These will be initialized after config is loaded
engine = None
oai = None


# ---------- Helper Functions ----------


def _yesterday_utc_window():
    """Return start and end dates as YYYY-MM-DD strings"""
    now = datetime.now(timezone.utc)
    end_date = now.date()
    start_date = end_date - timedelta(days=1)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def _fetch_subscribers_with_companies(conn) -> pd.DataFrame:
    """Fetch subscribers with ALL their active companies - now using user_companies only"""
    try:
        df = pd.read_sql_query(
            """
            SELECT s.user_id,
                   s.email,
                   uc.company_name,
                   uc.description AS company_description,
                   COALESCE(uc.city, '') AS city,
                   COALESCE(uc.state, '') AS state
            FROM digest_subscribers s
            INNER JOIN user_companies uc ON uc.user_id = s.user_id AND uc.is_active = TRUE
            WHERE COALESCE(s.is_enabled, TRUE) = TRUE
            """,
            conn
        )
    except Exception as e:
        logging.error("Error reading subscribers/companies: %s", e)
        return pd.DataFrame(columns=["user_id", "email", "company_name", "company_description", "city", "state"])
    return df.fillna('')


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
        df = pd.read_sql_query(
            sql, conn, params={"yesterday_date": start_date})
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
        logging.info(
            f"Stage 1: Filtering {len(notices)} notices to top {top_k} using embeddings")

        # Create text representations for comparison
        notice_texts = (notices["title"].fillna(
            "") + " " + notices["description"].fillna("")).str.slice(0, 2000).tolist()

        # Get company embedding
        company_response = oai.embeddings.create(
            model="text-embedding-3-small",
            input=[company_desc]
        )
        company_vector = np.array(
            company_response.data[0].embedding, dtype=np.float32)
        company_vector = company_vector / \
                         (np.linalg.norm(company_vector) + 1e-9)

        # Get notice embeddings in batches
        notice_vectors = []
        batch_size = 500

        for i in range(0, len(notice_texts), batch_size):
            batch = notice_texts[i:i + batch_size]
            batch_response = oai.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_vectors = [d.embedding for d in batch_response.data]
            notice_vectors.extend(batch_vectors)

        # Calculate similarities
        notice_matrix = np.array(notice_vectors, dtype=np.float32)
        notice_matrix = notice_matrix / \
                        (np.linalg.norm(notice_matrix, axis=1, keepdims=True) + 1e-9)
        similarities = notice_matrix @ company_vector

        # Sort by similarity and take top k
        notices_with_scores = notices.copy()
        notices_with_scores["similarity_score"] = similarities
        top_notices = notices_with_scores.sort_values(
            "similarity_score", ascending=False).head(top_k)

        logging.info(
            f"Stage 1 complete: {len(top_notices)} candidates selected (similarity range: {top_notices['similarity_score'].min():.3f} - {top_notices['similarity_score'].max():.3f})")

        return top_notices.drop(columns=["similarity_score"])

    except Exception as e:
        logging.warning(
            f"Stage 1 embedding filter failed: {e}. Using first {top_k} notices as fallback.")
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
        reasons_map = {r["notice_id"]: r.get(
            "overall_reason", "") for r in results}

        df = notices.copy()
        df["score"] = df["notice_id"].astype(str).map(scores_map).fillna(0.0)
        df["overall_reason"] = df["notice_id"].astype(
            str).map(reasons_map).fillna("")

        logging.info(
            f"Stage 2 complete: Scored {len(df)} notices (score range: {df['score'].min():.1f} - {df['score'].max():.1f})")

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
    """Send email via Gmail SMTP"""

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = to_email

    # Add plain text and HTML parts
    if text_body:
        msg.attach(MIMEText(text_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    try:
        # Connect to Gmail's SMTP server
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Secure the connection
            server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
            server.send_message(msg)

        logging.info("Sent to %s via Gmail SMTP", to_email)
    except Exception as e:
        logging.error("Gmail SMTP error to %s: %s", to_email, e)


def _generate_match_explanations(notices: pd.DataFrame, company_desc: str) -> dict[str, str]:
    """Generate AI explanations for why each notice matched the company"""
    if notices.empty or not company_desc.strip():
        return {}

    explanations = {}

    try:
        # Process in small batches to avoid token limits
        batch_size = 3
        for i in range(0, len(notices), batch_size):
            batch = notices.iloc[i:i + batch_size]

            # Prepare batch data
            batch_items = []
            for _, row in batch.iterrows():
                batch_items.append({
                    "notice_id": str(row.get("notice_id", "")),
                    "title": (row.get("title", "") or "")[:200],
                    "description": (row.get("description", "") or "")[:400],
                    "score": float(row.get("score", 0))
                })

            system_prompt = """You explain why government solicitations match companies. Write concise, specific explanations using second person (your company, your capabilities, your services). Focus on relevant capabilities and requirements. Keep each explanation to 1-2 sentences maximum."""

            user_prompt = {
                "company_description": company_desc[:300],
                "solicitations": batch_items,
                "instructions": 'For each solicitation, explain in 1-2 sentences why it matches using "your company" language. Return JSON: {"explanations":[{"notice_id":"...","reason":"..."}]}'
            }

            response = oai.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)}
                ],
                temperature=0.2,
                max_tokens=800,
                timeout=30
            )

            # Parse response
            try:
                data = json.loads(response.choices[0].message.content or "{}")
                for item in data.get("explanations", []):
                    notice_id = str(item.get("notice_id", "")).strip()
                    reason = (item.get("reason", "") or "").strip()
                    if notice_id and reason:
                        explanations[notice_id] = reason
            except json.JSONDecodeError:
                logging.warning(
                    f"Failed to parse match explanations for batch {i // batch_size + 1}")
                continue

    except Exception as e:
        logging.warning(f"Failed to generate match explanations: {e}")

    return explanations


def _render_email(subscriber_email: str, company_desc: str, picks: pd.DataFrame,
                  company_name: str = "") -> tuple[str, str]:
    """Render email with company name in subject/header"""

    # Email base styles
    email_wrapper = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; max-width: 680px; margin: 0 auto; background-color: #ffffff;">
    """

    # Header
    header_link = f'{APP_BASE_URL}' if APP_BASE_URL else "#"
    # Show company name in header if provided
    header_subtitle = company_name if company_name else "Your Daily Federal Opportunity Digest"

    header = f"""
          <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 32px 24px; text-align: center; border-radius: 8px 8px 0 0;">
            <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 600; letter-spacing: -0.5px;">
              Knowledge Integration Platform
            </h1>
            <p style="color: #dbeafe; margin: 8px 0 0 0; font-size: 14px; font-weight: 400;">
              {header_subtitle}
            </p>
          </div>
        """

    # Footer
    footer = f"""
      <div style="background-color: #f8fafc; padding: 24px; border-top: 1px solid #e2e8f0; margin-top: 32px; border-radius: 0 0 8px 8px;">
        <div style="text-align: center; margin-bottom: 16px;">
          <a href="{header_link}" style="display: inline-block; background-color: #3b82f6; color: #ffffff; padding: 12px 28px; text-decoration: none; border-radius: 6px; font-weight: 500; font-size: 14px;">
            Open KIP Dashboard
          </a>
        </div>
        <p style="color: #64748b; font-size: 12px; line-height: 1.6; margin: 12px 0 0 0; text-align: center;">
          Match explanations are AI-generated based on your company profile.<br>
          Technical assessments use our detailed scoring matrix for accuracy.
        </p>
        <p style="color: #94a3b8; font-size: 11px; margin: 16px 0 0 0; text-align: center;">
          © 2025 Knowledge Integration Platform. All rights reserved.
        </p>
      </div>
    """

    if picks.empty:
        company_label = f" [{company_name}]" if company_name else ""
        subject = f"KIP Daily Digest{company_label}: No Close Matches Today"
        html = f"""
        {email_wrapper}
          {header}
          <div style="padding: 32px 24px;">
            <p style="color: #1e293b; font-size: 16px; line-height: 1.6; margin: 0 0 16px 0;">
              Hello,
            </p>
            <div style="background-color: #f1f5f9; border-left: 4px solid #64748b; padding: 16px 20px; border-radius: 4px; margin: 24px 0;">
              <p style="color: #475569; font-size: 15px; line-height: 1.6; margin: 0;">
                No close matches (score ≥ {MIN_SCORE}) were found for your company profile yesterday.
              </p>
            </div>
            <p style="color: #64748b; font-size: 14px; line-height: 1.6; margin: 16px 0 0 0;">
              You'll receive up to {MAX_RESULTS} new opportunities when there are strong matches for your company.
            </p>
          </div>
          {footer}
        </div>
        """
        return subject, html

    # Generate AI explanations for why each notice matched
    logging.info(f"Generating match explanations for {len(picks)} final candidates")
    match_explanations = _generate_match_explanations(picks, company_desc)

    # Score badge color based on score value
    def get_score_color(score):
        if score >= 85:
            return "#10b981"  # Green
        elif score >= 70:
            return "#3b82f6"  # Blue
        else:
            return "#f59e0b"  # Amber

    rows_html = []
    for idx, r in picks.iterrows():
        nid = str(r.get("notice_id", ""))
        title = (r.get("title", "") or "Untitled").strip()
        score = float(r.get("score", 0.0))
        reason = r.get("overall_reason", "AI assessment of match quality")
        url = _sam_public_url(nid, r.get("link", ""))

        # Get additional details
        posted_date = r.get("posted_date", "")
        response_date = r.get("response_date", "")
        pop_city = r.get("pop_city", "")
        pop_state = r.get("pop_state", "")
        naics = r.get("naics_code", "")

        # Format location
        location = ""
        if pop_city or pop_state:
            location = f"{pop_city}, {pop_state}".strip(", ")

        # Get AI match explanation
        match_explanation = match_explanations.get(
            nid, "This opportunity aligns with your company's capabilities.")

        score_color = get_score_color(score)

        # Build metadata line
        metadata_parts = []
        if location:
            metadata_parts.append(f'<span style="margin-right: 16px;">📍 {location}</span>')
        if response_date:
            metadata_parts.append(f'<span style="margin-right: 16px;">📅 Response: {response_date}</span>')
        if naics:
            metadata_parts.append(f'<span>🏢 NAICS: {naics}</span>')

        metadata_html = f'<div style="color: #64748b; font-size: 13px; margin: 8px 0;">{" ".join(metadata_parts)}</div>' if metadata_parts else ""

        rows_html.append(f"""
          <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);">
            <div style="display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 12px;">
              <div style="flex: 1;">
                <a href="{url}" style="color: #1e293b; text-decoration: none; font-size: 17px; font-weight: 600; line-height: 1.4; display: block; margin-bottom: 8px;">
                  {title}
                </a>
              </div>
            </div>

            {metadata_html}

            <div style="background-color: #f0f9ff; border-left: 3px solid #3b82f6; padding: 12px 16px; border-radius: 4px; margin: 12px 0;">
              <p style="color: #1e40af; font-size: 14px; line-height: 1.6; margin: 0; font-weight: 500;">
                💡 {match_explanation}
              </p>
            </div>

            {f'<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #f1f5f9;"><p style="color: #64748b; font-size: 13px; line-height: 1.5; margin: 0;"><em>Assessment: {reason}</em></p></div>' if reason else ''}

            <div style="margin-top: 16px;">
              <a href="{url}" style="color: #3b82f6; text-decoration: none; font-size: 14px; font-weight: 500;">
                View Full Details →
              </a>
            </div>
          </div>
        """)

    # Add company name to subject if provided
    company_label = f" [{company_name}]" if company_name else ""
    subject = f"KIP Daily Digest{company_label}: {len(rows_html)} Top {'Match' if len(rows_html) == 1 else 'Matches'} for {datetime.now().strftime('%B %d, %Y')}"

    html = f"""
    {email_wrapper}
      {header}
      <div style="padding: 32px 24px;">
        <p style="color: #1e293b; font-size: 16px; line-height: 1.6; margin: 0 0 8px 0;">
          Hello,
        </p>
        <p style="color: #475569; font-size: 15px; line-height: 1.6; margin: 0 0 24px 0;">
          We found <strong>{len(rows_html)} high-quality {'opportunity' if len(rows_html) == 1 else 'opportunities'}</strong> matching your company profile from yesterday's federal solicitations.
        </p>

        <div style="margin: 24px 0;">
          {''.join(rows_html)}
        </div>
      </div>
      {footer}
    </div>
    """
    return subject, html


# ---------- Main ----------


def main():
    global engine, oai

    # Load config first
    _load_config()

    # Now initialize database and OpenAI clients with loaded config
    engine = sa.create_engine(DB_URL, pool_pre_ping=True)
    oai = OpenAI(api_key=OPENAI_API_KEY)

    yesterday_date, today_date = _yesterday_utc_window()
    logging.info("Processing notices posted on: %s", yesterday_date)

    logging.info(
        "Two-stage filtering: Stage 1 → %d candidates, Stage 2 → detailed scoring", PREFILTER_CANDIDATES)

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

        subs = _fetch_subscribers_with_companies(conn)
        if subs.empty:
            logging.info("No active subscribers. Exiting.")
            return

        notices = _fetch_yesterday_notices(conn, yesterday_date, today_date)
        if notices.empty:
            logging.info(
                "No notices posted on %s. Sending 'no matches' to all subscribers.", yesterday_date)
            for _, s in subs.iterrows():
                subject, html = _render_email(
                    s["email"], s["company_description"], pd.DataFrame())
                _send_email(s["email"], subject, html)
            return

        logging.info("Found %d notices from %s, processing %d subscribers...", len(
            notices), yesterday_date, len(subs))

        # Group by user_id and email to handle multiple companies
        grouped = subs.groupby(['user_id', 'email'])

        for (user_id, email), group in grouped:
            # Send separate email for EACH company this user has
            for _, company_row in group.iterrows():
                company_name = company_row["company_name"]
                desc = (company_row["company_description"] or "").strip()

                if not desc:
                    logging.info(f"Skipping {company_name} for {email} (no description)")
                    continue

                logging.info(f"Processing {company_name} for {email}")

                # STAGE 1: Fast embedding-based pre-filter
                stage1_candidates = _stage1_embedding_filter(
                    notices, desc, top_k=PREFILTER_CANDIDATES)

                if stage1_candidates.empty:
                    logging.info(f"Stage 1: No candidates for {company_name}")
                    continue

                # STAGE 2: Detailed scoring
                scored_notices = _stage2_detailed_scoring(
                    stage1_candidates, company_row.to_dict())

                # Filter and send
                final_matches = scored_notices[scored_notices["score"] >= MIN_SCORE].head(MAX_RESULTS)

                if not final_matches.empty:
                    subject, html = _render_email(
                        email,
                        desc,
                        final_matches,
                        company_name=company_name  # NEW: distinguish emails
                    )
                    _send_email(email, subject, html)
                    logging.info(f"Sent {len(final_matches)} matches for {company_name} to {email}")

    logging.info("Daily digest complete.")


if __name__ == "__main__":
    main()