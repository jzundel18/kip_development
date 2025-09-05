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

from app import AIMatrixScorer, ai_matrix_score_solicitations
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
MIN_SCORE = float(os.getenv("DIGEST_MIN_SCORE", "60"))

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
    """Return start and end dates as YYYY-MM-DD strings to match database format"""
    now = datetime.now(timezone.utc)
    end_date = now.date()  # today
    start_date = end_date - timedelta(days=1)  # yesterday

    # Return as simple YYYY-MM-DD strings to match your database format
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
            WHERE COALESCE(s.is_active, TRUE) = TRUE
            """,
            conn
        )
    except Exception as e:
        logging.error("Error reading subscribers/company_profile: %s", e)
        return pd.DataFrame(columns=["user_id", "email", "company_description", "company_name", "city", "state"])
    return df.fillna("")


def _fetch_yesterday_notices(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch notices posted on the start_date (yesterday)
    Database format: YYYY-MM-DD, so we can do exact string match
    """
    cols = [
        "notice_id", "title", "description", "naics_code", "set_aside_code",
        "posted_date", "response_date", "link"
    ]
    try:
        # Use SQLAlchemy text() with proper parameter binding for PostgreSQL
        from sqlalchemy import text

        sql = text(f"""
            SELECT {", ".join(cols)}
            FROM solicitationraw
            WHERE posted_date = :yesterday_date
        """)

        df = pd.read_sql_query(
            sql,
            conn,
            params={"yesterday_date": start_date}
        )
        logging.info(f"Found {len(df)} notices posted on {start_date}")
    except Exception as e:
        logging.error("Error reading solicitationraw: %s", e)
        return pd.DataFrame(columns=cols)
    return df.fillna("")


def _score_notices_for_subscriber(notices: pd.DataFrame, subscriber: dict) -> pd.DataFrame:
    """Score notices using matrix scorer for a single subscriber"""
    if notices.empty:
        return notices.copy()

    # Build company profile from subscriber data
    company_profile = {
        "description": subscriber.get("company_description", "").strip(),
        "company_name": subscriber.get("company_name", "").strip(),
        "city": subscriber.get("city", "").strip(),
        "state": subscriber.get("state", "").strip()
    }

    if not company_profile["description"]:
        df = notices.copy()
        df["score"] = 0.0
        df["overall_reason"] = "No company description available"
        return df

    # Use the same scoring function from app.py
    results = ai_matrix_score_solicitations(
        df=notices,
        company_profile=company_profile,
        api_key=OPENAI_API_KEY,
        top_k=len(notices),  # Score all notices
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

    return df.sort_values("score", ascending=False)

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
    logging.info("Looking for notices posted on: %s", yesterday_date)

    with engine.connect() as conn:
        # Debug: show what dates we have (fixed for PostgreSQL)
        try:
            from sqlalchemy import text
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
            logging.info(
                "No notices posted on %s. Sending 'no matches' to all subscribers.", yesterday_date)
            for _, s in subs.iterrows():
                subject, html = _render_email(
                    s["email"], s["company_description"], pd.DataFrame())
                _send_email(s["email"], subject, html)
            return

        # Rest of the function stays the same...
        logging.info("Found %d notices from %s, processing subscribers...", len(
            notices), yesterday_date)

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

            # Score notices using matrix scorer
            logging.info(f"Scoring notices for {email}...")
            scored_notices = _score_notices_for_subscriber(notices, s.to_dict())

            # Filter by minimum score and limit results
            df = scored_notices[scored_notices["score"] >= MIN_SCORE].head(MAX_RESULTS)
            logging.info(f"Found {len(df)} matches above {MIN_SCORE} for {email}")

            subject, html = _render_email(email, desc, df)
            _send_email(email, subject, html)

    logging.info("Done.")


if __name__ == "__main__":
    main()
