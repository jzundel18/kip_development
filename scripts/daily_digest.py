#!/usr/bin/env python3
"""
Daily email digest - CORRECTED FOR TECHNOLOGY_AREAS TABLE
Sends emails directly to email addresses stored in technology_areas table.
Each technology area with emails gets its own digest based on its description.

UPDATED: Now processes ALL solicitations (not just research category)
UPDATED: Now includes solicitation description summary in email

Required env vars:
  SUPABASE_DB_URL, OPENAI_API_KEY, GMAIL_EMAIL, GMAIL_PASSWORD
Optional:
  APP_BASE_URL, DIGEST_MAX_RESULTS, DIGEST_MIN_SCORE
  DIGEST_PREFILTER_CANDIDATES (controls stage 1 output)
  KIP_MEME_URL (URL to Kip meme image to include in emails)
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

import random

# Actual Kip quotes from Napoleon Dynamite
KIP_QUOTES = [
    "Napoleon, don't be jealous that I've been chatting online with babes all day.",
    "I'm just really trying to raise a few bucks now so I can bring her out for a few days.",
    "I love technology, but not as much as you, you see...",
    "But I STILL love technology... Always and forever.",
    "LaFawnduh is the best thing that has ever happened to me.",
    "I'm training to become a cage fighter.",
    "Your mom goes to college.",
    "Well, things are getting pretty serious right now. I mean, we chat online for, like, two hours every day so I guess you could say things are gettin' pretty serious.",
    "Napoleon, like anyone can even know that.",
    "I've already looked into it for myself.",
]


def get_random_kip_quote():
    """Return a random Kip quote from Napoleon Dynamite"""
    return random.choice(KIP_QUOTES)


# Module-level config
DB_URL = None
OPENAI_API_KEY = None
GMAIL_EMAIL = None
GMAIL_PASSWORD = None
FROM_EMAIL = None
APP_BASE_URL = None
MAX_RESULTS = 5
MIN_SCORE = 60.0
PREFILTER_CANDIDATES = 25
KIP_MEME_URL = None



def _load_config():
    """Load configuration from environment variables"""
    global DB_URL, OPENAI_API_KEY, GMAIL_EMAIL, GMAIL_PASSWORD, FROM_EMAIL, APP_BASE_URL
    global MAX_RESULTS, MIN_SCORE, PREFILTER_CANDIDATES, KIP_MEME_URL

    DB_URL = os.getenv("SUPABASE_DB_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GMAIL_EMAIL = os.getenv("GMAIL_EMAIL", "kipmatchemail@gmail.com")
    GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD", "kenaidefense!")
    FROM_EMAIL = os.getenv("FROM_EMAIL") or GMAIL_EMAIL
    APP_BASE_URL = os.getenv("APP_BASE_URL", "").rstrip("/")
    KIP_MEME_URL = os.getenv("KIP_MEME_URL", "")

    MAX_RESULTS = int(os.getenv("DIGEST_MAX_RESULTS", "10"))
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

engine = None
oai = None

# Healthcare keywords to filter out
HEALTHCARE_KEYWORDS = [
    "healthcare", "health care", "medical", "hospital", "clinical",
    "pharmaceutical", "pharmacy", "nursing", "patient care", "medicare",
    "medicaid", "hhs", "nih", "cdc", "fda", "cms", "veterans health",
    "mental health", "dental", "laboratory services", "biomedical",
    "health services", "healthcare services", "medical supplies",
    "medical equipment", "ambulance", "emergency medical"
]


def _filter_out_healthcare(notices: pd.DataFrame) -> pd.DataFrame:
    """Remove healthcare-related solicitations from consideration"""
    if notices.empty:
        return notices

    def is_healthcare(row):
        title = (row.get("title", "") or "").lower()
        description = (row.get("description", "") or "").lower()
        naics = str(row.get("naics_code", "") or "")
        combined = f"{title} {description}"

        # Check keywords
        for keyword in HEALTHCARE_KEYWORDS:
            if keyword in combined:
                return True

        # Healthcare NAICS codes (621xxx, 622xxx, 623xxx, 624xxx are health/social services)
        if naics.startswith(("621", "622", "623", "624")):
            return True

        return False

    original_count = len(notices)
    filtered = notices[~notices.apply(is_healthcare, axis=1)]
    removed_count = original_count - len(filtered)

    if removed_count > 0:
        logging.info(f"Filtered out {removed_count} healthcare-related solicitations")

    return filtered.reset_index(drop=True)


def _today_utc_date():
    """Return today's date as YYYY-MM-DD string"""
    now = datetime.now(timezone.utc)
    today_date = now.date()
    return today_date.strftime("%Y-%m-%d")


def _fetch_technology_areas_with_emails(conn) -> pd.DataFrame:
    """Fetch technology areas that have email addresses configured"""
    try:
        df = pd.read_sql_query(
            """
            SELECT id as technology_id,
                   emails,
                   technology_name,
                   description as technology_description
            FROM technology_areas
            WHERE emails IS NOT NULL
              AND emails != ''
              AND description IS NOT NULL
              AND description != ''
            ORDER BY technology_name
            """,
            conn
        )
        logging.info(f"Found {len(df)} technology areas with email addresses configured")
    except Exception as e:
        logging.error("Error reading technology areas: %s", e)
        return pd.DataFrame(columns=["technology_id", "emails", "technology_name", "technology_description"])
    return df.fillna('')


def _parse_email_addresses(email_field: str) -> list[str]:
    """
    Parse email field that may contain multiple comma-separated addresses.
    Returns list of cleaned email addresses.
    """
    if not email_field or email_field.strip() == '':
        return []

    # Split by comma and clean each email
    emails = [e.strip() for e in email_field.split(',')]
    # Filter out empty strings
    emails = [e for e in emails if e]

    return emails


def _fetch_today_notices(conn, today_date: str) -> pd.DataFrame:
    """Fetch notices pulled today (based on pulled_at date) - ALL CATEGORIES"""
    cols = [
        "notice_id", "title", "description", "naics_code", "set_aside_code",
        "posted_date", "response_date", "link", "pop_city", "pop_state", "category"
    ]
    try:
        sql = text(f"""
            SELECT {", ".join(cols)}
            FROM solicitationraw
            WHERE DATE(pulled_at) = :today_date
        """)
        df = pd.read_sql_query(
            sql, conn, params={"today_date": today_date})
        logging.info(f"Found {len(df)} notices pulled today ({today_date})")

        # Debug: Check if descriptions are populated
        if len(df) > 0:
            sample_desc = df["description"].iloc[0] if "description" in df.columns else "COLUMN MISSING"
            logging.info(f"Sample description length: {len(str(sample_desc))} chars")
            logging.info(f"Sample description preview: {str(sample_desc)[:100]}...")
            null_count = df["description"].isna().sum()
            empty_count = (df["description"] == "").sum()
            logging.info(f"Descriptions - Null: {null_count}, Empty string: {empty_count}, Total: {len(df)}")

            # Category breakdown
            if "category" in df.columns:
                category_counts = df["category"].value_counts()
                logging.info(f"Category breakdown: {category_counts.to_dict()}")
    except Exception as e:
        logging.error("Error reading solicitationraw: %s", e)
        return pd.DataFrame(columns=cols)
    return df.fillna("")


def _stage1_embedding_filter(notices: pd.DataFrame, tech_desc: str, top_k: int = 50) -> pd.DataFrame:
    """Stage 1: Fast embedding-based filtering"""
    if notices.empty or not tech_desc.strip():
        return notices.head(top_k)

    try:
        logging.info(
            f"Stage 1: Filtering {len(notices)} notices to top {top_k} using embeddings")

        notice_texts = (notices["title"].fillna(
            "") + " " + notices["description"].fillna("")).str.slice(0, 2000).tolist()

        tech_response = oai.embeddings.create(
            model="text-embedding-3-small",
            input=[tech_desc]
        )
        tech_vector = np.array(
            tech_response.data[0].embedding, dtype=np.float32)
        tech_vector = tech_vector / (np.linalg.norm(tech_vector) + 1e-9)

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

        notice_matrix = np.array(notice_vectors, dtype=np.float32)
        notice_matrix = notice_matrix / \
                        (np.linalg.norm(notice_matrix, axis=1, keepdims=True) + 1e-9)
        similarities = notice_matrix @ tech_vector

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


def _stage2_detailed_scoring(notices: pd.DataFrame, tech_area_row: dict) -> pd.DataFrame:
    """Stage 2: Detailed matrix scoring on pre-filtered candidates"""
    if notices.empty:
        return notices.copy()

    logging.info(f"Stage 2: Detailed scoring of {len(notices)} candidates")

    # Build profile for scoring
    tech_profile = {
        "description": tech_area_row.get("technology_description", "").strip(),
        "company_name": tech_area_row.get("technology_name", "").strip(),
        "city": "",  # Not in technology_areas table
        "state": ""  # Not in technology_areas table
    }

    if not tech_profile["description"]:
        df = notices.copy()
        df["score"] = 50.0
        df["overall_reason"] = "No technology description available for detailed scoring"
        return df

    try:
        results = ai_matrix_score_solicitations(
            df=notices,
            company_profile=tech_profile,
            api_key=OPENAI_API_KEY,
            top_k=len(notices),
            model="gpt-4o-mini"
        )

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

    if text_body:
        msg.attach(MIMEText(text_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(GMAIL_EMAIL, GMAIL_PASSWORD)
            server.send_message(msg)

        logging.info("Sent to %s via Gmail SMTP", to_email)
    except Exception as e:
        logging.error("Gmail SMTP error to %s: %s", to_email, e)


def _generate_description_summaries(notices: pd.DataFrame) -> dict[str, str]:
    """Generate AI summaries of solicitation descriptions (~500 characters each)"""
    if notices.empty:
        return {}

    summaries = {}

    try:
        batch_size = 3
        for i in range(0, len(notices), batch_size):
            batch = notices.iloc[i:i + batch_size]

            batch_items = []
            for _, row in batch.iterrows():
                # Access description directly from the Series
                desc = str(row["description"]) if "description" in row else ""
                # Check if description is actually populated (not just empty string)
                if pd.notna(row["description"]) and desc.strip():
                    desc_to_use = desc.strip()
                else:
                    desc_to_use = ""

                nid = str(row["notice_id"]) if "notice_id" in row else ""

                # Debug: log if description is empty
                if not desc_to_use:
                    logging.warning(f"Empty/missing description for notice_id: {nid}")
                else:
                    logging.info(f"Notice {nid} has description: {len(desc_to_use)} chars")

                batch_items.append({
                    "notice_id": nid,
                    "description": desc_to_use[:2000] if desc_to_use else "No description available."
                })

            system_prompt = """You summarize government solicitation descriptions concisely. Focus on the core requirements, objectives, and key technical details. Keep summaries to approximately 500 characters (3-4 sentences)."""

            user_prompt = {
                "solicitations": batch_items,
                "instructions": 'For each solicitation, provide a concise summary of approximately 500 characters. Return JSON: {"summaries":[{"notice_id":"...","summary":"..."}]}'
            }

            response = oai.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_prompt)}
                ],
                temperature=0.2,
                max_tokens=1000,
                timeout=30
            )

            try:
                data = json.loads(response.choices[0].message.content or "{}")
                for item in data.get("summaries", []):
                    notice_id = str(item.get("notice_id", "")).strip()
                    summary = (item.get("summary", "") or "").strip()
                    if notice_id and summary:
                        summaries[notice_id] = summary
                        logging.info(f"Generated summary for notice_id {notice_id}: {len(summary)} chars")
                    else:
                        logging.warning(f"Empty summary returned for notice_id: {notice_id}")
            except json.JSONDecodeError:
                logging.warning(
                    f"Failed to parse description summaries for batch {i // batch_size + 1}")
                continue

    except Exception as e:
        logging.warning(f"Failed to generate description summaries: {e}")

    logging.info(f"Successfully generated {len(summaries)} description summaries")
    return summaries


def _generate_match_explanations(notices: pd.DataFrame, tech_desc: str) -> dict[str, str]:
    """Generate AI explanations for why each notice matched the technology area"""
    if notices.empty or not tech_desc.strip():
        return {}

    explanations = {}

    try:
        batch_size = 3
        for i in range(0, len(notices), batch_size):
            batch = notices.iloc[i:i + batch_size]

            batch_items = []
            for _, row in batch.iterrows():
                batch_items.append({
                    "notice_id": str(row.get("notice_id", "")),
                    "title": (row.get("title", "") or "")[:200],
                    "description": (row.get("description", "") or "")[:400],
                    "score": float(row.get("score", 0))
                })

            system_prompt = """You explain why government solicitations match technology areas. Write concise, specific explanations focusing on relevant capabilities and requirements. Keep each explanation to 1-2 sentences maximum."""

            user_prompt = {
                "technology_description": tech_desc[:300],
                "solicitations": batch_items,
                "instructions": 'For each solicitation, explain in 1-2 sentences why it matches this technology area. Return JSON: {"explanations":[{"notice_id":"...","reason":"..."}]}'
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


def _get_funny_kip_greeting(num_matches: int, tech_name: str) -> str:
    """Generate a funny Kip-themed greeting based on number of matches"""
    greetings = [
        f"Napoleon, don't be jealous - we've been chatting with SAM.gov all day and found <strong>{num_matches} {'babe' if num_matches == 1 else 'babes'}</strong>... I mean, {'opportunity' if num_matches == 1 else 'opportunities'} for {tech_name}.",
        f"Yes, we love technology (and federal contracts), always and forever. Here {'is' if num_matches == 1 else 'are'} <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name}.",
        f"We've been training to be cage fighters... of opportunity matching. Found <strong>{num_matches} solid {'match' if num_matches == 1 else 'matches'}</strong> for {tech_name}.",
        f"Things are getting pretty serious right now. We found <strong>{num_matches} high-quality {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name}.",
        f"LaFawnduh would be proud - we found <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name} from today's federal solicitations.",
        f"Your mom goes to college... to learn about these <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong> we found for {tech_name}.",
        f"Tina, you fat lard, come get some federal contracts! We've got <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name}.",
        f"I caught you a delicious {'bass' if num_matches == 1 else 'bass'}... er, <strong>{num_matches} federal {'solicitation' if num_matches == 1 else 'solicitations'}</strong> for {tech_name}.",
        f"Vote for Pedro... or check out these <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name}. Your call.",
        f"Girls only want boyfriends who have great skills. Like finding <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name}.",
        f"Forget the time machine, Napoleon. We found <strong>{num_matches} real {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name}.",
        f"I see you're drinking 1% milk. Is that 'cause you think you're fat? Anyway, here {'is' if num_matches == 1 else 'are'} <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name}.",
        f"Lucky! We've been hunting wolverines with a frickin' 12-gauge and found <strong>{num_matches} federal {'opportunity' if num_matches == 1 else 'opportunities'}</strong> for {tech_name} instead.",
        f"Do the chickens have large talons? More importantly, does {tech_name} want these <strong>{num_matches} {'opportunity' if num_matches == 1 else 'opportunities'}</strong>?",
    ]
    return random.choice(greetings)


def _render_email(recipient_email: str, tech_desc: str, picks: pd.DataFrame,
                  tech_name: str = "") -> tuple[str, str]:
    """Render email with technology area name in subject/header"""

    email_wrapper = """
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; max-width: 680px; margin: 0 auto; background-color: #ffffff;">
    """

    header_link = f'{APP_BASE_URL}' if APP_BASE_URL else "#"
    # Get a random Kip quote for the banner
    kip_quote = get_random_kip_quote()
    header_subtitle = f"&ldquo;{kip_quote}&rdquo;"
    header = f"""
          <div style="background-color: #2563eb; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 32px 24px; text-align: center; border-radius: 8px 8px 0 0; border: 2px solid #1e3a8a;">
            <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 600; letter-spacing: -0.5px; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
              Kenai Infinite Pipeline (KIP)
            </h1>
            <p style="color: #ffffff; margin: 8px 0 0 0; font-size: 14px; font-weight: 400; text-shadow: 1px 1px 2px rgba(0,0,0,0.3); font-style: italic;">
              {header_subtitle}
            </p>
          </div>
        """

    # Add Kip meme section if URL is configured
    kip_meme_section = ""
    if KIP_MEME_URL:
        kip_meme_section = f"""
        <div style="text-align: center; margin: 24px 0; padding: 16px; background-color: #f8fafc; border-radius: 8px;">
          <img src="{KIP_MEME_URL}" alt="Kip from Napoleon Dynamite" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);">
        </div>
        """

    footer = f"""
      <div style="background-color: #f8fafc; padding: 24px; border-top: 1px solid #e2e8f0; margin-top: 32px; border-radius: 0 0 8px 8px;">
        <div style="text-align: center; margin-bottom: 16px;">
        </div>
        <p style="color: #64748b; font-size: 12px; line-height: 1.6; margin: 12px 0 0 0; text-align: center;">
          Match explanations are AI-generated based on your technology area profile.<br>
          Technical assessments use our detailed scoring matrix for accuracy.
        </p>
        <p style="color: #94a3b8; font-size: 11px; margin: 16px 0 0 0; text-align: center;">
          © 2025 Kenai Infinite Pipeline (KIP). All rights reserved.
        </p>
      </div>
    """

    if picks.empty:
        tech_label = f" [{tech_name}]" if tech_name else ""
        subject = f"KIP Daily Digest{tech_label}: No Close Matches Today"
        html = f"""
        {email_wrapper}
          {header}
          <div style="padding: 32px 24px;">
            <p style="color: #1e293b; font-size: 16px; line-height: 1.6; margin: 0 0 16px 0;">
              Napoleon, like anyone could even know if there'd be matches today...
            </p>
            <div style="background-color: #f1f5f9; border-left: 4px solid #64748b; padding: 16px 20px; border-radius: 4px; margin: 24px 0;">
              <p style="color: #475569; font-size: 15px; line-height: 1.6; margin: 0;">
                No close matches (score ≥ {MIN_SCORE}) were found for {tech_name} in today's pulled solicitations.
              </p>
            </div>
            <p style="color: #64748b; font-size: 14px; line-height: 1.6; margin: 16px 0 0 0;">
              You'll receive up to {MAX_RESULTS} new opportunities when there are strong matches for this technology area.
            </p>
            {kip_meme_section}
          </div>
          {footer}
        </div>
        """
        return subject, html

    logging.info(f"Generating description summaries for {len(picks)} final candidates")

    # CRITICAL DEBUG: Check what's actually in the DataFrame
    if len(picks) > 0:
        logging.info("=" * 60)
        logging.info("DEBUG: Checking DataFrame before summary generation")
        logging.info(f"DataFrame columns: {list(picks.columns)}")
        logging.info(f"DataFrame shape: {picks.shape}")

        # Check first row in detail
        first_row = picks.iloc[0]
        logging.info(f"First row notice_id: {first_row['notice_id']}")
        logging.info(f"First row has 'description' column: {'description' in picks.columns}")

        if 'description' in picks.columns:
            first_desc = first_row['description']
            logging.info(f"First row description type: {type(first_desc)}")
            logging.info(f"First row description is None: {first_desc is None}")
            logging.info(f"First row description is NaN: {pd.isna(first_desc)}")
            if pd.notna(first_desc):
                desc_str = str(first_desc)
                logging.info(f"First row description length: {len(desc_str)}")
                logging.info(f"First row description preview: {desc_str[:100]}...")
            else:
                logging.info("First row description is NULL/NaN")
        else:
            logging.error("❌ CRITICAL: 'description' column not in DataFrame!")
            logging.info(f"Available columns: {list(picks.columns)}")
        logging.info("=" * 60)

    description_summaries = _generate_description_summaries(picks)

    logging.info(f"Generating match explanations for {len(picks)} final candidates")
    match_explanations = _generate_match_explanations(picks, tech_desc)

    def get_score_color(score):
        if score >= 85:
            return "#10b981"
        elif score >= 70:
            return "#3b82f6"
        else:
            return "#f59e0b"

    rows_html = []
    for idx, r in picks.iterrows():
        nid = str(r.get("notice_id", ""))
        title = (r.get("title", "") or "Untitled").strip()
        score = float(r.get("score", 0.0))
        reason = r.get("overall_reason", "AI assessment of match quality")
        url = _sam_public_url(nid, r.get("link", ""))

        posted_date = r.get("posted_date", "")
        response_date = r.get("response_date", "")
        pop_city = r.get("pop_city", "")
        pop_state = r.get("pop_state", "")
        naics = r.get("naics_code", "")
        category = r.get("category", "")

        location = ""
        if pop_city or pop_state:
            location = f"{pop_city}, {pop_state}".strip(", ")

        # Get AI-generated summary, fallback to actual description if AI fails
        if nid in description_summaries:
            description_summary = description_summaries[nid]
        else:
            # Fallback: get description directly from row
            raw_desc = str(r["description"]) if "description" in r else ""
            # Check if it's actually populated
            if pd.notna(r["description"]) and raw_desc.strip():
                description_summary = raw_desc.strip()[:500]
            else:
                description_summary = "No description available."
                logging.warning(f"No description found for notice_id {nid} (in fallback)")

        match_explanation = match_explanations.get(
            nid, "This opportunity aligns with this technology area.")

        score_color = get_score_color(score)

        metadata_parts = []
        if category:
            category_emoji = "🔬" if category == "research" else "🔧" if category == "parts_services" else "📋"
            metadata_parts.append(f'<span style="margin-right: 16px;">{category_emoji} {category.upper()}</span>')
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

            <div style="background-color: #f8fafc; border-left: 3px solid #94a3b8; padding: 12px 16px; border-radius: 4px; margin: 12px 0;">
              <p style="color: #475569; font-size: 14px; line-height: 1.6; margin: 0;">
                {description_summary}
              </p>
            </div>

            <div style="background-color: #f0f9ff; border-left: 3px solid #3b82f6; padding: 12px 16px; border-radius: 4px; margin: 12px 0;">
              <p style="color: #1e40af; font-size: 14px; line-height: 1.6; margin: 0; font-weight: 500;">
                💡 {match_explanation}
              </p>
            </div>

            <div style="margin-top: 16px;">
              <a href="{url}" style="color: #3b82f6; text-decoration: none; font-size: 14px; font-weight: 500;">
                View Full Details →
              </a>
            </div>
          </div>
        """)

    tech_label = f" [{tech_name}]" if tech_name else ""
    subject = f"KIP Daily Digest{tech_label}: {len(rows_html)} Top {'Match' if len(rows_html) == 1 else 'Matches'} for {datetime.now().strftime('%B %d, %Y')}"

    # Get funny Kip greeting
    funny_greeting = _get_funny_kip_greeting(len(rows_html), tech_name)

    html = f"""
    {email_wrapper}
      {header}
      <div style="padding: 32px 24px;">
        <p style="color: #475569; font-size: 15px; line-height: 1.6; margin: 0 0 24px 0;">
          {funny_greeting}
        </p>

        {kip_meme_section}

        <div style="margin: 24px 0;">
          {''.join(rows_html)}
        </div>
      </div>
      {footer}
    </div>
    """
    return subject, html


def main():
    global engine, oai

    _load_config()

    engine = sa.create_engine(DB_URL, pool_pre_ping=True)
    oai = OpenAI(api_key=OPENAI_API_KEY)

    today_date = _today_utc_date()
    logging.info("Processing notices pulled today: %s", today_date)

    logging.info(
        "Two-stage filtering: Stage 1 → %d candidates, Stage 2 → detailed scoring", PREFILTER_CANDIDATES)

    with engine.connect() as conn:
        # Debug: show recent dates
        try:
            recent_dates = conn.execute(text("""
                SELECT DATE(pulled_at) as pull_date, category, COUNT(*) 
                FROM solicitationraw 
                WHERE pulled_at IS NOT NULL 
                GROUP BY DATE(pulled_at), category
                ORDER BY pull_date DESC, category
                LIMIT 15
            """)).fetchall()
            logging.info("Recent pull dates in database (by category):")
            for date_val, cat, count in recent_dates:
                logging.info("  %s [%s]: %d records", date_val, cat or "NULL", count)
        except Exception as e:
            logging.warning("Could not fetch recent dates: %s", e)

        tech_areas = _fetch_technology_areas_with_emails(conn)
        if tech_areas.empty:
            logging.info("No technology areas with email addresses configured. Exiting.")
            return

        notices = _fetch_today_notices(conn, today_date)
        notices = _filter_out_healthcare(notices)
        if notices.empty:
            logging.info(
                "No notices pulled today (%s). Sending 'no matches' to all technology areas.", today_date)
            for _, tech_area in tech_areas.iterrows():
                emails = _parse_email_addresses(tech_area["emails"])
                for email in emails:
                    subject, html = _render_email(
                        email, tech_area["technology_description"], pd.DataFrame(),
                        tech_name=tech_area["technology_name"])
                    _send_email(email, subject, html)
            return

        logging.info("Found %d notices pulled today (%s), processing %d technology areas...", len(
            notices), today_date, len(tech_areas))

        # Process each technology area separately
        for _, tech_area in tech_areas.iterrows():
            tech_name = tech_area["technology_name"]
            desc = (tech_area["technology_description"] or "").strip()

            if not desc:
                logging.info(f"Skipping {tech_name} (no description)")
                continue

            logging.info(f"Processing {tech_name}")

            # STAGE 1: Fast embedding-based pre-filter
            stage1_candidates = _stage1_embedding_filter(
                notices, desc, top_k=PREFILTER_CANDIDATES)

            if stage1_candidates.empty:
                logging.info(f"Stage 1: No candidates for {tech_name}")
                continue

            # STAGE 2: Detailed scoring
            scored_notices = _stage2_detailed_scoring(
                stage1_candidates, tech_area.to_dict())

            # Filter and send
            final_matches = scored_notices[scored_notices["score"] >= MIN_SCORE].head(MAX_RESULTS)

            # Parse email addresses (may be comma-separated)
            emails = _parse_email_addresses(tech_area["emails"])

            if not emails:
                logging.warning(f"No valid email addresses for {tech_name}")
                continue

            if not final_matches.empty:
                subject, html = _render_email(
                    emails[0],  # Primary email for recipient field
                    desc,
                    final_matches,
                    tech_name=tech_name
                )

                # Send to all configured email addresses
                for email in emails:
                    _send_email(email, subject, html)
                    logging.info(f"Sent {len(final_matches)} matches for {tech_name} to {email}")
            else:
                logging.info(f"No matches above threshold for {tech_name}")

    logging.info("Daily digest complete.")


if __name__ == "__main__":
    main()