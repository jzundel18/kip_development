#!/usr/bin/env python3
"""
Send a daily AI-matched solicitation digest for items posted yesterday.

Env vars required:
  SUPABASE_DB_URL     - same DB as app (e.g., postgresql+psycopg2://...)
  OPENAI_API_KEY      - for embeddings + ranking
  SENDGRID_API_KEY    - to send emails
  FROM_EMAIL          - verified sender, e.g., 'alerts@yourdomain.com'

Optional:
  APP_BASE_URL        - include links back to your app (not required)

Run daily via cron:
  7:30am local, e.g.:
    30 7 * * * /usr/bin/env bash -lc 'cd /path/to/app && OPENAI_API_KEY=... SENDGRID_API_KEY=... FROM_EMAIL=... SUPABASE_DB_URL=... python3 daily_digest.py'
"""

import os
import json
import re
import time
import logging
import requests
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from openai import OpenAI

# ---------- Config ----------
DB_URL = os.getenv("SUPABASE_DB_URL", "sqlite:///app.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "")
APP_BASE_URL = os.getenv("APP_BASE_URL", "")

if not (OPENAI_API_KEY and SENDGRID_API_KEY and FROM_EMAIL):
    raise SystemExit(
        "Missing one or more required env vars: OPENAI_API_KEY, SENDGRID_API_KEY, FROM_EMAIL")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s", datefmt="%H:%M:%S")

engine = sa.create_engine(DB_URL, pool_pre_ping=True)
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Helpers (lightweight copies) ----------


def _s(v):
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return "" if v is None else str(v)


def make_sam_public_url(notice_id: str, link: str | None = None) -> str:
    if link and "api.sam.gov" not in link:
        return link
    nid = (notice_id or "").strip()
    return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"


def _embed_texts(texts: list[str]) -> np.ndarray:
    # normalize L2
    resp = client.embeddings.create(
        model="text-embedding-3-small", input=texts)
    X = np.array([d.embedding for d in resp.data], dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def ai_downselect_df(company_desc: str, df: pd.DataFrame, top_k: int = 80) -> pd.DataFrame:
    if df.empty:
        return df
    texts = (df["title"].fillna("") + " " +
             df["description"].fillna("")).str.slice(0, 2000).tolist()
    q = client.embeddings.create(
        model="text-embedding-3-small", input=[company_desc])
    Xq = np.array(q.data[0].embedding, dtype=np.float32)
    Xq = Xq / (np.linalg.norm(Xq) + 1e-9)

    X_list = []
    for i in range(0, len(texts), 500):
        r = client.embeddings.create(
            model="text-embedding-3-small", input=texts[i:i+500])
        X_list.extend([d.embedding for d in r.data])
    X = np.array(X_list, dtype=np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    sims = X @ Xq
    df2 = df.copy()
    df2["__sim"] = sims
    df2 = df2.sort_values("__sim", ascending=False).head(
        int(top_k)).drop(columns="__sim")
    return df2.reset_index(drop=True)


def ai_rank_solicitations_by_fit(df: pd.DataFrame, company_desc: str, top_k: int = 10) -> list[dict]:
    if df.empty:
        return []
    items = []
    for _, r in df.iterrows():
        items.append({
            "notice_id": str(r.get("notice_id", "")),
            "title": _s(r.get("title"))[:300],
            "description": _s(r.get("description"))[:1500],
            "naics_code": _s(r.get("naics_code")),
            "set_aside_code": _s(r.get("set_aside_code")),
            "response_date": _s(r.get("response_date")),
            "posted_date": _s(r.get("posted_date")),
            "link": _s(r.get("link")),
        })
    system_msg = (
        "You are a contracts analyst. Rank solicitations by how well they match the company description. "
        "Consider title, description, NAICS, set-aside, and recency."
    )
    user_msg = {
        "company_description": company_desc,
        "solicitations": items,
        "instructions": (
            f"Return the top {top_k} as JSON: "
            '{"ranked":[{"notice_id":"...","score":0-100,"reason":"..."}]}. '
            "Score reflects strength of fit; keep reasons short."
        ),
    }
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_msg}, {
            "role": "user", "content": json.dumps(user_msg)}],
        temperature=0.2,
    )
    try:
        data = json.loads(r.choices[0].message.content or "{}")
        ranked = data.get("ranked", [])
    except Exception:
        return []
    seen = set()
    out = []
    keep_ids = set(df["notice_id"].astype(str))
    for it in ranked:
        nid = str(it.get("notice_id", ""))
        if nid in keep_ids and nid not in seen:
            seen.add(nid)
            out.append({"notice_id": nid, "score": float(
                it.get("score", 0)), "reason": _s(it.get("reason", ""))})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


def _send_email_via_sendgrid(to_email: str, subject: str, html: str):
    url = "https://api.sendgrid.com/v3/mail/send"
    payload = {
        "personalizations": [{"to": [{"email": to_email}], "subject": subject}],
        "from": {"email": FROM_EMAIL},
        "content": [{"type": "text/html", "value": html}]
    }
    r = requests.post(url, headers={"Authorization": f"Bearer {SENDGRID_API_KEY}", "Content-Type": "application/json"},
                      json=payload, timeout=20)
    if r.status_code >= 300:
        logging.warning("SendGrid error (%s): %s", r.status_code, r.text[:300])

# ---------- Main ----------


def main():
    # 1) Pull yesterday’s solicitations (UTC)
    today = datetime.now(timezone.utc).date()
    y0 = datetime.combine(today - timedelta(days=1),
                          datetime.min.time(), tzinfo=timezone.utc)
    y1 = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)

    with engine.connect() as conn:
        df = pd.read_sql_query("""
            SELECT notice_id, title, description, naics_code, set_aside_code,
                   posted_date, response_date, link
            FROM solicitationraw
        """, conn)

    if df.empty:
        logging.info("No solicitations in DB.")
        return

    # normalize posted_date to UTC and filter to yesterday
    df["posted_dt"] = pd.to_datetime(
        df["posted_date"], errors="coerce", utc=True)
    ydf = df[(df["posted_dt"] >= y0) & (
        df["posted_dt"] < y1)].drop(columns=["posted_dt"])
    if ydf.empty:
        logging.info("No new solicitations posted yesterday.")
        return

    # 2) Get enabled subscribers
    with engine.connect() as conn:
        subs = pd.read_sql_query("""
            SELECT id, user_id, email, is_enabled, min_score, max_per_day, company_desc_override
            FROM digest_subscribers
            WHERE is_enabled = TRUE
        """, conn)

        # for each subscriber, try to pull profile description if override is empty
        profiles = pd.read_sql_query("""
            SELECT u.id as user_id, u.email, cp.description
            FROM users u
            LEFT JOIN company_profile cp ON cp.user_id = u.id
        """, conn)

    if subs.empty:
        logging.info("No enabled subscribers.")
        return

    # join in profile descriptions by email or user_id
    profiles_by_user = {int(r["user_id"]): _s(r["description"])
                        for _, r in profiles.iterrows() if pd.notna(r["user_id"])}
    profiles_by_email = {_s(r["email"]).lower(): _s(
        r["description"]) for _, r in profiles.iterrows()}

    # 3) Build and send each email
    for _, s in subs.iterrows():
        email = _s(s["email"]).strip()
        if not email:
            continue
        desc = _s(s.get("company_desc_override", "")).strip()
        if not desc:
            if pd.notna(s.get("user_id")) and int(s["user_id"]) in profiles_by_user:
                desc = profiles_by_user[int(s["user_id"])]
            elif email.lower() in profiles_by_email:
                desc = profiles_by_email[email.lower()]

        if not desc:
            logging.info("Skip %s: no company description.", email)
            continue

        min_score = int(s.get("min_score", 70))
        max_per = int(s.get("max_per_day", 5))
        max_per = max(1, min(5, max_per))

        # pre-trim then rank
        pretrim = ai_downselect_df(desc, ydf, top_k=80)
        ranked = ai_rank_solicitations_by_fit(pretrim, desc, top_k=10)

        # filter by score cutoff & cap results
        ranked = [r for r in ranked if r.get(
            "score", 0) >= min_score][:max_per]
        if not ranked:
            subject = "Daily KIP Matches: No close matches yesterday"
            html = f"""
            <p>Good morning,</p>
            <p>We checked SAM.gov for new opportunities posted yesterday and didn’t find any close matches for your profile (cutoff ≥ {min_score}).</p>
            <p>— KIP</p>
            """
            _send_email_via_sendgrid(email, subject, html)
            logging.info("Sent 'no matches' to %s", email)
            continue

        # build map notice_id -> row
        ydf_map = {str(r["notice_id"]): r for _, r in pretrim.iterrows()}

        items_html = []
        for i, it in enumerate(ranked, 1):
            rec = ydf_map.get(it["notice_id"])
            if not rec:
                continue
            nid = str(rec["notice_id"])
            title = _s(rec["title"]) or "Untitled"
            reason = _s(it.get("reason", ""))
            score = int(round(float(it.get("score", 0))))
            sam_url = make_sam_public_url(nid, _s(rec.get("link", "")))
            naics = _s(rec.get("naics_code", ""))
            due = _s(rec.get("response_date", ""))
            items_html.append(f"""
              <li>
                <strong><a href="{sam_url}">{title}</a></strong><br/>
                Score: {score} &nbsp; {('NAICS: ' + naics) if naics else ''} &nbsp; {('Due: ' + due) if due else ''}<br/>
                <em>{reason}</em>
              </li>
            """)

        subject = f"Daily KIP Matches: {len(items_html)} close {'match' if len(items_html)==1 else 'matches'}"
        html = f"""
        <p>Good morning,</p>
        <p>Here are your top AI-matched opportunities posted yesterday (cutoff ≥ {min_score}, max {max_per}):</p>
        <ol>
          {''.join(items_html)}
        </ol>
        {"<p>View more in KIP: <a href='"+APP_BASE_URL+"'>"+APP_BASE_URL+"</a></p>" if APP_BASE_URL else ""}
        <p>— KIP</p>
        """
        _send_email_via_sendgrid(email, subject, html)
        logging.info("Sent digest (%d items) to %s", len(items_html), email)


if __name__ == "__main__":
    main()
