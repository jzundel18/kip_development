# scoring.py - Complete scoring module for KIP
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI


@dataclass
class MatrixComponent:
    key: str
    label: str
    weight: float
    description: str
    hints: list[str]
    scoring_method: str = "llm_assessment"


def make_sam_public_url(notice_id: str, link: str | None = None) -> str:
    """Return a human-viewable SAM.gov URL for this notice."""
    if link and isinstance(link, str) and "api.sam.gov" not in link:
        return link
    nid = (notice_id or "").strip()
    return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"


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
            # Use detailed fallback scoring when AI fails
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


def ai_matrix_score_solicitations(df: pd.DataFrame, company_profile: dict, api_key: str,
                                  top_k: int = 10, model: str = "gpt-4o-mini", max_candidates: int = 60) -> list[dict]:
    """Enhanced matrix scoring with complete scoring components and optimized processing"""
    if df is None or df.empty:
        return []

    # Prepare data for scoring
    cols = ["notice_id", "title", "description", "naics_code", "set_aside_code",
            "response_date", "posted_date", "link", "pop_city", "pop_state"]
    use_df = df[[c for c in cols if c in df.columns]].copy()

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
            continue

    if not results:
        return []

    # Combine results with original data
    enhanced_df = use_df.copy()
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


# Compatibility function for other modules
def ai_score_and_rank_solicitations_by_fit(df: pd.DataFrame, company_desc: str, company_profile: Dict[str, str], api_key: str, top_k: int = 10) -> list[dict]:
    """Compatibility wrapper for the function used in app.py"""
    prof = dict(company_profile or {})
    prof["description"] = company_desc or prof.get("description", "") or ""
    return ai_matrix_score_solicitations(df=df, company_profile=prof, api_key=api_key, top_k=int(top_k), model="gpt-4o-mini", max_candidates=60)
