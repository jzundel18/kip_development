# enhanced_matching.py - Robust Solicitation Matching System
"""
Enhanced solicitation matching system with improved criteria clarity and robustness.
This system uses a three-stage approach for maximum accuracy and transparency.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI
import logging


class MatchingStage(Enum):
    """Stages of the matching pipeline"""
    PREFILTER = "prefilter"
    EMBEDDING = "embedding"
    DETAILED = "detailed"


@dataclass
class MatchingCriteria:
    """Clear definition of what makes a good solicitation match"""

    # Technical Alignment (40%)
    # Company's main services align with solicitation needs
    core_services_match: float = 25.0
    industry_expertise: float = 15.0       # Relevant industry domain knowledge

    # Business Qualification (30%)
    contract_experience: float = 15.0      # Government contracting experience
    business_size_fit: float = 10.0        # Set-aside and size requirements match
    capability_depth: float = 5.0          # Depth of relevant capabilities

    # Practical Feasibility (20%)
    # Location compatibility with place of performance
    geographic_alignment: float = 8.0
    contract_value_fit: float = 7.0        # Contract size appropriate for company
    timing_feasibility: float = 5.0        # Adequate time to prepare and perform

    # Strategic Value (10%)
    growth_opportunity: float = 5.0        # Potential for business growth
    competitive_advantage: float = 3.0     # Company has unique advantages
    # Opportunity to build agency relationships
    relationship_building: float = 2.0

    @property
    def total_weight(self) -> float:
        return sum([
            self.core_services_match, self.industry_expertise,
            self.contract_experience, self.business_size_fit, self.capability_depth,
            self.geographic_alignment, self.contract_value_fit, self.timing_feasibility,
            self.growth_opportunity, self.competitive_advantage, self.relationship_building
        ])


@dataclass
class CompanyProfile:
    """Structured company profile for matching"""
    # Core Information
    company_name: str
    description: str
    state: str = ""
    states_perform_work: str = ""

    # Business Details
    primary_services: List[str] = None
    industries_served: List[str] = None
    certifications: List[str] = None
    size_category: str = ""  # "small", "large", etc.
    set_aside_eligibilities: List[str] = None

    # NAICS codes
    NAICS: str = ""
    other_NAICS: str = ""

    # Contact information
    email: str = ""
    phone: str = ""
    contact: str = ""

    # Capabilities
    core_competencies: List[str] = None
    past_performance_sectors: List[str] = None
    contract_value_range: Tuple[int, int] = (0, 0)  # (min, max)

    def __post_init__(self):
        # Initialize empty lists if None
        if self.primary_services is None:
            self.primary_services = []
        if self.industries_served is None:
            self.industries_served = []
        if self.certifications is None:
            self.certifications = []
        if self.set_aside_eligibilities is None:
            self.set_aside_eligibilities = []
        if self.core_competencies is None:
            self.core_competencies = []
        if self.past_performance_sectors is None:
            self.past_performance_sectors = []

class EnhancedMatcher:
    """Enhanced three-stage matching system"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.criteria = MatchingCriteria()
        self.logger = logging.getLogger(__name__)

        # Service taxonomy for better matching
        self.service_taxonomy = self._load_service_taxonomy()

        # Industry classifications
        self.industry_keywords = self._load_industry_keywords()

    def _load_service_taxonomy(self) -> Dict[str, List[str]]:
        """Comprehensive service taxonomy for better matching"""
        return {
            "manufacturing": [
                "machining", "cnc", "fabrication", "welding", "assembly", "molding",
                "casting", "tooling", "prototyping", "production", "machining centers",
                "precision manufacturing", "metal working", "plastic manufacturing"
            ],
            "engineering": [
                "design", "analysis", "simulation", "cad", "fem", "testing",
                "validation", "systems engineering", "mechanical engineering",
                "electrical engineering", "software engineering", "civil engineering"
            ],
            "it_services": [
                "software development", "cybersecurity", "cloud services", "data analytics",
                "system integration", "help desk", "network management", "database",
                "application development", "it consulting", "digital transformation"
            ],
            "professional_services": [
                "consulting", "training", "project management", "technical writing",
                "business analysis", "process improvement", "compliance",
                "audit", "advisory services", "strategic planning"
            ],
            "maintenance_services": [
                "facility maintenance", "equipment maintenance", "preventive maintenance",
                "repair services", "janitorial", "grounds keeping", "hvac maintenance",
                "electrical maintenance", "mechanical maintenance"
            ],
            "logistics": [
                "transportation", "warehousing", "supply chain", "distribution",
                "freight", "shipping", "inventory management", "fulfillment"
            ],
            "construction": [
                "building construction", "renovation", "infrastructure", "electrical work",
                "plumbing", "roofing", "flooring", "painting", "demolition"
            ]
        }

    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Industry-specific keywords for domain matching"""
        return {
            "defense": [
                "military", "army", "navy", "air force", "marines", "dod", "defense",
                "weapon systems", "aircraft", "vehicles", "communications", "radar",
                "security clearance", "itar", "dfars"
            ],
            "aerospace": [
                "aircraft", "spacecraft", "satellite", "aviation", "flight", "propulsion",
                "avionics", "flight test", "faa", "nasa", "space systems"
            ],
            "healthcare": [
                "medical", "hospital", "clinic", "pharmaceutical", "medical device",
                "healthcare", "patient", "clinical", "fda", "medical equipment"
            ],
            "energy": [
                "power", "utility", "renewable energy", "solar", "wind", "nuclear",
                "oil", "gas", "electricity", "grid", "transmission"
            ],
            "transportation": [
                "highway", "bridge", "transit", "rail", "airport", "seaport",
                "dot", "faa", "transportation infrastructure"
            ],
            "telecommunications": [
                "communications", "network", "wireless", "broadband", "fiber",
                "telecommunications", "radio", "satellite communications"
            ],
            "environmental": [
                "environmental", "epa", "remediation", "waste", "water treatment",
                "air quality", "environmental compliance", "sustainability"
            ]
        }

    def parse_company_profile(self, raw_profile: Dict[str, str]) -> CompanyProfile:
        """Parse and enrich company profile using AI"""
        description = raw_profile.get("description", "").strip()

        if not description:
            return CompanyProfile(
                company_name=raw_profile.get("company_name", ""),
                description="",
                city=raw_profile.get("city", ""),
                state=raw_profile.get("state", "")
            )

        try:
            # Use AI to extract structured information from description
            system_prompt = """You are a business analyst. Extract structured information from company descriptions.
            
            Focus on:
            - Primary services offered
            - Industries served  
            - Key certifications/credentials
            - Core competencies
            - Past performance areas
            
            Return JSON with these exact keys: primary_services, industries_served, certifications, core_competencies, past_performance_sectors"""

            user_prompt = f"""Company: {raw_profile.get('company_name', 'Unknown')}
            Description: {description}
            
            Extract the key business information into structured format."""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use smaller model for parsing
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )

            parsed_data = json.loads(
                response.choices[0].message.content or "{}")

            return CompanyProfile(
                company_name=raw_profile.get("company_name", ""),
                description=description,
                city=raw_profile.get("city", ""),
                state=raw_profile.get("state", ""),
                primary_services=parsed_data.get("primary_services", []),
                industries_served=parsed_data.get("industries_served", []),
                certifications=parsed_data.get("certifications", []),
                core_competencies=parsed_data.get("core_competencies", []),
                past_performance_sectors=parsed_data.get(
                    "past_performance_sectors", [])
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to parse company profile with AI: {e}")
            # Fallback to basic profile
            return CompanyProfile(
                company_name=raw_profile.get("company_name", ""),
                description=description,
                city=raw_profile.get("city", ""),
                state=raw_profile.get("state", "")
            )

    def stage1_prefilter(self, solicitations: pd.DataFrame, company_profile: CompanyProfile) -> pd.DataFrame:
        """Stage 1: Fast rule-based prefiltering to eliminate obvious non-matches"""
        if solicitations.empty:
            return solicitations

        self.logger.info(
            f"Stage 1: Prefiltering {len(solicitations)} solicitations")

        filtered_df = solicitations.copy()

        # Filter out expired solicitations
        today = pd.Timestamp.now().date()

        def is_expired(response_date_str):
            if not response_date_str or str(response_date_str).lower() in ("none", "nan", ""):
                return False
            try:
                response_date = pd.to_datetime(response_date_str).date()
                return response_date < today
            except:
                return False

        pre_count = len(filtered_df)
        filtered_df = filtered_df[~filtered_df["response_date"].apply(
            is_expired)]
        expired_filtered = pre_count - len(filtered_df)

        # Filter out award/justification notices
        notice_type_mask = ~filtered_df["notice_type"].str.lower().str.contains(
            "award|justification", case=False, na=False
        )
        filtered_df = filtered_df[notice_type_mask]

        # Basic keyword filtering for obvious matches
        if company_profile.primary_services:
            text_columns = ["title", "description"]
            combined_text = filtered_df[text_columns].fillna("").apply(
                lambda x: " ".join(x).lower(), axis=1
            )

            service_keywords = []
            for service in company_profile.primary_services:
                if service.lower() in self.service_taxonomy:
                    service_keywords.extend(
                        self.service_taxonomy[service.lower()])
                else:
                    service_keywords.append(service.lower())

            if service_keywords:
                keyword_mask = combined_text.apply(
                    lambda text: any(kw in text for kw in service_keywords)
                )
                # Keep solicitations that match OR have no clear service indication
                unclear_mask = ~combined_text.str.contains(
                    "|".join(list(self.service_taxonomy.keys())), case=False, na=False
                )
                filtered_df = filtered_df[keyword_mask | unclear_mask]

        self.logger.info(
            f"Stage 1 complete: {len(filtered_df)} remaining ({expired_filtered} expired filtered)")
        return filtered_df.reset_index(drop=True)

    def stage2_embedding_filter(self, solicitations: pd.DataFrame, company_profile: CompanyProfile,
                                top_k: int = 50) -> pd.DataFrame:
        """Stage 2: Semantic similarity using embeddings"""
        if solicitations.empty or not company_profile.description:
            return solicitations.head(top_k)

        self.logger.info(
            f"Stage 2: Embedding filter on {len(solicitations)} solicitations")

        try:
            # Create enhanced company representation
            company_text = self._create_company_text_representation(
                company_profile)

            # Create solicitation representations
            solicitation_texts = solicitations.apply(
                lambda row: self._create_solicitation_text_representation(row), axis=1
            ).tolist()

            # Get embeddings
            company_embedding = self._get_embedding(company_text)
            solicitation_embeddings = self._get_embeddings_batch(
                solicitation_texts)

            # Calculate similarities
            similarities = np.array([
                np.dot(company_embedding, sol_emb) /
                (np.linalg.norm(company_embedding) * np.linalg.norm(sol_emb))
                for sol_emb in solicitation_embeddings
            ])

            # Sort by similarity and take top k
            df_with_similarity = solicitations.copy()
            df_with_similarity["similarity_score"] = similarities

            top_solicitations = df_with_similarity.sort_values(
                "similarity_score", ascending=False
            ).head(top_k)

            self.logger.info(
                f"Stage 2 complete: {len(top_solicitations)} candidates selected")
            return top_solicitations.drop(columns=["similarity_score"])

        except Exception as e:
            self.logger.warning(f"Stage 2 embedding filter failed: {e}")
            return solicitations.head(top_k)

    def stage3_detailed_scoring(self, solicitations: pd.DataFrame,
                                company_profile: CompanyProfile) -> List[Dict[str, Any]]:
        """Stage 3: Detailed LLM-based scoring with explicit criteria"""
        if solicitations.empty:
            return []

        self.logger.info(
            f"Stage 3: Detailed scoring of {len(solicitations)} candidates")

        results = []

        # Process in small batches for reliability
        batch_size = 3
        for i in range(0, len(solicitations), batch_size):
            batch = solicitations.iloc[i:i+batch_size]

            try:
                batch_results = self._score_batch_detailed(
                    batch, company_profile)
                results.extend(batch_results)
            except Exception as e:
                self.logger.error(
                    f"Failed to score batch {i//batch_size + 1}: {e}")
                # Add fallback scores for this batch
                for _, row in batch.iterrows():
                    results.append({
                        "notice_id": str(row.get("notice_id", "")),
                        "score": 50.0,
                        "reasoning": f"Scoring failed: {str(e)[:100]}",
                        "component_scores": {}
                    })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        self.logger.info(
            f"Stage 3 complete: {len(results)} solicitations scored")
        return results

    def _create_company_text_representation(self, profile: CompanyProfile) -> str:
        """Create rich text representation of company for embedding"""
        parts = []

        parts.append(f"Company: {profile.company_name}")
        parts.append(f"Description: {profile.description}")

        if profile.primary_services:
            parts.append(
                f"Primary Services: {', '.join(profile.primary_services)}")

        if profile.industries_served:
            parts.append(f"Industries: {', '.join(profile.industries_served)}")

        if profile.certifications:
            parts.append(f"Certifications: {', '.join(profile.certifications)}")

        if profile.core_competencies:
            parts.append(
                f"Core Competencies: {', '.join(profile.core_competencies)}")

        if profile.NAICS:
            parts.append(f"NAICS: {profile.NAICS}")

        if profile.other_NAICS:
            parts.append(f"Other NAICS: {profile.other_NAICS}")

        if profile.state:
            parts.append(f"Location: {profile.state}")

        if profile.states_perform_work:
            parts.append(f"Work Locations: {profile.states_perform_work}")

        return " | ".join(parts)

    def _create_solicitation_text_representation(self, row: pd.Series) -> str:
        """Create rich text representation of solicitation"""
        parts = []

        title = str(row.get("title", "")).strip()
        if title and title != "nan":
            parts.append(f"Title: {title}")

        description = str(row.get("description", "")).strip()
        if description and description != "nan":
            parts.append(f"Description: {description[:500]}")  # Limit length

        naics = str(row.get("naics_code", "")).strip()
        if naics and naics != "nan":
            parts.append(f"NAICS: {naics}")

        set_aside = str(row.get("set_aside_code", "")).strip()
        if set_aside and set_aside != "nan":
            parts.append(f"Set-Aside: {set_aside}")

        pop_location = []
        city = str(row.get("pop_city", "")).strip()
        state = str(row.get("pop_state", "")).strip()
        if city and city != "nan":
            pop_location.append(city)
        if state and state != "nan":
            pop_location.append(state)
        if pop_location:
            parts.append(f"Location: {', '.join(pop_location)}")

        return " | ".join(parts)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [
                np.array(d.embedding, dtype=np.float32) for d in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    def _score_batch_detailed(self, batch: pd.DataFrame,
                              company_profile: CompanyProfile) -> List[Dict[str, Any]]:
        """Score a batch of solicitations with detailed criteria"""

        system_prompt = f"""You are an expert federal contracting analyst. Score how well each solicitation matches the company using these EXACT criteria (weights in parentheses):

TECHNICAL ALIGNMENT (40% total):
- Core Services Match ({self.criteria.core_services_match}%): Company's main services align with solicitation needs
- Industry Expertise ({self.criteria.industry_expertise}%): Relevant industry domain knowledge

BUSINESS QUALIFICATION (30% total):
- Contract Experience ({self.criteria.contract_experience}%): Government contracting experience 
- Business Size Fit ({self.criteria.business_size_fit}%): Set-aside and size requirements match
- Capability Depth ({self.criteria.capability_depth}%): Depth of relevant capabilities

PRACTICAL FEASIBILITY (20% total):
- Geographic Alignment ({self.criteria.geographic_alignment}%): Location compatibility
- Contract Value Fit ({self.criteria.contract_value_fit}%): Appropriate contract size
- Timing Feasibility ({self.criteria.timing_feasibility}%): Adequate preparation time

STRATEGIC VALUE (10% total):
- Growth Opportunity ({self.criteria.growth_opportunity}%): Business growth potential
- Competitive Advantage ({self.criteria.competitive_advantage}%): Unique advantages
- Relationship Building ({self.criteria.relationship_building}%): Agency relationship potential

Score each component 1-10, then calculate weighted total. Return JSON with exact structure:
{{"results": [{{"notice_id": "...", "total_score": 75.5, "components": [{{"name": "Core Services Match", "score": 8, "weight": {self.criteria.core_services_match}, "reasoning": "..."}}], "overall_reasoning": "..."}}]}}"""

        # Prepare solicitation data
        solicitation_data = []
        for _, row in batch.iterrows():
            solicitation_data.append({
                "notice_id": str(row.get("notice_id", "")),
                "title": str(row.get("title", ""))[:200],
                "description": str(row.get("description", ""))[:800],
                "naics_code": str(row.get("naics_code", "")),
                "set_aside_code": str(row.get("set_aside_code", "")),
                "location": f"{row.get('pop_city', '')} {row.get('pop_state', '')}".strip(),
                "response_date": str(row.get("response_date", ""))
            })

        user_prompt = {
            "company_profile": {
                "name": company_profile.company_name,
                "description": company_profile.description[:500],
                "location": f"{company_profile.city} {company_profile.state}".strip(),
                # Limit for token efficiency
                "primary_services": company_profile.primary_services[:5],
                "industries_served": company_profile.industries_served[:5],
                "certifications": company_profile.certifications[:3]
            },
            "solicitations": solicitation_data
        }

        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)}
            ],
            temperature=0.1,
            max_tokens=2500
        )

        try:
            data = json.loads(response.choices[0].message.content or "{}")
            results = []

            for result in data.get("results", []):
                notice_id = result.get("notice_id", "")
                score = float(result.get("total_score", 0))
                reasoning = result.get("overall_reasoning", "")
                components = result.get("components", [])

                # Convert component scores to simple dict
                component_scores = {}
                for comp in components:
                    component_scores[comp.get("name", "")] = {
                        "score": comp.get("score", 0),
                        "reasoning": comp.get("reasoning", "")
                    }

                results.append({
                    "notice_id": notice_id,
                    "score": score,
                    "reasoning": reasoning,
                    "component_scores": component_scores
                })

            return results

        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse detailed scoring response: {e}")
            raise

    def match_solicitations(self, solicitations: pd.DataFrame, raw_company_profile: Dict[str, str],
                            prefilter_limit: int = 200, embedding_limit: int = 50,
                            final_limit: int = 10) -> List[Dict[str, Any]]:
        """Complete three-stage matching pipeline"""

        if solicitations.empty:
            return []

        self.logger.info(
            f"Starting matching pipeline with {len(solicitations)} solicitations")

        # Parse company profile
        company_profile = self.parse_company_profile(raw_company_profile)

        # Stage 1: Prefilter
        stage1_results = self.stage1_prefilter(solicitations, company_profile)
        if len(stage1_results) > prefilter_limit:
            stage1_results = stage1_results.head(prefilter_limit)

        if stage1_results.empty:
            self.logger.info("No solicitations passed Stage 1 prefiltering")
            return []

        # Stage 2: Embedding filter
        stage2_results = self.stage2_embedding_filter(
            stage1_results, company_profile, top_k=embedding_limit
        )

        if stage2_results.empty:
            self.logger.info(
                "No solicitations passed Stage 2 embedding filter")
            return []

        # Stage 3: Detailed scoring
        final_results = self.stage3_detailed_scoring(
            stage2_results, company_profile)

        # Limit final results
        final_results = final_results[:final_limit]

        # Add additional metadata
        for result in final_results:
            notice_id = result["notice_id"]
            sol_row = solicitations[solicitations["notice_id"].astype(
                str) == notice_id].iloc[0]

            result.update({
                "title": str(sol_row.get("title", "")),
                "link": self._make_sam_public_url(notice_id, sol_row.get("link", "")),
                "response_date": str(sol_row.get("response_date", "")),
                "naics_code": str(sol_row.get("naics_code", "")),
                "set_aside_code": str(sol_row.get("set_aside_code", "")),
                "pop_location": f"{sol_row.get('pop_city', '')} {sol_row.get('pop_state', '')}".strip()
            })

        self.logger.info(
            f"Matching complete: {len(final_results)} final matches")
        return final_results

    def _make_sam_public_url(self, notice_id: str, link: str = None) -> str:
        """Create public SAM.gov URL"""
        if link and isinstance(link, str) and "api.sam.gov" not in link:
            return link
        nid = (notice_id or "").strip()
        return f"https://sam.gov/opp/{nid}/view" if nid else "https://sam.gov/"


# Usage Example and Integration Guide
def integrate_with_streamlit(df: pd.DataFrame, company_profile_dict: Dict[str, str],
                             openai_api_key: str) -> List[Dict[str, Any]]:
    """
    Integration function for your Streamlit app.
    
    Replace your existing AI scoring with:
    
    matcher = EnhancedMatcher(api_key=OPENAI_API_KEY)
    results = matcher.match_solicitations(
        solicitations=df,
        raw_company_profile={
            "company_name": profile.get("company_name", ""),
            "description": company_desc,
            "city": profile.get("city", ""),
            "state": profile.get("state", "")
        },
        final_limit=top_k
    )
    """
    matcher = EnhancedMatcher(api_key=openai_api_key)

    return matcher.match_solicitations(
        solicitations=df,
        raw_company_profile=company_profile_dict,
        prefilter_limit=200,
        embedding_limit=50,
        final_limit=10
    )
