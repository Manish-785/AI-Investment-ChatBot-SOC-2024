"""Domain-aware NLP classifier for the investment chatbot.

This module adds deterministic finance-domain phrase matching before semantic
fallback. It is intentionally standalone so benchmark changes can be compared
against the current production classifier without silently changing the app.
"""

import re
from typing import Dict, Tuple

import spacy
from sentence_transformers import SentenceTransformer, util

SECTOR_ALIASES = {
    "technology": ["tech", "software", "it companies", "semiconductor", "semiconductors", "chip companies"],
    "finance": ["financial companies", "banks", "banking", "financial services", "insurers", "insurance companies"],
    "healthcare": ["health", "medical", "medicine", "pharma", "pharmaceutical", "pharmaceuticals", "biotech", "biotechnology"],
    "consumer goods": ["consumer products", "fmcg", "household products", "personal care"],
    "energy": ["oil", "gas", "oil and gas", "petroleum", "oil companies", "energy companies"],
    "utilities": ["power companies", "electric utilities", "electricity providers", "water utilities"],
    "materials": ["chemicals", "chemical companies", "metals", "mining", "commodities", "construction materials"],
    "industrials": ["engineering", "manufacturing", "industrial companies", "machinery", "capital goods"],
    "telecommunications": ["telecom", "telecommunications", "wireless", "mobile carriers", "phone companies"],
    "real estate": ["property", "reit", "reits", "realty", "property companies"],
    "consumer services": ["retail", "ecommerce", "e-commerce", "restaurants", "hospitality", "consumer businesses"],
    "transportation": ["airlines", "aviation", "logistics", "shipping", "railways", "railroad", "transport companies"],
    "agriculture": ["farming", "agri", "agricultural", "fertilizer", "fertilizers", "crop companies"],
    "media and entertainment": ["media", "entertainment", "streaming", "gaming", "video games", "broadcasting"],
    "government": ["defense", "defence", "aerospace defense", "government contractors"],
}

GOAL_ALIASES = {
    "retirement": ["retirement", "retire", "retirement fund", "retirement savings"],
    "education": ["education", "college", "university", "higher education", "tuition", "school fees"],
    "home purchase": ["house deposit", "home deposit", "down payment", "buy a house", "buying a house", "buy a home", "buying a home"],
    "wealth accumulation": ["build wealth", "building wealth", "overall wealth", "grow my wealth", "accumulate wealth", "asset accumulation"],
    "emergency savings": ["emergency fund", "emergency savings", "financial safety net", "rainy day fund", "unexpected expenses"],
    "major purchases": ["major purchase", "large purchase", "buying a car", "buy a car", "home renovation"],
    "vacation": ["vacation", "holiday", "travel fund", "travel savings", "trip savings"],
    "debt repayment": ["pay off debt", "paying off debt", "debt repayment", "debt reduction", "loan repayment", "pay off my loan"],
    "healthcare": ["medical expenses", "medical costs", "health expenses", "healthcare expenses", "health savings"],
    "charitable giving": ["charity", "charitable giving", "donations", "donate", "philanthropy"],
    "estate planning": ["estate planning", "plan my estate", "inheritance planning", "heirs"],
    "business investment": ["business investment", "start a business", "starting a business", "entrepreneurship", "business growth"],
    "tax planning": ["tax planning", "tax minimization", "reduce my taxes", "tax liability", "tax savings"],
    "legacy building": ["legacy", "leave a legacy", "legacy building", "future generations", "lasting impact"],
    "growth": ["investment growth", "long term growth", "long-term growth", "capital appreciation", "grow my investment", "value increase"],
}

_RISK_RE = re.compile(r"\b(low|medium|high)\b", re.I)


class InvestmentNLP:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.sectors = list(SECTOR_ALIASES)
        self.goals = list(GOAL_ALIASES)
        self.sector_phrases = [(s, p) for s, ps in SECTOR_ALIASES.items() for p in ps]
        self.goal_phrases = [(g, p) for g, ps in GOAL_ALIASES.items() for p in ps]
        self.sector_embeddings = self.model.encode(
            [p for _, p in self.sector_phrases], convert_to_tensor=True, normalize_embeddings=True
        )
        self.goal_embeddings = self.model.encode(
            [p for _, p in self.goal_phrases], convert_to_tensor=True, normalize_embeddings=True
        )

    @staticmethod
    def _exact_match(text: str, phrases) -> str | None:
        lowered = text.lower()
        matches = [(len(p), category) for category, p in phrases
                   if re.search(rf"\b{re.escape(p.lower())}\b", lowered)]
        if not matches:
            return None
        return max(matches)[1]

    def sector(self, text: str) -> str:
        exact = self._exact_match(text, self.sector_phrases)
        if exact:
            return exact
        emb = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(emb, self.sector_embeddings)[0]
        idx = int(scores.argmax().item())
        return self.sector_phrases[idx][0] if float(scores[idx]) > 0.45 else "others"

    def goal(self, text: str) -> str:
        exact = self._exact_match(text, self.goal_phrases)
        if exact:
            return exact
        emb = self.model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.cos_sim(emb, self.goal_embeddings)[0]
        idx = int(scores.argmax().item())
        return self.goal_phrases[idx][0] if float(scores[idx]) > 0.50 else "others"

    def risk(self, text: str) -> str:
        matches = [m.lower() for m in _RISK_RE.findall(text)]
        if "high" in matches:
            return "high"
        if "medium" in matches:
            return "medium"
        if "low" in matches:
            return "low"
        return "medium"

    def classify(self, text: str) -> Dict[str, str]:
        return {"sector": self.sector(text), "goal": self.goal(text), "risk": self.risk(text)}
