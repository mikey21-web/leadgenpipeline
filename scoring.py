"""
scoring.py — heuristic + optional LLM lead scorer.

Heuristic runs always (fast, free). LLM layer activates when api_key is provided.
LLM re-scores heuristic results using pitch context for deeper reasoning.
"""

import asyncio
import itertools
import json
import logging
import httpx
from models import ResearchResult, QualifiedLead

log = logging.getLogger(__name__)


class KeyPool:
    """Round-robin key rotation. On 429, skips the exhausted key and tries next."""

    def __init__(self, keys: list[str]):
        self._keys = [k.strip() for k in keys if k.strip()]
        self._cycle = itertools.cycle(self._keys)
        self._lock = asyncio.Lock()

    def __len__(self):
        return len(self._keys)

    async def next(self) -> str:
        async with self._lock:
            return next(self._cycle)

    async def call(self, base_url: str, model: str, messages: list, **kwargs) -> dict:
        """Try keys in rotation. On 429 try next key, up to len(pool) attempts."""
        last_err = None
        for _ in range(len(self._keys) or 1):
            key = await self.next()
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    resp = await client.post(
                        f"{base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={"model": model, "messages": messages, **kwargs},
                    )
                if resp.status_code == 429:
                    log.debug("Key rate-limited, rotating to next key")
                    last_err = f"429 on key …{key[-6:]}"
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    last_err = f"429 on key …{key[-6:]}"
                    continue
                raise
        raise RuntimeError(f"All keys rate-limited: {last_err}")

_WEIGHTS = {
    "fb_ads_active":        3,
    "has_lead_form":        2,
    "hiring_relevant_role": 2,
    "has_fb_pixel":         1,
    "no_booking_widget":    1,
    "no_chat_widget":       1,
}

_LLM_PROVIDERS = {
    "groq":     {"base_url": "https://api.groq.com/openai/v1",  "default_model": "llama-3.1-8b-instant"},
    "deepseek": {"base_url": "https://api.deepseek.com",         "default_model": "deepseek-v4-flash"},
}


def _heuristic(result: ResearchResult) -> tuple[int, str]:
    score = 0
    reasons = []

    if result.fb_ads_active:
        score += _WEIGHTS["fb_ads_active"]
        note = f"{result.fb_ads_count} active" if result.fb_ads_count else "active"
        reasons.append(f"FB ads ({note})")

    if result.has_lead_form:
        score += _WEIGHTS["has_lead_form"]
        reasons.append("lead form")

    if result.hiring_relevant_role:
        score += _WEIGHTS["hiring_relevant_role"]
        reasons.append(f"hiring '{result.relevant_job_title}'")

    if result.has_fb_pixel and not result.fb_ads_active:
        score += _WEIGHTS["has_fb_pixel"]
        reasons.append("FB pixel")

    if not result.has_booking_widget:
        score += _WEIGHTS["no_booking_widget"]
        reasons.append("no booking automation")

    if not result.has_chat_widget:
        score += _WEIGHTS["no_chat_widget"]
        reasons.append("no chat widget")

    reason = "; ".join(reasons) if reasons else "no strong signals"
    return min(score, 10), reason


async def _llm_score(
    result: ResearchResult,
    pitch_brief: str,
    pool: "KeyPool",
    provider: str,
    model: str,
) -> tuple[int, str]:
    """One cheap LLM call per contact via key pool. Returns (score, reason)."""
    cfg = _LLM_PROVIDERS.get(provider, _LLM_PROVIDERS["groq"])
    base_url = cfg["base_url"]
    model = model or cfg["default_model"]

    signals = {
        "company":              result.contact.company,
        "city":                 result.contact.city,
        "country":              result.contact.country,
        "has_lead_form":        result.has_lead_form,
        "fb_ads_active":        result.fb_ads_active,
        "fb_ads_count":         result.fb_ads_count,
        "has_fb_pixel":         result.has_fb_pixel,
        "has_booking_widget":   result.has_booking_widget,
        "has_chat_widget":      result.has_chat_widget,
        "has_crm_script":       result.has_crm_script,
        "hiring_relevant_role": result.hiring_relevant_role,
        "relevant_job_title":   result.relevant_job_title,
    }

    prompt = f"""You are a B2B lead qualification expert.

Product: {pitch_brief}

Prospect signals (scraped, factual):
{json.dumps(signals, indent=2)}

Score this prospect 0-10 for fit with the product.
Reply with JSON only: {{"score": <int>, "reason": "<one sentence>"}}"""

    data = await pool.call(
        base_url, model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=80,
    )
    content = data["choices"][0]["message"]["content"].strip()
    content = content[content.find("{"):content.rfind("}") + 1]
    parsed = json.loads(content)
    return int(parsed["score"]), parsed["reason"]


async def score_contact(
    result: ResearchResult,
    pitch_brief: str = "",
    pool: "KeyPool | None" = None,
    provider: str = "groq",
    model: str = "",
) -> tuple[int, str]:
    """Heuristic always runs. LLM replaces it when a key pool is provided."""
    if pool and len(pool) > 0 and pitch_brief:
        try:
            return await _llm_score(result, pitch_brief, pool, provider, model)
        except Exception as e:
            result.research_errors.append(f"llm_score: {e}")
            log.warning("LLM score failed, falling back to heuristic: %s", e)
    return _heuristic(result)


async def build_qualified_lead(
    result: ResearchResult,
    threshold: int = 5,
    pitch_brief: str = "",
    pool: "KeyPool | None" = None,
    provider: str = "groq",
    model: str = "",
) -> QualifiedLead | None:
    score, reason = await score_contact(result, pitch_brief, pool, provider, model)
    if score < threshold:
        return None
    return QualifiedLead(
        contact=result.contact,
        research=result,
        score=score,
        score_reason=reason,
    )
