"""
emails.py — template matching + LLM slot-fill.

Templates live in templates/ as .txt files with {{SLOT}} placeholders.
LLM fills slots using ONLY real researched facts — never invents.
Falls back to heuristic fill if no LLM pool provided.
"""

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from models import QualifiedLead

if TYPE_CHECKING:
    from scoring import KeyPool

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Load all templates once at startup
_TEMPLATES: dict[str, str] = {}
for _p in TEMPLATES_DIR.glob("*.txt"):
    _TEMPLATES[_p.stem] = _p.read_text(encoding="utf-8").strip()


# ── Template selection ───────────────────────────────────────────────────────

def _pick_template(lead: QualifiedLead) -> tuple[str, str]:
    """Returns (template_name, template_text)."""
    r = lead.research

    if r.hiring_relevant_role and "hiring-pain" in _TEMPLATES:
        return "hiring-pain", _TEMPLATES["hiring-pain"]
    if r.fb_ads_active and not r.has_booking_widget and "ads-no-followup" in _TEMPLATES:
        return "ads-no-followup", _TEMPLATES["ads-no-followup"]
    if r.has_lead_form and not r.fb_ads_active and "form-no-ads" in _TEMPLATES:
        return "form-no-ads", _TEMPLATES["form-no-ads"]
    if r.fb_ads_active and "high-volume" in _TEMPLATES:
        return "high-volume", _TEMPLATES["high-volume"]
    return "generic", _TEMPLATES.get("generic", "")


# ── Heuristic fill (no LLM) ──────────────────────────────────────────────────

def _heuristic_fill(lead: QualifiedLead, template: str, sender_name: str = "") -> tuple[str, str]:
    c = lead.contact
    r = lead.research

    first = c.name.split()[0] if c.name else "there"
    industry = c.industry or "business"
    ad_platform = "Facebook" if r.fb_ads_active else "online"
    volume_signal = (
        f"{r.fb_ads_count} active Facebook ads"
        if r.fb_ads_count else "solid online presence"
    )

    filled = (template
        .replace("{{FIRST_NAME}}",   first)
        .replace("{{COMPANY}}",      c.company or "your business")
        .replace("{{INDUSTRY}}",     industry)
        .replace("{{AD_PLATFORM}}",  ad_platform)
        .replace("{{JOB_TITLE}}",    r.relevant_job_title or "sales/intake role")
        .replace("{{VOLUME_SIGNAL}}", volume_signal)
        .replace("{{SENDER_NAME}}",  sender_name or "[Your name]")
    )

    lines = filled.splitlines()
    subject = lines[0].replace("Subject:", "").strip() if lines else ""
    body = "\n".join(lines[2:]).strip() if len(lines) > 2 else filled
    return subject, body


# ── LLM slot-fill ────────────────────────────────────────────────────────────

async def _llm_fill(
    lead: QualifiedLead,
    template: str,
    pool: "KeyPool",
    provider: str,
    model: str,
    sender_name: str,
) -> tuple[str, str]:
    from scoring import _LLM_PROVIDERS

    cfg = _LLM_PROVIDERS.get(provider, _LLM_PROVIDERS["groq"])
    base_url = cfg["base_url"]
    model = model or cfg["default_model"]

    c = lead.contact
    r = lead.research
    facts = {
        "first_name":           c.name.split()[0] if c.name else "",
        "full_name":            c.name,
        "company":              c.company,
        "industry":             c.industry or "business",
        "city":                 c.city,
        "country":              c.country,
        "fb_ads_active":        r.fb_ads_active,
        "fb_ads_count":         r.fb_ads_count,
        "has_lead_form":        r.has_lead_form,
        "hiring_role":          r.relevant_job_title,
        "no_booking_system":    not r.has_booking_widget,
        "no_chat_widget":       not r.has_chat_widget,
        "sender_name":          sender_name or "[Your name]",
    }

    prompt = f"""You are filling in a cold email template with real prospect data.

TEMPLATE (do not change structure or tone — only fill {{{{SLOTS}}}}):
{template}

REAL FACTS about this prospect (use ONLY these — never invent details):
{json.dumps(facts, indent=2)}

Rules:
- Fill every {{{{SLOT}}}} using the facts above
- If a slot can't be filled from facts, use a natural generic phrase
- Keep every sentence from the original template — do not add or remove sentences
- Reply with ONLY the filled email: Subject line first, blank line, then body"""

    data = await pool.call(
        base_url, model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=350,
    )
    content = data["choices"][0]["message"]["content"].strip()
    lines = content.splitlines()
    subject = lines[0].replace("Subject:", "").strip() if lines else ""
    body = "\n".join(lines[2:]).strip() if len(lines) > 2 else content
    return subject, body


# ── Public API ───────────────────────────────────────────────────────────────

async def generate_email(
    lead: QualifiedLead,
    pool: "KeyPool | None" = None,
    provider: str = "groq",
    model: str = "",
    sender_name: str = "",
) -> QualifiedLead:
    _, template = _pick_template(lead)
    if not template:
        lead.email_subject = f"AI voice agent for {lead.contact.company}"
        lead.email_body = "Template not found."
        return lead

    try:
        if pool and len(pool) > 0:
            subject, body = await _llm_fill(lead, template, pool, provider, model, sender_name)
        else:
            subject, body = _heuristic_fill(lead, template, sender_name)
    except Exception:
        subject, body = _heuristic_fill(lead, template, sender_name)

    lead.email_subject = subject
    lead.email_body = body
    return lead
