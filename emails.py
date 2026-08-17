"""
emails.py — template selection + slot filling stub.

Templates pulled from: https://github.com/gtm-skills/gtm
For now: selects the right template file name and fills known slots from research.
LLM slot-fill (for natural-sounding prose) drops in here later.
"""

from models import QualifiedLead

# Template selector — maps score signals to template intent
def _pick_template(lead: QualifiedLead) -> str:
    r = lead.research
    if r.fb_ads_active and r.has_lead_form and not r.has_booking_widget:
        return "ads-leads-no-followup"   # running ads, capturing leads, losing them
    if r.hiring_relevant_role:
        return "hiring-pain"             # paying humans to do what AI does
    if r.has_lead_form and not r.fb_ads_active:
        return "form-no-ads"             # has form but not maximizing inbound
    return "generic-voice-agent"


def _fill_slots(lead: QualifiedLead, template: str) -> tuple[str, str]:
    c = lead.contact
    r = lead.research

    pain_line = lead.score_reason.split(";")[0]  # strongest signal

    subjects = {
        "ads-leads-no-followup": f"Quick question about your {c.company} leads",
        "hiring-pain":           f"Re: {r.relevant_job_title} role at {c.company}",
        "form-no-ads":           f"{c.company} — are your leads being followed up?",
        "generic-voice-agent":   f"AI voice agent for {c.company}",
    }
    subject = subjects.get(template, f"Quick question for {c.name or c.company}")

    # Slot-filled body (no LLM — factual slots only)
    ads_line = (
        f"Noticed {c.company} is running Facebook ads"
        if r.fb_ads_active else
        f"Noticed {c.company} has a lead capture form on the website"
    )
    hiring_line = (
        f"Also saw you're hiring a '{r.relevant_job_title}' — "
        "that's usually a sign leads are coming in faster than the team can handle."
        if r.hiring_relevant_role else ""
    )
    automation_line = (
        "" if r.has_booking_widget
        else "Most leads expect a response in under 5 minutes — "
             "without automation, the majority go cold before anyone calls back."
    )

    body = f"""Hey {c.name.split()[0] if c.name else "there"},

{ads_line} — but it looks like there's no automated follow-up in place.

{hiring_line}

{automation_line}

We built an AI voice agent that calls new leads in under 5 seconds, qualifies them, and books appointments — 24/7, no human needed.

Worth a quick 15-min call to see if it fits?

[Your name]
[Your company]
[Calendar link]""".strip()

    # Remove blank lines from optional slots
    body = "\n".join(line for line in body.splitlines() if line.strip() or line == "")
    return subject, body


def generate_email(lead: QualifiedLead) -> QualifiedLead:
    template = _pick_template(lead)
    subject, body = _fill_slots(lead, template)
    lead.email_subject = subject
    lead.email_body = body
    return lead
