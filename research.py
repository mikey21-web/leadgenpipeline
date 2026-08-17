"""
research.py — parallel research engine for one contact.

Runs 3 checks concurrently:
  1. Website scrape  → lead form, pixels, booking widget, tech stack
  2. Facebook Ad Library → active ads for this company
  3. JobSpy → hiring roles AI voice could replace

All checks are deterministic (no LLM). Errors are non-fatal and logged per contact.
"""

import asyncio
import re
import logging
from typing import List

import httpx
from bs4 import BeautifulSoup

from models import Contact, ResearchResult

log = logging.getLogger(__name__)

# ── Signal patterns (checked against raw HTML) ──────────────────────────────

_FB_PIXEL = ["fbq(", "connect.facebook.net/en_US/fbevents.js", "facebook-jssdk"]
_GA = ["gtag(", "google-analytics.com/analytics.js", "googletagmanager.com/gtag"]
_BOOKING = [
    "calendly.com", "acuityscheduling.com", "opentable.com",
    "resy.com", "booksy.com", "setmore.com", "square.site",
]
_CHAT = ["intercom", "drift.com", "tidio", "crisp.chat", "livechat.com", "tawk.to"]
_CRM = ["hs-scripts.com", "hubspot", "salesforce", "pipedrive", "zoho"]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── 1. Website scrape ────────────────────────────────────────────────────────

async def _scrape_website(contact: Contact, result: ResearchResult) -> None:
    if not contact.website:
        return
    url = contact.website
    if not url.startswith("http"):
        url = f"https://{url}"
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
        html_lower = resp.text.lower()
        soup = BeautifulSoup(resp.text, "html.parser")

        result.has_fb_pixel = any(p in html_lower for p in _FB_PIXEL)
        result.has_ga_tag = any(p in html_lower for p in _GA)
        result.has_booking_widget = any(p in html_lower for p in _BOOKING)
        result.has_chat_widget = any(p in html_lower for p in _CHAT)
        result.has_crm_script = any(p in html_lower for p in _CRM)
        result.phone_on_site = bool(
            re.search(r"[\+\(]?[0-9][0-9 \-\(\)]{7,}[0-9]", resp.text)
        )

        # Lead form = form containing email or tel input
        for form in soup.find_all("form"):
            input_types = [i.get("type", "").lower() for i in form.find_all("input")]
            if any(t in input_types for t in ["email", "tel"]) or \
               any("email" in (i.get("name", "") + i.get("placeholder", "")).lower()
                   for i in form.find_all("input")):
                result.has_lead_form = True
                break

    except Exception as e:
        result.research_errors.append(f"website: {e}")
        log.debug("website error %s: %s", contact.company, e)


# ── 2. Facebook Ad Library ───────────────────────────────────────────────────

async def _check_fb_ads(contact: Contact, result: ResearchResult) -> None:
    """
    Queries the public FB Ad Library search endpoint.
    Falls back to FB pixel detection (already set by website scrape).
    """
    if not contact.company:
        return
    query = contact.company.replace(" ", "+")
    country = (contact.country or "US")[:2].upper()
    url = (
        f"https://www.facebook.com/ads/library/?"
        f"active_status=active&ad_type=all&country={country}"
        f"&q={query}&search_type=keyword_unordered&media_type=all"
    )
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url, headers=_HEADERS)
        # FB Ad Library embeds result counts in the page JSON
        match = re.search(r'"totalCount"\s*:\s*(\d+)', resp.text)
        if match:
            count = int(match.group(1))
            result.fb_ads_active = count > 0
            result.fb_ads_count = count
        elif result.has_fb_pixel:
            # Pixel present = they run/have run ads (reliable fallback)
            result.fb_ads_active = True
    except Exception as e:
        # Non-fatal: pixel detection on website already partially covers this
        result.research_errors.append(f"fb_ads: {e}")
        log.debug("fb_ads error %s: %s", contact.company, e)
        if result.has_fb_pixel:
            result.fb_ads_active = True


# ── 3. Job scraping ──────────────────────────────────────────────────────────

# Roles that AI voice agents replace — derived at runtime from pitch keywords,
# but these defaults cover the common "inbound lead handling" pain.
_DEFAULT_PAIN_ROLES = [
    "receptionist", "inside sales", "bdc representative", "isa",
    "intake coordinator", "front desk", "appointment setter",
    "customer service representative", "call center",
]

async def _check_jobs(
    contact: Contact,
    result: ResearchResult,
    pain_roles: List[str],
) -> None:
    if not contact.company:
        return
    try:
        from jobspy import scrape_jobs  # type: ignore

        location = " ".join(filter(None, [contact.city, contact.country])) or "United States"
        country_map = {
            "USA": "USA", "US": "USA", "UK": "UK", "GB": "UK",
            "AU": "Australia", "CA": "Canada", "IN": "India",
        }
        country_indeed = country_map.get((contact.country or "USA").upper(), "USA")

        jobs = await asyncio.to_thread(
            scrape_jobs,
            site_name=["indeed", "linkedin"],
            search_term=f"{contact.company} {pain_roles[0]}",
            location=location,
            results_wanted=10,
            country_indeed=country_indeed,
            hours_old=720,  # last 30 days
        )
        if jobs is not None and len(jobs) > 0:
            mask = jobs["company"].str.contains(
                contact.company, case=False, na=False, regex=False
            )
            company_jobs = jobs[mask]
            if len(company_jobs) > 0:
                title = company_jobs.iloc[0]["title"]
                # Confirm title matches a pain role
                if any(r in title.lower() for r in pain_roles):
                    result.hiring_relevant_role = True
                    result.relevant_job_title = title
    except Exception as e:
        result.research_errors.append(f"jobs: {e}")
        log.debug("jobs error %s: %s", contact.company, e)


# ── Public API ───────────────────────────────────────────────────────────────

async def research_contact(
    contact: Contact,
    pain_roles: List[str] | None = None,
) -> ResearchResult:
    """
    Run all research checks concurrently for a single contact.
    Returns ResearchResult regardless of individual check failures.
    """
    if pain_roles is None:
        pain_roles = _DEFAULT_PAIN_ROLES

    result = ResearchResult(contact=contact)

    await asyncio.gather(
        _scrape_website(contact, result),
        _check_fb_ads(contact, result),
        _check_jobs(contact, result, pain_roles),
        return_exceptions=True,
    )

    return result
