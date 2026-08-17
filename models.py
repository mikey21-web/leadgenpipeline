from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Contact:
    name: str = ""
    company: str = ""
    website: str = ""
    email: str = ""
    phone: str = ""
    city: str = ""
    country: str = "USA"
    industry: str = ""  # auto-detected if blank
    raw: dict = field(default_factory=dict)  # original CSV row


@dataclass
class ResearchResult:
    contact: Contact = field(default_factory=Contact)

    # Website signals
    has_lead_form: bool = False
    has_fb_pixel: bool = False
    has_ga_tag: bool = False
    has_booking_widget: bool = False
    has_chat_widget: bool = False
    has_crm_script: bool = False
    phone_on_site: bool = False

    # Ads signals
    fb_ads_active: bool = False
    fb_ads_count: int = 0

    # Jobs signals — hiring humans for roles AI voice replaces
    hiring_relevant_role: bool = False
    relevant_job_title: str = ""

    # Errors (non-fatal, logged per contact)
    research_errors: List[str] = field(default_factory=list)

    def signal_count(self) -> int:
        """Number of positive pain signals found."""
        signals = [
            self.has_lead_form,
            self.has_fb_pixel,
            self.fb_ads_active,
            self.hiring_relevant_role,
            not self.has_booking_widget,   # no automation = pain
            not self.has_chat_widget,      # no automation = pain
        ]
        return sum(signals)


@dataclass
class QualifiedLead:
    contact: Contact = field(default_factory=Contact)
    research: ResearchResult = field(default_factory=ResearchResult)
    score: int = 0          # 0–10
    score_reason: str = ""
    email_subject: str = ""
    email_body: str = ""
