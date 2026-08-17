"""
pipeline.py — Lead qualification + cold email pipeline

Usage:
  python pipeline.py --csv contacts.csv --pitch pitch.txt
  python pipeline.py --csv contacts.csv --pitch pitch.txt --threshold 6 --concurrency 10

Output:
  qualified_leads.csv     — scored leads above threshold
  email_drafts/           — one .txt file per qualified lead

CSV expected columns (flexible, auto-mapped):
  name, company, website, email, phone, city, country, industry
  (any subset works — unknown columns are preserved in raw)
"""

import asyncio
import csv
import logging
import os
import re
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from models import Contact, QualifiedLead
from research import research_contact
from scoring import build_qualified_lead
from emails import generate_email

log = logging.getLogger(__name__)
console = Console()

# ── CSV column auto-mapping ──────────────────────────────────────────────────

_COL_MAP = {
    "name":     ["name", "full name", "contact name", "first name", "person"],
    "company":  ["company", "company name", "business", "organization", "org", "account"],
    "website":  ["website", "url", "web", "site", "homepage", "domain"],
    "email":    ["email", "email address", "e-mail", "mail"],
    "phone":    ["phone", "phone number", "tel", "telephone", "mobile", "cell"],
    "city":     ["city", "town", "location", "city/state"],
    "country":  ["country", "country code", "nation", "region"],
    "industry": ["industry", "sector", "vertical", "niche", "type"],
}

def _map_columns(header: list[str]) -> dict[str, str]:
    """Returns {field: csv_column} for each recognized field."""
    lower = {h.lower().strip(): h for h in header}
    mapping = {}
    for field, aliases in _COL_MAP.items():
        for alias in aliases:
            if alias in lower:
                mapping[field] = lower[alias]
                break
    return mapping


def _load_csv(path: str) -> list[Contact]:
    contacts = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        col_map = _map_columns(reader.fieldnames or [])
        for row in reader:
            c = Contact(raw=dict(row))
            for field, col in col_map.items():
                setattr(c, field, (row.get(col) or "").strip())
            if not c.country:
                c.country = "USA"
            contacts.append(c)
    return contacts


# ── Output writers ───────────────────────────────────────────────────────────

def _write_csv(leads: list[QualifiedLead], out_path: str) -> None:
    if not leads:
        return
    fieldnames = [
        "score", "score_reason",
        "name", "company", "email", "phone", "website",
        "city", "country", "industry",
        "has_lead_form", "fb_ads_active", "fb_ads_count",
        "hiring_relevant_role", "relevant_job_title",
        "has_booking_widget", "has_chat_widget",
        "email_subject",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for lead in sorted(leads, key=lambda l: l.score, reverse=True):
            writer.writerow({
                "score":                 lead.score,
                "score_reason":          lead.score_reason,
                "name":                  lead.contact.name,
                "company":               lead.contact.company,
                "email":                 lead.contact.email,
                "phone":                 lead.contact.phone,
                "website":               lead.contact.website,
                "city":                  lead.contact.city,
                "country":               lead.contact.country,
                "industry":              lead.contact.industry,
                "has_lead_form":         lead.research.has_lead_form,
                "fb_ads_active":         lead.research.fb_ads_active,
                "fb_ads_count":          lead.research.fb_ads_count,
                "hiring_relevant_role":  lead.research.hiring_relevant_role,
                "relevant_job_title":    lead.research.relevant_job_title,
                "has_booking_widget":    lead.research.has_booking_widget,
                "has_chat_widget":       lead.research.has_chat_widget,
                "email_subject":         lead.email_subject,
            })


def _write_emails(leads: list[QualifiedLead], out_dir: str) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    for lead in leads:
        slug = re.sub(r"[^\w]", "_", lead.contact.company or lead.contact.name or "lead")[:40]
        path = Path(out_dir) / f"{lead.score:02d}_{slug}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"To: {lead.contact.email}\n")
            f.write(f"Subject: {lead.email_subject}\n")
            f.write(f"Score: {lead.score}/10 — {lead.score_reason}\n")
            f.write("-" * 60 + "\n\n")
            f.write(lead.email_body)
            f.write("\n")


# ── Main pipeline ────────────────────────────────────────────────────────────

async def run(
    contacts: list[Contact],
    pain_roles: list[str],
    threshold: int,
    concurrency: int,
) -> list[QualifiedLead]:
    sem = asyncio.Semaphore(concurrency)
    leads: list[QualifiedLead] = []

    async def process(contact: Contact, progress, task_id) -> None:
        async with sem:
            try:
                result = await research_contact(contact, pain_roles)
                lead = await build_qualified_lead(result, threshold)
                if lead:
                    generate_email(lead)
                    leads.append(lead)
                    console.print(
                        f"  [green]✓[/green] {contact.company or contact.name} "
                        f"[bold]{lead.score}/10[/bold] — {lead.score_reason}"
                    )
                else:
                    from scoring import score_contact
                    score, _ = score_contact(result)
                    console.print(
                        f"  [dim]✗ {contact.company or contact.name} {score}/10 — below threshold[/dim]"
                    )
            except Exception as e:
                console.print(f"  [red]! {contact.company}: {e}[/red]")
            finally:
                progress.advance(task_id)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Researching contacts…", total=len(contacts))
        await asyncio.gather(*[process(c, progress, task_id) for c in contacts])

    return leads


# ── CLI ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--csv",         "csv_path",   required=True,  help="Path to contacts CSV")
@click.option("--pitch",       "pitch_path", default=None,   help="Path to pitch deck / product description (txt/md)")
@click.option("--threshold",   default=5,    show_default=True, help="Min score to qualify (0–10)")
@click.option("--concurrency", default=8,    show_default=True, help="Parallel research workers")
@click.option("--out-csv",     default="qualified_leads.csv", help="Output CSV path")
@click.option("--out-emails",  default="email_drafts",        help="Output folder for email drafts")
def main(csv_path, pitch_path, threshold, concurrency, out_csv, out_emails):
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # Load pitch deck for custom pain roles (optional)
    pain_roles = None
    if pitch_path and Path(pitch_path).exists():
        pitch_text = Path(pitch_path).read_text(encoding="utf-8").lower()
        # Extract pain role keywords mentioned in pitch (simple heuristic)
        role_keywords = re.findall(
            r"\b(receptionist|inside sales|bdc|isa|intake|appointment setter|"
            r"front desk|call center|coordinator|customer service)\b",
            pitch_text,
        )
        if role_keywords:
            pain_roles = list(dict.fromkeys(role_keywords))  # dedupe, preserve order
            console.print(f"[cyan]Pitch pain roles:[/cyan] {', '.join(pain_roles)}")

    contacts = _load_csv(csv_path)
    console.print(f"[bold]Loaded {len(contacts)} contacts[/bold] from {csv_path}")
    console.print(f"Threshold: {threshold}/10 | Workers: {concurrency}\n")

    leads = asyncio.run(run(contacts, pain_roles, threshold, concurrency))

    _write_csv(leads, out_csv)
    _write_emails(leads, out_emails)

    # Summary table
    table = Table(title=f"\n{len(leads)} qualified leads")
    table.add_column("Score", style="bold green")
    table.add_column("Company")
    table.add_column("Email")
    table.add_column("Top Signal")
    for lead in sorted(leads, key=lambda l: l.score, reverse=True)[:20]:
        table.add_row(
            str(lead.score),
            lead.contact.company or lead.contact.name,
            lead.contact.email or "—",
            lead.score_reason.split(";")[0],
        )
    console.print(table)
    console.print(f"\n[bold green]→ {out_csv}[/bold green]")
    console.print(f"[bold green]→ {out_emails}/[/bold green]")


if __name__ == "__main__":
    main()
