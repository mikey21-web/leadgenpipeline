"""
api.py — FastAPI backend for the lead gen pipeline.

Start: uvicorn api:app --reload --port 8000
Dashboard: http://localhost:8000
"""

import asyncio
import csv
import io
import re
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import Contact
from research import research_contact
from scoring import build_qualified_lead, KeyPool
from emails import generate_email

log = logging.getLogger(__name__)
app = FastAPI(title="Lead Gen Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store — good enough for one user / small runs
_jobs: dict[str, dict] = {}

# ── CSV helpers ──────────────────────────────────────────────────────────────

_COL_MAP = {
    "name":     ["name", "full name", "contact name", "first name", "person"],
    "company":  ["company", "company name", "business", "organization", "org", "account"],
    "website":  ["website", "url", "web", "site", "homepage", "domain"],
    "email":    ["email", "email address", "e-mail", "mail"],
    "phone":    ["phone", "phone number", "tel", "telephone", "mobile", "cell"],
    "city":     ["city", "town", "location"],
    "country":  ["country", "country code", "nation"],
    "industry": ["industry", "sector", "vertical", "niche", "type"],
}

def _map_columns(header: list[str]) -> dict[str, str]:
    lower = {h.lower().strip(): h for h in header}
    mapping = {}
    for field, aliases in _COL_MAP.items():
        for alias in aliases:
            if alias in lower:
                mapping[field] = lower[alias]
                break
    return mapping


def _parse_csv(content: bytes) -> list[Contact]:
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    col_map = _map_columns(reader.fieldnames or [])
    contacts = []
    for row in reader:
        c = Contact(raw=dict(row))
        for field, col in col_map.items():
            setattr(c, field, (row.get(col) or "").strip())
        if not c.country:
            c.country = "USA"
        contacts.append(c)
    return contacts


def _extract_pain_roles(pitch_text: str) -> list[str]:
    roles = re.findall(
        r"\b(receptionist|inside sales|bdc|isa|intake|appointment setter|"
        r"front desk|call center|coordinator|customer service)\b",
        pitch_text.lower(),
    )
    return list(dict.fromkeys(roles)) or [
        "receptionist", "inside sales", "bdc representative",
        "intake coordinator", "appointment setter",
    ]


# ── Background job runner ────────────────────────────────────────────────────

async def _run_job(
    job_id: str,
    contacts: list[Contact],
    pitch_brief: str,
    pain_roles: list[str],
    threshold: int,
    concurrency: int,
    api_keys: list[str],
    provider: str,
    model: str,
) -> None:
    job = _jobs[job_id]
    job["total"] = len(contacts)
    pool = KeyPool(api_keys) if api_keys else None
    sem = asyncio.Semaphore(concurrency)

    async def process(contact: Contact) -> None:
        async with sem:
            try:
                result = await research_contact(contact, pain_roles)
                lead = await build_qualified_lead(
                    result, threshold, pitch_brief, pool, provider, model
                )
                if lead:
                    generate_email(lead)
                    job["leads"].append({
                        "score":                lead.score,
                        "score_reason":         lead.score_reason,
                        "name":                 lead.contact.name,
                        "company":              lead.contact.company,
                        "email":                lead.contact.email,
                        "phone":                lead.contact.phone,
                        "website":              lead.contact.website,
                        "city":                 lead.contact.city,
                        "country":              lead.contact.country,
                        "industry":             lead.contact.industry,
                        "has_lead_form":        lead.research.has_lead_form,
                        "fb_ads_active":        lead.research.fb_ads_active,
                        "fb_ads_count":         lead.research.fb_ads_count,
                        "hiring_relevant_role": lead.research.hiring_relevant_role,
                        "relevant_job_title":   lead.research.relevant_job_title,
                        "has_booking_widget":   lead.research.has_booking_widget,
                        "has_chat_widget":      lead.research.has_chat_widget,
                        "email_subject":        lead.email_subject,
                        "email_body":           lead.email_body,
                        "errors":               lead.research.research_errors,
                    })
            except Exception as e:
                job["errors"].append(f"{contact.company}: {e}")
                log.exception("job %s contact %s failed", job_id, contact.company)
            finally:
                job["progress"] += 1

    await asyncio.gather(*[process(c) for c in contacts])
    job["status"] = "done"
    # Sort leads by score desc in place
    job["leads"].sort(key=lambda l: l["score"], reverse=True)


# ── API routes ───────────────────────────────────────────────────────────────

@app.post("/api/run")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    pitch_file: Optional[UploadFile] = File(None),
    pitch_text: str = Form(""),
    threshold: int = Form(5),
    concurrency: int = Form(8),
    api_keys: str = Form(""),   # newline or comma-separated keys
    provider: str = Form("groq"),
    model: str = Form("llama-3.1-8b-instant"),
):
    csv_bytes = await csv_file.read()
    contacts = _parse_csv(csv_bytes)
    if not contacts:
        return JSONResponse({"error": "CSV is empty or could not be parsed"}, status_code=400)

    pitch_brief = pitch_text
    if pitch_file:
        raw = await pitch_file.read()
        pitch_brief = raw.decode("utf-8", errors="replace") or pitch_text

    pain_roles = _extract_pain_roles(pitch_brief)

    # Parse key list — split on newlines or commas
    import re as _re
    key_list = [k.strip() for k in _re.split(r"[\n,]+", api_keys) if k.strip()]

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status":    "running",
        "progress":  0,
        "total":     len(contacts),
        "leads":     [],
        "errors":    [],
        "using_llm": bool(key_list),
        "key_count": len(key_list),
    }

    background_tasks.add_task(
        _run_job, job_id, contacts, pitch_brief, pain_roles,
        threshold, concurrency, key_list, provider, model,
    )
    return {"job_id": job_id, "total": len(contacts)}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "status":    job["status"],
        "progress":  job["progress"],
        "total":     job["total"],
        "qualified": len(job["leads"]),
        "using_llm": job.get("using_llm", False),
        "key_count": job.get("key_count", 0),
        "leads":     job["leads"],
        "errors":    job["errors"][-5:],  # last 5 only
    }


@app.get("/api/download/{job_id}")
def download_csv(job_id: str):
    job = _jobs.get(job_id)
    if not job or not job["leads"]:
        return JSONResponse({"error": "no leads"}, status_code=404)

    fields = [
        "score", "score_reason", "name", "company", "email", "phone",
        "website", "city", "country", "industry",
        "has_lead_form", "fb_ads_active", "fb_ads_count",
        "hiring_relevant_role", "relevant_job_title",
        "has_booking_widget", "has_chat_widget", "email_subject",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(job["leads"])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=leads_{job_id}.csv"},
    )


# ── Serve static frontend ────────────────────────────────────────────────────
Path("static").mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
