"""
api.py — FastAPI backend for the lead gen pipeline.

Start: python -m uvicorn api:app --reload --port 8000
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

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from models import Contact
from research import research_contact
from scoring import build_qualified_lead, KeyPool
from emails import generate_email
from sender import send_email
import db

log = logging.getLogger(__name__)
app = FastAPI(title="Lead Gen Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (jobs re-run if server restarts — fine for local use)
_jobs: dict[str, dict] = {}

# Current scoring weights (retunable at runtime)
_current_weights: dict = {}


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

def _parse_keys(raw: str) -> list[str]:
    return [k.strip() for k in re.split(r"[\n,]+", raw) if k.strip()]

def _sender_name(from_email: str) -> str:
    """Extract display name from 'Name <email>' or return email prefix."""
    m = re.match(r"^([^<]+)<", from_email)
    return m.group(1).strip() if m else from_email.split("@")[0]


# ── Background job ───────────────────────────────────────────────────────────

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
    sender_name: str,
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
                    await generate_email(lead, pool, provider, model, sender_name)
                    row = {
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
                        "fb_ads_active":        lead.research.fb_ads_active,
                        "fb_ads_count":         lead.research.fb_ads_count,
                        "has_lead_form":        lead.research.has_lead_form,
                        "hiring_relevant_role": lead.research.hiring_relevant_role,
                        "relevant_job_title":   lead.research.relevant_job_title,
                        "has_booking_widget":   lead.research.has_booking_widget,
                        "has_chat_widget":      lead.research.has_chat_widget,
                        "email_subject":        lead.email_subject,
                        "email_body":           lead.email_body,
                        "errors":               lead.research.research_errors,
                        "outcome":              "pending",
                        "db_id":                None,
                    }
                    job["leads"].append(row)
            except Exception as e:
                job["errors"].append(f"{contact.company}: {e}")
                log.exception("job %s contact %s failed", job_id, contact.company)
            finally:
                job["progress"] += 1

    await asyncio.gather(*[process(c) for c in contacts])
    job["leads"].sort(key=lambda l: l["score"], reverse=True)
    job["status"] = "done"

    # Persist leads to SQLite and store db IDs back in job
    await db.save_leads(job_id, job["leads"])
    # Retrieve IDs
    import sqlite3 as _sq
    with _sq.connect(db.DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id FROM leads WHERE job_id = ? ORDER BY id", (job_id,)
        ).fetchall()
    for i, (row,) in enumerate(rows):
        if i < len(job["leads"]):
            job["leads"][i]["db_id"] = row


# ── Pipeline run ─────────────────────────────────────────────────────────────

@app.post("/api/run")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    csv_file: UploadFile = File(...),
    pitch_file: Optional[UploadFile] = File(None),
    pitch_text: str = Form(""),
    threshold: int = Form(5),
    concurrency: int = Form(8),
    api_keys: str = Form(""),
    provider: str = Form("groq"),
    model: str = Form("llama-3.1-8b-instant"),
    from_email: str = Form(""),
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
    key_list = _parse_keys(api_keys)
    s_name = _sender_name(from_email) if from_email else ""

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status":    "running",
        "progress":  0,
        "total":     len(contacts),
        "leads":     [],
        "errors":    [],
        "using_llm": bool(key_list),
        "key_count": len(key_list),
        "from_email": from_email,
        "resend_key": "",  # stored separately via /api/config
    }

    background_tasks.add_task(
        _run_job, job_id, contacts, pitch_brief, pain_roles,
        threshold, concurrency, key_list, provider, model, s_name,
    )
    return {"job_id": job_id, "total": len(contacts)}


# ── Status ───────────────────────────────────────────────────────────────────

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
        "errors":    job["errors"][-5:],
    }


# ── Send single email ────────────────────────────────────────────────────────

@app.post("/api/send/{job_id}/{lead_idx}")
async def send_one(
    job_id: str,
    lead_idx: int,
    resend_key: str = Body(..., embed=True),
    from_email: str = Body(..., embed=True),
):
    job = _jobs.get(job_id)
    if not job or lead_idx >= len(job["leads"]):
        return JSONResponse({"error": "not found"}, status_code=404)

    lead = job["leads"][lead_idx]
    result = await send_email(
        api_key=resend_key,
        from_email=from_email,
        to_email=lead["email"],
        subject=lead["email_subject"],
        body=lead["email_body"],
    )

    if not result.get("error"):
        lead["outcome"] = "sent"
        db_id = lead.get("db_id")
        if db_id:
            await db.mark_sent(db_id)

    return result


# ── Mark outcome ─────────────────────────────────────────────────────────────

@app.post("/api/outcome/{job_id}/{lead_idx}")
async def mark_outcome(
    job_id: str,
    lead_idx: int,
    outcome: str = Body(..., embed=True),
):
    job = _jobs.get(job_id)
    if not job or lead_idx >= len(job["leads"]):
        return JSONResponse({"error": "not found"}, status_code=404)

    valid = {"sent", "replied", "booked", "skipped"}
    if outcome not in valid:
        return JSONResponse({"error": f"outcome must be one of {valid}"}, status_code=400)

    lead = job["leads"][lead_idx]
    lead["outcome"] = outcome
    db_id = lead.get("db_id")
    if db_id:
        await db.mark_outcome(db_id, outcome)
    return {"ok": True}


# ── Retune weights ───────────────────────────────────────────────────────────

@app.post("/api/retune")
async def retune():
    weights = await db.compute_weights()
    if weights is None:
        return JSONResponse(
            {"error": "Not enough outcome data yet (need at least 10 sent leads with outcomes marked)"},
            status_code=400,
        )
    # Apply to scoring module at runtime
    import scoring
    scoring._WEIGHTS.update(weights)
    _current_weights.update(weights)
    return {"weights": weights, "message": "Scoring weights updated for this session"}


# ── Outcome stats ─────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def stats():
    s = await db.outcome_stats()
    return {**s, "current_weights": _current_weights or None}


# ── Download CSV ─────────────────────────────────────────────────────────────

@app.get("/api/download/{job_id}")
def download_csv(job_id: str):
    job = _jobs.get(job_id)
    if not job or not job["leads"]:
        return JSONResponse({"error": "no leads"}, status_code=404)

    fields = [
        "score", "score_reason", "name", "company", "email", "phone",
        "website", "city", "country", "industry",
        "fb_ads_active", "fb_ads_count", "has_lead_form",
        "hiring_relevant_role", "relevant_job_title",
        "has_booking_widget", "has_chat_widget",
        "email_subject", "outcome",
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


# ── Static frontend ───────────────────────────────────────────────────────────
Path("static").mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
