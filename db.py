"""
db.py — SQLite persistence for leads and outcomes.

Jobs stay in-memory (local tool, fine to re-run).
Leads persist across restarts so outcomes can be tracked.
"""

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("leadgen.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _setup() -> None:
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS leads (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id      TEXT    NOT NULL,
                score       INTEGER,
                score_reason TEXT,
                name        TEXT, company TEXT, email TEXT,
                phone       TEXT, website TEXT, city TEXT,
                country     TEXT, industry TEXT,
                fb_ads_active      INTEGER DEFAULT 0,
                fb_ads_count       INTEGER DEFAULT 0,
                has_lead_form      INTEGER DEFAULT 0,
                hiring_relevant    INTEGER DEFAULT 0,
                relevant_job_title TEXT,
                has_booking_widget INTEGER DEFAULT 0,
                has_chat_widget    INTEGER DEFAULT 0,
                email_subject TEXT,
                email_body    TEXT,
                sent_at       TIMESTAMP,
                outcome       TEXT DEFAULT 'pending'
            );
            CREATE INDEX IF NOT EXISTS idx_leads_job ON leads(job_id);
            CREATE INDEX IF NOT EXISTS idx_leads_outcome ON leads(outcome);
        """)


_setup()


# ── Write ────────────────────────────────────────────────────────────────────

def _insert_leads_sync(leads: list[dict]) -> None:
    with _connect() as conn:
        conn.executemany("""
            INSERT INTO leads (
                job_id, score, score_reason,
                name, company, email, phone, website, city, country, industry,
                fb_ads_active, fb_ads_count, has_lead_form,
                hiring_relevant, relevant_job_title,
                has_booking_widget, has_chat_widget,
                email_subject, email_body
            ) VALUES (
                :job_id, :score, :score_reason,
                :name, :company, :email, :phone, :website, :city, :country, :industry,
                :fb_ads_active, :fb_ads_count, :has_lead_form,
                :hiring_relevant_role, :relevant_job_title,
                :has_booking_widget, :has_chat_widget,
                :email_subject, :email_body
            )
        """, leads)


async def save_leads(job_id: str, leads: list[dict]) -> None:
    rows = [{**lead, "job_id": job_id} for lead in leads]
    await asyncio.to_thread(_insert_leads_sync, rows)


# ── Outcome ──────────────────────────────────────────────────────────────────

def _mark_sent_sync(lead_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE leads SET sent_at = CURRENT_TIMESTAMP, outcome = 'sent' WHERE id = ?",
            (lead_id,)
        )


def _mark_outcome_sync(lead_id: int, outcome: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE leads SET outcome = ? WHERE id = ?", (outcome, lead_id))


async def mark_sent(lead_id: int) -> None:
    await asyncio.to_thread(_mark_sent_sync, lead_id)


async def mark_outcome(lead_id: int, outcome: str) -> None:
    await asyncio.to_thread(_mark_outcome_sync, lead_id, outcome)


# ── Retune: compute new weights from outcomes ────────────────────────────────

def _compute_weights_sync() -> Optional[dict]:
    """
    For each signal, calculate: booked / (sent + replied + booked).
    Returns new weight dict, or None if not enough data (< 10 outcomes).
    """
    with _connect() as conn:
        rows = conn.execute("""
            SELECT fb_ads_active, has_lead_form, hiring_relevant,
                   has_booking_widget, has_chat_widget, fb_ads_count,
                   outcome
            FROM leads
            WHERE outcome IN ('sent', 'replied', 'booked')
        """).fetchall()

    if len(rows) < 10:
        return None

    signals = {
        "fb_ads_active":        "fb_ads_active",
        "has_lead_form":        "has_lead_form",
        "hiring_relevant_role": "hiring_relevant",
        "no_booking_widget":    None,   # inverted: !has_booking_widget
        "no_chat_widget":       None,   # inverted: !has_chat_widget
        "has_fb_pixel":         "fb_ads_count",  # proxy: fb_ads_count > 0
    }

    weights = {}
    for signal_key, col in signals.items():
        total = booked = 0
        for row in rows:
            if col is None:
                # inverted signals
                val = not row["has_booking_widget"] if "booking" in signal_key else not row["has_chat_widget"]
            elif col == "fb_ads_count":
                val = (row[col] or 0) > 0
            else:
                val = bool(row[col])

            if val:
                total += 1
                if row["outcome"] == "booked":
                    booked += 1

        rate = (booked / total) if total > 5 else 0.5
        # Scale to 1–4 range, round to nearest 0.5
        raw = 1 + (rate * 3)
        weights[signal_key] = round(raw * 2) / 2  # nearest 0.5

    return weights


async def compute_weights() -> Optional[dict]:
    return await asyncio.to_thread(_compute_weights_sync)


# ── Stats ────────────────────────────────────────────────────────────────────

def _outcome_stats_sync() -> dict:
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'sent'    THEN 1 ELSE 0 END) as sent,
                SUM(CASE WHEN outcome = 'replied' THEN 1 ELSE 0 END) as replied,
                SUM(CASE WHEN outcome = 'booked'  THEN 1 ELSE 0 END) as booked,
                SUM(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) as skipped
            FROM leads
        """).fetchone()
    return dict(row) if row else {}


async def outcome_stats() -> dict:
    return await asyncio.to_thread(_outcome_stats_sync)
