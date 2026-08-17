"""
sender.py — email sending via Resend.

Free tier: 100 emails/day, 3,000/month — enough for review-and-send workflow.
"""

import asyncio
import logging

log = logging.getLogger(__name__)


async def send_email(
    api_key: str,
    from_email: str,
    to_email: str,
    subject: str,
    body: str,
) -> dict:
    """
    Send one email via Resend. Returns {"id": "...", "error": None} or {"error": "..."}.
    from_email format: "Name <email@domain.com>" or just "email@domain.com"
    """
    if not api_key:
        return {"error": "No Resend API key configured"}
    if not to_email:
        return {"error": "No recipient email address"}

    try:
        import resend as _resend  # type: ignore
        _resend.api_key = api_key

        result = await asyncio.to_thread(
            _resend.Emails.send,
            {
                "from": from_email,
                "to": [to_email],
                "subject": subject,
                "text": body,
            }
        )
        log.info("Sent to %s — id: %s", to_email, result.get("id"))
        return {"id": result.get("id"), "error": None}

    except Exception as e:
        log.error("Send failed to %s: %s", to_email, e)
        return {"error": str(e)}
