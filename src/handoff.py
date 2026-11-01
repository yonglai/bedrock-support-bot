"""Human handoff: records a ticket and says honestly whether a live agent or a callback is available."""

import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from rag_common import ROOT

LOG = (
    ROOT / ".cache" / "handoffs.jsonl"
)  # gitignored; never stores the customer's message text
TZ = ZoneInfo("America/Toronto")
REASONS = [
    "bereavement",
    "complaint",
    "appeal",
    "legal",
    "asked_for_person",
    "crisis",
    "cannot_answer",
    "guardrail",
    "other",
]


def availability(now=None):
    now = now or datetime.now(TZ)
    if now.weekday() < 5 and 8 <= now.hour < 20:
        return True, "A team member is available now."
    return False, (
        "Our team is available 8am to 8pm Eastern, Monday to Friday. "
        "I've requested a callback, and you'll hear back within 1 business day."
    )


def create_handoff(reason, summary, customer_id, now=None):
    live, line = availability(now)
    rec = {
        "ticket": uuid.uuid4().hex[:8],
        "ts": (now or datetime.now(TZ)).isoformat(timespec="seconds"),
        "customer_id": customer_id,
        "reason": reason,
        "summary": summary,
        "live": live,
    }
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return {**rec, "availability": line}
