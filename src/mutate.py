"""Inject errors into a correct answer, by construction, to test a verifier."""

import re

PCTS = ["50%", "80%", "90%", "100%"]
AMTS = ["$100", "$150", "$250", "$500", "$1,000", "$1,500", "$2,000", "$2,500"]


def mutations(answer):
    """(kind, mutated answer). The wrong value is never one that the answer already contains."""
    out = []
    m = re.search(r"\d+%", answer)
    new = m and next((p for p in PCTS if p != m.group(0) and p not in answer), None)
    if new:
        out.append(("percent", answer[: m.start()] + new + answer[m.end() :]))
    m = re.search(r"\$\d[\d,]*", answer)
    new = m and next((a for a in AMTS if a != m.group(0) and a not in answer), None)
    if new:
        out.append(("amount", answer[: m.start()] + new + answer[m.end() :]))
    has_s, has_e = "Standard" in answer, "Enhanced" in answer
    if has_s != has_e:
        a, b = ("Standard", "Enhanced") if has_s else ("Enhanced", "Standard")
        out.append(("plan", answer.replace(a, b)))
    return out
