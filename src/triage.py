"""Rule-based triage: deterministic checks that run BEFORE any model call or guardrail.

Sensitive situations get fixed, human-written replies. The rules are deliberately narrow:
anything they miss is left to the model's escalate_to_human tool.
"""

import re

# Order matters: the most serious category wins.
RULES = [
    (
        "crisis",
        (
            r"(end(ing)?|take|taking) (my|his|her) (own )?life|kill (myself|me)\b|suicid|want to die"
            r"|don'?t want to (be alive|live)|no (point|reason) (in|to) (living|go on)|better off dead"
            r"|don'?t see the point in anything"
        ),
    ),
    (
        "bereavement",
        r"passed away|passed on\b|\bdied\b|\bdeceased\b|death of|\bbereave|funeral|\bwidow",
    ),
    ("legal", r"\blawyer|\battorney|legal action|ombudsman"),
    (
        "appeal",
        r"\b(want|need|like|going|plan|trying) to appeal\b|\bappealing\b|\bfile an appeal\b|\bmy appeal\b",
    ),
    ("complaint", r"\bcomplain|\bunacceptable\b|\bfurious\b"),
    (
        "asked_for_person",
        (
            r"(speak|talk|connect|transfer)( me)? (to|with) (a |an )?(real |live |human )?"
            r"(person|human|agent|representative|someone)|\breal person\b|\bhuman agent\b"
        ),
    ),
]

REPLIES = {
    "crisis": (
        "I'm really sorry you're going through this. You don't have to face it alone. "
        "In Canada you can call or text 9-8-8 at any time to reach the Suicide Crisis Helpline, "
        "and if you are in immediate danger please call 911. I'm also connecting you with a team member."
    ),
    "bereavement": (
        "I'm very sorry for your loss. I'm connecting you with a team member who will help "
        "with the claim and handle this with care, so you won't need to repeat the details."
    ),
    "complaint": "I'm sorry about the trouble. I'm connecting you with a team member who can look into this.",
    "appeal": "I'm connecting you with a team member who handles appeals and can review the decision with you.",
    "legal": "I'm connecting you with a team member who can help with that.",
    "asked_for_person": "Of course. I'm connecting you with a team member now.",
}

_COMPILED = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in RULES]


def triage(text):
    """Return the escalation reason for the first matching rule, or None."""
    for name, rx in _COMPILED:
        if rx.search(text):
            return name
    return None
