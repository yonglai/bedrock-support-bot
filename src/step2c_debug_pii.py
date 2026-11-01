"""Debug why ANONYMIZE cases showed action=NONE. Prints raw ApplyGuardrail responses.
Run:  uv run python src/step2c_debug_pii.py
"""

import json
import pathlib

import boto3

from config import REGION

state = json.loads(
    (pathlib.Path(__file__).resolve().parent.parent / ".guardrail.json").read_text()
)
rt = boto3.client("bedrock-runtime", region_name=REGION)

CASES = [
    ("email only", "You can reach me at jane.doe@gmail.com for updates."),
    ("name only", "My name is John Smith and I work at a bank."),
    ("phone only", "Call me back at 416-555-0199 please."),
    ("address", "I live at 221 Queen Street West, Toronto, Ontario."),
    ("policy regex", "My policy number is EL-12345678."),
    (
        "all together",
        "Hi, I'm Maria Chen, email maria.chen@example.com, policy EL-12345678.",
    ),
]

for source in ("INPUT", "OUTPUT"):
    print(f"\n######## source={source}")
    for label, text in CASES:
        r = rt.apply_guardrail(
            guardrailIdentifier=state["id"],
            guardrailVersion=state["version"],
            source=source,
            content=[{"text": {"text": text}}],
        )
        out = r["outputs"][0]["text"] if r.get("outputs") else "(no output)"
        print(f"\n[{label}] action={r['action']}")
        print("  output:", out)
        sip = [
            a.get("sensitiveInformationPolicy")
            for a in r.get("assessments", [])
            if a.get("sensitiveInformationPolicy")
        ]
        print("  sensitiveInformationPolicy:", json.dumps(sip)[:400] if sip else "-")
