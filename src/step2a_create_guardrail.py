"""Step 2a: create the Bedrock Guardrail, or update it in place if it already exists,
then publish a new version and save it to .guardrail.json.
Run:  uv run python src/step2a_create_guardrail.py
"""

import json
import pathlib

import boto3

from config import REGION

STATE = pathlib.Path(__file__).resolve().parent.parent / ".guardrail.json"
bd = boto3.client("bedrock", region_name=REGION)


def pii(type_, action):
    """Set input AND output behaviour explicitly (the legacy single 'action' did not mask inputs)."""
    return {
        "type": type_,
        "action": action,
        "inputAction": action,
        "outputAction": action,
        "inputEnabled": True,
        "outputEnabled": True,
    }


def topic(name, definition, examples):
    return {
        "name": name,
        "definition": definition,
        "examples": examples,
        "type": "DENY",
    }


cfg = dict(
    name="evergreen-support-guardrail",
    description="Customer-service chatbot: PII, denied topics, prompt attacks, grounding",
    topicPolicyConfig={
        "topicsConfig": [
            topic(
                "LegalAdvice",
                "Giving legal advice or opinions on suing, lawsuits or legal rights.",
                [
                    "Can I sue my employer?",
                    "Do I have a legal case against the insurer?",
                ],
            ),
            topic(
                "MedicalAdvice",
                "Diagnosing conditions or recommending medical treatment or medication.",
                ["What medication should I take for my pain?", "Do I have diabetes?"],
            ),
            topic(
                "InvestmentAdvice",
                "Recommending specific investments, securities or financial products to buy.",
                ["Which stocks should I buy?", "Should I put my savings in crypto?"],
            ),
        ]
    },
    contentPolicyConfig={
        "filtersConfig": [
            {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
            {"type": "VIOLENCE", "inputStrength": "MEDIUM", "outputStrength": "MEDIUM"},
            {
                "type": "MISCONDUCT",
                "inputStrength": "MEDIUM",
                "outputStrength": "MEDIUM",
            },
            {
                "type": "PROMPT_ATTACK",
                "inputStrength": "HIGH",
                "outputStrength": "NONE",
            },
        ]
    },
    sensitiveInformationPolicyConfig={
        "piiEntitiesConfig": [
            pii("NAME", "ANONYMIZE"),
            pii("EMAIL", "ANONYMIZE"),
            pii("PHONE", "ANONYMIZE"),
            pii("ADDRESS", "ANONYMIZE"),
            pii("CA_HEALTH_NUMBER", "ANONYMIZE"),
            # Never accept these in chat: block the whole message
            pii("CA_SOCIAL_INSURANCE_NUMBER", "BLOCK"),
            pii("CREDIT_DEBIT_CARD_NUMBER", "BLOCK"),
        ],
        "regexesConfig": [
            {
                "name": "PolicyNumber",
                "description": "Evergreen policy number EL-12345678",
                "pattern": r"EL-\d{8}",
                "action": "ANONYMIZE",
                "inputAction": "ANONYMIZE",
                "outputAction": "ANONYMIZE",
                "inputEnabled": True,
                "outputEnabled": True,
            },
        ],
    },
    contextualGroundingPolicyConfig={
        "filtersConfig": [
            {"type": "GROUNDING", "threshold": 0.75},
            {"type": "RELEVANCE", "threshold": 0.5},
        ]
    },
    blockedInputMessaging="I can't help with that request here. I can connect you with a team member.",
    blockedOutputsMessaging="I can't provide that answer. I can connect you with a team member.",
)

if STATE.exists():
    old = json.loads(STATE.read_text())
    resp = bd.update_guardrail(guardrailIdentifier=old["id"], **cfg)
    verb = "Updated"
else:
    resp = bd.create_guardrail(**cfg)
    verb = "Created"
gid = resp["guardrailId"]
ver = bd.create_guardrail_version(guardrailIdentifier=gid, description="auto")[
    "version"
]
arn = (
    resp.get("guardrailArn") or json.loads(STATE.read_text())["arn"]
    if STATE.exists()
    else resp["guardrailArn"]
)
STATE.write_text(json.dumps({"id": gid, "arn": arn, "version": ver}))
print(f"{verb} guardrail {gid}, published version {ver}")
