"""Claim-level verifier (experiment). A second model call splits an answer into factual claims and checks each one
against the tool results, including whether the figure belongs to this customer's plan, person and service.
Bedrock's contextual grounding score only asks "is this text supported somewhere in the sources?".
"""

import json

from pipeline import MODEL_ID, rt

SYSTEM = (
    "You are a strict fact checker for an insurance assistant. You get a QUESTION, SOURCES (tool results, including "
    "the customer's own record) and an ANSWER. Split the ANSWER into its factual claims: every number, percentage, "
    "limit, time period and every statement about who or what a rule applies to. For each claim decide whether the "
    "SOURCES state it AND that it applies to the customer or situation the answer applies it to. A figure that the "
    "sources give for another plan, for adults when the answer applies it to a child, or for another service is NOT "
    "supported. Arithmetic is supported if it follows from supported numbers. Reply with JSON only: "
    '{"claims": [{"claim": "...", "supported": true or false, "evidence": "short quote from the sources, or why not"}]}'
)


def parse(text):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the verifier reply")
    return json.loads(text[a : b + 1])


def verify(question, answer, sources):
    """Returns (ok, unsupported_claims). Raises ValueError if the reply cannot be parsed."""
    prompt = (
        f"QUESTION:\n{question}\n\nSOURCES:\n"
        + "\n".join(sources)
        + f"\n\nANSWER:\n{answer}"
    )
    resp = rt.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1200, "temperature": 0},
    )
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    bad = [c for c in parse(text)["claims"] if not c.get("supported", True)]
    return not bad, bad
