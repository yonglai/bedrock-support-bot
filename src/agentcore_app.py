"""What runs inside the AgentCore Runtime: the same run_agent as locally, behind one JSON payload.
Payload: {"question": str, "customer_id": str, "history": [...]}. Returns the run_agent result dict.
The wiring to BedrockAgentCoreApp is in agentcore_entry.py, so this file imports (and tests) without that package."""

import json


def handle(payload):
    question, customer_id = payload.get("question"), payload.get("customer_id")
    if not question or not customer_id:
        raise ValueError("payload needs 'question' and 'customer_id'")
    from agent import run_agent  # imported late: loads the guardrail and KB state files

    result = run_agent(question, customer_id, payload.get("history") or [])
    return json.loads(json.dumps(result, default=str))
