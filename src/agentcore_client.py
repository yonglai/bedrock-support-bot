"""Call the deployed AgentCore Runtime. Used by run_agent when AGENT_REMOTE=agentcore, so the eval scripts run unchanged."""

import json
import uuid

import boto3

from config import REGION
from rag_common import ROOT

STATE = ROOT / ".agentcore.json"


def invoke_remote(question, customer_id, history=None, client=None, state=None):
    state = state or json.loads(STATE.read_text())
    client = client or boto3.client("bedrock-agentcore", region_name=REGION)
    payload = {
        "question": question,
        "customer_id": customer_id,
        "history": history or [],
    }
    resp = client.invoke_agent_runtime(
        agentRuntimeArn=state["runtime_arn"],
        runtimeSessionId="eval-"
        + uuid.uuid4().hex,  # must be 33+ characters; one session per question
        payload=json.dumps(payload).encode(),
    )
    body = resp["response"].read()
    result = json.loads(body)
    if isinstance(result, dict) and "error" in result and "status" not in result:
        raise RuntimeError(f"runtime error: {result['error']}")
    return result
