"""Step 8z: delete the managed Bedrock Agent (aliases, agent) and its IAM role. Safe to run twice.
The knowledge base is removed by step6z_kb_teardown.py. Run:  uv run python src/step8z_agent_teardown.py
"""

import json
import time

import boto3
from botocore.exceptions import ClientError

from config import REGION
from rag_common import ROOT

STATE = ROOT / ".agent.json"
if not STATE.exists():
    raise SystemExit("no .agent.json: nothing to delete")
state = json.loads(STATE.read_text())
agent = boto3.client("bedrock-agent", region_name=REGION)
iam = boto3.client("iam")
GONE = ("ResourceNotFoundException", "NoSuchEntity")
left = []


def attempt(label, fn, **kw):
    try:
        fn(**kw)
        print("deleted", label)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in GONE:
            print("already gone:", label)
        else:
            print(f"could not delete {label}: {code}")
            left.append(label)


agent_id = state.get("agent_id")
if agent_id:
    try:
        for al in agent.list_agent_aliases(agentId=agent_id)["agentAliasSummaries"]:
            attempt(
                f"alias {al['agentAliasId']}",
                agent.delete_agent_alias,
                agentId=agent_id,
                agentAliasId=al["agentAliasId"],
            )
    except ClientError as e:
        print("alias lookup failed:", e.response["Error"]["Code"])
    attempt(
        f"agent {agent_id}",
        agent.delete_agent,
        agentId=agent_id,
        skipResourceInUseCheck=True,
    )
    for _ in range(24):  # deletion is asynchronous
        try:
            agent.get_agent(agentId=agent_id)
        except ClientError:
            break
        time.sleep(5)
if state.get("role"):
    attempt(
        "role policy",
        iam.delete_role_policy,
        RoleName=state["role"],
        PolicyName="agent-access",
    )
    attempt("iam role", iam.delete_role, RoleName=state["role"])

if left:
    print("\nStill to remove by hand:", *left, sep="\n  - ")
else:
    print("\nNothing left.")
    STATE.unlink()
