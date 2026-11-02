"""Step 8a: create a managed Bedrock Agent from code: IAM role, agent, the knowledge base from step 6 (VARIANT, default
hierarchical), an action group whose functions return control to this code, a prepared version and an alias.
Needs a knowledge base from step6a_kb_create.py. State goes to .agent.json.
Run:  uv run python src/step8a_agent_create.py [variant]
Then: AGENT_DRIVER=bedrock uv run python src/step5_eval_agent.py agent
Remove with step8z_agent_teardown.py.
"""

import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

from bedrock_agent_common import (
    FUNCTIONS,
    INSTRUCTION,
    MODEL,
    ROLE_NAME,
    role_policy,
    trust_policy,
)
from config import REGION
from rag_common import ROOT

STATE = ROOT / ".agent.json"
variant = sys.argv[1] if len(sys.argv) > 1 else "hierarchical"
kb_state = json.loads((ROOT / ".kb.json").read_text())
kb_id = kb_state["kbs"][variant]["kb_id"]
account = boto3.client("sts").get_caller_identity()["Account"]
iam = boto3.client("iam")
agent = boto3.client("bedrock-agent", region_name=REGION)
state = json.loads(STATE.read_text()) if STATE.exists() else {}


def save():
    STATE.write_text(json.dumps(state, indent=1))


def wait(fn, done, what, bad=("FAILED",), every=4, limit=300):
    t0 = time.time()
    while time.time() - t0 < limit:
        status = fn()
        if status in done:
            return
        if status in bad:
            raise SystemExit(f"{what} ended in {status}")
        time.sleep(every)
    raise SystemExit(f"timed out waiting for {what}")


# 1. role
try:
    iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(trust_policy(account, REGION)),
    )
    print("created role", ROLE_NAME)
    time.sleep(10)
except ClientError as e:
    if e.response["Error"]["Code"] != "EntityAlreadyExists":
        raise
iam.put_role_policy(
    RoleName=ROLE_NAME,
    PolicyName="agent-access",
    PolicyDocument=json.dumps(role_policy(account, REGION, kb_id)),
)
role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
state.update({"role": ROLE_NAME, "kb_id": kb_id})

# 2. agent (a new role can take a few seconds to be assumable)
if not state.get("agent_id"):
    for attempt in range(8):
        try:
            a = agent.create_agent(
                agentName="evergreen-support-agent",
                agentResourceRoleArn=role_arn,
                foundationModel=MODEL,
                instruction=INSTRUCTION,
                idleSessionTTLInSeconds=600,
            )["agent"]
            break
        except ClientError as e:
            if attempt == 7 or e.response["Error"]["Code"] != "ValidationException":
                raise
            print("  not ready, retrying ...", e.response["Error"]["Message"][:90])
            time.sleep(10)
    state["agent_id"] = a["agentId"]
    save()
agent_id = state["agent_id"]
wait(
    lambda: agent.get_agent(agentId=agent_id)["agent"]["agentStatus"],
    {"NOT_PREPARED", "PREPARED"},
    "agent creation",
)
print("agent", agent_id)

# 3. knowledge base and the functions that return control to us
agent.associate_agent_knowledge_base(
    agentId=agent_id,
    agentVersion="DRAFT",
    knowledgeBaseId=kb_id,
    description="Evergreen Life group benefits policy documents: plan rules, limits, claims and escalation policy.",
    knowledgeBaseState="ENABLED",
)
try:
    agent.create_agent_action_group(
        agentId=agent_id,
        agentVersion="DRAFT",
        actionGroupName="customer-tools",
        description="Customer-specific functions; control returns to the calling application.",
        actionGroupExecutor={"customControl": "RETURN_CONTROL"},
        functionSchema={"functions": FUNCTIONS},
    )
except ClientError as e:
    if e.response["Error"]["Code"] != "ConflictException":
        raise
    print("action group already exists")

# 4. prepare and alias
agent.prepare_agent(agentId=agent_id)
wait(
    lambda: agent.get_agent(agentId=agent_id)["agent"]["agentStatus"],
    {"PREPARED"},
    "agent preparation",
)
if not state.get("alias_id"):
    state["alias_id"] = agent.create_agent_alias(
        agentId=agent_id, agentAliasName="live"
    )["agentAlias"]["agentAliasId"]
    save()
wait(
    lambda: agent.get_agent_alias(agentId=agent_id, agentAliasId=state["alias_id"])[
        "agentAlias"
    ]["agentAliasStatus"],
    {"PREPARED"},
    "alias",
    bad=("FAILED", "DISSOCIATED"),
)
save()
print("ready:", state)
print(
    "next:  AGENT_DRIVER=bedrock RETRIEVER=kb VARIANT=hierarchical uv run python src/step5_eval_agent.py agent"
)
