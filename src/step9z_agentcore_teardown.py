"""Step 9z: delete the AgentCore runtime, the code bucket and the IAM role. Safe to run twice.
Run:  uv run python src/step9z_agentcore_teardown.py
"""

import json
import time

import boto3
from botocore.exceptions import ClientError

from config import REGION
from rag_common import ROOT

STATE = ROOT / ".agentcore.json"
if not STATE.exists():
    raise SystemExit("no .agentcore.json: nothing to delete")
state = json.loads(STATE.read_text())
ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
s3, iam = boto3.client("s3", region_name=REGION), boto3.client("iam")
GONE = ("ResourceNotFoundException", "NoSuchEntity", "NoSuchBucket", "NoSuchKey")
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


rid = state.get("runtime_id")
if rid:
    for ep in ctl.list_agent_runtime_endpoints(agentRuntimeId=rid).get(
        "runtimeEndpoints", []
    ):
        if ep["name"] != "DEFAULT":  # DEFAULT goes with the runtime
            attempt(
                f"endpoint {ep['name']}",
                ctl.delete_agent_runtime_endpoint,
                agentRuntimeId=rid,
                endpointName=ep["name"],
            )
    attempt(f"runtime {rid}", ctl.delete_agent_runtime, agentRuntimeId=rid)
    for _ in range(36):  # deletion is asynchronous
        try:
            ctl.get_agent_runtime(agentRuntimeId=rid)
        except ClientError:
            break
        time.sleep(5)
if state.get("bucket"):
    attempt(
        "code zip",
        s3.delete_object,
        Bucket=state["bucket"],
        Key=state.get("key", "agent/code.zip"),
    )
    attempt("bucket", s3.delete_bucket, Bucket=state["bucket"])
if state.get("role"):
    attempt(
        "role policy",
        iam.delete_role_policy,
        RoleName=state["role"],
        PolicyName="runtime-access",
    )
    attempt("iam role", iam.delete_role, RoleName=state["role"])

if left:
    print("\nStill to remove by hand:", *left, sep="\n  - ")
else:
    print("\nNothing left.")
    STATE.unlink()
