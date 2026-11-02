"""Step 9a: deploy the agent to Amazon Bedrock AgentCore Runtime from code, no Docker: build a Linux zip (code + deps),
upload to S3, create the IAM role and the runtime, wait until READY. State goes to .agentcore.json.
Needs: a knowledge base (step6a), the guardrail (step2a), MODEL_ID in the environment.
Run:    uv run python src/step9a_agentcore_deploy.py
Try:    AGENT_REMOTE=agentcore RETRIEVER=kb VARIANT=hierarchical uv run python src/step5_eval_agent.py escalation
Remove: uv run python src/step9z_agentcore_teardown.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

from agentcore_common import (
    PACKAGES,
    ROLE_NAME,
    RUNTIME_NAME,
    SRC_FILES,
    STATE_FILES,
    bucket_name,
    role_policy,
    runtime_env,
    trust_policy,
)
from config import REGION
from rag_common import ROOT

STATE = ROOT / ".agentcore.json"
BUILD = ROOT / ".build"
PLATFORM = os.environ.get(
    "PLATFORM", "aarch64-manylinux2014"
)  # the runtime's CPU is unverified: x86_64-manylinux2014 if it fails
PYTHON = "3.12"
KEY = "agent/code.zip"
env = runtime_env(os.environ)
state = json.loads(STATE.read_text()) if STATE.exists() else {}
account = boto3.client("sts").get_caller_identity()["Account"]
bucket = bucket_name(account, REGION)
s3, iam = boto3.client("s3", region_name=REGION), boto3.client("iam")
ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
kb_id = json.loads((ROOT / ".kb.json").read_text())["kbs"][env["VARIANT"]]["kb_id"]
guardrail_arn = json.loads((ROOT / ".guardrail.json").read_text())["arn"]


def save():
    STATE.write_text(json.dumps(state, indent=1))


# 1. zip: dependencies for the runtime's platform, then our code and the two state files
shutil.rmtree(BUILD, ignore_errors=True)
pkg = BUILD / "pkg"
subprocess.run(
    ["uv", "pip", "install", "--quiet", "--target", str(pkg), "--python-platform", PLATFORM,
     "--python-version", PYTHON, "--only-binary", ":all:", *PACKAGES],
    check=True,
)  # fmt: skip
(pkg / "src").mkdir()
for name in SRC_FILES:
    shutil.copy(ROOT / "src" / name, pkg / "src" / name)
shutil.copy(ROOT / "src/agentcore_entry.py", pkg / "app.py")
shutil.copytree(ROOT / "data", pkg / "data")
for name in STATE_FILES:
    shutil.copy(ROOT / name, pkg / name)
zip_path = BUILD / "code.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            z.write(p, p.relative_to(pkg))
size = zip_path.stat().st_size / 1e6
print(f"built {zip_path} ({size:.1f} MB zipped)")

# 2. bucket and upload
try:
    if REGION == "us-east-1":
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
    print("created bucket", bucket)
except ClientError as e:
    if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou",):
        raise
state.update({"bucket": bucket, "key": KEY, "role": ROLE_NAME})
save()
s3.upload_file(str(zip_path), bucket, KEY)

# 3. role
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
    PolicyName="runtime-access",
    PolicyDocument=json.dumps(
        role_policy(account, REGION, env["MODEL_ID"], guardrail_arn, kb_id, bucket)
    ),
)
role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

# 4. runtime (a new role can take a few seconds to be assumable)
if not state.get("runtime_id"):
    for attempt in range(8):
        try:
            r = ctl.create_agent_runtime(
                agentRuntimeName=RUNTIME_NAME,
                agentRuntimeArtifact={
                    "codeConfiguration": {
                        "code": {"s3": {"bucket": bucket, "prefix": KEY}},
                        "runtime": "PYTHON_3_12",
                        "entryPoint": ["app.py"],
                    }
                },
                roleArn=role_arn,
                networkConfiguration={"networkMode": "PUBLIC"},
                protocolConfiguration={"serverProtocol": "HTTP"},
                # memory keeps billing while a session idles (CPU does not), so use the shortest allowed idle time
                lifecycleConfiguration={
                    "idleRuntimeSessionTimeout": 60,
                    "maxLifetime": 900,
                },
                environmentVariables=env,
            )
            break
        except ClientError as e:
            if attempt == 7 or e.response["Error"]["Code"] != "ValidationException":
                raise
            print("  not ready, retrying ...", e.response["Error"]["Message"][:100])
            time.sleep(10)
    state.update(
        {"runtime_id": r["agentRuntimeId"], "runtime_arn": r["agentRuntimeArn"]}
    )
    save()

# 5. wait
t0 = time.time()
while time.time() - t0 < 600:
    got = ctl.get_agent_runtime(agentRuntimeId=state["runtime_id"])
    status = got["status"]
    print("  status", status)
    if status == "READY":
        break
    if status.endswith("FAILED"):
        sys.exit(f"runtime {status}: {got.get('failureReason', 'no reason given')}")
    time.sleep(10)
else:
    sys.exit("timed out waiting for READY")
print("ready:", state["runtime_arn"])
