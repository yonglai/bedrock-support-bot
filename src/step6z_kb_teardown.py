"""Step 6z: delete everything step6a_kb_create.py made: knowledge bases, their vector indexes, the vector bucket, the IAM
role and the documents bucket. Reads .kb.json. Safe to run twice; anything it cannot delete is listed at the end.
Run:  uv run python src/step6z_kb_teardown.py
"""

import json
import time

import boto3
from botocore.exceptions import ClientError

from rag_common import ROOT

STATE = ROOT / ".kb.json"
if not STATE.exists():
    raise SystemExit(
        "no .kb.json: nothing to delete (check the console if you made things by hand)"
    )
state = json.loads(STATE.read_text())
region = state.get("region", "us-east-1")
agent = boto3.client("bedrock-agent", region_name=region)
s3v = boto3.client("s3vectors", region_name=region)
leftovers = []
GONE = (
    "ResourceNotFoundException",
    "NoSuchBucket",
    "NotFoundException",
    "NoSuchEntity",
)


def attempt(label, fn, **kw):
    try:
        fn(**kw)
        print("deleted", label)
        return True
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in GONE:
            print("already gone:", label)
            return True
        print(f"could not delete {label}: {code}")
        leftovers.append(label)
        return False


for variant, rec in state.get("kbs", {}).items():
    kb_id = rec.get("kb_id")
    if kb_id:
        try:
            for ds in agent.list_data_sources(knowledgeBaseId=kb_id)[
                "dataSourceSummaries"
            ]:
                attempt(
                    f"{variant}: data source {ds['dataSourceId']}",
                    agent.delete_data_source,
                    knowledgeBaseId=kb_id,
                    dataSourceId=ds["dataSourceId"],
                )
        except ClientError as e:
            print(f"{variant}: data source lookup failed:", e.response["Error"]["Code"])
        ok = attempt(
            f"{variant}: knowledge base {kb_id}",
            agent.delete_knowledge_base,
            knowledgeBaseId=kb_id,
        )
        for _ in range(
            24
        ):  # deletion is asynchronous; the index can only go once the KB is gone
            try:
                agent.get_knowledge_base(knowledgeBaseId=kb_id)
            except ClientError:
                break
            time.sleep(5)
    if rec.get("index_arn"):
        attempt(f"{variant}: vector index", s3v.delete_index, indexArn=rec["index_arn"])

if state.get("vector_bucket_arn"):
    attempt(
        "vector bucket",
        s3v.delete_vector_bucket,
        vectorBucketArn=state["vector_bucket_arn"],
    )

if state.get("role"):
    iam = boto3.client("iam")
    attempt(
        "role policy",
        iam.delete_role_policy,
        RoleName=state["role"],
        PolicyName="kb-access",
    )
    attempt("iam role", iam.delete_role, RoleName=state["role"])

if state.get("bucket"):
    s3 = boto3.resource("s3", region_name=region)
    try:
        s3.Bucket(state["bucket"]).objects.all().delete()
        s3.Bucket(state["bucket"]).delete()
        print("deleted bucket", state["bucket"])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            print("bucket already gone")
        else:
            print("could not delete bucket:", e.response["Error"]["Code"])
            leftovers.append(f"bucket {state['bucket']}")

if leftovers:
    print("\nStill to remove by hand:", *leftovers, sep="\n  - ")
else:
    print("\nNothing left. Check the console once to be sure.")
    STATE.unlink()
