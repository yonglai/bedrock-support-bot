"""Step 6a: create everything for the Knowledge Base experiment from code: S3 bucket + policy files, S3 Vectors bucket
and index, IAM role, the Knowledge Base, its data source, and the first ingestion. State goes to .kb.json.
Run:  uv run python src/step6a_kb_create.py default            (one variant)
      uv run python src/step6a_kb_create.py default none hierarchical
Variants: see kb_common.VARIANTS. Then:  VARIANT=default QFILE=questions_combined.jsonl uv run python src/step6_kb_retrieval_eval.py
Remove everything with step6z_kb_teardown.py.
"""

import json
import sys
import time

import boto3
from botocore.exceptions import ClientError

from config import REGION
from kb_common import DIMS, TITAN, ingestion_config, names, role_policy, trust_policy
from rag_common import ROOT

STATE = ROOT / ".kb.json"
variants = sys.argv[1:] or ["default"]
for v in variants:
    ingestion_config(v)  # fail early on a typo

account = boto3.client("sts").get_caller_identity()["Account"]
n0 = names(account, REGION, variants[0])
s3 = boto3.client("s3", region_name=REGION)
s3v = boto3.client("s3vectors", region_name=REGION)
iam = boto3.client("iam")
agent = boto3.client("bedrock-agent", region_name=REGION)
state = json.loads(STATE.read_text()) if STATE.exists() else {"kbs": {}}
state.update(
    {
        "region": REGION,
        "bucket": n0["bucket"],
        "vector_bucket": n0["vector_bucket"],
        "role": n0["role"],
    }
)


def save():
    STATE.write_text(json.dumps(state, indent=1))


def exists_ok(e, *codes):
    return e.response["Error"]["Code"] in codes


# 1. documents bucket
args = {"Bucket": n0["bucket"]}
if REGION != "us-east-1":
    args["CreateBucketConfiguration"] = {"LocationConstraint": REGION}
try:
    s3.create_bucket(**args)
    print("created bucket", n0["bucket"])
except ClientError as e:
    if not exists_ok(e, "BucketAlreadyOwnedByYou"):
        raise
s3.put_public_access_block(
    Bucket=n0["bucket"],
    PublicAccessBlockConfiguration=dict.fromkeys(
        [
            "BlockPublicAcls",
            "IgnorePublicAcls",
            "BlockPublicPolicy",
            "RestrictPublicBuckets",
        ],
        True,
    ),
)
files = sorted((ROOT / "data" / "policies").glob("*.md"))
for p in files:
    s3.upload_file(str(p), n0["bucket"], f"policies/{p.name}")
print(f"uploaded {len(files)} policy files")
save()

# 2. vector bucket
try:
    s3v.create_vector_bucket(vectorBucketName=n0["vector_bucket"])
    print("created vector bucket", n0["vector_bucket"])
except ClientError as e:
    if not exists_ok(e, "ConflictException"):
        raise
vector_bucket_arn = s3v.get_vector_bucket(vectorBucketName=n0["vector_bucket"])[
    "vectorBucket"
]["vectorBucketArn"]
state["vector_bucket_arn"] = vector_bucket_arn
save()

# 3. IAM role (shared by every variant)
try:
    iam.create_role(
        RoleName=n0["role"],
        AssumeRolePolicyDocument=json.dumps(trust_policy(account, REGION)),
    )
    print("created role", n0["role"])
    time.sleep(10)
except ClientError as e:
    if not exists_ok(e, "EntityAlreadyExists"):
        raise
iam.put_role_policy(
    RoleName=n0["role"],
    PolicyName="kb-access",
    PolicyDocument=json.dumps(
        role_policy(account, REGION, n0["bucket"], vector_bucket_arn)
    ),
)
role_arn = iam.get_role(RoleName=n0["role"])["Role"]["Arn"]


def kb_status(kb_id):
    return agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"]


def job_info(kb_id, ds_id, job_id):
    return agent.get_ingestion_job(
        knowledgeBaseId=kb_id, dataSourceId=ds_id, ingestionJobId=job_id
    )["ingestionJob"]


def wait(fn, args, done, bad, what, every=5, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        status = fn(*args)
        if status in done:
            return status
        if status in bad:
            raise SystemExit(f"{what} ended in {status}")
        time.sleep(every)
    raise SystemExit(f"timed out waiting for {what}")


for variant in variants:
    n = names(account, REGION, variant)
    rec = state["kbs"].get(variant, {})
    print(f"\n== variant {variant}")

    # 4. vector index (embeddings must match the model's dimensions; chunk text is stored as non-filterable metadata)
    try:
        index_arn = s3v.create_index(
            vectorBucketName=n["vector_bucket"],
            indexName=n["index"],
            dataType="float32",
            dimension=DIMS,
            distanceMetric="cosine",
            metadataConfiguration={
                "nonFilterableMetadataKeys": [
                    "AMAZON_BEDROCK_TEXT",
                    "AMAZON_BEDROCK_METADATA",
                ]
            },
        )["indexArn"]
        print("created index", n["index"])
    except ClientError as e:
        if not exists_ok(e, "ConflictException"):
            raise
        index_arn = s3v.get_index(
            vectorBucketName=n["vector_bucket"], indexName=n["index"]
        )["index"]["indexArn"]
    rec["index_arn"] = index_arn
    state["kbs"][variant] = rec
    save()

    # 5. knowledge base (a new role can take a few seconds to be assumable)
    if not rec.get("kb_id"):
        for attempt in range(8):
            try:
                kb = agent.create_knowledge_base(
                    name=n["kb"],
                    roleArn=role_arn,
                    knowledgeBaseConfiguration={
                        "type": "VECTOR",
                        "vectorKnowledgeBaseConfiguration": {
                            "embeddingModelArn": f"arn:aws:bedrock:{REGION}::foundation-model/{TITAN}",
                            "embeddingModelConfiguration": {
                                "bedrockEmbeddingModelConfiguration": {
                                    "dimensions": DIMS,
                                    "embeddingDataType": "FLOAT32",
                                }
                            },
                        },
                    },
                    storageConfiguration={
                        "type": "S3_VECTORS",
                        "s3VectorsConfiguration": {"indexArn": index_arn},
                    },
                )["knowledgeBase"]
                break
            except ClientError as e:
                if attempt == 7 or not exists_ok(e, "ValidationException"):
                    raise
                print(
                    "  role not ready yet, retrying ...",
                    e.response["Error"]["Message"][:80],
                )
                time.sleep(10)
        rec["kb_id"] = kb["knowledgeBaseId"]
        save()
    kb_id = rec["kb_id"]
    wait(kb_status, (kb_id,), {"ACTIVE"}, {"FAILED"}, "knowledge base")
    print("knowledge base active:", kb_id)

    # 6. data source + ingestion
    if not rec.get("data_source_id"):
        kw = {}
        if ingestion_config(variant):
            kw["vectorIngestionConfiguration"] = ingestion_config(variant)
        ds = agent.create_data_source(
            knowledgeBaseId=kb_id,
            name="policies",
            dataDeletionPolicy="DELETE",
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{n['bucket']}",
                    "inclusionPrefixes": ["policies/"],
                },
            },
            **kw,
        )["dataSource"]
        rec["data_source_id"] = ds["dataSourceId"]
        save()
    job = agent.start_ingestion_job(
        knowledgeBaseId=kb_id, dataSourceId=rec["data_source_id"]
    )["ingestionJob"]
    args = (kb_id, rec["data_source_id"], job["ingestionJobId"])
    wait(
        lambda *a: job_info(*a)["status"],
        args,
        {"COMPLETE"},
        {"FAILED", "STOPPED"},
        "ingestion",
    )
    done = job_info(*args)
    print("ingestion:", done.get("statistics"), done.get("failureReasons", ""))
    print(
        f"next:  VARIANT={variant} QFILE=questions_combined.jsonl uv run python src/step6_kb_retrieval_eval.py"
    )

print("\nsaved", STATE)
