"""Step 3b: create an OpenSearch Serverless NextGen vector collection (scale-to-zero, capped at 2 OCU) + index.
Run:  uv run python src/step3b_opensearch_setup.py
Tear down when finished:  uv run python src/step3z_teardown.py
"""

import json
import time

import boto3
from botocore.exceptions import ClientError

from config import REGION
from oss import *

aoss = boto3.client("opensearchserverless", region_name=REGION)
sts, iam = boto3.client("sts", region_name=REGION), boto3.client("iam")


def ok_if_exists(label, fn, **kw):
    try:
        r = fn(**kw)
        print(f"created {label}")
        return r
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConflictException":
            print(f"{label} already exists")
            return None
        raise


# who gets data access: the role behind your current credentials
arn = sts.get_caller_identity()["Arn"]
principal = (
    iam.get_role(RoleName=arn.split("/")[1])["Role"]["Arn"]
    if ":assumed-role/" in arn
    else arn
)
print("data access principal:", principal)

res = f"collection/{COLLECTION}"
ok_if_exists(
    "encryption policy",
    aoss.create_security_policy,
    type="encryption",
    name=POLICIES["encryption"],
    policy=json.dumps(
        {
            "Rules": [{"ResourceType": "collection", "Resource": [res]}],
            "AWSOwnedKey": True,
        }
    ),
)
ok_if_exists(
    "network policy",
    aoss.create_security_policy,
    type="network",
    name=POLICIES["network"],
    policy=json.dumps(
        [
            {
                "Rules": [{"ResourceType": "collection", "Resource": [res]}],
                "AllowFromPublic": True,
            }
        ]
    ),
)
ok_if_exists(
    "data access policy",
    aoss.create_access_policy,
    type="data",
    name=POLICIES["data"],
    policy=json.dumps(
        [
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [res],
                        "Permission": ["aoss:*"],
                    },
                    {
                        "ResourceType": "index",
                        "Resource": [f"index/{COLLECTION}/*"],
                        "Permission": ["aoss:*"],
                    },
                ],
                "Principal": [principal],
            }
        ]
    ),
)

# NextGen group: scale to zero when idle, hard cap of 2 OCU each for indexing and search
ok_if_exists(
    "collection group",
    aoss.create_collection_group,
    name=GROUP,
    standbyReplicas="ENABLED",
    generation="NEXTGEN",
    capacityLimits={
        "minIndexingCapacityInOCU": 0.0,
        "minSearchCapacityInOCU": 0.0,
        "maxIndexingCapacityInOCU": 2.0,
        "maxSearchCapacityInOCU": 2.0,
    },
)
ok_if_exists(
    "collection",
    aoss.create_collection,
    name=COLLECTION,
    type="VECTORSEARCH",
    collectionGroupName=GROUP,
)

print("waiting for collection to become ACTIVE ...")
for _ in range(60):
    d = aoss.batch_get_collection(names=[COLLECTION])["collectionDetails"][0]
    if d["status"] == "ACTIVE":
        break
    if d["status"] == "FAILED":
        raise SystemExit(
            f"collection failed: {d.get('failureCode')} {d.get('failureMessage')}"
        )
    time.sleep(10)
else:
    raise SystemExit("timed out waiting for ACTIVE")

STATE.write_text(
    json.dumps(
        {
            "endpoint": d["collectionEndpoint"],
            "collection_id": d["id"],
            "group": GROUP,
            "collection": COLLECTION,
            "index": INDEX,
        }
    )
)
print("collection ACTIVE; group info:", d.get("collectionGroupName"))
create_index(client())
print("done. Next: uv run python src/step3d_opensearch_ingest_query.py")
