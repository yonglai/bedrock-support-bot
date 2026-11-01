"""Delete everything step 3b created (collection, group, policies) so nothing keeps billing.
Run:  uv run python src/step3z_teardown.py
"""

import time

import boto3
from botocore.exceptions import ClientError

from config import REGION
from oss import COLLECTION, GROUP, POLICIES, STATE

aoss = boto3.client("opensearchserverless", region_name=REGION)


def attempt(label, fn, **kw):
    try:
        fn(**kw)
        print("deleted", label)
    except ClientError as e:
        print(f"{label}: {e.response['Error']['Code']} (skipping)")


for c in aoss.batch_get_collection(names=[COLLECTION])["collectionDetails"]:
    attempt("collection", aoss.delete_collection, id=c["id"])
print("waiting for the collection to disappear ...")
for _ in range(60):
    if not aoss.batch_get_collection(names=[COLLECTION])["collectionDetails"]:
        break
    time.sleep(10)
for g in aoss.batch_get_collection_group(names=[GROUP])["collectionGroupDetails"]:
    attempt("collection group", aoss.delete_collection_group, id=g["id"])
attempt(
    "data access policy", aoss.delete_access_policy, type="data", name=POLICIES["data"]
)
attempt(
    "network policy",
    aoss.delete_security_policy,
    type="network",
    name=POLICIES["network"],
)
attempt(
    "encryption policy",
    aoss.delete_security_policy,
    type="encryption",
    name=POLICIES["encryption"],
)
if STATE.exists():
    STATE.unlink()
print(
    "teardown finished. Also check the OpenSearch console and Billing for anything left."
)
