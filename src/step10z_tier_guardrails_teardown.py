"""Step 10z: delete the two grounding-tier guardrails. Safe to run twice.
Run:  uv run python src/step10z_tier_guardrails_teardown.py
"""

import json

import boto3
from botocore.exceptions import ClientError

from config import REGION
from rag_common import ROOT

STATE = ROOT / ".guardrail_tiers.json"
if not STATE.exists():
    raise SystemExit("no .guardrail_tiers.json: nothing to delete")
state = json.loads(STATE.read_text())
bd = boto3.client("bedrock", region_name=REGION)
left = []
for tier, g in state.items():
    try:
        bd.delete_guardrail(guardrailIdentifier=g["id"])
        print("deleted", tier, g["id"])
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "ResourceNotFoundException":
            print("already gone:", tier)
        else:
            print(f"could not delete {tier}: {code}")
            left.append(tier)
if left:
    print("\nStill to remove by hand:", *left, sep="\n  - ")
else:
    print("\nNothing left.")
    STATE.unlink()
