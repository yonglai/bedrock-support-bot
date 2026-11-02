"""Step 10a: create two guardrails that differ only in the contextual-grounding tier (Classic and Standard), each with
only the grounding and relevance checks, same thresholds as the main guardrail. Writes .guardrail_tiers.json.
Standard needs cross-Region inference through a guardrail profile (GUARDRAIL_PROFILE, default us.guardrail.v1:0) and is only
offered in some Regions; if the create call is refused, the error says why.
Run:     uv run python src/step10a_tier_guardrails_create.py
Remove:  uv run python src/step10z_tier_guardrails_teardown.py
"""

import json
import os

import boto3
from botocore.exceptions import ClientError

from config import REGION
from grounding_tiers_common import THRESHOLDS, TIERS
from rag_common import ROOT

STATE = ROOT / ".guardrail_tiers.json"
PROFILE = os.environ.get("GUARDRAIL_PROFILE", "us.guardrail.v1:0")
bd = boto3.client("bedrock", region_name=REGION)
state = json.loads(STATE.read_text()) if STATE.exists() else {}

for tier in TIERS:
    if tier in state:
        print(tier, "exists:", state[tier]["id"])
        continue
    kwargs = dict(
        name=f"evergreen-grounding-{tier.lower()}",
        description=f"Contextual grounding only, {tier} tier (experiment)",
        contextualGroundingPolicyConfig={
            "filtersConfig": [
                {"type": t, "threshold": v} for t, v in THRESHOLDS.items()
            ],
            "tierConfig": {"tierName": tier},
        },
        blockedInputMessaging="blocked",
        blockedOutputsMessaging="blocked",
    )
    if tier == "STANDARD":
        kwargs["crossRegionConfig"] = {"guardrailProfileIdentifier": PROFILE}
    try:
        r = bd.create_guardrail(**kwargs)
    except ClientError as e:
        raise SystemExit(
            f"{tier} guardrail was refused: {e.response['Error']['Code']}: {e.response['Error']['Message']}"
        )
    ver = bd.create_guardrail_version(
        guardrailIdentifier=r["guardrailId"], description="auto"
    )["version"]
    state[tier] = {"id": r["guardrailId"], "arn": r["guardrailArn"], "version": ver}
    STATE.write_text(json.dumps(state, indent=1))
    print("created", tier, r["guardrailId"], "version", ver)
print("ready:", {t: state[t]["id"] for t in state})
