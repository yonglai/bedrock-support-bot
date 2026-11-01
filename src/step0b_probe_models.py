"""Probe which chat models actually work for your account (tiny call each).
Run:  uv run python src/step0b_probe_models.py
"""

import boto3
from botocore.exceptions import ClientError

from config import REGION

CANDIDATES = [
    "us.anthropic.claude-haiku-5-5",
    "us.anthropic.claude-sonnet-5-5",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.amazon.nova-2-lite-v1:0",
    "us.amazon.nova-pro-v1:0",
    "amazon.nova-lite-v1:0",
    "amazon.nova-micro-v1:0",
    "openai.gpt-oss-120b-1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "mistral.mistral-large-3-675b-instruct",
]
rt = boto3.client("bedrock-runtime", region_name=REGION)
for mid in CANDIDATES:
    try:
        r = rt.converse(
            modelId=mid,
            messages=[{"role": "user", "content": [{"text": "Say OK."}]}],
            inferenceConfig={"maxTokens": 20},
        )
        print(
            f"OK      {mid}   ({r['usage']['inputTokens']} in / {r['usage']['outputTokens']} out)"
        )
    except ClientError as e:
        print(
            f"DENIED  {mid}   {e.response['Error']['Code']}: {e.response['Error']['Message'][:90]}"
        )
    except Exception as e:  # noqa: BLE001 - report any failure, keep probing
        print(f"ERROR   {mid}   {str(e)[:90]}")
