"""Step 1: first Bedrock Converse call, with latency and token usage.
Run:  MODEL_ID=<id from step 0> python src/step1_converse.py "What is a deductible?"
"""
import sys, time
import boto3
from config import REGION, MODEL_ID

if not MODEL_ID:
    sys.exit("Set MODEL_ID first (see step0_list_models.py).")

client = boto3.client("bedrock-runtime", region_name=REGION)
question = " ".join(sys.argv[1:]) or "What is a deductible?"

t0 = time.perf_counter()
resp = client.converse(
    modelId=MODEL_ID,
    system=[{"text": "You are a concise insurance customer-service assistant."}],
    messages=[{"role": "user", "content": [{"text": question}]}],
    inferenceConfig={"maxTokens": 300, "temperature": 0.2},
)
dt = time.perf_counter() - t0

print(resp["output"]["message"]["content"][0]["text"])
u = resp["usage"]
print(f"\n--- latency {dt:.2f}s | in {u['inputTokens']} tok | out {u['outputTokens']} tok "
      f"| stop: {resp['stopReason']}")
