"""Step 0: see which text and embedding models your account can use in this region.
Run:  python src/step0_list_models.py
"""
import boto3
from config import REGION

br = boto3.client("bedrock", region_name=REGION)
print(f"Region: {REGION}\n")

print("== Text-generation models (on-demand) ==")
for m in br.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]:
    if "ON_DEMAND" in m.get("inferenceTypesSupported", []):
        print(f'{m["modelId"]:60} {m["providerName"]}')

print("\n== Inference profiles (cross-region; some newer models need these) ==")
try:
    for p in br.list_inference_profiles()["inferenceProfileSummaries"]:
        print(f'{p["inferenceProfileId"]:60} {p["inferenceProfileName"]}')
except Exception as e:
    print("could not list profiles:", e)

print("\n== Embedding models ==")
for m in br.list_foundation_models(byOutputModality="EMBEDDING")["modelSummaries"]:
    print(f'{m["modelId"]:60} {m["providerName"]}')
