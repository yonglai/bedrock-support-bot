"""Step 6b: show what each Knowledge Base variant returns (chunk length and start), to explain the retrieval numbers.
Run:  uv run python src/step6b_kb_inspect.py ["a question"]
"""

import json
import sys

import boto3

from rag_common import ROOT

state = json.loads((ROOT / ".kb.json").read_text())
rt = boto3.client("bedrock-agent-runtime", region_name=state["region"])
question = " ".join(sys.argv[1:]) or "physiotherapy coverage limits"
print("question:", question)
for variant, rec in state["kbs"].items():
    r = rt.retrieve(
        knowledgeBaseId=rec["kb_id"],
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}},
    )
    print(f"\n== {variant}")
    for x in r["retrievalResults"]:
        text = x["content"]["text"]
        doc = x["location"]["s3Location"]["uri"].rsplit("/", 1)[-1]
        print(f"  {doc:28} {len(text):5} chars  {text[:70]!r}")
