"""Step 6: score a Bedrock Knowledge Base on the same retrieval questions as the hand-built pipeline.
Run:  KB_ID=<knowledge base id> uv run python src/step6_kb_retrieval_eval.py
"""

import json
import os
import pathlib
import sys

import boto3

from config import REGION
from rag_common import ROOT, evaluate, load_questions

KB_ID = os.environ.get("KB_ID") or sys.exit("Set KB_ID to your knowledge base id.")
agent_rt = boto3.client("bedrock-agent-runtime", region_name=REGION)


def retrieve(question, k):
    r = agent_rt.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": k}},
    )
    return [
        (
            pathlib.PurePosixPath(x["location"]["s3Location"]["uri"]).name,
            x["content"]["text"],
        )
        for x in r["retrievalResults"]
    ]


questions = load_questions()
retrieve(questions[0]["question"], 3)  # warm-up
res = evaluate(retrieve, questions)
print("\nBedrock Knowledge Base (managed retrieval), 11 questions:")
for k, v in res.items():
    print(f"  {k:20} {v}")
print("\nHand-built, line_with_title (step 3d): doc@3 1.0, fact@3 1.0, MRR 0.95")
print("Hand-built, whole_doc (step 3a):       doc@3 0.91, fact@3 0.91, MRR 0.93")
(ROOT / "eval" / "results_kb.json").write_text(json.dumps(res, indent=2))
print("saved eval/results_kb.json")
