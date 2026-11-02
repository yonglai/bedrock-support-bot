"""Step 6: score a Bedrock Knowledge Base on the same retrieval questions as the hand-built pipeline.
Run:  VARIANT=default QFILE=questions_combined.jsonl uv run python src/step6_kb_retrieval_eval.py
(reads the knowledge base id from .kb.json; or set KB_ID yourself)
"""

import json
import os
import pathlib
import sys

import boto3

from config import REGION
from rag_common import ROOT, evaluate, load_questions

VARIANT = os.environ.get("VARIANT", "default")
_state = ROOT / ".kb.json"
KB_ID = (
    os.environ.get("KB_ID")
    or (
        json.loads(_state.read_text())["kbs"].get(VARIANT, {}).get("kb_id")
        if _state.exists()
        else None
    )
    or sys.exit("No knowledge base: run step6a_kb_create.py first, or set KB_ID.")
)
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
print(f"\nBedrock Knowledge Base, variant {VARIANT}, {len(questions)} questions:")
for k, v in res.items():
    print(f"  {k:20} {v}")
if (
    os.environ.get("QFILE") == "questions_combined.jsonl"
):  # local step 3c, vector only, same 15 questions
    print("\nReference, hand-built (doc@3 / fact@3 / MRR):")
    print("  line_with_title   0.73 / 0.73 / 0.75")
    print("  whole_doc         0.80 / 0.80 / 0.77")
    print(
        "  fixed_200_ov40    1.00 / 0.93 / 0.97  (200 characters; Bedrock's sizes are tokens)"
    )
(ROOT / "eval" / f"results_kb_{VARIANT}.json").write_text(json.dumps(res, indent=2))
print(f"saved eval/results_kb_{VARIANT}.json")
