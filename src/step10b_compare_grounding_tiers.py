"""Step 10b: does the Standard tier of the Bedrock contextual grounding check score wrong answers lower than Classic does?
Re-scores text that already exists, no new generation except two agent runs to rebuild sources for Q501 and Q506:
  - the 26 clean answers and their sources (.cache/verifier_pairs.json, from step 7),
  - the 31 answers with an injected error (same mutations as step 7),
  - Q501 and Q506: the stored wrong answers and a fresh re-run's answers, scored against the re-run's sources.
Ground truth for the re-run answers was read by hand (Q501 new = correct, 80%; Q506 new = wrong, invented child's eye exam).
Reports score ranges, how many are blocked at the guardrail threshold, a threshold-free separation (AUROC), and the verifier's
flags from step 7/7b for comparison. Thresholds are NOT re-tuned on these data.
Needs: step10a, a Knowledge Base (for the two re-runs), MODEL_ID.
Run:   RETRIEVER=kb VARIANT=hierarchical MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 uv run python src/step10b_compare_grounding_tiers.py
"""

import json
import os
import statistics
import time

import boto3

from agent import run_agent
from config import REGION
from grounding_tiers_common import THRESHOLDS, TIERS, auroc, content_blocks, scores
from mutate import mutations
from rag_common import ROOT

STATE = json.loads((ROOT / ".guardrail_tiers.json").read_text())
rt = boto3.client("bedrock-runtime", region_name=REGION)
pairs = json.loads((ROOT / ".cache" / "verifier_pairs.json").read_text())

# (id, kind, question, answer, sources, is_error)
items = []
for p in pairs:
    items.append((p["id"], "clean", p["question"], p["answer"], p["sources"], False))
    for kind, wrong in mutations(p["answer"]):
        items.append((p["id"], kind, p["question"], wrong, p["sources"], True))

stored = {
    r["id"]: r["answer"]
    for r in json.loads((ROOT / "eval/results_agent_fresh.json").read_text())["rows"]
}
fresh = {
    q["id"]: q
    for q in map(
        json.loads, (ROOT / "eval/questions_fresh.jsonl").read_text().splitlines()
    )
}
for qid, new_is_error in ((501, False), (506, True)):
    q = fresh[qid]
    r = run_agent(q["question"], q["customer_id"])
    items.append((qid, "real_stored", q["question"], stored[qid], r["sources"], True))
    items.append(
        (qid, "real_new", q["question"], r["answer"], r["sources"], new_is_error)
    )

verifier = {}
for row in json.loads((ROOT / "eval/results_verifier.json").read_text()):
    verifier[(row["id"], row["kind"])] = row["flagged"]

rows = []
for n, (qid, kind, question, answer, sources, is_error) in enumerate(items, 1):
    row = {
        "id": qid,
        "kind": kind,
        "is_error": is_error,
        "verifier_flagged": verifier.get((qid, kind)),
    }
    for tier in TIERS:
        g = STATE[tier]
        t0 = time.perf_counter()
        resp = rt.apply_guardrail(
            guardrailIdentifier=g["id"],
            guardrailVersion=g["version"],
            source="OUTPUT",
            content=content_blocks(question, answer, sources),
        )
        row[tier] = {
            **scores(resp),
            "blocked": resp["action"] != "NONE",
            "ms": round((time.perf_counter() - t0) * 1000),
        }
    rows.append(row)
    if n % 10 == 0:
        print(f"scored {n}/{len(items)}")
OUT = os.environ.get("OUT", "eval/results_grounding_tiers.json")
(ROOT / OUT).write_text(json.dumps(rows, indent=1))


def rng(vals):
    return (
        f"{min(vals):.2f}-{max(vals):.2f} (median {statistics.median(vals):.2f})"
        if vals
        else "n/a"
    )


groups = {
    "clean": [r for r in rows if r["kind"] == "clean"],
    "injected": [r for r in rows if r["kind"] in ("percent", "amount", "plan")],
    "real errors": [r for r in rows if r["kind"].startswith("real") and r["is_error"]],
}
for tier in TIERS:
    print(
        f"\n=== {tier}  (blocked = the guardrail's own decision at GROUNDING {THRESHOLDS['GROUNDING']})"
    )
    print(f"{'group':12} {'n':>3} {'blocked':>8}  GROUNDING score range")
    for name, g in groups.items():
        print(
            f"{name:12} {len(g):>3} {sum(r[tier]['blocked'] for r in g):>8}  {rng([r[tier]['GROUNDING'] for r in g if 'GROUNDING' in r[tier]])}"
        )
    errs = [
        r[tier]["GROUNDING"] for r in rows if r["is_error"] and "GROUNDING" in r[tier]
    ]
    cleans = [r[tier]["GROUNDING"] for r in groups["clean"] if "GROUNDING" in r[tier]]
    if errs and cleans:
        print(
            f"AUROC (error scores lower than clean scores): {auroc(errs, cleans):.2f}"
        )
    print(
        f"guardrail call latency ms: median {statistics.median(r[tier]['ms'] for r in rows):.0f}"
    )
new501 = next(r for r in rows if r["kind"] == "real_new" and r["id"] == 501)
print(
    "\nQ501 new answer is CORRECT (80%): blocked? "
    + ", ".join(f"{t}={new501[t]['blocked']}" for t in TIERS)
)
print(
    "\nVerifier (step 7) for reference: clean flagged",
    sum(bool(r["verifier_flagged"]) for r in groups["clean"]),
    "of",
    len(groups["clean"]),
    "| injected flagged",
    sum(bool(r["verifier_flagged"]) for r in groups["injected"]),
    "of",
    len(groups["injected"]),
)
print(f"\nsaved {OUT}")
