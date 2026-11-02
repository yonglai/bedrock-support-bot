"""Step 7b: can the claim-level verifier catch two REAL errors the agent made on the fresh set (not injected ones)?
Q501: counsellor answer said 90% where the mental-health rule is 80%. Q506: the adult eye exam rule applied to the son.
The stored answers (eval/results_agent_fresh.json) are checked against sources rebuilt by running the agent again on the same
question, because the old run did not save its sources. So the sources are a reconstruction, and the new answer (which may or
may not repeat the mistake) is checked too and must be read by hand.
This uses 2 of the 12 fresh questions, for recall only; nothing is tuned on them.
Run:  RETRIEVER=kb VARIANT=hierarchical MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 uv run python src/step7b_verifier_known_bad.py
"""

import json

from agent import run_agent
from rag_common import ROOT
from verify import verify

IDS = (501, 506)
stored = {
    r["id"]: r["answer"]
    for r in json.loads((ROOT / "eval/results_agent_fresh.json").read_text())["rows"]
}
qs = {
    q["id"]: q
    for q in map(
        json.loads, (ROOT / "eval/questions_fresh.jsonl").read_text().splitlines()
    )
}
out = []
for qid in IDS:
    q = qs[qid]
    r = run_agent(q["question"], q["customer_id"])
    print(f"\n=== Q{qid}: {q['question']}  (customer {q['customer_id']})")
    print("tools this run:", r["tools_called"], "| status:", r["status"])
    for label, answer in (
        ("STORED (had the error)", stored[qid]),
        ("NEW run", r["answer"]),
    ):
        ok, bad = verify(q["question"], answer, r["sources"])
        print(f"\n[{label}] verifier: {'ok' if ok else 'FLAGGED'}")
        print("answer:", answer.strip())
        for c in bad:
            print("  unsupported:", c.get("claim"), "->", c.get("evidence"))
        out.append({"id": qid, "which": label, "flagged": not ok, "claims": bad})
(ROOT / "eval/results_verifier_known_bad.json").write_text(json.dumps(out, indent=1))
print("\nsaved eval/results_verifier_known_bad.json")
