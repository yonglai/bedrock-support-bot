"""Step 7: does a claim-level verifier catch wrong figures that the grounding score misses?
1. Collect answers (and the tool results they came from) from the agent on the dev, held-out and agent questions.
   Cached in .cache/verifier_pairs.json; pass --refresh to collect again.
2. Run the verifier on the untouched answers (assumed correct; every flag is read by hand, not trusted).
3. Inject errors by construction (swap a percentage, a dollar amount, or the plan name) and run the verifier on those.
Run:  MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 uv run python src/step7_eval_verifier.py
The fresh set (501-512) is not used here.
"""

import json
import statistics
import sys
import time

from agent import run_agent
from mutate import mutations
from rag_common import ROOT
from verify import verify

PAIRS = ROOT / ".cache" / "verifier_pairs.json"
FILES = ["questions.jsonl", "questions_heldout.jsonl", "questions_agent.jsonl"]


def load_questions():
    qs = []
    for name in FILES:
        for line in (ROOT / "eval" / name).read_text().splitlines():
            q = json.loads(line) if line.strip() else None
            if q and q["type"] in ("single", "multi_doc", "agent"):
                qs.append(q)
    return qs


def collect():
    pairs = []
    for q in load_questions():
        r = run_agent(q["question"], q.get("customer_id", "C1001"))
        if r["status"] == "answered" and r.get("sources"):
            pairs.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": r["answer"],
                    "sources": r["sources"],
                }
            )
            print("collected", q["id"])
    PAIRS.parent.mkdir(exist_ok=True)
    PAIRS.write_text(json.dumps(pairs, indent=1))
    return pairs


def check(item, answer):
    t0 = time.perf_counter()
    try:
        ok, bad = verify(item["question"], answer, item["sources"])
        err = None
    except (ValueError, KeyError) as e:  # unparsable reply: counted separately
        ok, bad, err = True, [], str(e)[:100]
    return {
        "flagged": not ok,
        "claims": bad,
        "error": err,
        "ms": round((time.perf_counter() - t0) * 1000),
    }


pairs = (
    json.loads(PAIRS.read_text())
    if PAIRS.exists() and "--refresh" not in sys.argv
    else collect()
)
rows = []
for it in pairs:
    rows.append(
        {
            "id": it["id"],
            "kind": "clean",
            "answer": it["answer"],
            **check(it, it["answer"]),
        }
    )
    for kind, mut in mutations(it["answer"]):
        rows.append({"id": it["id"], "kind": kind, "answer": mut, **check(it, mut)})

print(f"\n{len(pairs)} answered questions")
print(f"{'kind':8} {'n':>3} {'flagged':>8} {'rate':>6} {'unparsable':>11}")
for kind in ("clean", "percent", "amount", "plan"):
    g = [r for r in rows if r["kind"] == kind]
    if g:
        f = sum(r["flagged"] for r in g)
        print(
            f"{kind:8} {len(g):>3} {f:>8} {f / len(g):>6.2f} {sum(bool(r['error']) for r in g):>11}"
        )
print(
    "verifier latency ms: median",
    statistics.median(r["ms"] for r in rows),
    "max",
    max(r["ms"] for r in rows),
)

print(
    "\nFlagged clean answers (read these: false positive, or a real error in the answer?)"
)
for r in rows:
    if r["kind"] == "clean" and r["flagged"]:
        print(f"  Q{r['id']}:", [c.get("claim") for c in r["claims"]])
print("\nMissed injected errors")
for r in rows:
    if r["kind"] != "clean" and not r["flagged"]:
        print(f"  Q{r['id']} [{r['kind']}]:", r["answer"].replace("\n", " ")[:160])

out = ROOT / "eval" / "results_verifier.json"
out.write_text(json.dumps(rows, indent=1))
print("\nsaved", out)
