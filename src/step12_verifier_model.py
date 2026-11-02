"""Step 12: does a different, larger verifier model change the verifier results?
The verifier in step 7 is Claude Haiku 4.5, the same model that wrote the answers. This re-runs it with another model.
  - The 26 clean + 31 injected answers come from .cache/verifier_pairs.json, so inputs are identical to step 7.
    The Haiku flags are read from eval/results_verifier.json (temperature 0, not re-run).
  - Q501 and Q506: the agent is run once more and its sources are SAVED. The stored wrong answer and the new answer are checked
    by BOTH models against those same sources. New answers must be read by hand; nothing is scored automatically.
Needs: step 7 results and cache, a Knowledge Base (for the two agent runs), MODEL_ID (agent), VERIFIER_MODEL_ID (the other model).
Run:   RETRIEVER=kb VARIANT=hierarchical MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 \\
       VERIFIER_MODEL_ID=<inference profile of the larger model> uv run python src/step12_verifier_model.py
"""

import json
import os
import time

from agent import run_agent
from config import MODEL_ID
from mutate import mutations
from rag_common import ROOT
from verify import verify

OTHER = os.environ.get("VERIFIER_MODEL_ID", "")
if not OTHER or OTHER == MODEL_ID:
    raise SystemExit("Set VERIFIER_MODEL_ID to a model other than MODEL_ID.")


def check(question, answer, sources, model_id):
    t0 = time.perf_counter()
    try:
        ok, bad = verify(question, answer, sources, model_id=model_id)
        err = None
    except (ValueError, KeyError) as e:
        ok, bad, err = True, [], str(e)[:100]
    return {
        "flagged": not ok,
        "claims": bad,
        "error": err,
        "ms": round((time.perf_counter() - t0) * 1000),
    }


haiku = {
    (r["id"], r["kind"], r["answer"]): r["flagged"]
    for r in json.loads((ROOT / "eval/results_verifier.json").read_text())
}
pairs = json.loads((ROOT / ".cache/verifier_pairs.json").read_text())

rows = []
for p in pairs:
    versions = [("clean", p["answer"])] + list(mutations(p["answer"]))
    for kind, answer in versions:
        r = check(p["question"], answer, p["sources"], OTHER)
        r.update(
            id=p["id"],
            kind=kind,
            answer=answer,
            haiku_flagged=haiku.get((p["id"], kind, answer)),
        )
        rows.append(r)
    print("checked", p["id"])

print(f"\nverifier {OTHER} vs Haiku (Haiku flags from step 7)")
print(f"{'kind':8} {'n':>3} {'other':>6} {'haiku':>6} {'unparsable':>11}")
for kind in ("clean", "percent", "amount", "plan"):
    g = [r for r in rows if r["kind"] == kind]
    print(
        f"{kind:8} {len(g):>3} {sum(r['flagged'] for r in g):>6} "
        f"{sum(bool(r['haiku_flagged']) for r in g):>6} {sum(bool(r['error']) for r in g):>11}"
    )
print("\nDisagreements (read these)")
for r in rows:
    if bool(r["haiku_flagged"]) != r["flagged"]:
        print(
            f"  Q{r['id']} [{r['kind']}] other={r['flagged']} haiku={r['haiku_flagged']}"
        )
        for c in r["claims"]:
            print("     ", c.get("claim"), "->", c.get("evidence"))
print("median latency ms:", sorted(r["ms"] for r in rows)[len(rows) // 2])

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
real = []
for qid in (501, 506):
    q = fresh[qid]
    run = run_agent(q["question"], q["customer_id"])
    for label, answer in (("stored", stored[qid]), ("new", run["answer"])):
        both = {
            "haiku": check(q["question"], answer, run["sources"], MODEL_ID),
            "other": check(q["question"], answer, run["sources"], OTHER),
        }
        real.append(
            {
                "id": qid,
                "which": label,
                "answer": answer,
                "sources": run["sources"],
                **both,
            }
        )
        print(
            f"\n=== Q{qid} {label}: haiku={'FLAG' if both['haiku']['flagged'] else 'ok'}  other={'FLAG' if both['other']['flagged'] else 'ok'}"
        )
        print("answer:", answer.strip())
        for name in ("haiku", "other"):
            for c in both[name]["claims"]:
                print(
                    f"  [{name}] unsupported:", c.get("claim"), "->", c.get("evidence")
                )

out = ROOT / "eval/results_verifier_model.json"
out.write_text(
    json.dumps({"verifier_model": OTHER, "rows": rows, "real": real}, indent=1)
)
print("\nsaved", out)
