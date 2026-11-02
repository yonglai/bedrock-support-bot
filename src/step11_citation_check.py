"""Step 11: would checking each statement against ONLY the passages it cites (ALCE-style) catch the errors that the grounding
check missed? Same answers as steps 7 and 10: 26 clean, 31 injected (original sources, so the [S#] numbers are exact), plus
Q501/Q506. For the real errors, the stored wrong answers are checked against sources rebuilt by a re-run, so their [S#]
numbers may not line up with the original run; fresh answers from the re-run are printed and must be read and labelled by
hand. The check calls the model, so it shares the verifier's same-family caveat.
Run:  RETRIEVER=kb VARIANT=hierarchical MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 uv run python src/step11_citation_check.py
"""

import json

from agent import run_agent
from cite_check import check, parse_sources
from mutate import mutations
from rag_common import ROOT

pairs = json.loads((ROOT / ".cache" / "verifier_pairs.json").read_text())
items = []  # (id, kind, answer, sources, is_error or None)
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
rerun = []
for qid in (501, 506):
    q = fresh[qid]
    r = run_agent(q["question"], q["customer_id"])
    items.append((qid, "real_stored", q["question"], stored[qid], r["sources"], True))
    items.append(
        (qid, "real_new", q["question"], r["answer"], r["sources"], None)
    )  # label by hand
    rerun.append((qid, r["answer"], r["sources"]))
    print(f"\n--- Q{qid} fresh re-run answer (label it by hand):\n{r['answer']}\n")
    print("source numbers available:", sorted(parse_sources(r["sources"])))

rows = []
for n, (qid, kind, question, answer, sources, is_error) in enumerate(items, 1):
    try:
        res = check(answer, sources)
        err = None
    except (ValueError, KeyError) as e:
        res, err = (
            {"cited": 0, "uncited": 0, "unsupported": [], "missing": []},
            str(e)[:100],
        )
    rows.append(
        {
            "id": qid,
            "kind": kind,
            "is_error": is_error,
            "answer": answer,
            **res,
            "error": err,
            "flagged": bool(res["unsupported"]),
        }
    )
    if n % 10 == 0:
        print(f"checked {n}/{len(items)}")
(ROOT / "eval/results_citation_check.json").write_text(json.dumps(rows, indent=1))

print("\nkind          n  flagged  with-uncited-sentences  unparsable")
for kind in ("clean", "percent", "amount", "plan", "real_stored", "real_new"):
    g = [r for r in rows if r["kind"] == kind]
    if g:
        print(
            f"{kind:12} {len(g):>3} {sum(r['flagged'] for r in g):>8} {sum(r['uncited'] > 0 for r in g):>14} {sum(bool(r['error']) for r in g):>14}"
        )
cited = sum(r["cited"] for r in rows if r["kind"] == "clean")
uncited = sum(r["uncited"] for r in rows if r["kind"] == "clean")
print(
    f"\nclean answers: {cited} cited statements checked, {uncited} statements without a citation (not checked)"
)
print("\nFlagged clean answers (read: false positive or real?):")
for r in rows:
    if r["kind"] == "clean" and r["flagged"]:
        print(f"  Q{r['id']}: {r['unsupported']}")
print("\nReal-error rows:")
for r in rows:
    if r["kind"].startswith("real"):
        print(
            f"  Q{r['id']} {r['kind']}: flagged={r['flagged']} cited={r['cited']} uncited={r['uncited']} missing={len(r['missing'])}"
        )
        for s, why in r["unsupported"]:
            print(f"     unsupported: {s[:110]} -> {why[:140]}")
print("\nsaved eval/results_citation_check.json")
