"""Step 3d: load the 58 line_with_title chunks into OpenSearch and re-score the retrieval eval there.
Expect results close to the local numbers from step 3a (vector-only, line_with_title).
Run:  uv run python src/step3d_opensearch_ingest_query.py
"""

import json
import time

from oss import INDEX, client, create_index
from rag_common import ROOT, build_chunks, embed, evaluate, load_questions, save_cache

c = client()
chunks = build_chunks()
print(f"{len(chunks)} chunks; embedding (cached where possible) ...")
vecs = [embed(ch["text"]) for ch in chunks]
save_cache()

create_index(
    c, recreate=True
)  # no custom ids on vector collections, so rebuild to avoid duplicates
lines = []
for ch, v in zip(chunks, vecs):
    lines += [
        json.dumps({"index": {"_index": INDEX}}),
        json.dumps({"doc": ch["doc"], "text": ch["text"], "embedding": v}),
    ]
resp = c.bulk(body="\n".join(lines) + "\n")
if resp.get("errors"):
    bad = [i for i in resp["items"] if list(i.values())[0].get("error")]
    raise SystemExit(f"bulk errors, first: {bad[0]}")

print("waiting for documents to become searchable ...")
for _ in range(30):
    n = c.count(index=INDEX)["count"]
    if n >= len(chunks):
        break
    time.sleep(3)
print(f"indexed {n}/{len(chunks)}")


def retrieve(question, k):
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": embed(question), "k": k}}},
        "_source": ["doc", "text"],
    }
    hits = c.search(index=INDEX, body=body)["hits"]["hits"]
    return [(h["_source"]["doc"], h["_source"]["text"]) for h in hits]


questions = load_questions()
for q in questions:  # make sure question embeddings are cached
    embed(q["question"])
save_cache()
retrieve(questions[0]["question"], 3)  # warm-up (first query after idle can be slow)
res = evaluate(retrieve, questions)
print("\nOpenSearch NextGen, vector-only, line_with_title:")
for k, v in res.items():
    print(f"  {k:20} {v}")
(ROOT / "eval" / "results_opensearch.json").write_text(json.dumps(res, indent=2))
print(
    "\nSaved eval/results_opensearch.json   (local step-3a reference: fact@3=1.0, doc@3=1.0, MRR~0.95)"
)
