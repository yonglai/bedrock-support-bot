"""Step 3a: compare chunking strategies for retrieval, entirely local (no OpenSearch yet).
Embeds chunks with Titan (cached on disk), ranks by cosine similarity, and scores each strategy
against eval/questions.jsonl.

Run:  uv run python src/step3a_retrieval_experiment.py
      uv run python src/step3a_retrieval_experiment.py --mock     # offline test, no AWS calls
"""

import hashlib
import json
import pathlib
import re
import sys

import numpy as np

from config import EMBED_MODEL_ID, REGION

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = EMBED_MODEL_ID or "amazon.titan-embed-text-v2:0"
DIMS = 512
MOCK = "--mock" in sys.argv
CACHE = ROOT / ".cache" / "embeddings.json"
KS = (1, 3, 5)


# ---------- chunking strategies ----------
def title_of(text):
    m = re.match(r"#\s+(.+)", text)
    return m.group(1).strip() if m else ""


def whole_doc(name, text):
    return [text.strip()]


def line_with_title(name, text):
    t = title_of(text)
    lines = [
        l.strip() for l in text.splitlines() if l.strip() and not l.startswith("# ")
    ]
    return [f"{t}\n{l}" for l in lines]


def fixed(size, overlap):
    def f(name, text):
        out, i = [], 0
        while i < len(text):
            out.append(text[i : i + size].strip())
            i += size - overlap
        return [c for c in out if c]

    return f


STRATEGIES = {
    "whole_doc": whole_doc,
    "line_with_title": line_with_title,
    "fixed_200_ov40": fixed(200, 40),
    "fixed_400_ov80": fixed(400, 80),
}

# ---------- embeddings ----------
_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
_client = None


def mock_embed(text):
    v = np.zeros(DIMS)
    for w in re.findall(r"[a-z0-9$%\-]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIMS] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def embed(text):
    if MOCK:
        return mock_embed(text)
    global _client
    key = hashlib.sha1(f"{MODEL}|{DIMS}|{text}".encode()).hexdigest()
    if key not in _cache:
        if _client is None:
            import boto3

            _client = boto3.client("bedrock-runtime", region_name=REGION)
        r = _client.invoke_model(
            modelId=MODEL,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text, "dimensions": DIMS, "normalize": True}),
        )
        _cache[key] = json.loads(r["body"].read())["embedding"]
    return _cache[key]


def save_cache():
    if not MOCK:
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(_cache))


# ---------- evaluation ----------
docs = {p.name: p.read_text() for p in sorted((ROOT / "data/policies").glob("*.md"))}
questions = [
    json.loads(l)
    for l in (ROOT / "eval/questions.jsonl").read_text().splitlines()
    if l.strip()
]
scored = [q for q in questions if q["type"] in ("single", "multi_doc")]
q_emb = {q["id"]: np.array(embed(q["question"])) for q in scored}


def expected_docs(q):
    e = q["expected_doc"]
    return set(e if isinstance(e, list) else [e])


results = {}
for name, fn in STRATEGIES.items():
    chunks = [(d, c) for d, t in docs.items() for c in fn(d, t)]
    M = np.array([embed(c) for _, c in chunks])
    stats = {f"doc_hit@{k}": 0 for k in KS}
    stats.update({f"fact@{k}": 0 for k in KS})
    rr_sum, misses = 0.0, []
    for q in scored:
        order = np.argsort(-(M @ q_emb[q["id"]]))
        want = expected_docs(q)
        # reciprocal rank of the first chunk from an expected doc
        rank = next(
            (i + 1 for i, idx in enumerate(order) if chunks[idx][0] in want), None
        )
        rr_sum += 1 / rank if rank else 0
        for k in KS:
            top = order[:k]
            top_docs = {chunks[i][0] for i in top}
            # doc@1 counts a hit if the single top chunk is from ANY expected doc (a multi-doc question
            # cannot have all its docs in one chunk); for k>1 all expected docs must be present.
            if (want & top_docs) if k == 1 else (want <= top_docs):
                stats[f"doc_hit@{k}"] += 1
            # facts only count if they appear in a retrieved chunk from an EXPECTED doc
            # (otherwise "$100" or "50%" can be matched by an unrelated document)
            text = "\n".join(chunks[i][1] for i in top if chunks[i][0] in want)
            if all(f in text for f in q["key_facts"]):
                stats[f"fact@{k}"] += 1
            elif k == 3:
                misses.append(q["id"])
    n = len(scored)
    results[name] = {
        "chunks": len(chunks),
        "avg_chars": round(sum(len(c) for _, c in chunks) / len(chunks)),
        **{k: round(v / n, 2) for k, v in stats.items()},
        "MRR": round(rr_sum / n, 2),
        "fact@3_misses": misses,
    }
save_cache()

print(
    f"\nEmbedding: {'MOCK (hashed words)' if MOCK else MODEL}   questions scored: {len(scored)}\n"
)
hdr = f"{'strategy':18}{'chunks':>7}{'chars':>7}{'doc@1':>7}{'doc@3':>7}{'fact@1':>8}{'fact@3':>8}{'fact@5':>8}{'MRR':>6}"
print(hdr + "\n" + "-" * len(hdr))
for n, r in results.items():
    print(
        f"{n:18}{r['chunks']:>7}{r['avg_chars']:>7}{r['doc_hit@1']:>7}{r['doc_hit@3']:>7}"
        f"{r['fact@1']:>8}{r['fact@3']:>8}{r['fact@5']:>8}{r['MRR']:>6}"
    )
print("\nQuestions missing key facts in top-3:")
for n, r in results.items():
    print(f"  {n:18} {r['fact@3_misses'] or 'none'}")

out = (
    ROOT
    / "eval"
    / ("results_retrieval_mock.json" if MOCK else "results_retrieval.json")
)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved {out.relative_to(ROOT)}")
