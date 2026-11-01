"""Step 3c: does adding keyword (BM25) search to vector search fix the misses?
Uses cached Titan embeddings, so it makes no AWS calls.
Methods: vector only, BM25 only, hybrid RRF (rank fusion), hybrid weighted (min-max, alpha=0.5).
Run:  uv run python src/step3c_hybrid_experiment.py
"""

import hashlib
import json
import math
import pathlib
import re
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL, DIMS = "amazon.titan-embed-text-v2:0", 512
cache = json.loads((ROOT / ".cache" / "embeddings.json").read_text())
KS = (1, 3, 5)


def emb(t):
    return np.array(cache[hashlib.sha1(f"{MODEL}|{DIMS}|{t}".encode()).hexdigest()])


# ---- chunkers (same as step3a) ----
def title_of(t):
    m = re.match(r"#\s+(.+)", t)
    return m.group(1).strip() if m else ""


def whole(n, t):
    return [t.strip()]


def line_title(n, t):
    ti = title_of(t)
    return [
        f"{ti}\n{l.strip()}"
        for l in t.splitlines()
        if l.strip() and not l.startswith("# ")
    ]


def fixed(size, ov):
    def f(n, t):
        out, i = [], 0
        while i < len(t):
            out.append(t[i : i + size].strip())
            i += size - ov
        return [c for c in out if c]

    return f


STRATS = {
    "whole_doc": whole,
    "line_with_title": line_title,
    "fixed_200_ov40": fixed(200, 40),
    "fixed_400_ov80": fixed(400, 80),
}

# ---- BM25 ----
STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "be",
    "to",
    "of",
    "and",
    "or",
    "for",
    "in",
    "on",
    "at",
    "my",
    "i",
    "me",
    "do",
    "does",
    "what",
    "how",
    "much",
    "can",
    "it",
    "this",
    "that",
    "with",
    "from",
    "your",
    "you",
    "our",
    "who",
    "when",
    "where",
    "which",
    "if",
    "so",
    "as",
    "by",
    "have",
    "has",
    "had",
    "get",
    "yearly",
    "per",
}


def tok(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(tok(t)) for t in texts]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        self.k1, self.b, self.N = k1, b, len(texts)
        df = Counter(w for d in self.docs for w in d)
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()
        }

    def scores(self, query):
        q = tok(query)
        out = np.zeros(self.N)
        for i, d in enumerate(self.docs):
            for w in q:
                if w in d:
                    f = d[w]
                    out[i] += (
                        self.idf[w]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                    )
        return out


def ranks(scores):  # rank position (1 = best) for each item
    order = np.argsort(-scores)
    r = np.empty(len(scores))
    r[order] = np.arange(1, len(scores) + 1)
    return r


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def rank_order(method, vec, bm):
    if method == "vector":
        s = vec
    elif method == "bm25":
        s = bm + 1e-9 * vec  # vector breaks ties among zero-keyword chunks
    elif method == "hybrid_rrf":
        s = 1 / (60 + ranks(vec)) + 1 / (60 + ranks(bm + 1e-9 * vec))
    elif method == "hybrid_w50":
        s = 0.5 * minmax(vec) + 0.5 * minmax(bm)
    return np.argsort(-s)


# ---- evaluation ----
docs = {p.name: p.read_text() for p in sorted((ROOT / "data/policies").glob("*.md"))}
qs = [
    json.loads(l)
    for l in (ROOT / "eval/questions.jsonl").read_text().splitlines()
    if l.strip()
]
qs = [q for q in qs if q["type"] in ("single", "multi_doc")]
METHODS = ["vector", "bm25", "hybrid_rrf", "hybrid_w50"]
results, detail = {}, {}

for sname, fn in STRATS.items():
    chunks = [(d, c) for d, t in docs.items() for c in fn(d, t)]
    M = np.array([emb(c) for _, c in chunks])
    bm25 = BM25([c for _, c in chunks])
    for m in METHODS:
        st = {f"doc@{k}": 0 for k in KS}
        st.update({f"fact@{k}": 0 for k in KS})
        rr = 0
        miss = []
        for q in qs:
            vec = M @ emb(q["question"])
            bm = bm25.scores(q["question"])
            order = rank_order(m, vec, bm)
            want = set(
                q["expected_doc"]
                if isinstance(q["expected_doc"], list)
                else [q["expected_doc"]]
            )
            rk = next(
                (i + 1 for i, ix in enumerate(order) if chunks[ix][0] in want), None
            )
            rr += 1 / rk if rk else 0
            for k in KS:
                top = order[:k]
                td = {chunks[i][0] for i in top}
                if (want & td) if k == 1 else (want <= td):
                    st[f"doc@{k}"] += 1
                text = "\n".join(chunks[i][1] for i in top if chunks[i][0] in want)
                ok = all(f in text for f in q["key_facts"])
                st[f"fact@{k}"] += ok
                if k == 3 and not ok:
                    miss.append(q["id"])
            if q["id"] == 2:
                detail[(sname, m)] = [f"{chunks[i][0]}" for i in order[:3]]
        n = len(qs)
        results[(sname, m)] = {
            **{k: round(v / n, 2) for k, v in st.items()},
            "MRR": round(rr / n, 2),
            "fact@3_miss": miss,
        }

print(
    f"\n{len(qs)} questions. fact@k counts only facts found in chunks from the expected document(s).\n"
)
hdr = f"{'strategy':17}{'method':12}{'doc@1':>6}{'doc@3':>6}{'fact@1':>7}{'fact@3':>7}{'fact@5':>7}{'MRR':>6}  fact@3 misses"
print(hdr + "\n" + "-" * (len(hdr) + 6))
for (s, m), r in results.items():
    print(
        f"{s:17}{m:12}{r['doc@1']:>6}{r['doc@3']:>6}{r['fact@1']:>7}{r['fact@3']:>7}{r['fact@5']:>7}{r['MRR']:>6}  {r['fact@3_miss'] or '-'}"
    )
    if m == METHODS[-1]:
        print()

print("Q2 (yearly deductible) top-3 documents:")
for (s, m), d in detail.items():
    print(f"  {s:17}{m:12}{d}")
(ROOT / "eval" / "results_hybrid.json").write_text(
    json.dumps({f"{s}|{m}": r for (s, m), r in results.items()}, indent=2)
)
print("\nSaved eval/results_hybrid.json")
