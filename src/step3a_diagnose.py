"""Per-question diagnosis of the retrieval experiment, using the cached embeddings (no AWS calls).
Run:  uv run python src/step3a_diagnose.py
"""

import hashlib
import json
import pathlib
import re
import sys

import numpy as np

sys.argv = [a for a in sys.argv if a != "--mock"]
ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL, DIMS = "amazon.titan-embed-text-v2:0", 512
cache = json.loads((ROOT / ".cache" / "embeddings.json").read_text())


def emb(t):
    return np.array(cache[hashlib.sha1(f"{MODEL}|{DIMS}|{t}".encode()).hexdigest()])


# same chunkers as step3a
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


S = {
    "whole_doc": whole,
    "line_with_title": line_title,
    "fixed_200_ov40": fixed(200, 40),
    "fixed_400_ov80": fixed(400, 80),
}

docs = {p.name: p.read_text() for p in sorted((ROOT / "data/policies").glob("*.md"))}
qs = [
    json.loads(l)
    for l in (ROOT / "eval/questions.jsonl").read_text().splitlines()
    if l.strip()
]
qs = [q for q in qs if q["type"] in ("single", "multi_doc")]

for name, fn in S.items():
    chunks = [(d, c) for d, t in docs.items() for c in fn(d, t)]
    M = np.array([emb(c) for _, c in chunks])
    print(f"\n=== {name} ===")
    for q in qs:
        order = np.argsort(-(M @ emb(q["question"])))
        want = (
            q["expected_doc"]
            if isinstance(q["expected_doc"], list)
            else [q["expected_doc"]]
        )
        top1 = chunks[order[0]][0]
        ranks = {
            w: next(i + 1 for i, ix in enumerate(order) if chunks[ix][0] == w)
            for w in want
        }
        text3 = "\n".join(chunks[i][1] for i in order[:3])
        ok3 = all(f in text3 for f in q["key_facts"])
        flag = "" if (top1 in want and len(want) == 1) else "   <-- check"
        print(
            f"  Q{q['id']:<2} top1={top1:28} first-rank-of-expected={ranks} facts@3={'ok' if ok3 else 'MISS'}{flag}"
        )
