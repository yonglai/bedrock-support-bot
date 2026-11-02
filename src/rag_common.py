"""Shared RAG helpers: chunking, cached Titan embeddings, and the retrieval metrics used in step 3."""

import hashlib
import json
import os
import pathlib
import re
import statistics
import time

from config import EMBED_MODEL_ID, REGION

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL = EMBED_MODEL_ID or "amazon.titan-embed-text-v2:0"
DIMS = 512
CACHE = ROOT / ".cache" / "embeddings.json"
_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
_client = None


def embed(text):
    key = hashlib.sha1(f"{MODEL}|{DIMS}|{text}".encode()).hexdigest()
    if key not in _cache:
        global _client
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
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(_cache))


def load_docs():
    return {
        p.name: p.read_text() for p in sorted((ROOT / "data/policies").glob("*.md"))
    }


def load_questions(types=("single", "multi_doc")):
    qs = [
        json.loads(l)
        for l in (ROOT / "eval" / os.environ.get("QFILE", "questions.jsonl"))
        .read_text()
        .splitlines()
        if l.strip()
    ]
    return [q for q in qs if q["type"] in types]


def title_of(text):
    m = re.match(r"#\s+(.+)", text)
    return m.group(1).strip() if m else ""


CHUNKING = os.environ.get("CHUNKING", "fixed_200_ov40")


def _fixed(text, size=200, overlap=40):
    """Yield (chunk_text, context). The chunk is what gets embedded and scored. The context is what the model reads:
    the whole document (each policy file is one short section, ~150 tokens). A 200-character window cuts lines in half
    and drops the lines around it, e.g. the Standard-plan rate next to the Enhanced one, which led the model to apply
    the wrong plan's figure. With longer documents the parent would be the section, not the file."""
    i = 0
    while i < len(text):
        chunk = text[i : i + size].strip()
        if chunk:
            yield chunk, text.strip()
        i += size - overlap


def build_chunks(strategy=None):
    """Chunking chosen in step 3c (docs/notes.md): fixed_200_ov40 (200-character windows, 40 overlap, same as the
    experiment). 'line_with_title' (one line per chunk plus the doc title) was the earlier default and missed
    every physiotherapy question."""
    strategy = strategy or CHUNKING
    out = []
    for name, text in load_docs().items():
        if strategy == "fixed_200_ov40":
            out += [{"doc": name, "text": c, "context": ctx} for c, ctx in _fixed(text)]
        elif strategy == "line_with_title":
            t = title_of(text)
            for line in text.splitlines():
                if line.strip() and not line.startswith("# "):
                    out.append({"doc": name, "text": f"{t}\n{line.strip()}"})
        else:
            raise ValueError(f"unknown chunking strategy: {strategy}")
    return out


def expected_docs(q):
    e = q["expected_doc"]
    return set(e if isinstance(e, list) else [e])


def evaluate(retrieve, questions, ks=(1, 3, 5)):
    """retrieve(question_text, k) -> list of (doc, text), best first. Same metric definitions as step 3a."""
    st = {f"doc@{k}": 0 for k in ks}
    st.update({f"fact@{k}": 0 for k in ks})
    rr, lat, misses = 0.0, [], []
    for q in questions:
        t0 = time.perf_counter()
        hits = retrieve(q["question"], 10)
        lat.append((time.perf_counter() - t0) * 1000)
        want = expected_docs(q)
        rank = next((i + 1 for i, (d, _) in enumerate(hits) if d in want), None)
        rr += 1 / rank if rank else 0
        for k in ks:
            top = hits[:k]
            docs = {d for d, _ in top}
            if (want & docs) if k == 1 else (want <= docs):
                st[f"doc@{k}"] += 1
            text = "\n".join(t for d, t in top if d in want)
            ok = all(f in text for f in q["key_facts"])
            st[f"fact@{k}"] += ok
            if k == 3 and not ok:
                misses.append(q["id"])
    n = len(questions)
    res = {k: round(v / n, 2) for k, v in st.items()}
    res.update(
        {
            "MRR@10": round(rr / n, 2),
            "fact@3_misses": misses,
            "latency_ms_median": round(statistics.median(lat)),
            "latency_ms_max": round(max(lat)),
        }
    )
    return res
