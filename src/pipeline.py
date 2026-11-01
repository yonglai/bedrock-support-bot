"""Core chatbot pipeline:  input guardrail -> retrieve (OpenSearch) -> grounded generation (Bedrock) -> result dict.

Env vars:  MODEL_ID (required)   TOP_K (default 3)   GROUNDING=0 to send sources as plain text (no grounding check)
"""
import json, os, time
import boto3
from config import REGION, MODEL_ID
from oss import client as oss_client, INDEX
from rag_common import embed, save_cache, ROOT

GUARD = json.loads((ROOT / ".guardrail.json").read_text())
GROUNDING = os.environ.get("GROUNDING", "1") != "0"
TOP_K = int(os.environ.get("TOP_K", "3"))
HANDOFF = "I'll connect you with a team member who can help with that."

SYSTEM = (
    "You are a customer-service assistant for Evergreen Life, a fictional insurer. "
    "Answer ONLY from the numbered sources provided and cite them like [1]. Be concise. "
    "If the sources do not contain the answer, say you don't have that information and offer to connect "
    "the customer with a team member. Never give legal, medical or investment advice. "
    "Never ask for or repeat SINs, card numbers or passwords."
)

rt = boto3.client("bedrock-runtime", region_name=REGION)
_os = None


def _any_blocked(assessments):
    for a in assessments:
        sip = a.get("sensitiveInformationPolicy", {})
        groups = [a.get("topicPolicy", {}).get("topics", []), a.get("contentPolicy", {}).get("filters", []),
                  sip.get("piiEntities", []), sip.get("regexes", [])]
        if any(item.get("action") == "BLOCKED" for g in groups for item in g):
            return True
    return False


def check_input(text):
    """Run the guardrail on the user's message. Returns (status, sanitized_text); status is ok|redacted|blocked."""
    r = rt.apply_guardrail(
        guardrailIdentifier=GUARD["id"], guardrailVersion=GUARD["version"], source="INPUT",
        content=[{"text": {"text": text, "qualifiers": ["guard_content"]}}],
    )
    if r["action"] == "NONE":
        return "ok", text
    if _any_blocked(r.get("assessments", [])):
        return "blocked", HANDOFF
    return "redacted", r["outputs"][0]["text"]


def retrieve(question, k=TOP_K):
    global _os
    _os = _os or oss_client()
    body = {"size": k, "query": {"knn": {"embedding": {"vector": embed(question), "k": k}}}, "_source": ["doc", "text"]}
    hits = _os.search(index=INDEX, body=body)["hits"]["hits"]
    return [{"doc": h["_source"]["doc"], "text": h["_source"]["text"]} for h in hits]


def _grounding_scores(resp):
    out = []
    for assessments in resp.get("trace", {}).get("guardrail", {}).get("outputAssessments", {}).values():
        for a in assessments:
            for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
                out.append(f"{f['type']}={f.get('score')}/{f.get('threshold')}:{f.get('action')}")
    return out


def generate(question, sources, history=None):
    src_text = "\n".join(f"[{i + 1}] ({s['doc']}) {s['text']}" for i, s in enumerate(sources))
    if GROUNDING:   # tagged blocks let the guardrail compare the answer with the retrieved sources
        content = [
            {"guardContent": {"text": {"text": src_text, "qualifiers": ["grounding_source"]}}},
            {"guardContent": {"text": {"text": question, "qualifiers": ["query", "guard_content"]}}},
        ]
    else:
        content = [{"text": f"Sources:\n{src_text}\n\nQuestion: {question}"}]
    messages = list(history or []) + [{"role": "user", "content": content}]
    resp = rt.converse(
        modelId=MODEL_ID, system=[{"text": SYSTEM}], messages=messages,
        inferenceConfig={"maxTokens": 400, "temperature": 0.1},
        guardrailConfig={"guardrailIdentifier": GUARD["id"], "guardrailVersion": GUARD["version"], "trace": "enabled"},
    )
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    return text, resp


def answer(question, history=None):
    t0 = time.perf_counter()
    status, clean = check_input(question)
    t1 = time.perf_counter()
    if status == "blocked":
        return {"status": "blocked", "answer": clean, "sources": [], "ms": {"input_guard": round((t1 - t0) * 1000)}}
    sources = retrieve(clean)
    t2 = time.perf_counter()
    text, resp = generate(clean, sources, history)
    t3 = time.perf_counter()
    save_cache()
    intervened = resp["stopReason"] == "guardrail_intervened"
    return {
        "status": "guardrail_blocked" if intervened else ("answered_redacted" if status == "redacted" else "answered"),
        "answer": text, "sources": [s["doc"] for s in sources], "grounding": _grounding_scores(resp),
        "tokens": resp["usage"], "stop": resp["stopReason"],
        "ms": {"input_guard": round((t1 - t0) * 1000), "retrieve": round((t2 - t1) * 1000), "generate": round((t3 - t2) * 1000)},
    }
