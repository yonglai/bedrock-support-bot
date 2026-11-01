"""Step 4b: find out WHY the grounding check scores an answer low.
Generates the answer without the guardrail, then scores it against the retrieved sources with apply_guardrail:
the whole answer, each sentence alone, the answer without [n] citation markers, and the answer with only the
relevant source lines. Makes a handful of cheap calls and no changes.

Run:  MODEL_ID=... uv run python src/step4b_debug_grounding.py ["question"]
"""

import re
import sys

from pipeline import GUARD, MODEL_ID, SYSTEM, check_input, retrieve, rt

QUESTION = (
    " ".join(sys.argv[1:])
    or "My daughter is 12 and needs braces and new glasses. What will I get back?"
)


def score(sources_text, question, answer):
    r = rt.apply_guardrail(
        guardrailIdentifier=GUARD["id"],
        guardrailVersion=GUARD["version"],
        source="OUTPUT",
        content=[
            {"text": {"text": sources_text, "qualifiers": ["grounding_source"]}},
            {"text": {"text": question, "qualifiers": ["query"]}},
            {"text": {"text": answer, "qualifiers": ["guard_content"]}},
        ],
    )
    out = {}
    for a in r.get("assessments", []):
        for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
            out[f["type"]] = f["score"]
    return out.get("GROUNDING"), out.get("RELEVANCE")


status, clean = check_input(QUESTION)
sources = retrieve(clean)
src_text = "\n".join(
    f"[{i + 1}] ({s['doc']}) {s['text']}" for i, s in enumerate(sources)
)
print("SOURCES GIVEN TO THE MODEL:\n" + src_text + "\n")

resp = rt.converse(
    modelId=MODEL_ID,
    system=[{"text": SYSTEM}],
    messages=[
        {
            "role": "user",
            "content": [{"text": f"Sources:\n{src_text}\n\nQuestion: {clean}"}],
        }
    ],
    inferenceConfig={"maxTokens": 400, "temperature": 0.1},
)
answer = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
print("UNGUARDED ANSWER:\n" + answer + "\n")

print(f"{'variant':34} grounding  relevance")
g, rel = score(src_text, clean, answer)
print(f"{'whole answer':34} {g}  {rel}")
no_cites = re.sub(r"\s*\[\d+\]", "", answer)
g, rel = score(src_text, clean, no_cites)
print(f"{'whole answer, no [n] markers':34} {g}  {rel}")
for i, sent in enumerate(
    s for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if len(s.strip()) > 3
):
    g, rel = score(src_text, clean, sent.strip())
    print(f"sentence {i + 1}: {sent.strip()[:60]!r:44} {g}  {rel}")
