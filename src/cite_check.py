"""Citation check (experiment): is each cited statement entailed by ONLY the passages it cites? ALCE-style "citation recall".
Unlike the claim-level verifier (verify.py), this sees just the cited passages, never the whole source list or the customer's
record, and it does not ask whether a figure applies to this customer. Statements without a [S#] marker are not judged; they
are counted so the coverage is visible.
"""

import json
import re

from pipeline import MODEL_ID, rt

MARKER = re.compile(r"\[S(\d+)\]")
SOURCE_START = re.compile(r"(?m)^\[S(\d+)\] ")
SYSTEM = (
    "You check citations. You get numbered STATEMENTS, each with the PASSAGES it cites. For each statement decide whether "
    "the cited passages, and nothing else, state it or directly imply it. A number, percentage or time period in the "
    "statement must appear in the cited passages for the same thing the statement says it is for. Reply with JSON only: "
    '{"results": [{"i": 0, "supported": true or false, "reason": "short quote or why not"}]}'
)


def parse_sources(sources):
    """{3: 'text of [S3]'} from the source strings, which may hold several [S#] blocks each."""
    text = "\n".join(sources)
    parts = SOURCE_START.split(text)  # ['', '1', 'text1', '2', 'text2', ...]
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def statements(answer):
    """[(sentence, [cited numbers])]. A fragment made only of markers belongs to the sentence before it."""
    pieces = [p for p in re.split(r"(?<=[.!?])\s+|\n+", answer) if p.strip()]
    out = []
    for p in pieces:
        nums = [int(n) for n in MARKER.findall(p)]
        if not MARKER.sub("", p).strip(" .-*\u2022") and out:  # markers only
            out[-1][1].extend(nums)
            continue
        out.append([p.strip(), nums])
    return [(s, list(dict.fromkeys(n))) for s, n in out]


def parse(text):
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the reply")
    return json.loads(text[a : b + 1])["results"]


def check(answer, sources):
    """Returns {'cited': n, 'uncited': n, 'unsupported': [(statement, reason)], 'missing': [...]}; raises ValueError on a bad reply."""
    passages = parse_sources(sources)
    stmts = statements(answer)
    judged, missing, uncited = [], [], 0
    for s, nums in stmts:
        if not nums:
            uncited += 1
            continue
        have = [n for n in nums if n in passages]
        if not have:  # cites a number that is not a source
            missing.append(s)
            continue
        judged.append((s, have))
    unsupported = []
    if judged:
        body = "\n\n".join(
            f"STATEMENT {i}: {s}\nPASSAGES:\n"
            + "\n".join(f"[S{n}] {passages[n]}" for n in have)
            for i, (s, have) in enumerate(judged)
        )
        resp = rt.converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"text": body}]}],
            inferenceConfig={"maxTokens": 1500, "temperature": 0},
        )
        results = parse(resp["output"]["message"]["content"][0]["text"])
        for r in results:
            if not r.get("supported", True):
                unsupported.append((judged[r["i"]][0], r.get("reason", "")))
    return {
        "cited": len(judged),
        "uncited": uncited,
        "unsupported": unsupported,
        "missing": missing,
    }
