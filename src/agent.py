"""Step 5: the agent. Order of defences:
  1. rule-based triage (deterministic, fixed wording)  ->  2. input guardrail  ->  3. tool-using model loop
  ->  4. output check (grounding against the tool results, blocked content)  ->  retry once  ->  human handoff.

Run a question:  MODEL_ID=... uv run python src/agent.py C1001 "How much dental do I have left this year?"
"""

import sys
import time

from handoff import create_handoff
from pipeline import GUARD, HANDOFF, MODEL_ID, _any_blocked, check_input, rt
from rag_common import save_cache
from tools import TOOL_SPECS, Session
from triage import REPLIES, triage

AGENT_SYSTEM = (
    "You are a customer-service assistant for Evergreen Life, a fictional insurer, talking to one authenticated "
    "customer. Tools: search_policy (plan rules and limits), get_my_plan (this customer's own plan, dependants and "
    "amounts used this year) and escalate_to_human. Rules: answer ONLY from tool results and cite policy sources like "
    "[S1]. Do arithmetic carefully and show the numbers. State only facts from the tool results: no greetings, "
    "estimates, assumptions or general advice. You can only see the current customer's data and cannot look up anyone "
    "else; politely say so if asked. Call escalate_to_human when the customer is distressed, grieving, complaining, "
    "appealing a decision, asks for a person, or when the tools do not contain the answer, then tell the customer what "
    "the tool said about availability. Never give legal, medical or investment advice. Never ask for or repeat SINs, "
    "card numbers or passwords. Do not mention tool names to the customer."
)
RETRY_PROMPT = (
    "Your previous answer included statements the tool results do not support. Rewrite it using only facts in the "
    "tool results, as short bullet points, each with a [S#] citation. No introduction or closing remarks."
)


def _grounding_blocked(assessments):
    return any(
        f.get("action") == "BLOCKED"
        for a in assessments
        for f in a.get("contextualGroundingPolicy", {}).get("filters", [])
    )


def _scores(assessments):
    return [
        f"{f['type']}={f.get('score')}"
        for a in assessments
        for f in a.get("contextualGroundingPolicy", {}).get("filters", [])
    ]


def check_output(text, question, sources):
    """Returns (ok, reason, scores). Anonymisation alone is not a failure: the authenticated customer
    may see their own details, and the model only ever receives that customer's data."""
    if sources:
        content = [
            {"text": {"text": "\n".join(sources), "qualifiers": ["grounding_source"]}},
            {"text": {"text": question, "qualifiers": ["query"]}},
            {"text": {"text": text, "qualifiers": ["guard_content"]}},
        ]
    else:
        content = [{"text": {"text": text}}]
    r = rt.apply_guardrail(
        guardrailIdentifier=GUARD["id"],
        guardrailVersion=GUARD["version"],
        source="OUTPUT",
        content=content,
    )
    assessments = r.get("assessments", [])
    if r["action"] == "NONE":
        return True, None, _scores(assessments)
    if _grounding_blocked(assessments):
        return False, "grounding", _scores(assessments)
    if _any_blocked(assessments):
        return False, "blocked_content", _scores(assessments)
    return True, None, _scores(assessments)


def run_agent(question, customer_id, history=None, max_steps=6):
    t0 = time.perf_counter()
    sess = Session(customer_id)
    history = list(history or [])

    def done(status, text, clean=None, **extra):
        save_cache()
        return {
            "status": status,
            "answer": text,
            "tools_called": sess.tools_called,
            "ms_total": round((time.perf_counter() - t0) * 1000),
            "history": history
            + [
                {"role": "user", "content": [{"text": clean or question}]},
                {"role": "assistant", "content": [{"text": text}]},
            ],
            **extra,
        }

    reason = triage(question)  # runs first, even before the guardrail
    if reason:
        rec = create_handoff(
            reason,
            f"Rule-based escalation ({reason}); message text not stored.",
            customer_id,
        )
        return done(
            "escalated",
            f"{REPLIES[reason]} {rec['availability']}",
            clean="[escalated before processing]",
            via="rule",
            reason=reason,
        )

    verdict, clean = check_input(question)
    if verdict == "blocked":
        rec = create_handoff(
            "guardrail",
            "Message blocked by the input guardrail; text not stored.",
            customer_id,
        )
        return done(
            "blocked",
            f"{HANDOFF} {rec['availability']}",
            clean="[blocked by guardrail]",
            via="guardrail",
        )

    messages = history + [{"role": "user", "content": [{"text": clean}]}]
    retries = 1
    for _ in range(max_steps):
        resp = rt.converse(
            modelId=MODEL_ID,
            system=[{"text": AGENT_SYSTEM}],
            messages=messages,
            toolConfig={"tools": TOOL_SPECS},
            inferenceConfig={"maxTokens": 600, "temperature": 0.1},
        )
        msg = resp["output"]["message"]
        messages.append(msg)
        if resp["stopReason"] == "tool_use":
            results = []
            for block in msg["content"]:
                if "toolUse" in block:
                    use = block["toolUse"]
                    out = sess.call(use["name"], use["input"])
                    results.append(
                        {
                            "toolResult": {
                                "toolUseId": use["toolUseId"],
                                "content": [{"text": out}],
                            }
                        }
                    )
            messages.append({"role": "user", "content": results})
            continue

        text = "".join(b.get("text", "") for b in msg["content"])
        ok, why, scores = check_output(text, clean, sess.sources)
        if ok:
            status = "escalated" if sess.escalation else "answered"
            return done(
                status,
                text,
                clean=clean,
                via="tool" if sess.escalation else None,
                retried=retries == 0,
                grounding=scores,
            )
        if why == "grounding" and retries > 0:
            retries -= 1
            messages.append({"role": "user", "content": [{"text": RETRY_PROMPT}]})
            continue
        rec = create_handoff(
            "guardrail", f"Answer blocked by the output check ({why}).", customer_id
        )
        return done(
            "escalated",
            "I want to be sure you get accurate information, so I'm connecting you with a team member. "
            + rec["availability"],
            clean=clean,
            via=why,
            retried=retries == 0,
            grounding=scores,
        )

    rec = create_handoff("cannot_answer", "Agent step limit reached.", customer_id)
    return done(
        "escalated",
        "I wasn't able to complete that, so I'm connecting you with a team member. "
        + rec["availability"],
        clean=clean,
        via="step_limit",
    )


if __name__ == "__main__":
    cid, q = sys.argv[1], " ".join(sys.argv[2:])
    r = run_agent(q, cid)
    print(r["answer"])
    print(
        f"\n[{r['status']} via={r.get('via')} tools={r['tools_called']} retried={r.get('retried')} "
        f"grounding={r.get('grounding')} {r['ms_total']} ms]"
    )
