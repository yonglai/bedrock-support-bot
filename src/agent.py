"""Step 5: the agent. Order of defences:
  1. rule-based triage (deterministic, fixed wording)  ->  2. input guardrail  ->  3. tool-using model loop
  ->  4. output check (grounding against the tool results, blocked content)  ->  retry once  ->  human handoff.

Run a question:  MODEL_ID=... uv run python src/agent.py C1001 "How much dental do I have left this year?"
"""

import os
import re
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
    "amounts used this year), calculate (exact arithmetic) and escalate_to_human. Rules: answer ONLY from tool results and cite policy sources like "
    "[S1]. Plans differ: when you quote a rate or limit, use the one for the customer's own plan (from get_my_plan) and never apply the other plan's figure to them. Use calculate for every sum, difference or remainder; never calculate in your head. State only facts from the tool results: no greetings, "
    "estimates, assumptions or general advice. You can only see the current customer's data and cannot look up anyone "
    "else; politely say so if asked. Mild frustration or a passing remark is not a reason to escalate: answer the question. Call escalate_to_human when the customer is seriously distressed, grieving, complaining about a specific problem, "
    "appealing a decision, asks for a person, or when the tools do not contain the answer, then tell the customer what "
    "the tool said about availability. Never give legal, medical or investment advice. Never ask for or repeat SINs, "
    "card numbers or passwords. Do not mention tool names to the customer."
)
# a paraphrase of the availability line counts as already stated; only append when none of these appear
AVAILABILITY_MARKERS = ("8am to 8pm", "available now")
# A reply that offers or promises a handoff must be backed by one (Q308: "I can connect you with a team member" while
# no ticket existed). The escalation policy says to transfer when the documents cannot answer.
PROMISE = re.compile(
    r"connect(?:ing)? you (?:with|to)|callback|team member (?:will|can) (?:contact|call|follow|help)",
    re.IGNORECASE,
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


class StepLimit(Exception):
    """The driver used all its model calls without producing a final answer."""


class ConverseDriver:
    """The hand-written tool loop on the Bedrock Converse API. A framework driver replaces only this class: triage,
    input guardrail, output check, retry and handoff stay in run_agent."""

    def __init__(self, sess, history, max_steps=6):
        self.sess, self.messages, self.max_steps = sess, list(history), max_steps

    def reply(self, prompt, force_tool=False):
        self.messages.append({"role": "user", "content": [{"text": prompt}]})
        for _ in range(self.max_steps):
            resp = rt.converse(
                modelId=MODEL_ID,
                system=[{"text": AGENT_SYSTEM}],
                messages=self.messages,
                toolConfig={
                    "tools": TOOL_SPECS,
                    **(
                        {"toolChoice": {"tool": {"name": "search_policy"}}}
                        if force_tool
                        else {}
                    ),
                },
                inferenceConfig={"maxTokens": 600, "temperature": 0.1},
            )
            force_tool = False
            msg = resp["output"]["message"]
            self.messages.append(msg)
            if resp["stopReason"] != "tool_use":
                return "".join(b.get("text", "") for b in msg["content"])
            results = []
            for block in msg["content"]:
                if "toolUse" in block:
                    use = block["toolUse"]
                    out = self.sess.call(use["name"], use["input"])
                    results.append(
                        {
                            "toolResult": {
                                "toolUseId": use["toolUseId"],
                                "content": [{"text": out}],
                            }
                        }
                    )
            self.messages.append({"role": "user", "content": results})
        raise StepLimit


def default_driver():
    """AGENT_DRIVER=strands runs the loop on the Strands SDK."""
    driver = os.environ.get("AGENT_DRIVER")
    if driver == "strands":
        from agent_strands import StrandsDriver

        return StrandsDriver
    return ConverseDriver


def run_agent(question, customer_id, history=None, max_steps=6, driver=None):
    if (
        os.environ.get("AGENT_REMOTE") == "agentcore"
    ):  # same code, running in the deployed AgentCore Runtime
        from agentcore_client import invoke_remote

        return invoke_remote(question, customer_id, history)
    t0 = time.perf_counter()
    sess = Session(customer_id)
    history = list(history or [])

    def done(status, text, clean=None, **extra):
        save_cache()
        return {
            "status": status,
            "answer": text,
            "tools_called": sess.tools_called,
            "sources": list(
                sess.sources
            ),  # kept for the verifier experiment (fictional data only)
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

    driver = (driver or default_driver())(sess, history, max_steps)
    # A reply with no tool call skips the grounding check (nothing to ground on) and the model then answers from
    # general knowledge (Q308 invented where to find an account manager). So the first call of a new conversation
    # must search the policies (any tool was not enough: Q210 satisfied it with get_my_plan alone and then answered from
    # memory); later calls and follow-up turns are free ("thanks" should not trigger a policy search).
    prompt, force_tool, retries = clean, not history, 1
    rejected = []  # drafts the output check refused; kept in the eval results for debugging only
    while True:
        try:
            text = driver.reply(prompt, force_tool)
        except StepLimit:
            rec = create_handoff(
                "cannot_answer", "Agent step limit reached.", customer_id
            )
            return done(
                "escalated",
                "I wasn't able to complete that, so I'm connecting you with a team member. "
                + rec["availability"],
                clean=clean,
                via="step_limit",
            )
        ok, why, scores = check_output(text, clean, sess.sources)
        if ok:
            if not sess.escalation and PROMISE.search(text):
                sess.escalation = create_handoff(
                    "cannot_answer",
                    "The reply offered a handoff without calling the tool.",
                    customer_id,
                )
            if sess.escalation and not any(m in text for m in AVAILABILITY_MARKERS):
                # the grounding retry rewrites to bullets and can drop the handoff notice, so state it in code
                text = (
                    f"{text.rstrip()}\n\nI've also asked a team member to follow up. "
                    f"{sess.escalation['availability']}"
                )
            status = "escalated" if sess.escalation else "answered"
            return done(
                status,
                text,
                clean=clean,
                via="tool" if sess.escalation else None,
                retried=retries == 0,
                grounding=scores,
                rejected=rejected,
            )
        rejected.append(text)
        if why == "grounding" and retries > 0:
            retries -= 1
            prompt, force_tool = RETRY_PROMPT, False
            continue
        if (
            sess.escalation
        ):  # the model already handed off: say so, and do not open a second ticket
            rec = sess.escalation
            text = "I'm connecting you with a team member who can look into this. "
        else:
            rec = create_handoff(
                "guardrail", f"Answer blocked by the output check ({why}).", customer_id
            )
            text = "I want to be sure you get accurate information, so I'm connecting you with a team member. "
        return done(
            "escalated",
            text + rec["availability"],
            clean=clean,
            via=why,
            retried=retries == 0,
            grounding=scores,
            rejected=rejected,
        )


if __name__ == "__main__":
    cid, q = sys.argv[1], " ".join(sys.argv[2:])
    r = run_agent(q, cid)
    for d in r.get("rejected", []):
        print(f"[rejected draft] {d}\n")
    print(r["answer"])
    print(
        f"\n[{r['status']} via={r.get('via')} tools={r['tools_called']} retried={r.get('retried')} "
        f"grounding={r.get('grounding')} {r['ms_total']} ms]"
    )
