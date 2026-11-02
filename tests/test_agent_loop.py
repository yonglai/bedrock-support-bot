import json
from datetime import datetime

import agent
import handoff
import tools

GROUND_BLOCK = {
    "action": "GUARDRAIL_INTERVENED",
    "assessments": [
        {
            "contextualGroundingPolicy": {
                "filters": [
                    {
                        "type": "GROUNDING",
                        "score": 0.5,
                        "threshold": 0.75,
                        "action": "BLOCKED",
                    }
                ]
            }
        }
    ],
}
OK = {"action": "NONE", "assessments": []}


def tool_use(name, args, tid="t1"):
    block = {"toolUse": {"toolUseId": tid, "name": name, "input": args}}
    return {
        "output": {"message": {"role": "assistant", "content": [block]}},
        "stopReason": "tool_use",
    }


def text(t):
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": t}]}},
        "stopReason": "end_turn",
    }


class FakeRT:
    def __init__(self, converse, guard):
        self.c, self.g, self.converse_calls, self.guard_calls = (
            list(converse),
            list(guard),
            [],
            [],
        )
        self.tool_configs = []

    def converse(self, **kw):
        self.converse_calls.append(json.loads(json.dumps(kw["messages"])))
        self.tool_configs.append(kw["toolConfig"])
        assert self.c, "unexpected extra converse call"
        return self.c.pop(0)

    def apply_guardrail(self, **kw):
        self.guard_calls.append(kw)
        return self.g.pop(0)


def setup(monkeypatch, tmp_path, converse=(), guard=()):
    fake = FakeRT(converse, guard)
    monkeypatch.setattr(agent, "rt", fake)
    monkeypatch.setattr(agent, "check_input", lambda t: ("ok", t))
    monkeypatch.setattr(agent, "save_cache", lambda: None)
    monkeypatch.setattr(
        tools,
        "retrieve",
        lambda q, k: [
            {
                "doc": "dental_coverage.md",
                "text": "**Annual dental maximum:** $1,500 on Standard",
            }
        ],
    )
    monkeypatch.setattr(handoff, "LOG", tmp_path / "handoffs.jsonl")
    return fake


def log(tmp_path):
    p = tmp_path / "handoffs.jsonl"
    return (
        [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    )


def test_two_tools_then_answer(monkeypatch, tmp_path):
    fake = setup(
        monkeypatch,
        tmp_path,
        converse=[
            tool_use("get_my_plan", {}, "a"),
            tool_use("search_policy", {"query": "dental max"}, "b"),
            text("You have $600 left [S1]."),
        ],
        guard=[OK],
    )
    r = agent.run_agent("How much dental do I have left?", "C1001")
    assert r["status"] == "answered" and r["tools_called"] == [
        "get_my_plan",
        "search_policy",
    ]
    last_user = fake.converse_calls[1][-1]["content"][0]["toolResult"]["content"][0][
        "text"
    ]
    assert "Maria Chen" in last_user and "Okafor" not in last_user
    source = fake.guard_calls[0]["content"][0]["text"]
    assert (
        "grounding_source" in source["qualifiers"]
        and "$1,500" in source["text"]
        and "used_this_year" in source["text"]
    )
    assert r["history"][-1]["content"][0]["text"] == "You have $600 left [S1]."


def test_model_cannot_pick_another_customer():
    s = tools.Session("C1001")
    out = s.call("get_my_plan", {"customer_id": "C1002"})
    assert "Maria Chen" in out and "Okafor" not in out


def test_unknown_tool_and_bad_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(handoff, "LOG", tmp_path / "h.jsonl")
    s = tools.Session("C1001")
    assert "Unknown tool" in s.call("delete_everything", {})
    s.call("escalate_to_human", {"reason": "made_up", "summary": "x"})
    assert s.escalation["reason"] == "other"


def test_grounding_block_then_retry_succeeds(monkeypatch, tmp_path):
    fake = setup(
        monkeypatch,
        tmp_path,
        converse=[text("plausible extra claim"), text("- $600 [S1]")],
        guard=[GROUND_BLOCK, OK],
    )
    r = agent.run_agent("dental left?", "C1001")
    assert (
        r["status"] == "answered"
        and r["retried"] is True
        and r["answer"] == "- $600 [S1]"
    )
    assert fake.converse_calls[1][-1]["content"][0]["text"] == agent.RETRY_PROMPT


def test_grounding_blocked_twice_hands_off(monkeypatch, tmp_path):
    setup(
        monkeypatch,
        tmp_path,
        converse=[text("a"), text("b")],
        guard=[GROUND_BLOCK, GROUND_BLOCK],
    )
    r = agent.run_agent("dental left?", "C1001")
    assert r["status"] == "escalated" and r["via"] == "grounding"
    assert log(tmp_path)[0]["reason"] == "guardrail"


def test_crisis_short_circuits_with_fixed_text(monkeypatch, tmp_path):
    fake = setup(monkeypatch, tmp_path)
    msg = "I don't see the point in anything anymore and I've been thinking about ending my life."
    r = agent.run_agent(msg, "C1001")
    assert (
        fake.converse_calls == [] and fake.guard_calls == []
    )  # no model, no guardrail call
    assert (
        r["status"] == "escalated"
        and r["reason"] == "crisis"
        and "9-8-8" in r["answer"]
        and "911" in r["answer"]
    )
    rec = log(tmp_path)[0]
    assert rec["reason"] == "crisis" and "ending my life" not in json.dumps(rec)
    assert "ending my life" not in json.dumps(r["history"])


def test_blocked_input_creates_a_real_handoff(monkeypatch, tmp_path):
    setup(monkeypatch, tmp_path)
    monkeypatch.setattr(agent, "check_input", lambda t: ("blocked", "blocked"))
    r = agent.run_agent("card 4111 1111 1111 1111", "C1001")
    assert r["status"] == "blocked" and log(tmp_path)[0]["reason"] == "guardrail"
    assert "4111" not in json.dumps(r["history"])


def test_availability_is_honest():
    tz = handoff.TZ
    live, line = handoff.availability(
        datetime(2026, 11, 4, 10, 0, tzinfo=tz)
    )  # Wednesday 10am
    assert live and "available now" in line
    live, line = handoff.availability(
        datetime(2026, 10, 31, 22, 0, tzinfo=tz)
    )  # Saturday 10pm
    assert not live and "callback" in line


def test_calculate_is_exact_and_safe():
    s = tools.Session("C1001")
    assert s.call("calculate", {"expression": "$2,500 - 2,300"}) == "2500 - 2300 = 200"
    assert s.call("calculate", {"expression": "1500 * 0.8"}) == "1500 * 0.8 = 1200"
    assert "Tool error" in s.call("calculate", {"expression": "__import__('os')"})
    assert "Tool error" in s.call("calculate", {"expression": "2 ** 99999999"})
    assert "Tool error" in s.call("calculate", {"expression": "1 / 0"})


def test_handoff_notice_survives_the_grounding_retry(monkeypatch, tmp_path):
    fake = setup(
        monkeypatch,
        tmp_path,
        converse=[
            tool_use("escalate_to_human", {"reason": "complaint", "summary": "x"}),
            text("unsupported extra claim"),
            text("- Nothing found [S1]"),
        ],
        guard=[GROUND_BLOCK, OK],
    )
    monkeypatch.setattr(
        handoff, "availability", lambda now=None: (False, "AVAIL LINE.")
    )
    r = agent.run_agent("Why was my claim late?", "C1001")
    assert r["status"] == "escalated" and r["retried"] is True
    assert r["answer"].endswith("AVAIL LINE.")  # the customer is always told
    assert (
        "AVAIL LINE." in fake.guard_calls[0]["content"][0]["text"]["text"]
    )  # and it counts as a grounding source


def test_first_call_forces_a_tool_and_follow_ups_do_not(monkeypatch, tmp_path):
    fake = setup(
        monkeypatch,
        tmp_path,
        converse=[
            tool_use("search_policy", {"query": "dental"}),
            text("$1,500 [S1]"),
            text("You're welcome."),
        ],
        guard=[OK, OK],
    )
    r = agent.run_agent("dental max?", "C1001")
    assert fake.tool_configs[0]["toolChoice"] == {"tool": {"name": "search_policy"}}
    assert (
        "toolChoice" not in fake.tool_configs[1]
    )  # after a tool result the model is free to answer
    agent.run_agent("thanks", "C1001", history=r["history"])
    assert "toolChoice" not in fake.tool_configs[2]


def test_paraphrased_availability_is_not_repeated(monkeypatch, tmp_path):
    setup(
        monkeypatch,
        tmp_path,
        converse=[
            tool_use("escalate_to_human", {"reason": "complaint", "summary": "x"}),
            text("Our hours are 8am to 8pm Eastern; someone will call you."),
        ],
        guard=[OK],
    )
    r = agent.run_agent("Where is my claim?", "C1001")
    assert r["answer"].count("8am to 8pm") == 1


def test_blocked_draft_after_a_model_handoff_opens_no_second_ticket(
    monkeypatch, tmp_path
):
    setup(
        monkeypatch,
        tmp_path,
        converse=[
            tool_use("escalate_to_human", {"reason": "complaint", "summary": "x"}),
            text("a"),
            text("b"),
        ],
        guard=[GROUND_BLOCK, GROUND_BLOCK],
    )
    r = agent.run_agent("Where is my claim?", "C1001")
    assert r["status"] == "escalated" and "accurate information" not in r["answer"]
    assert len(log(tmp_path)) == 1  # one ticket, not two


def test_an_offered_handoff_is_backed_by_a_real_ticket(monkeypatch, tmp_path):
    setup(
        monkeypatch,
        tmp_path,
        converse=[
            tool_use("search_policy", {"query": "account manager"}),
            text("I can connect you with a team member who can help."),
        ],
        guard=[OK],
    )
    monkeypatch.setattr(
        handoff, "availability", lambda now=None: (False, "AVAIL 8am to 8pm.")
    )
    r = agent.run_agent("Who is my account manager?", "C1001")
    assert r["status"] == "escalated" and len(log(tmp_path)) == 1
    assert log(tmp_path)[0]["reason"] == "cannot_answer" and "AVAIL" in r["answer"]


def test_a_normal_answer_creates_no_ticket(monkeypatch, tmp_path):
    setup(
        monkeypatch,
        tmp_path,
        converse=[tool_use("search_policy", {"query": "dental"}), text("$1,500 [S1].")],
        guard=[OK],
    )
    r = agent.run_agent("dental max?", "C1001")
    assert r["status"] == "answered" and log(tmp_path) == []
