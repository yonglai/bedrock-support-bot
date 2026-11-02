import json

import agent_bedrock
import bedrock_agent_common as common
import tools

STATE = {"agent_id": "A", "alias_id": "L"}


def chunk(t):
    return {"chunk": {"bytes": t.encode()}}


def kb_trace(*passages):
    refs = [
        {
            "content": {"text": p},
            "location": {"s3Location": {"uri": f"s3://b/policies/{d}"}},
        }
        for d, p in passages
    ]
    return {
        "trace": {
            "trace": {
                "orchestrationTrace": {
                    "observation": {
                        "knowledgeBaseLookupOutput": {"retrievedReferences": refs}
                    }
                }
            }
        }
    }


class FakeRT:
    def __init__(self, *streams):
        self.streams, self.calls = list(streams), []

    def invoke_agent(self, **kw):
        self.calls.append(kw)
        return {"completion": iter(self.streams.pop(0))}


def driver(rt):
    return agent_bedrock.AgentsDriver(tools.Session("C1001"), [], rt=rt, state=STATE)


def test_knowledge_base_passages_become_numbered_grounding_sources():
    rt = FakeRT(
        [
            kb_trace(("dental_coverage.md", "Annual dental maximum: $1,500")),
            chunk("$1,500 [S1]"),
        ]
    )
    d = driver(rt)
    assert d.reply("dental max?", force_tool=True) == "$1,500 [S1]"
    assert d.sess.sources == ["[S1] (dental_coverage.md) Annual dental maximum: $1,500"]
    assert d.sess.tools_called == ["knowledge_base"]
    assert (
        rt.calls[0]["enableTrace"] is True and rt.calls[0]["inputText"] == "dental max?"
    )


def test_returned_control_runs_the_function_in_our_session():
    ctl = {
        "returnControl": {
            "invocationId": "i1",
            "invocationInputs": [
                {
                    "functionInvocationInput": {
                        "actionGroup": "customer-tools",
                        "function": "calculate",
                        "parameters": [
                            {
                                "name": "expression",
                                "type": "string",
                                "value": "2500 - 2300",
                            }
                        ],
                    }
                }
            ],
        }
    }
    rt = FakeRT([kb_trace(("dental_coverage.md", "x")), ctl], [chunk("$200 left")])
    d = driver(rt)
    assert d.reply("how much left?") == "$200 left"
    state = rt.calls[1]["sessionState"]
    assert state["invocationId"] == "i1" and "inputText" not in rt.calls[1]
    body = state["returnControlInvocationResults"][0]["functionResult"]["responseBody"][
        "TEXT"
    ]["body"]
    assert body == "2500 - 2300 = 200"
    assert d.sess.tools_called == ["knowledge_base", "calculate"]


def test_one_nudge_when_the_agent_answers_without_searching():
    rt = FakeRT(
        [chunk("from memory")],
        [kb_trace(("dental_coverage.md", "p")), chunk("grounded")],
    )
    d = driver(rt)
    assert d.reply("dental max?", force_tool=True) == "grounded"
    assert rt.calls[1]["inputText"] == agent_bedrock.NUDGE
    rt2 = FakeRT([chunk("hello")])
    assert (
        driver(rt2).reply("thanks", force_tool=False) == "hello"
    )  # follow-ups are not nudged


def test_instruction_policies_and_function_schema():
    assert (
        "search_policy" not in common.INSTRUCTION
        and "knowledge base" in common.INSTRUCTION
    )
    assert {f["name"] for f in common.FUNCTIONS} == {
        "get_my_plan",
        "calculate",
        "escalate_to_human",
    }
    assert common.base_model_id(common.MODEL).startswith("anthropic.claude-haiku")
    pol = json.dumps(common.role_policy("123456789012", "us-east-1", "KB1"))
    assert "knowledge-base/KB1" in pol and '"Resource": "*"' not in pol
