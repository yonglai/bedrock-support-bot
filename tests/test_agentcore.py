import io
import json

import pytest

import agent
import agentcore_app
import agentcore_client
import agentcore_common as common


def test_handle_runs_the_agent_and_returns_json_safe_result(monkeypatch):
    seen = {}

    def fake(q, cid, history):
        seen.update(q=q, cid=cid, history=history)
        return {
            "status": "answered",
            "answer": "hi",
            "grounding": {"x": 0.9},
            "odd": {1, 2}.__class__,
        }

    monkeypatch.setattr(agent, "run_agent", fake)
    out = agentcore_app.handle({"question": "q", "customer_id": "C1001"})
    assert out["answer"] == "hi" and seen == {"q": "q", "cid": "C1001", "history": []}
    json.dumps(out)


def test_handle_rejects_incomplete_payload():
    with pytest.raises(ValueError):
        agentcore_app.handle({"question": "q"})


class FakeClient:
    def __init__(self, body):
        self.body, self.kw = body, None

    def invoke_agent_runtime(self, **kw):
        self.kw = kw
        return {"response": io.BytesIO(json.dumps(self.body).encode())}


def test_client_sends_payload_with_long_session_id():
    c = FakeClient({"status": "answered", "answer": "ok"})
    r = agentcore_client.invoke_remote(
        "q", "C1001", [{"role": "user"}], client=c, state={"runtime_arn": "arn:x"}
    )
    assert r["answer"] == "ok" and c.kw["agentRuntimeArn"] == "arn:x"
    assert len(c.kw["runtimeSessionId"]) >= 33
    assert json.loads(c.kw["payload"])["customer_id"] == "C1001"


def test_client_surfaces_runtime_errors():
    c = FakeClient({"error": "boom"})
    with pytest.raises(RuntimeError, match="boom"):
        agentcore_client.invoke_remote("q", "C1", client=c, state={"runtime_arn": "a"})


def test_policies_and_env():
    pol = json.dumps(
        common.role_policy(
            "123456789012",
            "us-east-1",
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "arn:g",
            "KB1",
            "b",
        )
    )
    assert (
        "knowledge-base/KB1" in pol
        and "foundation-model/anthropic.claude-haiku" in pol
        and '"Resource": "*"' not in pol
    )
    assert (
        common.trust_policy("1", "us-east-1")["Statement"][0]["Principal"]["Service"]
        == "bedrock-agentcore.amazonaws.com"
    )
    with pytest.raises(SystemExit):
        common.runtime_env({})
    env = common.runtime_env({"MODEL_ID": "m"})
    assert env["RETRIEVER"] == "kb" and env["CACHE_DIR"] == "/tmp/cache"
    assert (
        "agentcore_app.py" in common.SRC_FILES
        and "step5_eval_agent.py" not in common.SRC_FILES
    )
