import asyncio

import pytest

pytest.importorskip("strands")

from strands.models import BedrockModel

import agent_strands
import handoff
import tools


def test_tools_match_the_hand_written_ones_and_share_the_session(monkeypatch, tmp_path):
    monkeypatch.setattr(handoff, "LOG", tmp_path / "h.jsonl")
    sess = tools.Session("C1001")
    made = {t.tool_spec["name"]: t for t in agent_strands.make_tools(sess)}
    assert set(made) == {s["toolSpec"]["name"] for s in tools.TOOL_SPECS}
    # every tool needs a real description; the model picks tools from it
    by_name = {s["toolSpec"]["name"]: s["toolSpec"] for s in tools.TOOL_SPECS}
    for name, t in made.items():
        assert t.tool_spec["description"] != name, f"{name} has no description"
        assert t.tool_spec["description"][:30] == by_name[name]["description"][:30]
    for (
        reason
    ) in handoff.REASONS:  # the docstring must list every reason the handoff accepts
        assert (
            reason
            in made["escalate_to_human"].tool_spec["inputSchema"]["json"]["properties"][
                "reason"
            ]["description"]
        )
    assert (
        made["get_my_plan"].tool_spec["inputSchema"]["json"].get("required", []) == []
    )
    assert "Maria Chen" in made["get_my_plan"]()
    assert made["calculate"](expression="2500 - 2300") == "2500 - 2300 = 200"
    made["escalate_to_human"](reason="complaint", summary="waiting six weeks")
    assert sess.escalation["reason"] == "complaint"
    assert sess.tools_called == ["get_my_plan", "calculate", "escalate_to_human"]
    assert len(sess.sources) == 3  # same grounding sources as the hand-written loop


def test_first_model_call_forces_search_policy_once(monkeypatch):
    seen = []

    async def fake_stream(
        self, messages, tool_specs=None, system_prompt=None, *, tool_choice=None, **kw
    ):
        seen.append(tool_choice)
        yield {}

    monkeypatch.setattr(BedrockModel, "stream", fake_stream)
    model = agent_strands.ForcedFirstToolModel(model_id="x", region_name="us-east-1")
    model.force_tool = "search_policy"

    async def run():
        for _ in range(2):
            async for _ in model.stream([], [], ""):
                pass

    asyncio.run(run())
    assert seen == [{"tool": {"name": "search_policy"}}, None]
