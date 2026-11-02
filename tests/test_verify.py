import pytest

import verify as verify_mod
from mutate import mutations
from verify import parse


def test_parse_finds_json_inside_chatty_text():
    assert parse('Sure.\n{"claims": []}\nDone.') == {"claims": []}
    with pytest.raises(ValueError):
        parse("no json here")


def test_mutations_change_a_figure_to_one_not_already_in_the_answer():
    kinds = dict(mutations("Your Standard plan pays 80% up to $500 [S1]."))
    assert kinds["percent"] == "Your Standard plan pays 50% up to $500 [S1]."
    assert kinds["amount"] == "Your Standard plan pays 80% up to $100 [S1]."
    assert kinds["plan"] == "Your Enhanced plan pays 80% up to $500 [S1]."


def test_no_plan_swap_when_both_plans_are_named():
    kinds = dict(mutations("Standard is 80% and Enhanced is 90%."))
    assert "plan" not in kinds


def _fake_converse(seen):
    def converse(**kw):
        seen.append(kw["modelId"])
        return {"output": {"message": {"content": [{"text": '{"claims": []}'}]}}}

    return converse


def test_verifier_model_is_argument_then_env_then_agent_model(monkeypatch):
    seen = []
    monkeypatch.setattr(verify_mod.rt, "converse", _fake_converse(seen))
    monkeypatch.setattr(verify_mod, "MODEL_ID", "agent-model")
    monkeypatch.delenv("VERIFIER_MODEL_ID", raising=False)
    verify_mod.verify("q", "a", ["s"])
    monkeypatch.setenv("VERIFIER_MODEL_ID", "env-model")
    verify_mod.verify("q", "a", ["s"])
    verify_mod.verify("q", "a", ["s"], model_id="arg-model")
    assert seen == ["agent-model", "env-model", "arg-model"]
