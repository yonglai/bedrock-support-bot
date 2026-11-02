import pytest

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
