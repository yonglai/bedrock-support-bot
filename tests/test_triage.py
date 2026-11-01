import json
import pathlib

from triage import triage

EVAL = pathlib.Path(__file__).resolve().parent.parent / "eval"


def load(name):
    return [
        json.loads(line)
        for line in (EVAL / name).read_text().splitlines()
        if line.strip()
    ]


def test_escalation_set_rules():
    for q in load("questions_escalation.jsonl"):
        hit = triage(q["question"])
        if (
            q["id"] == 208
        ):  # "waiting six weeks": deliberately left to the model's escalate tool
            assert hit is None
        elif q["expect_escalate"]:
            assert hit == q["reason"], (q["id"], hit)
        else:
            assert hit is None, (q["id"], hit)


def test_no_over_triggering_on_eval_questions():
    for name in ("questions.jsonl", "questions_heldout.jsonl"):
        for q in load(name):
            hit = triage(q["question"])
            if q["id"] == 115:  # bereavement
                assert hit == "bereavement"
            else:
                assert hit is None, (q["id"], hit)
