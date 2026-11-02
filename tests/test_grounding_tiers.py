import pytest

import grounding_tiers_common as g


def test_content_blocks_mirror_the_agents_check():
    b = g.content_blocks("q?", "a.", ["[S1] x", "[S2] y"])
    assert b[0]["text"] == {
        "text": "[S1] x\n[S2] y",
        "qualifiers": ["grounding_source"],
    }
    assert b[1]["text"]["qualifiers"] == ["query"] and b[2]["text"]["text"] == "a."


def test_scores_reads_the_contextual_grounding_filters():
    resp = {
        "assessments": [
            {
                "contextualGroundingPolicy": {
                    "filters": [
                        {
                            "type": "GROUNDING",
                            "score": 0.94,
                            "threshold": 0.75,
                            "action": "NONE",
                        },
                        {
                            "type": "RELEVANCE",
                            "score": 1.0,
                            "threshold": 0.5,
                            "action": "NONE",
                        },
                    ]
                }
            }
        ]
    }
    assert g.scores(resp) == {"GROUNDING": 0.94, "RELEVANCE": 1.0}
    assert g.scores({"assessments": []}) == {}


def test_auroc_perfect_chance_and_ties():
    assert g.auroc([0.1, 0.2], [0.9, 0.95]) == 1.0
    assert g.auroc([0.9], [0.1]) == 0.0
    assert g.auroc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert g.auroc([0.4, 0.9], [0.8]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        g.auroc([], [0.5])
