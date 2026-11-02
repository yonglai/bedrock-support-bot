import rag_common


def test_context_is_the_whole_parent_document():
    chunks = [
        c
        for c in rag_common.build_chunks("fixed_200_ov40")
        if c["doc"] == "group_health_overview.md"
    ]
    hit = next(c for c in chunks if "Paramedical" in c["text"])
    assert len(hit["text"]) <= 200 < len(hit["context"])
    # the model sees both plans' rates next to the paramedical limits, so it can pick the customer's own plan
    for needle in ("**Standard plan:**", "**Enhanced plan:**", "Paramedical services"):
        assert needle in hit["context"]


def test_unknown_strategy_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        rag_common.build_chunks("nope")
