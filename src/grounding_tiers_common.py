"""Helpers for comparing the Classic and Standard tiers of the Bedrock contextual grounding check (no AWS calls)."""

THRESHOLDS = {"GROUNDING": 0.75, "RELEVANCE": 0.5}  # same as the main guardrail
TIERS = ("CLASSIC", "STANDARD")


def content_blocks(question, answer, sources):
    """Same shape as agent.check_output: all sources in one grounding_source block, the question as the query."""
    return [
        {"text": {"text": "\n".join(sources), "qualifiers": ["grounding_source"]}},
        {"text": {"text": question, "qualifiers": ["query"]}},
        {"text": {"text": answer, "qualifiers": ["guard_content"]}},
    ]


def scores(resp):
    """{'GROUNDING': 0.94, 'RELEVANCE': 1.0} from an apply_guardrail response; {} if the check returned nothing."""
    out = {}
    for a in resp.get("assessments", []):
        for f in a.get("contextualGroundingPolicy", {}).get("filters", []):
            out[f["type"]] = f["score"]
    return out


def auroc(error_scores, clean_scores):
    """Chance that a wrong answer scores LOWER than a clean one (1.0 = perfect separation, 0.5 = no better than chance;
    ties count half). The grounding score is high for good answers, so a low score is the alarm."""
    if not error_scores or not clean_scores:
        raise ValueError("need at least one error score and one clean score")
    wins = sum(
        1.0 if e < c else 0.5 if e == c else 0.0
        for e in error_scores
        for c in clean_scores
    )
    return wins / (len(error_scores) * len(clean_scores))
