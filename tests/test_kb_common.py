import json

import pytest

import kb_common as kb


def test_variants_and_names():
    assert kb.ingestion_config("default") is None
    assert (
        kb.ingestion_config("none")["chunkingConfiguration"]["chunkingStrategy"]
        == "NONE"
    )
    levels = kb.ingestion_config("hierarchical")["chunkingConfiguration"][
        "hierarchicalChunkingConfiguration"
    ]["levelConfigurations"]
    assert levels[0]["maxTokens"] > levels[1]["maxTokens"]  # parent larger than child
    with pytest.raises(ValueError):
        kb.ingestion_config("typo")
    n = kb.names("123456789012", "us-east-1", "fixed_200")
    assert n["kb"] == "evergreen-kb-fixed-200" and n["index"] == "policies-fixed-200"


def test_policies_are_scoped_and_valid_json():
    vb = "arn:aws:s3vectors:us-east-1:123456789012:bucket/evergreen-vec"
    pol = kb.role_policy("123456789012", "us-east-1", "docs", vb)
    text = json.dumps(pol)
    assert (
        "foundation-model/amazon.titan-embed-text-v2:0" in text
        and "arn:aws:s3:::docs/*" in text
    )
    assert f"{vb}/index/*" in text and '"Resource": "*"' not in text
    trust = kb.trust_policy("123456789012", "us-east-1")["Statement"][0]
    assert trust["Principal"]["Service"] == "bedrock.amazonaws.com"
    assert trust["Condition"]["StringEquals"]["aws:SourceAccount"] == "123456789012"
