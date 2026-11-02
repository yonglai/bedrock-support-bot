"""Pure builders for the Knowledge Base experiment (no AWS calls, unit-tested offline)."""

TITAN = "amazon.titan-embed-text-v2:0"
DIMS = 512

# One Knowledge Base per chunking variant: a KB has one vector index, and mixing chunk sets in it would blur the comparison.
VARIANTS = {
    "default": None,  # Bedrock's own default (fixed size, about 300 tokens)
    "none": {"chunkingConfiguration": {"chunkingStrategy": "NONE"}},
    "fixed_200": {
        "chunkingConfiguration": {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 200,
                "overlapPercentage": 20,
            },
        }
    },
    "hierarchical": {  # small child chunks are searched, the larger parent is returned
        "chunkingConfiguration": {
            "chunkingStrategy": "HIERARCHICAL",
            "hierarchicalChunkingConfiguration": {
                "levelConfigurations": [{"maxTokens": 300}, {"maxTokens": 60}],
                "overlapTokens": 15,
            },
        }
    },
}


def ingestion_config(variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    return VARIANTS[variant]


def names(account, region, variant):
    slug = variant.replace("_", "-")
    return {
        "bucket": f"evergreen-kb-{account}-{region}",
        "vector_bucket": f"evergreen-vec-{account}-{region}",
        "role": "evergreen-kb-role",
        "kb": f"evergreen-kb-{slug}",
        "index": f"policies-{slug}",
    }


def trust_policy(account, region):
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock:{region}:{account}:knowledge-base/*"
                    },
                },
            }
        ],
    }


def role_policy(account, region, bucket, vector_bucket_arn):
    """Least privilege for this one KB. If ingestion reports AccessDenied, add the action the error names."""
    owner = {"StringEquals": {"aws:ResourceAccount": account}}
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": "bedrock:InvokeModel",
                "Resource": f"arn:aws:bedrock:{region}::foundation-model/{TITAN}",
            },
            {
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": f"arn:aws:s3:::{bucket}",
                "Condition": owner,
            },
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket}/*",
                "Condition": owner,
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3vectors:GetIndex",
                    "s3vectors:QueryVectors",
                    "s3vectors:PutVectors",
                    "s3vectors:GetVectors",
                    "s3vectors:DeleteVectors",
                ],
                "Resource": [vector_bucket_arn, f"{vector_bucket_arn}/index/*"],
            },
        ],
    }
