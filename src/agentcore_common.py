"""Pure builders for the AgentCore Runtime deployment (no AWS calls; unit-tested offline)."""

ROLE_NAME = "AgentCoreRuntimeRole_evergreen"
RUNTIME_NAME = "evergreen_support_agent"  # letters, digits, underscores
# The modules the agent needs at run time (not the step scripts or experiments)
SRC_FILES = [
    "agent.py", "agent_strands.py", "agentcore_app.py", "config.py", "handoff.py", "oss.py",
    "pipeline.py", "rag_common.py", "tools.py", "triage.py",
]  # fmt: skip
STATE_FILES = [".guardrail.json", ".kb.json"]  # ids only, no secrets
PACKAGES = ["bedrock-agentcore", "strands-agents", "boto3", "opensearch-py"]


def base_model_id(profile_id):
    return (
        profile_id.split(".", 1)[1]
        if profile_id.startswith(("us.", "eu.", "apac.", "global."))
        else profile_id
    )


def bucket_name(account, region):
    return f"evergreen-agentcore-{account}-{region}"


def trust_policy(account, region):
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account},
                    "ArnLike": {
                        "aws:SourceArn": f"arn:aws:bedrock-agentcore:{region}:{account}:*"
                    },
                },
            }
        ],
    }


def role_policy(account, region, model_id, guardrail_arn, kb_id, bucket):
    """What the code inside the runtime calls: the model, the guardrail, the knowledge base, its own logs, its zip."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region}:{account}:inference-profile/{model_id}",
                    f"arn:aws:bedrock:*::foundation-model/{base_model_id(model_id)}",
                ],
            },
            {
                "Effect": "Allow",
                "Action": "bedrock:ApplyGuardrail",
                "Resource": guardrail_arn,
            },
            {
                "Effect": "Allow",
                "Action": "bedrock:Retrieve",
                "Resource": f"arn:aws:bedrock:{region}:{account}:knowledge-base/{kb_id}",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                "Resource": f"arn:aws:logs:{region}:{account}:log-group:/aws/bedrock-agentcore/runtimes/*",
            },
            {
                "Effect": "Allow",
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
        ],
    }


def runtime_env(env):
    """Environment variables the runtime needs, taken from the deployer's shell. Fails early if MODEL_ID is missing."""
    if not env.get("MODEL_ID"):
        raise SystemExit("set MODEL_ID first (same value as for the local runs)")
    out = {
        "AWS_REGION": env.get("AWS_REGION", "us-east-1"),
        "MODEL_ID": env["MODEL_ID"],
        "RETRIEVER": "kb",
        "VARIANT": env.get("VARIANT", "hierarchical"),
        "AGENT_DRIVER": "strands",
        "CACHE_DIR": "/tmp/cache",
    }
    if env.get("EMBED_MODEL_ID"):
        out["EMBED_MODEL_ID"] = env["EMBED_MODEL_ID"]
    return out
