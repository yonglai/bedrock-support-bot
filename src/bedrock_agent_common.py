"""Pure builders for the Bedrock Agents experiment (no AWS calls, unit-tested offline)."""

from agent import AGENT_SYSTEM
from handoff import REASONS

ROLE_NAME = "AmazonBedrockExecutionRoleForAgents_evergreen"  # Bedrock expects this prefix for agent roles
MODEL = (
    "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # inference profile, as for Converse
)

# The managed agent searches the attached knowledge base itself, so there is no search_policy tool to name.
INSTRUCTION = AGENT_SYSTEM.replace(
    "search_policy (plan rules and limits)",
    "the attached policy knowledge base (plan rules and limits)",
)

FUNCTIONS = [
    {
        "name": "get_my_plan",
        "description": "Get the current customer's own plan, dependants and amounts used this year. Takes no arguments; it only ever returns the customer you are talking to.",
        "parameters": {},
    },
    {
        "name": "calculate",
        "description": "Exact arithmetic for amounts, e.g. '1500 - 900'. Use it for every sum, difference, percentage or remainder instead of calculating yourself.",
        "parameters": {
            "expression": {
                "type": "string",
                "description": "Numbers with + - * / and brackets",
                "required": True,
            }
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human team member. Use when the customer is distressed, grieving, complaining, appealing a decision, asks for a person, or when the tools cannot answer the question.",
        "parameters": {
            "reason": {
                "type": "string",
                "description": "One of " + ", ".join(REASONS),
                "required": True,
            },
            "summary": {
                "type": "string",
                "description": "One or two sentences for the team member, with no personal identifiers",
                "required": True,
            },
        },
    },
]


def base_model_id(profile_id):
    """us.anthropic.claude-haiku-... -> anthropic.claude-haiku-..."""
    return (
        profile_id.split(".", 1)[1]
        if profile_id.startswith(("us.", "eu.", "apac.", "global."))
        else profile_id
    )


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
                        "aws:SourceArn": f"arn:aws:bedrock:{region}:{account}:agent/*"
                    },
                },
            }
        ],
    }


def role_policy(account, region, kb_id, model=MODEL):
    """Invoke the model through its inference profile (which can route to other US regions, hence the wildcard region on
    the foundation model) and retrieve from the one knowledge base."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:GetInferenceProfile",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region}:{account}:inference-profile/{model}",
                    f"arn:aws:bedrock:*::foundation-model/{base_model_id(model)}",
                ],
            },
            {
                "Effect": "Allow",
                "Action": "bedrock:Retrieve",
                "Resource": f"arn:aws:bedrock:{region}:{account}:knowledge-base/{kb_id}",
            },
        ],
    }


def kb_references(trace):
    """(document name, text) for each passage the agent's knowledge base lookup returned, from one trace event."""
    obs = trace.get("trace", {}).get("orchestrationTrace", {}).get("observation", {})
    refs = obs.get("knowledgeBaseLookupOutput", {}).get("retrievedReferences", [])
    out = []
    for r in refs:
        uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
        out.append(
            (uri.rsplit("/", 1)[-1] or "unknown", r.get("content", {}).get("text", ""))
        )
    return out
