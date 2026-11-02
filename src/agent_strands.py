"""The agent loop on the Strands Agents SDK (AWS's open-source agent framework).
Everything else is shared with the hand-written agent: the system prompt, the tools (they call Session.call, so the tool
results, the grounding sources and the tool log are identical), and triage, input guardrail, output check, retry and
handoff, which live in agent.run_agent. Only the loop differs.

Run:  AGENT_DRIVER=strands MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 uv run python src/step5_eval_agent.py agent
Needs:  uv pip install strands-agents
"""

from strands import Agent, tool
from strands.models import BedrockModel

from agent import AGENT_SYSTEM, StepLimit
from config import MODEL_ID, REGION


class ForcedFirstToolModel(BedrockModel):
    """Strands has no per-call tool_choice on Agent.__call__, so the model wrapper forces search_policy on the first
    model call of a conversation (the same rule as the hand-written loop) and then lets the model choose."""

    force_tool = None

    async def stream(
        self,
        messages,
        tool_specs=None,
        system_prompt=None,
        *,
        tool_choice=None,
        **kwargs,
    ):
        if self.force_tool and tool_choice is None:
            tool_choice = {"tool": {"name": self.force_tool}}
            self.force_tool = None
        async for event in super().stream(
            messages, tool_specs, system_prompt, tool_choice=tool_choice, **kwargs
        ):
            yield event


def make_tools(sess):
    """Strands builds the tool schema from the function signature and docstring. The identity rule is unchanged:
    get_my_plan takes no argument, so the model cannot ask for another customer."""

    @tool
    def search_policy(query: str) -> str:
        """Search the Evergreen Life benefits documents. Use for plan rules, limits and procedures.

        Args:
            query: What to look up
        """
        return sess.call("search_policy", {"query": query})

    @tool
    def get_my_plan() -> str:
        """Get the current customer's own plan, dependants and amounts used this year.
        Takes no arguments; it only ever returns the customer you are talking to."""
        return sess.call("get_my_plan", {})

    @tool
    def calculate(expression: str) -> str:
        """Exact arithmetic for amounts, e.g. '1500 - 900'. Use it for every sum, difference, percentage or remainder
        instead of calculating yourself.

        Args:
            expression: Numbers with + - * / and brackets
        """
        return sess.call("calculate", {"expression": expression})

    @tool
    def escalate_to_human(reason: str, summary: str) -> str:
        """Hand the conversation to a human team member. Use when the customer is distressed, grieving, complaining,
        appealing a decision, asks for a person, or when the tools cannot answer the question.

        Args:
            reason: One of bereavement, complaint, appeal, legal, asked_for_person, crisis, cannot_answer, guardrail, other
            summary: One or two sentences for the team member, with no personal identifiers
        """
        return sess.call("escalate_to_human", {"reason": reason, "summary": summary})

    return [search_policy, get_my_plan, calculate, escalate_to_human]


class StrandsDriver:
    def __init__(self, sess, history, max_steps=6):
        self.model = ForcedFirstToolModel(
            model_id=MODEL_ID, region_name=REGION, temperature=0.1, max_tokens=600
        )
        self.agent = Agent(
            model=self.model,
            system_prompt=AGENT_SYSTEM,
            tools=make_tools(sess),
            messages=list(history),
            callback_handler=None,  # no printing; run_agent decides what the customer sees
        )
        self.max_steps = max_steps

    def reply(self, prompt, force_tool=False):
        self.model.force_tool = "search_policy" if force_tool else None
        result = self.agent(prompt, limits={"turns": self.max_steps})
        if result.stop_reason == "limit_turns":
            raise StepLimit
        return "".join(b.get("text", "") for b in result.message.get("content", []))
