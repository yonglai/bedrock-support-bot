"""Agent tools. Identity comes from the session, never from the model: get_my_plan() takes no customer id,
so the model cannot ask for another customer's data."""

import json

from handoff import REASONS, create_handoff
from pipeline import retrieve
from rag_common import ROOT

CUSTOMERS = json.loads((ROOT / "data/customers.json").read_text())

TOOL_SPECS = [
    {
        "toolSpec": {
            "name": "search_policy",
            "description": "Search the Evergreen Life benefits documents. Use for plan rules, limits and procedures.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to look up"}
                    },
                    "required": ["query"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "get_my_plan",
            "description": (
                "Get the current customer's own plan, dependants and amounts used this year. "
                "Takes no arguments; it only ever returns the customer you are talking to."
            ),
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
    },
    {
        "toolSpec": {
            "name": "escalate_to_human",
            "description": (
                "Hand the conversation to a human team member. Use when the customer is distressed, grieving, "
                "complaining, appealing a decision, asks for a person, or when the tools cannot answer the question."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "enum": REASONS},
                        "summary": {
                            "type": "string",
                            "description": "One or two sentences for the team member, with no personal identifiers",
                        },
                    },
                    "required": ["reason", "summary"],
                }
            },
        }
    },
]


class Session:
    def __init__(self, customer_id):
        self.customer_id = customer_id
        self.sources = []  # text of every tool result, used as the grounding source
        self.n = 0
        self.escalation = None
        self.tools_called = []

    def search_policy(self, query):
        lines = []
        for hit in retrieve(query, 3):
            self.n += 1
            lines.append(f"[S{self.n}] ({hit['doc']}) {hit['text']}")
        return "\n".join(lines) or "No matching policy text found."

    def get_my_plan(self):
        customer = CUSTOMERS.get(self.customer_id)
        if not customer:
            return "No plan found for this customer."
        return "Customer data (current customer only): " + json.dumps(customer)

    def escalate_to_human(self, reason, summary):
        reason = reason if reason in REASONS else "other"
        self.escalation = create_handoff(reason, summary, self.customer_id)
        return "Handoff created. " + self.escalation["availability"]

    def call(self, name, args):
        self.tools_called.append(name)
        try:
            if name == "search_policy":
                out = self.search_policy(str(args.get("query", "")))
            elif name == "get_my_plan":
                out = self.get_my_plan()  # any customer_id the model passes is ignored
            elif name == "escalate_to_human":
                out = self.escalate_to_human(
                    args.get("reason", "other"), str(args.get("summary", ""))[:500]
                )
            else:
                out = f"Unknown tool: {name}"
        except Exception as e:  # noqa: BLE001 - a tool failure goes back to the model as text
            out = f"Tool error: {e}"
        if name != "escalate_to_human":
            self.sources.append(out)
        return out
