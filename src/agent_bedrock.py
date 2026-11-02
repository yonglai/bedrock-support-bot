"""The agent on a managed Bedrock Agent. Bedrock runs the loop, searches the attached knowledge base itself and returns
control to this code to run the customer-specific functions, so identity stays in our Session, not in a Lambda. Everything
else is shared with the other drivers: triage, input guardrail, output check, retry and handoff live in agent.run_agent.

Limits worth knowing: the conversation lives server-side under a session id (a new driver starts a new session, so the
`history` argument is not replayed), and a managed agent has no per-call tool_choice, so "search first" is a prompt plus
one nudge, not a guarantee.

Run:  AGENT_DRIVER=bedrock uv run python src/step5_eval_agent.py agent        (after step8a_agent_create.py)
"""

import json
import uuid

import boto3

from agent import StepLimit
from bedrock_agent_common import kb_references
from config import REGION
from rag_common import ROOT

STATE = ROOT / ".agent.json"
NUDGE = "Look this up in the policy knowledge base before answering."


class AgentsDriver:
    def __init__(self, sess, history, max_steps=6, rt=None, state=None):
        self.sess, self.max_steps = sess, max_steps
        self.state = state or json.loads(STATE.read_text())
        self.rt = rt or boto3.client("bedrock-agent-runtime", region_name=REGION)
        self.session_id = uuid.uuid4().hex
        self.looked_up = False

    def _invoke(self, kwargs):
        resp = self.rt.invoke_agent(
            agentId=self.state["agent_id"],
            agentAliasId=self.state["alias_id"],
            sessionId=self.session_id,
            enableTrace=True,
            **kwargs,
        )
        text, ret = [], None
        for ev in resp["completion"]:
            if "chunk" in ev:
                text.append(ev["chunk"]["bytes"].decode())
            elif "returnControl" in ev:
                ret = ev["returnControl"]
            elif "trace" in ev:
                refs = kb_references(ev["trace"])
                if refs:
                    self.looked_up = True
                    self.sess.tools_called.append("knowledge_base")
                    for doc, passage in refs:
                        self.sess.add_reference(doc, passage)
        return "".join(text), ret

    def _run_functions(self, ret):
        results = []
        for item in ret["invocationInputs"]:
            f = item["functionInvocationInput"]
            args = {p["name"]: p["value"] for p in f.get("parameters", [])}
            out = self.sess.call(f["function"], args)
            results.append(
                {
                    "functionResult": {
                        "actionGroup": f["actionGroup"],
                        "function": f["function"],
                        "responseBody": {"TEXT": {"body": out}},
                    }
                }
            )
        return {
            "invocationId": ret["invocationId"],
            "returnControlInvocationResults": results,
        }

    def reply(self, prompt, force_tool=False):
        kwargs, nudged = {"inputText": prompt}, False
        for _ in range(self.max_steps):
            text, ret = self._invoke(kwargs)
            if ret:
                kwargs = {"sessionState": self._run_functions(ret)}
                continue
            if force_tool and not self.looked_up and not nudged:
                nudged, kwargs = True, {"inputText": NUDGE}
                continue
            return text
        raise StepLimit
