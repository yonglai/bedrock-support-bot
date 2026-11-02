"""Zip-root app.py for AgentCore Runtime (copied there by step9a_agentcore_deploy.py)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

from bedrock_agentcore import BedrockAgentCoreApp

from agentcore_app import handle

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload):
    return handle(payload)


if __name__ == "__main__":
    app.run()
