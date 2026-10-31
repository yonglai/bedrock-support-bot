"""Central config. Override with environment variables, never hard-code credentials."""
import os

REGION = os.environ.get("AWS_REGION", "us-east-1")
# Set after running step0_list_models.py, e.g.  export MODEL_ID=...
MODEL_ID = os.environ.get("MODEL_ID", "")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "")
