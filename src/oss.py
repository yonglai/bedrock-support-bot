"""OpenSearch Serverless helpers: names, saved state, signed client, index creation."""

import json
import pathlib
import time

import boto3
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import AuthorizationException, RequestError

from config import REGION

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE = ROOT / ".opensearch.json"
GROUP, COLLECTION, INDEX = "support-bot-grp", "support-bot-kb", "policy-chunks"
POLICIES = {"encryption": "sb-enc", "network": "sb-net", "data": "sb-data"}
DIMS = 512


def state():
    return json.loads(STATE.read_text())


def client():
    host = state()["endpoint"].replace("https://", "")
    auth = AWSV4SignerAuth(boto3.Session().get_credentials(), REGION, "aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )


def create_index(c, recreate=False):
    if recreate and c.indices.exists(index=INDEX):
        c.indices.delete(index=INDEX)
    props = {
        "embedding": {"type": "knn_vector", "dimension": DIMS},
        "text": {"type": "text"},
        "doc": {"type": "keyword"},
    }
    bodies = [
        {"settings": {"index": {"knn": True}}, "mappings": {"properties": props}},
        {"mappings": {"properties": props}},
    ]  # fallback if the setting is rejected
    deadline = time.time() + 300
    while True:
        try:
            for i, body in enumerate(bodies):
                try:
                    c.indices.create(index=INDEX, body=body)
                    print(
                        "index created" + (" (without index.knn setting)" if i else "")
                    )
                    return
                except RequestError as e:
                    if "resource_already_exists" in str(e):
                        print("index already exists")
                        return
                    if i == len(bodies) - 1:
                        raise
                    print("retrying without index.knn setting:", str(e)[:120])
        except AuthorizationException:
            if time.time() > deadline:
                raise
            print("data access policy still propagating (403), waiting 15s ...")
            time.sleep(15)
