"""Show what search_policy returns for a few phrasings (needs the OpenSearch collection to be up).
Run:  uv run python src/step5b_debug_search.py            or with your own queries as arguments.
"""

import sys

from pipeline import retrieve

QUERIES = [
    "physiotherapy coverage",
    "physiotherapy coverage limits",
    "How much physiotherapy coverage do I have left this year?",
    "paramedical services physiotherapy chiropractic massage",
]

for q in sys.argv[1:] or QUERIES:
    print(f"\n{q}")
    for h in retrieve(q, 3):
        print(f"  {h['doc']:30} {h['text'][:80]!r}")
