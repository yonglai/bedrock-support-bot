"""Step 10c: how much do the grounding scores change between two runs of the same inputs?
Compares two result files from step 10b (second run saved with OUT=...). Only the 57 clean and injected items are compared:
their answers and sources are identical in both runs. The Q501/Q506 items are skipped because step 10b re-runs the agent for them.
Run:   uv run python src/step10c_repeat_check.py eval/results_grounding_tiers.json eval/results_grounding_tiers_run2.json
"""

import json
import sys
from pathlib import Path

from grounding_tiers_common import TIERS

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    rows = json.loads((ROOT / path).read_text())
    return {
        (r["id"], r["kind"], i): r
        for i, r in enumerate(rows)
        if not r["kind"].startswith("real")
    }


a, b = load(sys.argv[1]), load(sys.argv[2])
keys = [k for k in a if k in b]
print(f"{len(keys)} items compared")
for tier in TIERS:
    flips = [k for k in keys if a[k][tier]["blocked"] != b[k][tier]["blocked"]]
    diffs = [
        abs(a[k][tier]["GROUNDING"] - b[k][tier]["GROUNDING"])
        for k in keys
        if "GROUNDING" in a[k][tier] and "GROUNDING" in b[k][tier]
    ]
    print(
        f"{tier:9} block decision changed: {len(flips)}/{len(keys)}   "
        f"score changed: {sum(d > 0 for d in diffs)}/{len(diffs)}   max change: {max(diffs, default=0):.2f}"
    )
    for k in flips:
        print("   flipped:", k[0], k[1])
