"""Print a markdown table from eval_policy.py results (latest row per scenario/policy wins).

    python isaac/eval_summary.py experiments/v2_eval/results.jsonl
"""

import json
import sys

rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows[(r["scenario"], r["policy"])] = r
policies = list(dict.fromkeys(p for _, p in rows))
scenarios = list(dict.fromkeys(s for s, _ in rows))

for metric, fmt in [("success_rate", "{:.1%}"), ("tracking_error", "{:.3f}"), ("slip_speed", "{:.3f}")]:
    print(f"\n**{metric}**\n")
    print("| scenario | " + " | ".join(policies) + " |")
    print("|---|" + "---:|" * len(policies))
    for s in scenarios:
        cells = [fmt.format(rows[(s, p)][metric]) if (s, p) in rows else "–" for p in policies]
        print(f"| {s} | " + " | ".join(cells) + " |")
