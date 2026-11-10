"""Pick the sweep configuration with the best selection-split detection (compare.py --stage select output).

Reads <out>/models/<prefix>*/selection.json and prints one JSON line {name, checkpoint, d, L, reg, w_pred, key_detect}
for the winner. The criterion is the W5 one: mean detection within 2 s of the key events on w4_events among
detector settings whose false-alarm rate on w4_clean half B is at most 2x the target (compare.best_row).

  python -m models.world_model_jepa.sweep_pick --out experiments/w7_jepa_wm --prefix Bs_
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--prefix", default="Bs_")
    args = p.parse_args()
    rows = []
    for f in sorted((Path(args.out) / "models").glob(f"{args.prefix}*/selection.json")):
        sel = json.loads(f.read_text())
        det = sel["detectors"][sel["model"]]
        cfg = sel["model_config"]
        rows.append({"name": sel["model"], "checkpoint": sel["checkpoint"], "d": cfg["d"], "L": cfg["L"], "reg": cfg["reg"],
                     "w_pred": sel["args"]["w_pred"],
                     "key_detect": det["key_detect_mean"], "far_verify_max": det["far_verify_max"],
                     "detector": f"{det['stat']} {det['weights']} clip {det['clip']} k {det['k']}"})
    if not rows:
        raise SystemExit(f"no {args.prefix}* selections in {args.out}/models")
    rows.sort(key=lambda r: -r["key_detect"])
    for r in rows:
        print(f"# {r['name']:24s} key detection {r['key_detect']:.3f}  ({r['detector']})", flush=True)
    print(json.dumps(rows[0]))


if __name__ == "__main__":
    main()
