"""Convert a reference rollout (.npz from isaac/record_reference.py) to JSON that Unity can load.

Every array becomes {"shape": [...], "data": [...]} with data flattened row-major (C order), so a C#
class with one field per array (e.g. `public NdArray obs;` where NdArray has `int[] shape; float[] data;`)
can be filled by JsonUtility. String / scalar fields are stored as plain values. Floats are written as the
shortest decimal that round-trips to the same float32, so parsing into `float` is exact.

    ~/miniforge3/envs/env_isaaclab/bin/python export/npz_to_json.py experiments/w3_reference/go2_flat.npz
"""

import json
import sys
from pathlib import Path

import numpy as np


def convert(npz_path: Path) -> Path:
    data = np.load(npz_path)
    out = {}
    for key in data.files:
        arr = data[key]
        if arr.dtype.kind in "US":  # strings
            out[key] = arr.tolist()
        elif arr.ndim == 0:
            out[key] = arr.item()
        else:
            flat = arr.reshape(-1)
            if arr.dtype.kind == "f":
                values = [float(np.format_float_positional(x, unique=True, trim="-")) for x in flat.astype(np.float32)]
            elif arr.dtype.kind == "b":
                values = [int(x) for x in flat.tolist()]
            else:
                values = flat.tolist()
            out[key] = {"shape": list(arr.shape), "data": values}
    json_path = npz_path.with_suffix(".json")
    json_path.write_text(json.dumps(out, separators=(",", ":")))
    return json_path


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(convert(Path(p)))
