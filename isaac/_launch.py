"""Run one of Isaac Lab's stock RSL-RL scripts with the Break My Bot tasks registered.

Isaac Lab's train.py / play.py only import isaaclab_tasks, and hydra looks the task up in the gym registry
before anything else, so our gym ids must be registered in the same process first.
"""

import os
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_SCRIPTS = Path(os.environ.get("ISAACLAB_DIR", Path.home() / "IsaacLab")) / "scripts/reinforcement_learning/rsl_rl"


def run(script_name: str) -> None:
    script = RSL_RL_SCRIPTS / script_name
    sys.path.insert(0, str(RSL_RL_SCRIPTS))  # for the scripts' `import cli_args`
    sys.path.insert(0, str(REPO_ROOT))
    import isaac.bmb_tasks  # noqa: F401  (registers BMB-* gym ids)

    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")
