"""Isaac Lab's RSL-RL play.py with the Break My Bot tasks registered (same CLI; also exports ONNX).

    python isaac/play.py --task BMB-Go2-Rough-v2-Play --load_run <run> --checkpoint <model_N.pt>
"""

from _launch import run

run("play.py")
