# Training / evaluation image for the W7 world-model comparison (no Isaac Sim needed).
# CUDA 12.8 builds of PyTorch are required for RTX 50-series GPUs.
#   docker build -t bmb-wm -f cloud/jepa/Dockerfile cloud/jepa
#   docker run --gpus all --shm-size 8g -v "$PWD":/work -w /work bmb-wm cloud/jepa/run_all.sh
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONUNBUFFERED=1
WORKDIR /work
