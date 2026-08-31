# Base image: NVIDIA CUDA 12.8 Development image for RTX 5090 (Blackwell architecture)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
    CONFIG_PATH="conf/voxcpm_v2/voxcpm_finetune_lora.yaml" \
    PRETRAINED_PATH="/workspace/VoxCPM/pretrained_models/VoxCPM2" \
    TRAIN_MANIFEST="/workspace/VoxCPM/data/merged/train.jsonl" \
    VAL_MANIFEST="/workspace/VoxCPM/data/merged/val.jsonl" \
    BATCH_SIZE=4 \
    GRAD_ACCUM_STEPS=4 \
    LEARNING_RATE=0.0001 \
    SAVE_PATH="/workspace/VoxCPM/checkpoints/lora" \
    TENSORBOARD="/workspace/VoxCPM/logs/lora" \
    LORA_R=32 \
    LORA_ALPHA=32 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0" \
    CUDA_HOME=/usr/local/cuda

WORKDIR /workspace/VoxCPM

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-setuptools \
    git \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python3 -m pip install --no-cache-dir \
    "torch>=2.6.0" \
    "torchaudio>=2.6.0" \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY pyproject.toml README.md /workspace/VoxCPM/
COPY src /workspace/VoxCPM/src

RUN python3 -m pip install --no-cache-dir -e .

COPY . /workspace/VoxCPM

RUN chmod +x /workspace/VoxCPM/docker_entrypoint.sh

ENTRYPOINT ["/workspace/VoxCPM/docker_entrypoint.sh"]