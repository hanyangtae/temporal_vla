# LeRobot all-in-one: training, finetuning, evaluation
# Supports pi0, groot policies + pusht, libero simulation environments
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && apt install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y --no-install-recommends \
    git wget curl vim build-essential \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils \
    sudo cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} 2>/dev/null || true && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_NAME}
WORKDIR /temporal_vla

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Flash Attention
RUN pip install psutil packaging ninja
RUN pip install flash-attn --no-build-isolation

# Inference server
RUN pip install fastapi uvicorn colorlog

# LeRobot + policies + simulation envs
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# pi0 / pi05 policies 는 transformers fork (fix/lerobot_openpi) 가 필요.
# 다른 extras(groot, smolvla 등) 가 공식 transformers 로 덮어쓰므로 마지막에 재설치.
# fork 는 tokenizers>=0.21,<0.22 를 요구하므로 같이 다운그레이드.
RUN pip install --force-reinstall --no-deps \
    "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" && \
    pip install "tokenizers>=0.21,<0.22"

# requirements 가 torch 를 2.5.1 → 2.10 로 업그레이드시키는데, 먼저 설치한 flash-attn
# prebuilt wheel 은 torch 2.5.1 ABI 에 link 되어있어 undefined symbol 로 import 실패.
# 추론 정확도에 영향 없으므로 flash-attn 제거 (transformers 는 eager attention 으로 fallback).
RUN pip uninstall -y flash-attn

CMD ["/bin/bash"]
