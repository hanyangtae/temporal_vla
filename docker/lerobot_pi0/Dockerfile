# LeRobot pi0 / pi05 전용
# flash-attn 미설치: patched transformers(fix/lerobot_openpi)와 ABI 불일치로 크래시
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
ENV TORCHINDUCTOR_CACHE_DIR=/temporal_vla/tmp/torchinductor

RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Inference server
RUN pip install fastapi uvicorn colorlog

# LeRobot + pi0 policy + simulation envs
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# pi0 필수: patched transformers
RUN pip install "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"

CMD ["/bin/bash"]
