# UP-VLA Training & Inference Server
# Show-o backbone based VLA with future prediction pretraining
# torch 2.2.1 + xformers 0.0.25 (incompatible with xvla/dreamvla containers)
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
    libgl1-mesa-glx libglib2.0-0 ffmpeg libegl1 libgl1-mesa-dev \
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

RUN pip install --upgrade pip "wheel<0.43" && pip install packaging==22.0 setuptools==57.5.0

# PyTorch 2.2.1 (CUDA 12.1) — required by xformers 0.0.25
RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# UP-VLA dependencies (xformers, deepspeed, diffusers, etc.)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt 
# 가벼운 추가 패키지 (바뀔 가능성 있음) → 빠르게 재빌드
RUN pip install --no-build-isolation pyhash && \
    pip install "hydra-core==1.1.1" uvicorn fastapi

CMD ["/bin/bash"]
