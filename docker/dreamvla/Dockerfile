# DreamVLA Training & Inference Server
# Isolated environment to avoid dependency conflicts with robocasa
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt update && apt install -y --no-install-recommends \
    git wget curl vim build-essential \
    python3 python3-dev python3-venv python3-distutils \
    sudo cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} 2>/dev/null || true && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_NAME}
WORKDIR /temporal_vla

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Flash Attention
RUN pip install psutil packaging ninja
RUN pip install flash-attn --no-build-isolation

# DreamVLA dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# LeRobot (dataset loading only — installed from submodule at runtime)
# Dependencies: datasets, pyarrow, av, accelerate, jsonlines
RUN pip install --no-cache-dir datasets pyarrow av accelerate jsonlines

CMD ["/bin/bash"]
