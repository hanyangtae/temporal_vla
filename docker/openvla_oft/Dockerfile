# OpenVLA-OFT Inference & Evaluation
# Python 3.10, PyTorch 2.2.0 (openvla-oft 공식 권장), custom transformers fork
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# apt 미러를 카카오로 변경 (한국 내 빌드 속도 개선)
RUN sed -i 's|http://archive.ubuntu.com|http://mirror.kakao.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirror.kakao.com|g' /etc/apt/sources.list

RUN apt update && apt install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y --no-install-recommends \
    git wget curl vim build-essential \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils \
    sudo cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    libglew-dev libglfw3-dev libosmesa6-dev \
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

# PyTorch 2.2.0 (CUDA 12.1) — openvla-oft 공식 버전
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Flash Attention (openvla-oft 권장)
RUN pip install psutil packaging ninja
RUN pip install "flash-attn==2.5.5" --no-build-isolation

# openvla-oft 의존성 (requirements.txt로 관리)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
       --index-url https://download.pytorch.org/whl/cu121

CMD ["/bin/bash"]
