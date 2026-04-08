# GR00T N1.6 Inference Server
# Isaac-GR00T 공식 레포 기반, ZMQ 서버로 모델 서빙
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 1. System dependencies + Python 3.10
RUN apt update && apt install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    sudo \
    cmake \
    pkg-config \
    libegl1-mesa-dev \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# 2. User setup
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_NAME}
WORKDIR /temporal_vla
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# 3. Python packages
RUN pip install --upgrade pip setuptools wheel

# PyTorch 2.7.1 (CUDA 12.6)
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

# Flash Attention 2
RUN pip install psutil packaging ninja
RUN pip install flash-attn --no-build-isolation

# Isaac-GR00T dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Isaac-GR00T 패키지 설치 (volume mount 후 editable install)
# 컨테이너 시작 후 수동: pip install -e /temporal_vla/src/policies/Isaac-GR00T --no-deps

CMD ["/bin/bash"]
