# Base Image: CUDA 12.1 for PyTorch compatibility
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Arguments from docker-compose
ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 1. Install Python 3.11 via deadsnakes PPA + System Dependencies
RUN apt update && apt install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    sudo \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libglib2.0-0 \
    x11-apps \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libglu1-mesa \
    xfce4 \
    xfce4-goodies \
    libx11-dev \
    libxkbfile-dev \
    libsecret-1-dev \
    libgbm-dev \
    libnotify4 \
    libnss3 \
    libxss1 \
    libasound2 \
    xfonts-base \
    xfonts-100dpi \
    xfonts-75dpi \
    xfonts-cyrillic \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 2. Install KasmVNC (Ubuntu 22.04 Jammy)
RUN curl -L -O https://github.com/kasmtech/KasmVNC/releases/download/v1.3.1/kasmvncserver_jammy_1.3.1_amd64.deb && \
    apt update && \
    apt install -y ./kasmvncserver_jammy_1.3.1_amd64.deb && \
    rm kasmvncserver_jammy_1.3.1_amd64.deb && \
    rm -rf /var/lib/apt/lists/*

# 3. User Setup
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 4. Environment Setup
USER ${USER_NAME}
WORKDIR /temporal_vla

# Add local bin to PATH
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# 5. Python Setup
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1) - custom index url, cannot go in requirements.txt
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2 - requires --no-build-isolation, cannot go in requirements.txt
# psutil/packaging/ninja must be installed first as flash-attn needs them at build time
RUN pip install psutil packaging ninja
RUN pip install flash-attn --no-build-isolation

# Install EGL dependencies - requires system CMake, must come before requirements.txt
RUN pip install "egl_probe>=1.0.1" "hf-egl-probe>=1.0.2"

# Install all remaining dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Copy start script (if not mounted) - but we will mount it via docker-compose usually.
# But for safety, let's assume it's mounted or copied.
# Since we created scripts/start_vnc.sh locally, we rely on volume mount or user to rebuild.
# The entrypoint will be the new script.

ENTRYPOINT ["/temporal_vla/scripts/start_vnc.sh"]
