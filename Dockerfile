# Base Image: CUDA 12.1 for PyTorch compatibility
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Arguments from docker-compose
ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 1. Install System Dependencies (LeRobot requirements)
# - ffmpeg libs: Required for LeRobot video processing
# - libgl1: For OpenGL rendering (Simulators)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
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
    && rm -rf /var/lib/apt/lists/*

# 2. User Setup
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 3. Environment Setup
USER ${USER_NAME}
WORKDIR /temporal_vla

# Add local bin to PATH
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# 4. Python Setup
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1) - Pin version to avoid conflicts
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Pre-install Flash Attention build dependencies
RUN pip3 install psutil packaging ninja

# Install Flash Attention 2 (Optional but recommended for X-VLA)
RUN pip3 install flash-attn --no-build-isolation

# Pre-install tricky dependencies with system CMake (avoids conflict with pip cmake)
RUN pip3 install "egl_probe>=1.0.1" "hf-egl-probe>=1.0.2"

ENTRYPOINT ["/temporal_vla/scripts/setup_env.sh"]
CMD ["/bin/bash"]
