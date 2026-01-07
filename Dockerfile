# Base Image: CUDA 12.1 for PyTorch compatibility + Devel for compilation tools
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Arguments from docker-compose
ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 1. Install System Dependencies
# - libgl1-mesa-*: Required for OmniGibson/GUI rendering
# - git, wget, curl: Basic tools
# - python3-pip: Python environment
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
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libglib2.0-0 \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# 2. User Setup (Match Host User to avoid permission issues)
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 3. Environment Setup
USER ${USER_NAME}
WORKDIR /workspace/vla_tset

# Add local bin to PATH
ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# 4. Install Basic Python Tools
RUN pip3 install --upgrade pip setuptools wheel

# (Optional) Pre-install torch here if you want to bake it into the image
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

CMD ["/bin/bash"]

