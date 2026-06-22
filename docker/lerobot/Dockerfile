# LeRobot all-in-one: training, finetuning, evaluation.
# Uses the repo-pinned lerobot submodule, not the PyPI package.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

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
    python3.12 python3.12-dev python3.12-venv \
    sudo cmake pkg-config \
    libegl1 libgles2 libgl1-mesa-glx libglib2.0-0 libosmesa6 ffmpeg \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} 2>/dev/null || true && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_NAME}
WORKDIR /temporal_vla

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"
# A4000 (kanu) = sm_86. flash-attn(below) compiles for this arch at build time.
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV FLASH_ATTN_CUDA_ARCHS="86"

RUN pip install --upgrade pip setuptools wheel

# Build helpers used by flash-attn and some simulator dependencies.
RUN pip install psutil packaging ninja

# PyTorch (CUDA 12.8). Install the final torch ABI before flash-attn.
RUN pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# GR00T's Eagle backbone requests FlashAttention2. Build/install it after the
# final torch install so the native extension matches torch's ABI, and before
# LeRobot extras so pip sees the satisfied dependency.
RUN pip install --no-build-isolation --no-deps flash-attn

# Inference server
RUN pip install fastapi uvicorn colorlog

# LeRobot + policies + simulation envs from the repo-pinned submodule.
COPY --chown=${USER_ID}:${GROUP_ID} lerobot /opt/lerobot
COPY docker/lerobot/requirements.txt /tmp/requirements.txt
RUN pip install -e "/opt/lerobot[pi,groot,smolvla,pusht,libero]" && \
    pip install -r /tmp/requirements.txt

ENV PYTHONPATH="/temporal_vla:/temporal_vla/lerobot/src:/temporal_vla/scripts/utils"

CMD ["/bin/bash"]
