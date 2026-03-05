# CALVIN Simulation & Data Preprocessing Container
# Provides: CALVIN env (PyBullet), calvin_env, calvin_models, tensorflow-datasets
# Used for: data preprocessing (process_calvin.py), CALVIN evaluation rollout
#
# Build context: repo root (context: .) to access calvin/ submodule
# Usage: docker compose run --rm calvin python /temporal_vla/upvla/preprocess_data/process_calvin.py

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# System packages
# - libosmesa6: OSMesa software renderer (required by tacto's pyopengl fork for headless)
# - libgl/libegl: OpenGL/EGL for PyBullet headless rendering
RUN apt update && apt install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && apt install -y --no-install-recommends \
    git wget curl vim build-essential \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils \
    sudo pkg-config \
    libgl1-mesa-dev libgles2-mesa-dev libegl1-mesa-dev \
    libosmesa6-dev libglib2.0-0 \
    ffmpeg \
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

RUN pip install --upgrade pip wheel

# cmake==3.18.4 required by MulticoreTSNE (higher versions break its build)
RUN pip install cmake==3.18.4 setuptools==57.5.0

# PyTorch 2.2.1 (CUDA 12.1)
# calvin_models requires torch>=1.9; we install 2.2.1 to match upvla container
RUN pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ── tacto (touch sensor sim, required by calvin_env) ──────────────
# Needs a forked PyOpenGL for headless OSMesa support (built from git)
COPY --chown=${USER_NAME}:${GROUP_ID} calvin/calvin_env/tacto /tmp/tacto
RUN pip install /tmp/tacto

# ── calvin_env (PyBullet-based robot sim environment) ─────────────
COPY --chown=${USER_NAME}:${GROUP_ID} calvin/calvin_env /tmp/calvin_env
RUN pip install /tmp/calvin_env

# ── calvin_models (baselines; installed without torch==1.13.1 pin) ─
COPY --chown=${USER_NAME}:${GROUP_ID} calvin/calvin_models /tmp/calvin_models
RUN pip install --no-deps /tmp/calvin_models && \
    pip install --no-build-isolation pyhash MulticoreTSNE && \
    pip install \
        "hydra-core==1.1.1" hydra-colorlog omegaconf \
        plotly \
        "pytorch-lightning==1.8.6" lightning_lite \
        sentence-transformers moviepy \
        termcolor wandb gitpython

# ── tensorflow-datasets (for process_calvin.py) ────────────────────
RUN pip install tensorflow-cpu tensorflow-datasets

# ── numpy<2 required by PyBullet (compiled against NumPy 1.x) ─────
# tensorflow-datasets pulls in numpy 2.x which breaks pybullet at runtime
RUN pip install "numpy<2"

CMD ["/bin/bash"]
