# CALVIN Simulation & Data Preprocessing Container
# Provides: CALVIN env (PyBullet), calvin_env, calvin_models, tensorflow-datasets
# Used for: data preprocessing (process_calvin.py), CALVIN evaluation rollout
#
# Build context: repo root (context: .) to access calvin/ submodule
# Usage: docker compose run --rm calvin python /temporal_vla/src/benchmarks/calvin/...
#
# Follows official CALVIN requirements: Python 3.8, torch==1.13.1+cu117

FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ARG USER_NAME
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# System packages
# - libosmesa6: OSMesa software renderer (required by tacto's pyopengl fork for headless)
# - libgl/libegl: OpenGL/EGL for PyBullet headless rendering
# Ubuntu 20.04 ships Python 3.8 by default; no PPA needed
RUN apt update && apt install -y --no-install-recommends \
    git wget curl vim build-essential \
    python3.8 python3.8-dev python3-distutils \
    sudo pkg-config \
    libgl1-mesa-dev libgles2-mesa-dev libegl1-mesa-dev \
    libosmesa6-dev libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

RUN groupadd -g ${GROUP_ID} ${USER_NAME} 2>/dev/null || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} 2>/dev/null || true && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER_NAME}
WORKDIR /temporal_vla

ENV PATH="/home/${USER_NAME}/.local/bin:${PATH}"

# pip<24.1: 최신 pip 는 old setuptools==57.5.0 소스 빌드를 거부 (PEP 517 호환성).
RUN pip install --upgrade 'pip<24.1' wheel

# cmake==3.18.4.post1 required by MulticoreTSNE (higher versions break its build).
# 3.18.4 는 PyPI 에서 yanked 되었으므로 .post1 사용.
# setuptools==57.5.0 required by pyhash (matches calvin_models/requirements.txt pin).
RUN pip install cmake==3.18.4.post1 setuptools==57.5.0

# PyTorch 1.13.1 (CUDA 11.7) — pinned by calvin_models/requirements.txt
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# ── tacto (touch sensor sim, required by calvin_env) ──────────────
# Installs a forked PyOpenGL (mmatl/pyopengl) for headless OSMesa/EGL support
COPY --chown=${USER_NAME}:${GROUP_ID} src/benchmarks/calvin/calvin_env/tacto /tmp/tacto
RUN pip install /tmp/tacto

# ── calvin_env (PyBullet-based robot sim environment) ─────────────
COPY --chown=${USER_NAME}:${GROUP_ID} src/benchmarks/calvin/calvin_env /tmp/calvin_env
RUN pip install /tmp/calvin_env

# ── calvin_models ─────────────────────────────────────────────────
# pyhash and MulticoreTSNE require --no-build-isolation (legacy build system)
COPY --chown=${USER_NAME}:${GROUP_ID} src/benchmarks/calvin/calvin_models /tmp/calvin_models
RUN pip install /tmp/calvin_models && \
    pip install --no-build-isolation pyhash MulticoreTSNE

# ── tensorflow-datasets (for process_calvin.py) ────────────────────
RUN pip install tensorflow-cpu tensorflow-datasets "pyarrow<15.0.0" fastparquet

# ── numpy<2 required by PyBullet (compiled against NumPy 1.x) ─────
# tensorflow-datasets may pull numpy 2.x which breaks pybullet at runtime
RUN pip install "numpy>=1.20,<1.24"

# ── typing_extensions upgrade (wandb/pytorch_lightning 호환) ──────
# wandb imports Generator from typing_extensions, requires >=4.6
RUN pip install --upgrade typing_extensions

CMD ["/bin/bash"]
