# Build from the repository root. Keep dependency versions in python/pyproject.toml.
ARG BASE_IMAGE=nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.9 \
    MAX_JOBS=4 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv \
    build-essential cmake ninja-build pkg-config git ca-certificates \
    curl util-linux procps ffmpeg libnuma1 libnuma-dev libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# /usr/local/bin is also visible to the bash -lc used by the launch script.
RUN python3 -m venv /opt/venv \
    && ln -s /opt/venv/bin/python3 /usr/local/bin/python3 \
    && ln -s /opt/venv/bin/python /usr/local/bin/python
ENV PATH=/opt/venv/bin:${PATH}
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
RUN python3 -m pip install --index-url "${PIP_INDEX_URL}" \
    --upgrade pip setuptools wheel packaging ninja cmake numpy \
    && python3 -m pip install --index-url "${TORCH_INDEX_URL}" \
    'torch==2.9.1' 'torchaudio==2.9.1'

WORKDIR /sgl-workspace/sglang
COPY python/ ./python/
COPY README.md LICENSE ./python/
COPY scripts/videoedit_dual_service/ ./scripts/videoedit_dual_service/
COPY scripts/start_videoedit_container.sh ./scripts/start_videoedit_container.sh

# st_attn / vsa build against the already installed torch environment.
# This installs the local package, including uncommitted source changes.
RUN python3 -m pip install --index-url "${PIP_INDEX_URL}" \
    --no-build-isolation './python[diffusion]' \
    && ln -s /opt/venv/bin/sglang /usr/local/bin/sglang \
    && python3 -m pip check \
    && mkdir -p /opt/videoedit-build \
    && python3 -m pip freeze > /opt/videoedit-build/pip-freeze.txt

ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
EXPOSE 30000
CMD ["bash"]
