ARG BASE_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260601-373cadc9@sha256:0068fe3bf3f78f42d3d18314e8b0b6d7a721a4968b37a7d689257a6ea3915d36
FROM ${BASE_IMAGE}

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/zsh devuser

# Give devuser sudo access
RUN apt-get update && apt-get install -y sudo ffmpeg curl util-linux procps zsh && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# VideoEdit text processing and object storage dependencies.
# Keep these versions aligned with python/pyproject.toml. Install before
# switching users so both root service processes and devuser can import them.
RUN python3 -m pip install --no-cache-dir \
    ftfy==6.3.1 boto3==1.43.102 minio==7.2.20 && \
    python3 -c "import ftfy, boto3, minio"

# Copy optional shell/editor configuration; base image contents may vary.
# Keep actual copy errors fatal instead of masking them with `|| true`.
RUN set -eu; \
    for config in .oh-my-zsh .zshrc .vimrc .tmux.conf; do \
        if [ -e "/root/$config" ]; then \
            cp -R "/root/$config" /home/devuser/; \
        fi; \
    done; \
    if [ -f /home/devuser/.zshrc ]; then \
        sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc; \
    fi; \
    chown -R devuser:devuser /home/devuser/

# The repository is bind-mounted at runtime. Only change directory ownership;
# recursively chowning the base image's workspace is expensive and unnecessary.
WORKDIR /sgl-workspace/sglang
RUN chown devuser:devuser /sgl-workspace /sgl-workspace/sglang

# Switch to devuser
USER devuser

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
