FROM lmsysorg/sglang:dev

# Create non-root user for development.
# UID/GID will be automatically adjusted at container start
# by devcontainer's updateRemoteUserUID to match the host user.
RUN useradd -m -s /bin/zsh devuser

# Give devuser sudo access & install Node.js 20.x (Claude Code requires >= 18)
RUN apt-get update && apt-get install -y sudo curl ca-certificates gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Install AI coding assistants globally.
RUN npm install -g @openai/codex @anthropic-ai/claude-code

# Set up oh-my-zsh for devuser
RUN cp -r /root/.oh-my-zsh /home/devuser/.oh-my-zsh && \
    cp /root/.zshrc /home/devuser/.zshrc && \
    (cp /root/.vimrc /home/devuser/.vimrc 2>/dev/null || true) && \
    (cp /root/.tmux.conf /home/devuser/.tmux.conf 2>/dev/null || true) && \
    sed -i 's|/root/.oh-my-zsh|/home/devuser/.oh-my-zsh|g' /home/devuser/.zshrc && \
    chown -R devuser:devuser /home/devuser/

# Set workspace directory and ownership
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:devuser /sgl-workspace

# Install diffusion dependencies directly (avoid rebuilding sglang which needs Rust)
RUN pip install --no-deps \
    "PyYAML==6.0.1" \
    "cloudpickle==3.1.2" \
    "diffusers==0.37.0" \
    "imageio==2.36.0" \
    "imageio-ffmpeg==0.5.1" \
    "moviepy>=2.0.0" \
    "opencv-python-headless==4.10.0.84" \
    "remote-pdb==2.1.0" \
    "st_attn==0.0.7" \
    "vsa==0.0.4" \
    "runai_model_streamer>=0.15.7" \
    "cache-dit==1.3.0" \
    "addict==2.4.0" \
    "av==16.1.0" \
    "scikit-image==0.25.2" \
    "trimesh>=4.0.0" \
    "xatlas"

# Switch to devuser
USER devuser

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
