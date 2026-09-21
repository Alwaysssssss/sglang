# syntax=docker/dockerfile:1
# CUDA runtime libraries come from the official PyTorch cu126 wheels.
ARG BASE_IMAGE=python:3.11.12-bookworm
FROM ${BASE_IMAGE}
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/sglang/python \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor TRITON_CACHE_DIR=/cache/triton \
    HF_HOME=/cache/huggingface \
    VSR_MODE=single VSR_PORT=30176
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates libnuma1 libibverbs1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/sglang
COPY docs_always/add_new_mode/add_vsr/docker/requirements.txt /opt/vsr/requirements.txt
COPY docs_always/add_new_mode/add_vsr/docker/constraints.txt /opt/vsr/constraints.txt
RUN python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --no-cache-dir -c /opt/vsr/constraints.txt -r /opt/vsr/requirements.txt
# build.sh stages the locally validated torch 2.10 kernel and verifies its hash.
COPY docs_always/add_new_mode/add_vsr/docker/vendor/ /opt/vsr/vendor/
RUN cd /opt/vsr/vendor && sha256sum -c SHA256SUMS \
    && python -m pip install --no-cache-dir --no-deps ./*.whl \
    && rm -f ./*.whl
COPY python/ /opt/sglang/python/
COPY docs_always/add_new_mode/add_vsr/scripts/serve_vsr.py /opt/vsr/serve_vsr.py
COPY docs_always/add_new_mode/add_vsr/scripts/test_server_api.py /opt/vsr/test_server_api.py
COPY docs_always/add_new_mode/add_vsr/docker/check_runtime.py /opt/vsr/check_runtime.py
COPY docs_always/add_new_mode/add_vsr/docker/warmup.py /opt/vsr/warmup.py
COPY docs_always/add_new_mode/add_vsr/docker/entrypoint.sh /opt/vsr/entrypoint.sh
# Use the checkout via PYTHONPATH: the generic package metadata pins torch 2.9.1.
# No GPU is needed for these import/version checks; CUDA kernels are checked at startup.
RUN python -m pip check && python /opt/vsr/check_runtime.py \
    && python -m pip freeze > /opt/vsr/installed-requirements.txt \
    && chmod +x /opt/vsr/entrypoint.sh && mkdir -p /output /cache
EXPOSE 30176
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=5 \
    CMD python -c "import os,urllib.request; urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://127.0.0.1:'+os.environ.get('VSR_PORT','30176')+'/health',timeout=3)"
ENTRYPOINT ["/opt/vsr/entrypoint.sh"]
