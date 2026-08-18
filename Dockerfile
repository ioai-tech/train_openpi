# syntax=docker/dockerfile:1.7
# ==========================================================================
#  OpenPI Training Image
#
#  Build (from this directory):
#    docker build \
#      --build-arg MODEL_TYPE=pi0 \
#      -t ioaitech/train_openpi:pi0-cuda126 \
#      -f Dockerfile .
#
#  Upstream openpi is cloned at build time (not a git submodule).
#  Pin OPENPI_GIT_REF to a commit; do not point published images at main.
#
#  Layer order: keep volatile ARG/LABEL (BUILD_VERSION, VCS_REF) after the
#  weight download. Using them earlier busts BuildKit cache and forces
#  docker pull to re-fetch the multi-GB checkpoint layer.
# ==========================================================================

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04@sha256:4cf7f8137bdeeb099b1f2de126e505aa1f01b6e4471d13faf93727a9bf83d539

FROM ${CUDA_IMAGE}

ARG OPENPI_GIT_URL=https://github.com/Physical-Intelligence/openpi.git
ARG OPENPI_GIT_REF=15a9616a00943ada6c20a0f158e3adb39df2ccac
ARG MODEL_TYPE=pi0
ARG LEROBOT_VERSION=auto

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs curl ca-certificates \
        build-essential clang \
        linux-headers-generic \
        ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:${LD_LIBRARY_PATH:-}

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv

WORKDIR /app

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

RUN git clone --filter=blob:none "${OPENPI_GIT_URL}" /tmp/openpi \
    && git -C /tmp/openpi checkout --detach "${OPENPI_GIT_REF}" \
    && test "$(git -C /tmp/openpi rev-parse HEAD)" = "${OPENPI_GIT_REF}" \
    && git -C /tmp/openpi submodule update --init --recursive --depth 1 \
    && cp /tmp/openpi/uv.lock /tmp/openpi/pyproject.toml /app/ \
    && mkdir -p /app/packages/openpi-client \
    && cp /tmp/openpi/packages/openpi-client/pyproject.toml /app/packages/openpi-client/pyproject.toml \
    && cp -r /tmp/openpi/packages/openpi-client/src /app/packages/openpi-client/src \
    && cp -r /tmp/openpi/src /app/src \
    && cp -r /tmp/openpi/scripts /app/scripts \
    && cp -r /tmp/openpi/third_party /app/third_party \
    && cp -r /tmp/openpi/src/openpi/models_pytorch/transformers_replace /tmp/transformers_replace \
    && rm -rf /tmp/openpi

RUN --mount=type=cache,target=/root/.cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 \
    uv sync --frozen --no-install-project --no-dev

RUN uv pip install --python /.venv/bin/python pytest

RUN /.venv/bin/python -c \
    "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /tmp/transformers_replace/* {} \
    && rm -rf /tmp/transformers_replace

ENV OPENPI_DATA_HOME=/models
ENV PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app
ENV PATH="/.venv/bin:${PATH}"
ENV WANDB_MODE=disabled
ENV HF_HUB_OFFLINE=1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
ENV MODEL_TYPE=${MODEL_TYPE}
ENV LEROBOT_DATASET_VERSION=${LEROBOT_VERSION}

RUN pip install --no-deps google-cloud-storage 2>/dev/null || true

RUN python -c "\
from openpi.shared.download import maybe_download; \
maybe_download('gs://openpi-assets/checkpoints/${MODEL_TYPE}_base/params'); \
maybe_download('gs://big_vision/paligemma_tokenizer.model')"

COPY train_lerobot.py /app/train_lerobot.py
COPY lerobot_v3_compat.py /app/lerobot_v3_compat.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod 0755 /app/entrypoint.sh \
    && mkdir -p /data/input /data/output \
    && python -c "import jax; print('jax', jax.__version__)"

VOLUME ["/data/input", "/data/output"]
ENTRYPOINT ["/app/entrypoint.sh"]

# Volatile metadata only — do not move above the weight RUN.
ARG BUILD_VERSION=dev
ARG VCS_REF=unknown
ARG OPENPI_GIT_REF
ARG MODEL_TYPE
LABEL org.opencontainers.image.title="IOAI OpenPI Trainer" \
      org.opencontainers.image.description="OpenPI (Pi0 / Pi0.5) training for LeRobot v2/v3 datasets" \
      org.opencontainers.image.source="https://github.com/ioai-tech/train_openpi" \
      org.opencontainers.image.documentation="https://github.com/ioai-tech/train_openpi#readme" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      io.ioai.train-openpi.upstream-commit="${OPENPI_GIT_REF}" \
      io.ioai.train-openpi.model-type="${MODEL_TYPE}"
