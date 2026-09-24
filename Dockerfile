# =============================================================================
# acne-ai-engine - Motor de inferencia CNN y analisis estadistico (FastAPI)
#
# Python 3.11 is pinned here as the specification requires, which also sidesteps
# the study machine's constraint: it only has Python 3.14, where PyTorch has no
# stable wheels yet.
# =============================================================================

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1 is needed by SciPy/NumPy; libjpeg/zlib keep Pillow able to decode JPEG/PNG;
# libgl1 + libglib2.0-0 are what OpenCV (the admission screener, pulled in by ultralytics) loads.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 libjpeg62-turbo zlib1g libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --upgrade pip && pip install .

# WITH_TORCH=1 (default) installs the real CNN stack, so MODEL_BACKEND=torch works out of the box.
# WITH_TORCH=0 builds a light image that can only serve the deterministic stub.
#
# The `.[torch]` extra is NOT used on purpose: it pulls facenet-pytorch, which pins torch<2.3 and
# would silently downgrade PyTorch. Versions below are the ones the thesis models were trained and
# evaluated with (informe-acne.md §6.4).
#
# TORCH_INDEX picks the wheel: CPU by default; https://download.pytorch.org/whl/cu128 for NVIDIA.
ARG WITH_TORCH=1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN if [ "$WITH_TORCH" = "1" ]; then \
      pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX" \
   && pip install timm==1.0.29 captum==0.9.0 scikit-learn==1.9.1 ultralytics==8.4.155 pyyaml; \
    fi

# Checkpoints are not baked into the image (160 MB, and they change with every retraining):
# docker-compose mounts ../acne-ai-engine/checkpoints here, read-only.
RUN mkdir -p /app/checkpoints /app/runs

RUN useradd --create-home --uid 10001 acne && chown -R acne:acne /app
USER acne

# Ultralytics writes its settings file under $HOME; keep it inside the container user's home.
ENV YOLO_CONFIG_DIR=/home/acne/.config/Ultralytics

EXPOSE 8000

# start-period is long because loading the three models on CPU takes a while.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
